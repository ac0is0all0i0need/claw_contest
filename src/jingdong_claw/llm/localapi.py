from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterable
from types import SimpleNamespace

from ..models import ChatMessage, LLMResponse, ProviderConfig
from .base import BaseLLMClient, LLMError, run_with_retries


RESPONSES_LIFECYCLE_EVENTS = {
    "response.created",
    "response.in_progress",
    "response.queued",
    "response.output_item.added",
    "response.content_part.added",
    "response.content_part.done",
    "response.output_item.done",
    "response.audio.delta",
    "response.audio.done",
    "response.audio_transcript.delta",
    "response.audio_transcript.done",
    "response.reasoning.delta",
    "response.reasoning.done",
    "response.reasoning_summary.delta",
    "response.reasoning_summary.done",
    "rate_limits.updated",
}


def _event_value(event: object, key: str, default: object = None) -> object:
    if isinstance(event, dict):
        return event.get(key, default)
    return getattr(event, key, default)


def _normalize_role(role: str) -> str:
    if role in {"assistant", "system"}:
        return role
    return "user"


def build_responses_input(messages: list[ChatMessage]) -> dict[str, object]:
    input_items: list[dict[str, object]] = []
    instructions: list[str] = []

    for message in messages:
        role = _normalize_role(message["role"])
        content = message["content"].strip()
        if not content:
            continue

        if role == "system":
            instructions.append(content)
            continue

        input_items.append(
            {
                "role": role,
                "content": content,
            }
        )

    return {
        "input": input_items,
        "instructions": "\n\n".join(instructions).strip(),
    }


def _collect_output_text(payload: object) -> list[str]:
    texts: list[str] = []

    if isinstance(payload, str):
        return [payload]

    if isinstance(payload, list):
        for item in payload:
            texts.extend(_collect_output_text(item))
        return texts

    if not isinstance(payload, dict):
        return texts

    if payload.get("type") == "output_text" and isinstance(payload.get("text"), str):
        texts.append(payload["text"])

    output_text = payload.get("output_text")
    if isinstance(output_text, str):
        texts.append(output_text)

    text = payload.get("text")
    if isinstance(text, str) and payload.get("type") in {"response.output_text.done", "response.refusal.done"}:
        texts.append(text)

    for key in ("response", "item", "part", "output", "content"):
        if key in payload:
            texts.extend(_collect_output_text(payload[key]))

    return texts


def extract_responses_output_text(payload: object) -> str:
    texts = [text for text in _collect_output_text(payload) if text]
    if not texts:
        return ""

    deduped: list[str] = []
    for text in texts:
        if not deduped or deduped[-1] != text:
            deduped.append(text)
    return "".join(deduped)


def collect_stream_text(events: Iterable[object]) -> tuple[str, int]:
    chunks: list[str] = []
    emitted_text = ""
    event_count = 0

    for event in events:
        event_count += 1
        event_type = _event_value(event, "type", "")

        if not isinstance(event_type, str) or event_type in RESPONSES_LIFECYCLE_EVENTS:
            continue

        if event_type in {"response.output_text.delta", "response.refusal.delta"}:
            delta = _event_value(event, "delta", "")
            if not isinstance(delta, str):
                delta = ""
            if delta:
                chunks.append(delta)
                emitted_text = f"{emitted_text}{delta}"
            continue

        if event_type in {"response.output_text.done", "response.refusal.done", "response.completed"}:
            full_text = extract_responses_output_text(event)
            if full_text.startswith(emitted_text):
                remaining = full_text[len(emitted_text) :]
                if remaining:
                    chunks.append(remaining)
                    emitted_text = f"{emitted_text}{remaining}"
                continue
            if not emitted_text and full_text:
                chunks.append(full_text)
                emitted_text = full_text
            continue

        fallback_text = extract_responses_output_text(event)
        if fallback_text and fallback_text.startswith(emitted_text):
            remaining = fallback_text[len(emitted_text) :]
            if remaining:
                chunks.append(remaining)
                emitted_text = f"{emitted_text}{remaining}"

    return "".join(chunks), event_count


def iter_sse_payloads(response: object) -> Iterable[str]:
    event_lines: list[str] = []

    while True:
        raw_line = response.readline()
        if not raw_line:
            break

        line = raw_line.decode("utf-8", errors="replace").replace("\r\n", "\n").rstrip("\n")

        if line == "":
            payload_lines = [entry[5:].lstrip() for entry in event_lines if entry.startswith("data:")]
            payload = "\n".join(payload_lines).strip()
            if payload:
                yield payload
            event_lines = []
            continue

        event_lines.append(line)

    if event_lines:
        payload_lines = [entry[5:].lstrip() for entry in event_lines if entry.startswith("data:")]
        payload = "\n".join(payload_lines).strip()
        if payload:
            yield payload


class LocalAPIClient(BaseLLMClient):
    def generate_text(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        def _request() -> LLMResponse:
            payload = build_responses_input(messages)
            body: dict[str, object] = {
                "model": self.config.model,
                "input": payload["input"],
                "stream": True,
                "temperature": temperature,
            }
            instructions = payload["instructions"]
            if instructions:
                body["instructions"] = instructions
            if max_output_tokens is not None:
                body["max_output_tokens"] = max_output_tokens

            request = urllib.request.Request(
                f"{self.config.base_url.rstrip('/')}/responses",
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                    "Authorization": f"Bearer {self.config.api_key}",
                    "User-Agent": "jingdong-claw/0.1.0",
                },
                method="POST",
            )

            try:
                with urllib.request.urlopen(request, timeout=self.config.request_timeout) as response:
                    payload_count = 0
                    parsed_events: list[dict[str, object]] = []
                    for payload_text in iter_sse_payloads(response):
                        payload_count += 1
                        if payload_text == "[DONE]":
                            break
                        parsed_events.append(json.loads(payload_text))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace").strip()
                raise LLMError(f"Local API HTTP {exc.code}: {detail or exc.reason}") from exc
            except urllib.error.URLError as exc:
                raise LLMError(f"Local API connection failed: {exc.reason}") from exc

            streamed_text, event_count = collect_stream_text(parsed_events)
            text = streamed_text.strip()
            if not text:
                raise LLMError("Local API stream completed without any text output.")

            finish_reason = None
            for event in reversed(parsed_events):
                if event.get("type") == "response.completed":
                    response_payload = event.get("response")
                    if isinstance(response_payload, dict):
                        status = response_payload.get("status")
                        if isinstance(status, str):
                            finish_reason = status
                    break

            return LLMResponse(
                text=text,
                provider=self.config.provider,
                model=self.config.model,
                finish_reason=finish_reason,
                event_count=max(event_count, payload_count),
            )

        return run_with_retries(
            _request,
            provider=self.config.provider,
            max_retries=self.config.max_retries,
        )
