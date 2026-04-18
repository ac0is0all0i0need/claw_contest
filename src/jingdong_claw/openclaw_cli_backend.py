from __future__ import annotations

import ipaddress
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .models import ProviderConfig


class OpenClawCliBackendError(RuntimeError):
    """Raised when the OpenClaw CLI backend cannot complete a live run."""


@dataclass(slots=True)
class OpenClawCliToolResult:
    tool_name: str
    tool_call_id: str
    success: bool
    result: str
    error: str
    duration_ms: int


@dataclass(slots=True)
class OpenClawCliUsage:
    payload: dict[str, object]

    def model_dump(self) -> dict[str, object]:
        return self.payload


@dataclass(slots=True)
class OpenClawCliRunResult:
    request_id: str
    success: bool
    text: str
    error: str
    duration_ms: int
    usage: OpenClawCliUsage
    tool_results: list[OpenClawCliToolResult]
    raw_session_path: str


LOCALAPI_PROVIDER_ID = "localapi"
LOCALAPI_CONTEXT_TOKENS = 400_000
LOCALAPI_MAX_TOKENS = 128_000


def sync_workspace_skills(
    *,
    source_root: Path,
    workspace_root: Path,
    worker_skills: tuple[object, ...],
) -> Path:
    target_root = (workspace_root / "skills").resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    for spec in worker_skills:
        directory_name = str(getattr(spec, "directory_name"))
        source_dir = (source_root / directory_name).resolve()
        target_dir = (target_root / directory_name).resolve()

        if not source_dir.exists():
            continue
        if target_root not in target_dir.parents:
            raise OpenClawCliBackendError(f"refusing to sync skill outside workspace root: {target_dir}")
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)

    return target_root


def ensure_openclaw_profile(
    *,
    profile_name: str,
    workspace_root: Path,
    provider_config: ProviderConfig,
) -> Path:
    profile_root = (Path.home() / f".openclaw-{profile_name}").resolve()
    profile_root.mkdir(parents=True, exist_ok=True)
    config_path = profile_root / "openclaw.json"

    payload: dict[str, object] = {}
    if config_path.exists():
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise OpenClawCliBackendError(f"invalid OpenClaw profile config: {config_path}") from exc

    agents = payload.get("agents")
    if not isinstance(agents, dict):
        agents = {}
    defaults = agents.get("defaults")
    if not isinstance(defaults, dict):
        defaults = {}
    model = defaults.get("model")
    if not isinstance(model, dict):
        model = {}
    models = payload.get("models")
    if not isinstance(models, dict):
        models = {}
    providers = models.get("providers")
    if not isinstance(providers, dict):
        providers = {}

    defaults["workspace"] = str(workspace_root)
    model["primary"] = resolve_openclaw_model_id(provider_config)
    defaults["model"] = model
    agents["defaults"] = defaults
    payload["agents"] = agents
    if provider_config.provider == "localapi":
        providers[LOCALAPI_PROVIDER_ID] = _build_localapi_provider_config(provider_config)
        models["providers"] = providers
        payload["models"] = models

    config_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return profile_root


def build_openclaw_env(
    *,
    provider_config: ProviderConfig,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    env = dict(base_env or os.environ)
    if provider_config.provider == "deepseek":
        env["DEEPSEEK_API_KEY"] = provider_config.api_key
        env["DEEPSEEK_BASE_URL"] = provider_config.base_url
    else:
        env["OPENAI_API_KEY"] = provider_config.api_key
        env["OPENAI_BASE_URL"] = provider_config.base_url
    return env


def resolve_openclaw_model_id(provider_config: ProviderConfig) -> str:
    provider_prefix = "deepseek" if provider_config.provider == "deepseek" else LOCALAPI_PROVIDER_ID
    return f"{provider_prefix}/{provider_config.model}"


def _build_localapi_provider_config(provider_config: ProviderConfig) -> dict[str, object]:
    model_entry: dict[str, object] = {
        "id": provider_config.model,
        "name": provider_config.model,
        "reasoning": True,
        "input": ["text", "image"],
        "contextWindow": LOCALAPI_CONTEXT_TOKENS,
        "contextTokens": LOCALAPI_CONTEXT_TOKENS,
        "maxTokens": LOCALAPI_MAX_TOKENS,
    }
    provider_entry: dict[str, object] = {
        "api": "openai-responses",
        "baseUrl": provider_config.base_url,
        # Root-level `openclaw ...` launches do not inherit the Python-side env mapping
        # from LOCAL_API_KEY -> OPENAI_API_KEY, so persist the resolved key in the profile.
        "apiKey": provider_config.api_key,
        "models": [model_entry],
    }
    if _is_private_network_url(provider_config.base_url):
        provider_entry["request"] = {"allowPrivateNetwork": True}
    return provider_entry


def _is_private_network_url(url: str) -> bool:
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
    )


def list_openclaw_skills(
    *,
    profile_name: str,
    workspace_root: Path,
    env: dict[str, str],
) -> list[dict[str, object]]:
    command = _resolve_openclaw_command()
    completed = subprocess.run(
        [command, "--profile", profile_name, "skills", "list", "--json"],
        cwd=str(workspace_root),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        error_text = completed.stderr.strip() or completed.stdout.strip() or "skills list failed"
        raise OpenClawCliBackendError(error_text)

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise OpenClawCliBackendError("OpenClaw skills list did not return valid JSON.") from exc

    skills = payload.get("skills")
    if not isinstance(skills, list):
        raise OpenClawCliBackendError("OpenClaw skills list response is missing `skills`.")
    return [item for item in skills if isinstance(item, dict)]


def run_openclaw_local_agent(
    *,
    profile_name: str,
    agent_id: str,
    session_id: str,
    workspace_root: Path,
    prompt: str,
    env: dict[str, str],
    timeout_seconds: int,
    thinking_level: str,
    stdout_path: Path,
    stderr_path: Path,
) -> OpenClawCliRunResult:
    profile_root = (Path.home() / f".openclaw-{profile_name}").resolve()
    session_file = profile_root / "agents" / agent_id / "sessions" / f"{session_id}.jsonl"
    session_file.parent.mkdir(parents=True, exist_ok=True)
    if session_file.exists():
        session_file.unlink()

    command = _resolve_openclaw_command()
    command = [
        command,
        "--profile",
        profile_name,
        "agent",
        "--local",
        "--session-id",
        session_id,
        "--thinking",
        thinking_level,
        "--timeout",
        str(timeout_seconds),
        "--message",
        prompt,
    ]

    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
        "w",
        encoding="utf-8",
    ) as stderr_file:
        process = subprocess.Popen(
            command,
            cwd=str(workspace_root),
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            encoding="utf-8",
        )
        try:
            deadline = time.monotonic() + max(timeout_seconds, 30)
            final_message: dict[str, Any] | None = None
            events: list[dict[str, Any]] = []
            while time.monotonic() < deadline:
                if session_file.exists():
                    events = _load_session_events(session_file)
                    final_message = _find_terminal_assistant_message(events)
                    if final_message is not None:
                        time.sleep(1.0)
                        events = _load_session_events(session_file)
                        final_message = _find_terminal_assistant_message(events)
                        break
                if process.poll() is not None and not session_file.exists():
                    break
                time.sleep(1.0)

            if final_message is None:
                stderr_text = stderr_path.read_text(encoding="utf-8").strip() if stderr_path.exists() else ""
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                raise OpenClawCliBackendError(
                    stderr_text or "OpenClaw local agent did not produce a terminal assistant message."
                )

            _stop_process(process)
        finally:
            if process.poll() is None:
                _stop_process(process)

    last_user_index = _find_last_user_index(events)
    usage_payload = final_message.get("usage", {})
    if not isinstance(usage_payload, dict):
        usage_payload = {}
    response_id = str(final_message.get("responseId", "") or "")
    stop_reason = str(final_message.get("stopReason", "") or "")
    error_message = str(final_message.get("errorMessage", "") or "")
    success = stop_reason != "error" and not error_message

    return OpenClawCliRunResult(
        request_id=response_id,
        success=success,
        text=_extract_message_text(final_message),
        error=error_message,
        duration_ms=_compute_duration_ms(events, last_user_index, final_message),
        usage=OpenClawCliUsage(usage_payload),
        tool_results=_collect_tool_results(events, last_user_index),
        raw_session_path=str(session_file),
    )


def _resolve_openclaw_command() -> str:
    for candidate in ("openclaw.cmd", "openclaw.exe", "openclaw"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise OpenClawCliBackendError("OpenClaw CLI is not installed or not on PATH.")


def _stop_process(process: subprocess.Popen[str]) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _load_session_events(session_file: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw_line in session_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _find_last_user_index(events: list[dict[str, Any]]) -> int:
    for index in range(len(events) - 1, -1, -1):
        message = events[index].get("message")
        if not isinstance(message, dict):
            continue
        if message.get("role") == "user":
            return index
    return -1


def _find_terminal_assistant_message(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    last_user_index = _find_last_user_index(events)
    for event in reversed(events[last_user_index + 1 :]):
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue
        stop_reason = str(message.get("stopReason", "") or "")
        if stop_reason and stop_reason != "toolUse":
            return message
    return None


def _extract_message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if not isinstance(content, list):
        return ""
    text_parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            text = item.get("text")
            if isinstance(text, str):
                text_parts.append(text)
    return "\n".join(part.strip() for part in text_parts if part.strip()).strip()


def _parse_iso_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _compute_duration_ms(
    events: list[dict[str, Any]],
    last_user_index: int,
    final_message: dict[str, Any],
) -> int:
    start_timestamp: datetime | None = None
    for event in events[last_user_index + 1 :]:
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        if message.get("role") == "user":
            start_timestamp = _parse_iso_timestamp(str(event.get("timestamp", "") or ""))
            break

    if start_timestamp is None and last_user_index >= 0:
        start_timestamp = _parse_iso_timestamp(str(events[last_user_index].get("timestamp", "") or ""))

    end_timestamp = _parse_iso_timestamp(str(final_message.get("timestamp", "") or ""))
    if start_timestamp is None or end_timestamp is None:
        return 0
    return max(int((end_timestamp - start_timestamp).total_seconds() * 1000), 0)


def _collect_tool_results(
    events: list[dict[str, Any]],
    last_user_index: int,
) -> list[OpenClawCliToolResult]:
    tool_calls: dict[str, tuple[str, datetime | None]] = {}
    results: list[OpenClawCliToolResult] = []

    for event in events[last_user_index + 1 :]:
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        event_timestamp = _parse_iso_timestamp(str(event.get("timestamp", "") or ""))

        if role == "assistant":
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "toolCall":
                    continue
                tool_call_id = str(item.get("id", "") or "")
                tool_name = str(item.get("name", "") or "")
                if tool_call_id:
                    tool_calls[tool_call_id] = (tool_name, event_timestamp)
            continue

        if role != "toolResult":
            continue

        tool_call_id = str(message.get("toolCallId", "") or "")
        tool_name = str(message.get("toolName", "") or "")
        if tool_call_id in tool_calls:
            tool_name = tool_calls[tool_call_id][0] or tool_name
            start_timestamp = tool_calls[tool_call_id][1]
        else:
            start_timestamp = None

        result_text = _extract_message_text(message)
        is_error = bool(message.get("isError"))
        duration_ms = 0
        if start_timestamp is not None and event_timestamp is not None:
            duration_ms = max(int((event_timestamp - start_timestamp).total_seconds() * 1000), 0)

        results.append(
            OpenClawCliToolResult(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                success=not is_error,
                result="" if is_error else result_text,
                error=result_text if is_error else "",
                duration_ms=duration_ms,
            )
        )

    return results
