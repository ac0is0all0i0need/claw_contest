from __future__ import annotations

from openai import OpenAI

from ..models import ChatMessage, LLMResponse, ProviderConfig
from .base import BaseLLMClient, LLMError, run_with_retries


def _extract_chat_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


class StandardAPIClient(BaseLLMClient):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.request_timeout,
        )

    def generate_text(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        def _request() -> LLMResponse:
            request_kwargs: dict[str, object] = {
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_output_tokens is not None:
                request_kwargs["max_tokens"] = max_output_tokens

            response = self._client.chat.completions.create(**request_kwargs)
            if not response.choices:
                raise LLMError("Standard provider returned no choices.")

            choice = response.choices[0]
            text = _extract_chat_text(choice.message.content).strip()
            if not text:
                raise LLMError("Standard provider returned an empty message.")

            return LLMResponse(
                text=text,
                provider=self.config.provider,
                model=self.config.model,
                finish_reason=choice.finish_reason,
                event_count=1,
            )

        return run_with_retries(
            _request,
            provider=self.config.provider,
            max_retries=self.config.max_retries,
        )
