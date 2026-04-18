from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeVar

from ..models import ChatMessage, LLMResponse, ProviderConfig


class LLMError(RuntimeError):
    """Raised when a provider call fails or returns unusable output."""


_T = TypeVar("_T")


def run_with_retries(
    func: Callable[[], _T],
    *,
    provider: str,
    max_retries: int,
) -> _T:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as exc:  # pragma: no cover - exercised via provider integration
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(min(2**attempt, 4))
    raise LLMError(f"{provider} request failed after {max_retries + 1} attempts: {last_error}") from last_error


class BaseLLMClient(ABC):
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    @abstractmethod
    def generate_text(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        raise NotImplementedError
