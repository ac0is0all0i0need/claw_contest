from __future__ import annotations

from ..models import ProviderConfig
from .base import BaseLLMClient, LLMError
from .localapi import LocalAPIClient
from .standard import StandardAPIClient


def build_client(config: ProviderConfig) -> BaseLLMClient:
    if config.provider == "localapi":
        return LocalAPIClient(config)
    if config.provider == "deepseek":
        return StandardAPIClient(config)
    raise ValueError(f"Unsupported provider: {config.provider}")


__all__ = ["BaseLLMClient", "LLMError", "build_client"]
