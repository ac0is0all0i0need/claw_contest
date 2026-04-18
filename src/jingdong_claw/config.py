from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .models import ProviderConfig, ProviderName


class ConfigError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid."""


@dataclass(slots=True)
class ResearchConfig:
    breadth: int
    depth: int
    concurrency: int
    firecrawl_api_key: str
    firecrawl_base_url: str | None


@dataclass(slots=True)
class Settings:
    default_provider: ProviderName
    default_rounds: int
    output_root: Path
    request_timeout: float
    max_retries: int
    local_model: str | None
    local_api_key: str | None
    local_base_url: str | None
    deepseek_model: str | None
    deepseek_api_key: str | None
    deepseek_base_url: str | None
    research_breadth: int
    research_depth: int
    research_concurrency: int
    firecrawl_api_key: str | None
    firecrawl_base_url: str | None


def _clean_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        cleaned = value.strip().strip('"').strip("'")
        if cleaned:
            return cleaned
    return None


def _parse_int(name: str, default: int) -> int:
    raw = _clean_env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name} must be an integer.") from exc


def _parse_float(name: str, default: float) -> float:
    raw = _clean_env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name} must be a float.") from exc


def load_settings() -> Settings:
    load_dotenv(override=False)

    local_api_key = _clean_env("LOCAL_API_KEY")
    local_base_url = _clean_env("LOCAL_BASE_URL")
    deepseek_api_key = _clean_env("DEEPSEEK_API_KEY")
    deepseek_base_url = _clean_env("DEEPSEEK_BASE_URL")
    firecrawl_api_key = _clean_env("FIRECRAWL_API_KEY", "FIRECRAWL_API_KEK")
    firecrawl_base_url = _clean_env("FIRECRAWL_BASE_URL")

    local_model = _clean_env("LOCAL_MODEL", "JDCLAW_LOCAL_MODEL", "JDCLAW_MODEL")
    deepseek_model = _clean_env("DEEPSEEK_MODEL", "JDCLAW_DEEPSEEK_MODEL", "JDCLAW_MODEL") or "deepseek-chat"

    default_provider_raw = _clean_env("JDCLAW_PROVIDER")
    if default_provider_raw is None:
        # Prefer the hosted DeepSeek path for the recorded demo when both providers are configured.
        # The localapi route is still available via JDCLAW_PROVIDER=localapi or --provider localapi.
        default_provider = "deepseek" if deepseek_api_key and deepseek_base_url else "localapi"
    elif default_provider_raw in {"localapi", "deepseek"}:
        default_provider = default_provider_raw
    else:
        raise ConfigError("JDCLAW_PROVIDER must be either 'localapi' or 'deepseek'.")

    return Settings(
        default_provider=default_provider,
        default_rounds=_parse_int("JDCLAW_ROUNDS", 3),
        output_root=Path(_clean_env("JDCLAW_OUTPUT_DIR") or "runs"),
        request_timeout=_parse_float("JDCLAW_REQUEST_TIMEOUT", 120.0),
        max_retries=_parse_int("JDCLAW_MAX_RETRIES", 2),
        local_model=local_model,
        local_api_key=local_api_key,
        local_base_url=local_base_url,
        deepseek_model=deepseek_model,
        deepseek_api_key=deepseek_api_key,
        deepseek_base_url=deepseek_base_url,
        research_breadth=_parse_int("JDCLAW_RESEARCH_BREADTH", 3),
        research_depth=_parse_int("JDCLAW_RESEARCH_DEPTH", 2),
        research_concurrency=_parse_int("JDCLAW_RESEARCH_CONCURRENCY", 2),
        firecrawl_api_key=firecrawl_api_key,
        firecrawl_base_url=firecrawl_base_url,
    )


def resolve_provider_config(
    settings: Settings,
    *,
    provider: ProviderName | None = None,
    model: str | None = None,
) -> ProviderConfig:
    selected_provider = provider or settings.default_provider

    if selected_provider == "localapi":
        selected_model = model or settings.local_model
        missing = []
        if not settings.local_api_key:
            missing.append("LOCAL_API_KEY")
        if not settings.local_base_url:
            missing.append("LOCAL_BASE_URL")
        if not selected_model:
            missing.append("LOCAL_MODEL or JDCLAW_MODEL")
        if missing:
            missing_text = ", ".join(missing)
            raise ConfigError(f"Missing localapi configuration: {missing_text}.")
        return ProviderConfig(
            provider="localapi",
            model=selected_model,
            api_key=settings.local_api_key,
            base_url=settings.local_base_url,
            request_timeout=settings.request_timeout,
            max_retries=settings.max_retries,
        )

    selected_model = model or settings.deepseek_model
    missing = []
    if not settings.deepseek_api_key:
        missing.append("DEEPSEEK_API_KEY")
    if not settings.deepseek_base_url:
        missing.append("DEEPSEEK_BASE_URL")
    if not selected_model:
        missing.append("DEEPSEEK_MODEL or JDCLAW_MODEL")
    if missing:
        missing_text = ", ".join(missing)
        raise ConfigError(f"Missing deepseek configuration: {missing_text}.")
    return ProviderConfig(
        provider="deepseek",
        model=selected_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        request_timeout=settings.request_timeout,
        max_retries=settings.max_retries,
    )


def resolve_research_config(
    settings: Settings,
    *,
    breadth: int | None = None,
    depth: int | None = None,
    concurrency: int | None = None,
) -> ResearchConfig:
    selected_breadth = breadth if breadth is not None else settings.research_breadth
    selected_depth = depth if depth is not None else settings.research_depth
    selected_concurrency = concurrency if concurrency is not None else settings.research_concurrency

    if selected_breadth < 1:
        raise ConfigError("Research breadth must be at least 1.")
    if selected_depth < 1:
        raise ConfigError("Research depth must be at least 1.")
    if selected_concurrency < 1:
        raise ConfigError("Research concurrency must be at least 1.")
    if not settings.firecrawl_api_key:
        raise ConfigError("Missing Firecrawl configuration: FIRECRAWL_API_KEY (or FIRECRAWL_API_KEK).")

    return ResearchConfig(
        breadth=selected_breadth,
        depth=selected_depth,
        concurrency=selected_concurrency,
        firecrawl_api_key=settings.firecrawl_api_key,
        firecrawl_base_url=settings.firecrawl_base_url,
    )
