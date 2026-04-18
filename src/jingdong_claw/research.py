from __future__ import annotations

import asyncio
import importlib
import os
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

from firecrawl import FirecrawlApp
from openai import OpenAI

from .config import ResearchConfig
from .models import DraftDocument, ProviderConfig
from .llm.localapi import LocalAPIClient
from .parser import parse_document_output


class ResearchError(RuntimeError):
    """Raised when deep-research-backed initial draft generation fails."""


DEFAULT_FIRECRAWL_BASE_URL = "https://api.firecrawl.dev"


def build_research_query(topic: str) -> str:
    return (
        "Produce a research-grounded literature survey on the following topic.\n"
        "The output should emphasize representative prior work, comparisons across methods, "
        "clear section structure, open problems, and recent developments where available.\n\n"
        f"Topic: {topic}"
    )


def _model_to_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped

    return {}


def _normalize_firecrawl_item(item: object) -> dict[str, str]:
    payload = _model_to_dict(item)
    url = str(payload.get("url") or "").strip()
    title = str(payload.get("title") or "").strip()
    markdown = str(
        payload.get("markdown")
        or payload.get("content")
        or payload.get("description")
        or title
    ).strip()
    return {
        "url": url,
        "title": title,
        "markdown": markdown,
    }


class _CompatibleFirecrawl:
    """Normalize Firecrawl SDK responses to the shape expected by deep_research_py."""

    def __init__(self, api_key: str, api_url: str | None = None) -> None:
        self.app = FirecrawlApp(api_key=api_key, api_url=api_url)

    async def search(
        self,
        query: str,
        timeout: int = 15000,
        limit: int = 5,
    ) -> dict[str, list[dict[str, str]]]:
        del timeout

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.app.search(query=query),
        )

        if isinstance(response, dict) and "data" in response:
            data = response.get("data")
            if isinstance(data, list):
                return {"data": [_normalize_firecrawl_item(item) for item in data][:limit]}
            return {"data": []}

        payload = _model_to_dict(response)
        candidates: list[object] = []
        for key in ("data", "web", "news"):
            items = payload.get(key)
            if isinstance(items, list):
                candidates.extend(items)

        return {
            "data": [
                item
                for item in (_normalize_firecrawl_item(candidate) for candidate in candidates[:limit])
                if item["url"] and item["markdown"]
            ]
        }


class _LocalResearchChatCompletions:
    def __init__(self, provider_config: ProviderConfig) -> None:
        self.provider_config = provider_config
        self.client = LocalAPIClient(provider_config)

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, str] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        **_: Any,
    ) -> SimpleNamespace:
        request_messages = list(messages)
        if response_format and response_format.get("type") == "json_object":
            request_messages = [
                {
                    "role": "system",
                    "content": (
                        "Return only a valid JSON object. "
                        "Do not include markdown fences or explanatory text."
                    ),
                },
                *request_messages,
            ]

        client = self.client
        if model != self.provider_config.model:
            client = LocalAPIClient(replace(self.provider_config, model=model))

        response = client.generate_text(
            request_messages,
            temperature=temperature if temperature is not None else 0.2,
            max_output_tokens=max_tokens if max_tokens is not None else max_completion_tokens,
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response.text))]
        )


class _LocalResearchClient:
    def __init__(self, provider_config: ProviderConfig) -> None:
        self.chat = SimpleNamespace(
            completions=_LocalResearchChatCompletions(provider_config)
        )


def build_research_client(provider_config: ProviderConfig) -> Any:
    if provider_config.provider == "localapi":
        return _LocalResearchClient(provider_config)
    return OpenAI(
        api_key=provider_config.api_key,
        base_url=provider_config.base_url,
        timeout=provider_config.request_timeout,
    )


@dataclass(slots=True)
class DeepResearchDraftGenerator:
    provider_config: ProviderConfig
    research_config: ResearchConfig

    def generate(self, topic: str) -> DraftDocument:
        return asyncio.run(self._generate_async(topic))

    async def _generate_async(self, topic: str) -> DraftDocument:
        firecrawl_base_url = (
            self.research_config.firecrawl_base_url or DEFAULT_FIRECRAWL_BASE_URL
        )
        os.environ["OPENAI_API_KEY"] = self.provider_config.api_key
        os.environ["OPENAI_KEY"] = self.provider_config.api_key
        os.environ["OPENAI_API_ENDPOINT"] = self.provider_config.base_url
        os.environ["FIRECRAWL_API_KEY"] = self.research_config.firecrawl_api_key
        os.environ["FIRECRAWL_BASE_URL"] = firecrawl_base_url
        module = importlib.import_module("deep_research_py.deep_research")
        module.firecrawl = _CompatibleFirecrawl(
            api_key=self.research_config.firecrawl_api_key,
            api_url=firecrawl_base_url,
        )
        client = build_research_client(self.provider_config)
        research_query = build_research_query(topic)

        try:
            research_results = await module.deep_research(
                query=research_query,
                breadth=self.research_config.breadth,
                depth=self.research_config.depth,
                concurrency=self.research_config.concurrency,
                client=client,
                model=self.provider_config.model,
            )
            if not research_results["learnings"] and not research_results["visited_urls"]:
                raise ResearchError("Deep research completed without Firecrawl learnings or visited URLs.")
            report = await module.write_final_report(
                prompt=research_query,
                learnings=research_results["learnings"],
                visited_urls=research_results["visited_urls"],
                client=client,
                model=self.provider_config.model,
            )
        except Exception as exc:
            raise ResearchError(f"Deep research failed for topic '{topic}': {exc}") from exc

        if not isinstance(report, str) or not report.strip() or report.strip() == "Error generating report":
            raise ResearchError(f"Deep research returned an empty report for topic '{topic}'.")

        return parse_document_output(report, fallback_title=f"Survey on {topic}")
