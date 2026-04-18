from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, TypedDict


ProviderName = Literal["localapi", "deepseek"]
RunMode = Literal["topic", "paper"]


class ChatMessage(TypedDict):
    role: str
    content: str


@dataclass(slots=True)
class ProviderConfig:
    provider: ProviderName
    model: str
    api_key: str
    base_url: str
    request_timeout: float = 120.0
    max_retries: int = 2


@dataclass(slots=True)
class DraftDocument:
    title: str
    body: str


@dataclass(slots=True)
class LLMResponse:
    text: str
    provider: ProviderName
    model: str
    finish_reason: str | None = None
    event_count: int = 0


@dataclass(slots=True)
class ScoreResult:
    dimension: str
    score: int
    reason: str
    raw_text: str


@dataclass(slots=True)
class RoundResult:
    round_index: int
    draft_title: str
    draft_body: str
    scores: list[ScoreResult]
    total_score: int
    average_score: float
    feedback_summary: str
    provider: str
    model: str


@dataclass(slots=True)
class RunResult:
    run_id: str
    mode: RunMode
    topic: str | None
    rounds: list[RoundResult] = field(default_factory=list)
    best_round_index: int = -1
    best_total_score: int = 0
    output_dir: str = ""
    stopped_early: bool = False
    stop_reason: str = "not_started"
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
