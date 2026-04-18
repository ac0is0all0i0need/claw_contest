from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal
from collections.abc import Callable

from prompt_templates import VALID_DIMENSIONS
from pydantic import BaseModel, ConfigDict

from .models import DraftDocument, ProviderConfig, ScoreResult
from .openclaw_compat import load_openclaw
from .parser import ParseError, parse_document_output, parse_score_output
from .prompts import build_evaluation_messages, build_revision_messages

ExecutionMode = Literal["openclaw", "local_fallback", "local_only"]
FallbackReason = Literal[
    "agent_unavailable",
    "stale_agent_discovery",
    "transport_error",
    "executor_not_configured",
    "entrypoint_unavailable",
    "structured_output_parse_error",
    "agent_runtime_error",
]


@dataclass(slots=True)
class OpenClawAgentAvailability:
    available: bool
    backend: str
    mode: str
    reason: str = ""
    details: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "backend": self.backend,
            "mode": self.mode,
            "reason": self.reason,
            "details": self.details,
        }


@dataclass(slots=True)
class StageExecutionMetadata:
    execution_mode: ExecutionMode
    executor_backend: str
    fallback_reason: str = ""
    openclaw_request_id: str = ""
    openclaw_request_ids: list[str] = field(default_factory=list)
    details: dict[str, object] = field(default_factory=dict)

    @classmethod
    def local_only(cls) -> StageExecutionMetadata:
        return cls(
            execution_mode="local_only",
            executor_backend="local",
        )

    @classmethod
    def local_fallback(cls, *, fallback_reason: str) -> StageExecutionMetadata:
        return cls(
            execution_mode="local_fallback",
            executor_backend="local",
            fallback_reason=fallback_reason,
        )

    @classmethod
    def openclaw(
        cls,
        *,
        backend: str,
        request_ids: list[str],
        details: dict[str, object] | None = None,
    ) -> StageExecutionMetadata:
        return cls(
            execution_mode="openclaw",
            executor_backend=backend,
            openclaw_request_id=request_ids[0] if request_ids else "",
            openclaw_request_ids=request_ids,
            details=details or {},
        )

    def to_summary_fields(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "execution_mode": self.execution_mode,
            "executor_backend": self.executor_backend,
            "fallback_reason": self.fallback_reason,
            "openclaw_request_id": self.openclaw_request_id,
        }
        if self.openclaw_request_ids:
            payload["openclaw_request_ids"] = list(self.openclaw_request_ids)
        payload.update(self.details)
        return payload


@dataclass(slots=True)
class OpenClawExecutionRuntime:
    openclaw_backend: str
    provider_config: ProviderConfig
    availability: OpenClawAgentAvailability | None = None
    default_execution_mode: Literal["openclaw", "local_fallback"] = "local_fallback"
    latched_fallback_reason: str = ""
    client_factory: Callable[[], object] | None = None
    _client: object | None = field(default=None, init=False, repr=False)

    def get_client(self) -> object:
        if self._client is None:
            if self.client_factory is not None:
                self._client = self.client_factory()
            else:
                load_result = load_openclaw()
                self._client = load_result.OpenClaw.local()
        return self._client

    def close_client(self) -> None:
        if self._client is None:
            return
        close = getattr(self._client, "close", None)
        if callable(close):
            close()
        self._client = None

    def set_availability(self, availability: OpenClawAgentAvailability) -> OpenClawAgentAvailability:
        self.availability = availability
        if availability.available:
            self.default_execution_mode = "openclaw"
            self.latched_fallback_reason = ""
        else:
            self.default_execution_mode = "local_fallback"
            self.latched_fallback_reason = availability.reason
        return availability

    def latch_local_fallback(self, fallback_reason: str) -> None:
        self.default_execution_mode = "local_fallback"
        self.latched_fallback_reason = fallback_reason


@dataclass(slots=True)
class OpenClawScoreExecutionResult:
    scores: list[ScoreResult]
    metadata: StageExecutionMetadata


@dataclass(slots=True)
class OpenClawRevisionCandidateExecutionResult:
    draft: DraftDocument
    metadata: StageExecutionMetadata


class OpenClawStageExecutionError(RuntimeError):
    def __init__(self, fallback_reason: FallbackReason, message: str) -> None:
        super().__init__(message)
        self.fallback_reason = fallback_reason


class _ScoreOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str
    score: int


class _RevisionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    body: str


def build_execution_runtime(
    *,
    openclaw_backend: str,
    provider_config: ProviderConfig,
    client_factory: Callable[[], object] | None = None,
) -> OpenClawExecutionRuntime:
    return OpenClawExecutionRuntime(
        openclaw_backend=openclaw_backend,
        provider_config=provider_config,
        client_factory=client_factory,
    )


def probe_openclaw_agent(runtime: OpenClawExecutionRuntime) -> OpenClawAgentAvailability:
    if runtime.availability is not None:
        return runtime.set_availability(runtime.availability)

    try:
        from cmdop.transport.discovery import discover_agent
    except Exception as exc:  # pragma: no cover - defensive import boundary
        return runtime.set_availability(
            OpenClawAgentAvailability(
                available=False,
                backend=runtime.openclaw_backend,
                mode="local",
                reason="entrypoint_unavailable",
                details={"error": str(exc)},
            )
        )

    try:
        discovery_result = discover_agent()
    except Exception as exc:  # pragma: no cover - defensive transport wrapper
        return runtime.set_availability(
            OpenClawAgentAvailability(
                available=False,
                backend=runtime.openclaw_backend,
                mode="local",
                reason="transport_error",
                details={"error": str(exc)},
            )
        )

    if discovery_result.found:
        details: dict[str, object] = {}
        if discovery_result.discovery_path is not None:
            details["discovery_path"] = str(discovery_result.discovery_path)
        if discovery_result.agent_info is not None:
            details["transport"] = discovery_result.agent_info.transport.value
            details["address"] = discovery_result.agent_info.address
            details["pid"] = discovery_result.agent_info.pid
        return runtime.set_availability(
            OpenClawAgentAvailability(
                available=True,
                backend=runtime.openclaw_backend,
                mode="local",
                details=details,
            )
        )

    reason: FallbackReason = "agent_unavailable"
    if discovery_result.discovery_path is not None:
        reason = "stale_agent_discovery"

    details = {}
    if discovery_result.discovery_path is not None:
        details["discovery_path"] = str(discovery_result.discovery_path)
    if discovery_result.error:
        details["error"] = discovery_result.error
    return runtime.set_availability(
        OpenClawAgentAvailability(
            available=False,
            backend=runtime.openclaw_backend,
            mode="local",
            reason=reason,
            details=details,
        )
    )


def execute_score_draft_with_openclaw(
    runtime: OpenClawExecutionRuntime,
    draft: DraftDocument,
) -> OpenClawScoreExecutionResult:
    request_ids: list[str] = []
    scores: list[ScoreResult] = []

    for dimension in VALID_DIMENSIONS:
        messages = build_evaluation_messages(dimension, draft.title, draft.body)
        prompt = render_agent_prompt(messages)
        result = _run_openclaw_agent(
            runtime,
            prompt=prompt,
            output_model=_ScoreOutput,
        )
        request_ids.append(_read_request_id(result))
        score_text = _build_score_text(result)
        try:
            score = parse_score_output(score_text, dimension=dimension)
        except ParseError as exc:
            raise OpenClawStageExecutionError("structured_output_parse_error", str(exc)) from exc
        scores.append(score)

    return OpenClawScoreExecutionResult(
        scores=scores,
        metadata=StageExecutionMetadata.openclaw(
            backend=runtime.openclaw_backend,
            request_ids=request_ids,
        ),
    )


def execute_revision_candidate_with_openclaw(
    runtime: OpenClawExecutionRuntime,
    *,
    current_title: str,
    current_body: str,
    scores: list[ScoreResult],
    feedback_summary: str,
    candidate_index: int,
    candidate_count: int,
) -> OpenClawRevisionCandidateExecutionResult:
    messages = build_revision_messages(
        current_title=current_title,
        current_body=current_body,
        scores=scores,
        feedback_summary=feedback_summary,
        candidate_index=candidate_index,
        candidate_count=candidate_count,
    )
    prompt = render_agent_prompt(messages)
    result = _run_openclaw_agent(
        runtime,
        prompt=prompt,
        output_model=_RevisionOutput,
    )
    request_id = _read_request_id(result)
    try:
        draft = _build_document(result, fallback_title=current_title)
    except ParseError as exc:
        raise OpenClawStageExecutionError("structured_output_parse_error", str(exc)) from exc

    return OpenClawRevisionCandidateExecutionResult(
        draft=draft,
        metadata=StageExecutionMetadata.openclaw(
            backend=runtime.openclaw_backend,
            request_ids=[request_id],
        ),
    )


def render_agent_prompt(messages: list[dict[str, str]]) -> str:
    blocks: list[str] = []
    for message in messages:
        role = message.get("role", "user").upper()
        content = message.get("content", "").strip()
        blocks.append(f"{role}:\n{content}")
    return "\n\n".join(blocks).strip()


def _run_openclaw_agent(
    runtime: OpenClawExecutionRuntime,
    *,
    prompt: str,
    output_model: type[BaseModel],
) -> object:
    try:
        client = runtime.get_client()
    except Exception as exc:
        raise _classify_stage_error(exc) from exc

    agent = getattr(client, "agent", None)
    run = getattr(agent, "run", None)
    if not callable(run):
        raise OpenClawStageExecutionError(
            "entrypoint_unavailable",
            "OpenClaw client does not expose agent.run(...).",
        )

    kwargs = {
        "prompt": prompt,
        "output_model": output_model,
    }
    options = _build_agent_options(runtime)
    if options is not None:
        kwargs["options"] = options

    try:
        result = run(**kwargs)
    except TypeError:
        kwargs.pop("options", None)
        try:
            result = run(**kwargs)
        except Exception as exc:
            raise _classify_stage_error(exc) from exc
    except Exception as exc:
        raise _classify_stage_error(exc) from exc

    if not getattr(result, "success", False):
        error = getattr(result, "error", "") or "OpenClaw agent run failed."
        raise OpenClawStageExecutionError("agent_runtime_error", str(error))
    return result


def _classify_stage_error(exc: Exception) -> OpenClawStageExecutionError:
    try:
        from cmdop.exceptions import AgentNotRunningError, StalePortFileError
    except Exception:  # pragma: no cover - dependency import should exist in runtime
        AgentNotRunningError = tuple()  # type: ignore[assignment]
        StalePortFileError = tuple()  # type: ignore[assignment]

    if AgentNotRunningError and isinstance(exc, AgentNotRunningError):
        return OpenClawStageExecutionError("agent_unavailable", str(exc))
    if StalePortFileError and isinstance(exc, StalePortFileError):
        return OpenClawStageExecutionError("stale_agent_discovery", str(exc))
    if isinstance(exc, ParseError):
        return OpenClawStageExecutionError("structured_output_parse_error", str(exc))
    if isinstance(exc, (OSError, ConnectionError)):
        return OpenClawStageExecutionError("transport_error", str(exc))
    if isinstance(exc, AttributeError):
        return OpenClawStageExecutionError("entrypoint_unavailable", str(exc))
    return OpenClawStageExecutionError("agent_runtime_error", str(exc))


def _build_agent_options(runtime: OpenClawExecutionRuntime) -> object | None:
    try:
        from cmdop.models.agent import AgentRunOptions
    except Exception:  # pragma: no cover - dependency import should exist in runtime
        return None
    return AgentRunOptions(model=runtime.provider_config.model)


def _read_request_id(result: object) -> str:
    request_id = getattr(result, "request_id", "")
    return request_id if isinstance(request_id, str) else ""


def _build_score_text(result: object) -> str:
    data = getattr(result, "data", None)
    if data is not None and hasattr(data, "reason") and hasattr(data, "score"):
        reason = str(getattr(data, "reason")).strip()
        score = int(getattr(data, "score"))
        return f"<reason>{reason}</reason> <score>{score}</score>"

    output_json = getattr(result, "output_json", "")
    if isinstance(output_json, str) and output_json.strip():
        try:
            payload = json.loads(output_json)
        except json.JSONDecodeError:
            pass
        else:
            reason = str(payload.get("reason", "")).strip()
            score = payload.get("score", "")
            return f"<reason>{reason}</reason> <score>{score}</score>"

    text = getattr(result, "text", "")
    return text if isinstance(text, str) else ""


def _build_document(result: object, *, fallback_title: str) -> DraftDocument:
    data = getattr(result, "data", None)
    if data is not None and hasattr(data, "title") and hasattr(data, "body"):
        title = str(getattr(data, "title")).strip()
        body = str(getattr(data, "body")).strip()
        if title and body:
            return DraftDocument(title=title, body=body)

    output_json = getattr(result, "output_json", "")
    if isinstance(output_json, str) and output_json.strip():
        try:
            payload = json.loads(output_json)
        except json.JSONDecodeError:
            pass
        else:
            title = str(payload.get("title", "")).strip()
            body = str(payload.get("body", "")).strip()
            if title and body:
                return DraftDocument(title=title, body=body)

    text = getattr(result, "text", "")
    if not isinstance(text, str):
        raise ParseError("OpenClaw revision result text is not parseable.")
    return parse_document_output(text, fallback_title=fallback_title)
