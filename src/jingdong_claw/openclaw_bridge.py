from __future__ import annotations

import inspect
from dataclasses import dataclass, replace
from pathlib import Path

from .artifacts import ArtifactWriter, create_run_dir
from .config import Settings, load_settings, resolve_provider_config, resolve_research_config
from .llm import build_client
from .openclaw_execution import OpenClawAgentAvailability, build_execution_runtime
from .pipeline import SelfEvolutionPipeline
from .research import DeepResearchDraftGenerator
from .openclaw_workflow import (
    OpenClawWorkflowRequest,
    OpenClawWorkflowRuntime,
    run_local_openclaw_workflow,
)


class OpenClawBridgeError(ValueError):
    """Raised when an OpenClaw demo request is invalid."""


def _default_revision_candidates() -> int:
    parameter = inspect.signature(SelfEvolutionPipeline.__init__).parameters["revision_candidates"]
    default = parameter.default
    if isinstance(default, int) and default >= 1:
        return default
    return 1


DEFAULT_REVISION_CANDIDATES = _default_revision_candidates()
VALID_MODES = {"topic", "paper"}
VALID_PROVIDERS = {"localapi", "deepseek"}


@dataclass(slots=True)
class OpenClawDemoRequest:
    mode: str
    topic: str | None = None
    title: str | None = None
    body: str | None = None
    rounds: int | None = None
    provider: str | None = None
    model: str | None = None
    output_dir: str | None = None
    run_dir: str | None = None
    research_breadth: int | None = None
    research_depth: int | None = None
    research_concurrency: int | None = None
    revision_candidates: int | None = None
    force_local_only: bool = False


@dataclass(slots=True)
class OpenClawDemoResult:
    run_id: str
    mode: str
    topic: str | None
    round_count: int
    best_round_index: int
    best_total_score: int
    stop_reason: str
    stopped_early: bool
    output_dir: str
    summary_path: str
    draft_paths: list[str]
    score_paths: list[str]
    feedback_paths: list[str]
    openclaw_backend: str


def _clean_optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _normalize_positive_int(name: str, value: int | None, default: int) -> int:
    normalized = default if value is None else value
    if normalized < 1:
        raise OpenClawBridgeError(f"{name} must be at least 1.")
    return normalized


def normalize_request(request: OpenClawDemoRequest, settings: Settings) -> OpenClawDemoRequest:
    mode = _clean_optional_str(request.mode)
    if mode not in VALID_MODES:
        valid_modes = ", ".join(sorted(VALID_MODES))
        raise OpenClawBridgeError(f"mode must be one of: {valid_modes}.")

    provider = _clean_optional_str(request.provider)
    if provider is not None and provider not in VALID_PROVIDERS:
        valid_providers = ", ".join(sorted(VALID_PROVIDERS))
        raise OpenClawBridgeError(f"provider must be one of: {valid_providers}.")

    topic = _clean_optional_str(request.topic)
    title = _clean_optional_str(request.title)
    body = _clean_optional_str(request.body)
    model = _clean_optional_str(request.model)
    output_dir = _clean_optional_str(request.output_dir)
    run_dir = _clean_optional_str(request.run_dir)

    if mode == "topic" and topic is None:
        raise OpenClawBridgeError("topic mode requires a non-empty topic.")
    if mode == "paper":
        if title is None:
            raise OpenClawBridgeError("paper mode requires a non-empty title.")
        if body is None:
            raise OpenClawBridgeError("paper mode requires a non-empty body.")

    return replace(
        request,
        mode=mode,
        topic=topic,
        title=title,
        body=body,
        provider=provider,
        model=model,
        output_dir=output_dir,
        run_dir=run_dir,
        rounds=_normalize_positive_int("rounds", request.rounds, settings.default_rounds),
        research_breadth=_normalize_positive_int(
            "research_breadth",
            request.research_breadth,
            settings.research_breadth,
        ),
        research_depth=_normalize_positive_int(
            "research_depth",
            request.research_depth,
            settings.research_depth,
        ),
        research_concurrency=_normalize_positive_int(
            "research_concurrency",
            request.research_concurrency,
            settings.research_concurrency,
        ),
        revision_candidates=_normalize_positive_int(
            "revision_candidates",
            request.revision_candidates,
            DEFAULT_REVISION_CANDIDATES,
        ),
        force_local_only=bool(request.force_local_only),
    )


def _resolve_run_dir(*, output_root: Path, label: str, requested_run_dir: str | None) -> Path:
    if requested_run_dir is None:
        return create_run_dir(output_root, label)

    run_dir = Path(requested_run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise OpenClawBridgeError(f"run_dir already exists and is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _build_round_paths(output_dir: Path, round_count: int) -> tuple[list[str], list[str], list[str]]:
    draft_paths: list[str] = []
    score_paths: list[str] = []
    feedback_paths: list[str] = []
    for round_index in range(round_count):
        draft_paths.append(str(output_dir / f"draft_round_{round_index}.md"))
        score_paths.append(str(output_dir / f"scores_round_{round_index}.json"))
        feedback_paths.append(str(output_dir / f"feedback_round_{round_index}.md"))
    return draft_paths, score_paths, feedback_paths


def run_openclaw_demo(
    request: OpenClawDemoRequest,
    *,
    openclaw_backend: str,
) -> OpenClawDemoResult:
    settings = load_settings()
    normalized_request = normalize_request(request, settings)
    provider_config = resolve_provider_config(
        settings,
        provider=normalized_request.provider,
        model=normalized_request.model,
    )

    output_root = Path(normalized_request.output_dir) if normalized_request.output_dir else settings.output_root
    label = normalized_request.topic if normalized_request.mode == "topic" else normalized_request.title
    if label is None:
        raise OpenClawBridgeError("normalized request is missing its label.")
    run_dir = _resolve_run_dir(
        output_root=output_root,
        label=label,
        requested_run_dir=normalized_request.run_dir,
    )
    artifact_writer = ArtifactWriter(run_dir)
    client = build_client(provider_config)

    topic_draft_generator = None
    research_config = None
    if normalized_request.mode == "topic":
        research_config = resolve_research_config(
            settings,
            breadth=normalized_request.research_breadth,
            depth=normalized_request.research_depth,
            concurrency=normalized_request.research_concurrency,
        )
        topic_draft_generator = DeepResearchDraftGenerator(
            provider_config=provider_config,
            research_config=research_config,
        ).generate

    workflow_request = OpenClawWorkflowRequest(
        mode=normalized_request.mode,
        topic=normalized_request.topic,
        title=normalized_request.title,
        body=normalized_request.body,
        rounds=normalized_request.rounds or settings.default_rounds,
        revision_candidates=normalized_request.revision_candidates or DEFAULT_REVISION_CANDIDATES,
        openclaw_backend=_clean_optional_str(openclaw_backend) or "unknown",
        run_id=run_dir.name,
        run_dir=str(run_dir),
    )
    execution_runtime = build_execution_runtime(
        openclaw_backend=_clean_optional_str(openclaw_backend) or "unknown",
        provider_config=provider_config,
    )
    if normalized_request.force_local_only:
        execution_runtime.set_availability(
            OpenClawAgentAvailability(
                available=False,
                backend=_clean_optional_str(openclaw_backend) or "unknown",
                mode="local",
                reason="executor_not_configured",
                details={"source": "force_local_only"},
            )
        )

    workflow_runtime = OpenClawWorkflowRuntime(
        provider_config=provider_config,
        research_config=research_config,
        client=client,
        artifact_writer=artifact_writer,
        execution_runtime=execution_runtime,
        topic_draft_generator=topic_draft_generator,
    )
    workflow_result = run_local_openclaw_workflow(workflow_request, workflow_runtime)
    run_result = workflow_result.run_result

    output_dir = Path(run_result.output_dir)
    draft_paths, score_paths, feedback_paths = _build_round_paths(output_dir, len(run_result.rounds))

    return OpenClawDemoResult(
        run_id=run_result.run_id,
        mode=run_result.mode,
        topic=run_result.topic,
        round_count=len(run_result.rounds),
        best_round_index=run_result.best_round_index,
        best_total_score=run_result.best_total_score,
        stop_reason=run_result.stop_reason,
        stopped_early=run_result.stopped_early,
        output_dir=run_result.output_dir,
        summary_path=str(output_dir / "summary.json"),
        draft_paths=draft_paths,
        score_paths=score_paths,
        feedback_paths=feedback_paths,
        openclaw_backend=workflow_result.openclaw_backend,
    )
