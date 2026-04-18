from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable

from .artifacts import ArtifactWriter
from .config import ResearchConfig
from .models import DraftDocument, ProviderConfig, RoundResult, RunResult, ScoreResult
from .openclaw_execution import (
    OpenClawExecutionRuntime,
    OpenClawStageExecutionError,
    StageExecutionMetadata,
    execute_revision_candidate_with_openclaw,
    execute_score_draft_with_openclaw,
    probe_openclaw_agent,
)
from .pipeline import generate_initial_draft, select_best_revision_candidate
from .scoring import build_feedback_summary, build_round_result, evaluate_draft


def _timestamp() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass(slots=True)
class OpenClawWorkflowRequest:
    mode: str
    topic: str | None
    title: str | None
    body: str | None
    rounds: int
    revision_candidates: int
    openclaw_backend: str
    run_id: str
    run_dir: str


@dataclass(slots=True)
class OpenClawWorkflowRuntime:
    provider_config: ProviderConfig
    research_config: ResearchConfig | None
    client: object
    artifact_writer: ArtifactWriter
    execution_runtime: OpenClawExecutionRuntime
    topic_draft_generator: Callable[[str], DraftDocument] | None = None


@dataclass(slots=True)
class StageRecord:
    stage_name: str
    round_index: int | None
    status: str
    started_at: str
    ended_at: str
    summary: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class RoundState:
    draft: DraftDocument
    cached_scores: list[ScoreResult] | None = None
    cached_feedback_summary: str | None = None
    cached_score_metadata: StageExecutionMetadata | None = None


@dataclass(slots=True)
class StageScoreResult:
    scores: list[ScoreResult]
    total_score: int
    average_score: float
    feedback_summary: str
    cache_used: bool = False


@dataclass(slots=True)
class StageRevisionResult:
    candidate_count: int
    best_candidate_draft: DraftDocument
    best_candidate_scores: list[ScoreResult]
    best_candidate_feedback_summary: str
    best_candidate_total_score: int
    best_candidate_score_metadata: StageExecutionMetadata | None = None
    candidate_summaries: list[dict[str, object]] = field(default_factory=list)


@dataclass(slots=True)
class StageRoundFinalizationResult:
    round_result: RoundResult
    should_stop: bool
    stop_reason: str
    best_round_index: int
    best_total_score: int


@dataclass(slots=True)
class StageAdvanceDecisionResult:
    should_continue: bool
    stop_reason: str
    next_round_state: RoundState | None
    best_round_index: int
    best_total_score: int


@dataclass(slots=True)
class OpenClawWorkflowResult:
    run_result: RunResult
    stage_records: list[StageRecord]
    openclaw_backend: str
    failed_stage: str | None
    workflow_trace_path: str


def _serialize_stage_records(records: list[StageRecord]) -> list[dict[str, object]]:
    return [asdict(record) for record in records]


def _build_workflow_trace(
    request: OpenClawWorkflowRequest,
    *,
    stage_records: list[StageRecord],
    failed_stage: str | None,
    stop_reason: str,
    workflow_runtime: OpenClawWorkflowRuntime,
) -> dict[str, object]:
    return {
        "run_id": request.run_id,
        "mode": request.mode,
        "topic": request.topic,
        "openclaw_backend": request.openclaw_backend,
        "agent_availability": workflow_runtime.execution_runtime.availability.to_dict()
        if workflow_runtime.execution_runtime.availability is not None
        else None,
        "default_execution_mode": workflow_runtime.execution_runtime.default_execution_mode,
        "failed_stage": failed_stage,
        "stop_reason": stop_reason,
        "stage_records": _serialize_stage_records(stage_records),
    }


def _build_success_record(
    *,
    stage_name: str,
    round_index: int | None,
    started_at: str,
    summary: dict[str, object] | None = None,
) -> StageRecord:
    return StageRecord(
        stage_name=stage_name,
        round_index=round_index,
        status="completed",
        started_at=started_at,
        ended_at=_timestamp(),
        summary=summary or {},
    )


def _build_error_record(
    *,
    stage_name: str,
    round_index: int | None,
    started_at: str,
    error: Exception,
) -> StageRecord:
    return StageRecord(
        stage_name=stage_name,
        round_index=round_index,
        status="error",
        started_at=started_at,
        ended_at=_timestamp(),
        summary={"error": str(error)},
    )


def _merge_stage_summary(
    summary: dict[str, object] | None,
    metadata: StageExecutionMetadata,
) -> dict[str, object]:
    merged = dict(summary or {})
    merged.update(metadata.to_summary_fields())
    return merged


def _fallback_reason(runtime: OpenClawWorkflowRuntime) -> str:
    if runtime.execution_runtime.latched_fallback_reason:
        return runtime.execution_runtime.latched_fallback_reason
    availability = runtime.execution_runtime.availability
    if availability is not None and availability.reason:
        return availability.reason
    return "agent_unavailable"


def generate_initial_draft_stage(
    request: OpenClawWorkflowRequest,
    runtime: OpenClawWorkflowRuntime,
) -> tuple[DraftDocument, StageRecord]:
    started_at = _timestamp()
    if request.mode == "topic":
        if request.topic is None:
            raise ValueError("Topic workflow request is missing topic.")
        draft = generate_initial_draft(
            runtime.client,
            topic=request.topic,
            topic_draft_generator=runtime.topic_draft_generator,
        )
    else:
        if request.title is None or request.body is None:
            raise ValueError("Paper workflow request is missing title or body.")
        draft = DraftDocument(title=request.title, body=request.body)

    record = _build_success_record(
        stage_name="generate_initial_draft_stage",
        round_index=0,
        started_at=started_at,
        summary={
            "mode": request.mode,
            "title": draft.title,
        },
    )
    return draft, record


def dispatch_generate_initial_draft_stage(
    request: OpenClawWorkflowRequest,
    runtime: OpenClawWorkflowRuntime,
) -> tuple[DraftDocument, StageRecord]:
    draft, record = generate_initial_draft_stage(request, runtime)
    record.summary = _merge_stage_summary(record.summary, StageExecutionMetadata.local_only())
    return draft, record


def score_draft_stage(
    state: RoundState,
    runtime: OpenClawWorkflowRuntime,
    *,
    round_index: int,
) -> tuple[StageScoreResult, StageRecord]:
    started_at = _timestamp()

    if state.cached_scores is not None and state.cached_feedback_summary is not None:
        scores = state.cached_scores
        feedback_summary = state.cached_feedback_summary
        cache_used = True
    else:
        scores = evaluate_draft(runtime.client, state.draft)
        feedback_summary = build_feedback_summary(scores)
        cache_used = False

    total_score = sum(item.score for item in scores)
    average_score = total_score / len(scores)
    result = StageScoreResult(
        scores=scores,
        total_score=total_score,
        average_score=average_score,
        feedback_summary=feedback_summary,
        cache_used=cache_used,
    )
    record = _build_success_record(
        stage_name="score_draft_stage",
        round_index=round_index,
        started_at=started_at,
        summary={
            "total_score": total_score,
            "average_score": average_score,
            "cache_used": cache_used,
        },
    )
    return result, record


def dispatch_score_draft_stage(
    state: RoundState,
    runtime: OpenClawWorkflowRuntime,
    *,
    round_index: int,
) -> tuple[StageScoreResult, StageRecord]:
    started_at = _timestamp()

    if state.cached_scores is not None and state.cached_feedback_summary is not None:
        metadata = state.cached_score_metadata or StageExecutionMetadata.local_fallback(
            fallback_reason=_fallback_reason(runtime)
        )
        total_score = sum(item.score for item in state.cached_scores)
        average_score = total_score / len(state.cached_scores)
        result = StageScoreResult(
            scores=state.cached_scores,
            total_score=total_score,
            average_score=average_score,
            feedback_summary=state.cached_feedback_summary,
            cache_used=True,
        )
        record = _build_success_record(
            stage_name="score_draft_stage",
            round_index=round_index,
            started_at=started_at,
            summary=_merge_stage_summary(
                {
                    "total_score": total_score,
                    "average_score": average_score,
                    "cache_used": True,
                },
                metadata,
            ),
        )
        return result, record

    if runtime.execution_runtime.default_execution_mode == "openclaw":
        try:
            openclaw_result = execute_score_draft_with_openclaw(runtime.execution_runtime, state.draft)
        except OpenClawStageExecutionError as exc:
            runtime.execution_runtime.latch_local_fallback(exc.fallback_reason)
            result, record = score_draft_stage(state, runtime, round_index=round_index)
            metadata = StageExecutionMetadata.local_fallback(fallback_reason=exc.fallback_reason)
            record.summary = _merge_stage_summary(record.summary, metadata)
            return result, record

        feedback_summary = build_feedback_summary(openclaw_result.scores)
        total_score = sum(item.score for item in openclaw_result.scores)
        average_score = total_score / len(openclaw_result.scores)
        result = StageScoreResult(
            scores=openclaw_result.scores,
            total_score=total_score,
            average_score=average_score,
            feedback_summary=feedback_summary,
            cache_used=False,
        )
        record = _build_success_record(
            stage_name="score_draft_stage",
            round_index=round_index,
            started_at=started_at,
            summary=_merge_stage_summary(
                {
                    "total_score": total_score,
                    "average_score": average_score,
                    "cache_used": False,
                },
                openclaw_result.metadata,
            ),
        )
        return result, record

    result, record = score_draft_stage(state, runtime, round_index=round_index)
    record.summary = _merge_stage_summary(
        record.summary,
        StageExecutionMetadata.local_fallback(fallback_reason=_fallback_reason(runtime)),
    )
    return result, record


def revise_draft_stage(
    state: RoundState,
    score_result: StageScoreResult,
    runtime: OpenClawWorkflowRuntime,
    *,
    round_index: int,
    revision_candidates: int,
) -> tuple[StageRevisionResult, StageRecord]:
    started_at = _timestamp()
    best_candidate_draft, best_candidate_scores, best_candidate_feedback_summary = select_best_revision_candidate(
        runtime.client,
        draft=state.draft,
        scores=score_result.scores,
        feedback_summary=score_result.feedback_summary,
        revision_candidates=revision_candidates,
    )
    best_candidate_total_score = sum(item.score for item in best_candidate_scores)
    result = StageRevisionResult(
        candidate_count=revision_candidates,
        best_candidate_draft=best_candidate_draft,
        best_candidate_scores=best_candidate_scores,
        best_candidate_feedback_summary=best_candidate_feedback_summary,
        best_candidate_total_score=best_candidate_total_score,
    )
    record = _build_success_record(
        stage_name="revise_draft_stage",
        round_index=round_index,
        started_at=started_at,
        summary={
            "candidate_count": revision_candidates,
            "best_candidate_total_score": best_candidate_total_score,
        },
    )
    return result, record


def dispatch_revise_draft_stage(
    state: RoundState,
    score_result: StageScoreResult,
    runtime: OpenClawWorkflowRuntime,
    *,
    round_index: int,
    revision_candidates: int,
) -> tuple[StageRevisionResult, StageRecord]:
    started_at = _timestamp()
    if runtime.execution_runtime.default_execution_mode != "openclaw":
        result, record = revise_draft_stage(
            state,
            score_result,
            runtime,
            round_index=round_index,
            revision_candidates=revision_candidates,
        )
        metadata = StageExecutionMetadata.local_fallback(fallback_reason=_fallback_reason(runtime))
        result.best_candidate_score_metadata = metadata
        record.summary = _merge_stage_summary(record.summary, metadata)
        return result, record

    try:
        candidate_summaries: list[dict[str, object]] = []
        combined_request_ids: list[str] = []
        best_candidate_draft: DraftDocument | None = None
        best_candidate_scores: list[ScoreResult] | None = None
        best_candidate_feedback_summary: str | None = None
        best_candidate_total_score: int | None = None
        best_candidate_score_metadata: StageExecutionMetadata | None = None

        for candidate_index in range(1, revision_candidates + 1):
            revision_execution = execute_revision_candidate_with_openclaw(
                runtime.execution_runtime,
                current_title=state.draft.title,
                current_body=state.draft.body,
                scores=score_result.scores,
                feedback_summary=score_result.feedback_summary,
                candidate_index=candidate_index,
                candidate_count=revision_candidates,
            )
            combined_request_ids.extend(revision_execution.metadata.openclaw_request_ids)

            candidate_score_execution = execute_score_draft_with_openclaw(
                runtime.execution_runtime,
                revision_execution.draft,
            )
            combined_request_ids.extend(candidate_score_execution.metadata.openclaw_request_ids)
            candidate_feedback_summary = build_feedback_summary(candidate_score_execution.scores)
            candidate_total_score = sum(item.score for item in candidate_score_execution.scores)

            candidate_summary = {
                "candidate_index": candidate_index,
                "candidate_title": revision_execution.draft.title,
                "generation_execution_mode": revision_execution.metadata.execution_mode,
                "generation_executor_backend": revision_execution.metadata.executor_backend,
                "generation_request_id": revision_execution.metadata.openclaw_request_id,
                "score_execution_mode": candidate_score_execution.metadata.execution_mode,
                "score_executor_backend": candidate_score_execution.metadata.executor_backend,
                "score_request_id": candidate_score_execution.metadata.openclaw_request_id,
                "score_openclaw_request_ids": list(candidate_score_execution.metadata.openclaw_request_ids),
                "candidate_total_score": candidate_total_score,
                "selected": False,
            }
            candidate_summaries.append(candidate_summary)

            if best_candidate_total_score is None or candidate_total_score > best_candidate_total_score:
                best_candidate_draft = revision_execution.draft
                best_candidate_scores = candidate_score_execution.scores
                best_candidate_feedback_summary = candidate_feedback_summary
                best_candidate_total_score = candidate_total_score
                best_candidate_score_metadata = candidate_score_execution.metadata

        if (
            best_candidate_draft is None
            or best_candidate_scores is None
            or best_candidate_feedback_summary is None
            or best_candidate_total_score is None
            or best_candidate_score_metadata is None
        ):
            raise RuntimeError("Revision candidate selection produced no draft.")

        for candidate_summary in candidate_summaries:
            if candidate_summary["candidate_total_score"] == best_candidate_total_score and not any(
                item["selected"] for item in candidate_summaries
            ):
                candidate_summary["selected"] = True
                break

        result = StageRevisionResult(
            candidate_count=revision_candidates,
            best_candidate_draft=best_candidate_draft,
            best_candidate_scores=best_candidate_scores,
            best_candidate_feedback_summary=best_candidate_feedback_summary,
            best_candidate_total_score=best_candidate_total_score,
            best_candidate_score_metadata=best_candidate_score_metadata,
            candidate_summaries=candidate_summaries,
        )
        metadata = StageExecutionMetadata.openclaw(
            backend=runtime.execution_runtime.openclaw_backend,
            request_ids=combined_request_ids,
            details={"candidate_summaries": candidate_summaries},
        )
        record = _build_success_record(
            stage_name="revise_draft_stage",
            round_index=round_index,
            started_at=started_at,
            summary=_merge_stage_summary(
                {
                    "candidate_count": revision_candidates,
                    "best_candidate_total_score": best_candidate_total_score,
                    "candidate_summaries": candidate_summaries,
                },
                metadata,
            ),
        )
        return result, record
    except OpenClawStageExecutionError as exc:
        runtime.execution_runtime.latch_local_fallback(exc.fallback_reason)
        result, record = revise_draft_stage(
            state,
            score_result,
            runtime,
            round_index=round_index,
            revision_candidates=revision_candidates,
        )
        metadata = StageExecutionMetadata.local_fallback(fallback_reason=exc.fallback_reason)
        result.best_candidate_score_metadata = metadata
        record.summary = _merge_stage_summary(record.summary, metadata)
        return result, record


def finalize_scored_round_stage(
    state: RoundState,
    score_result: StageScoreResult,
    runtime: OpenClawWorkflowRuntime,
    *,
    round_index: int,
    rounds: int,
    best_round_index: int,
    best_total_score: int,
) -> tuple[StageRoundFinalizationResult, StageRecord]:
    started_at = _timestamp()
    round_result = build_round_result(
        round_index=round_index,
        draft=state.draft,
        scores=score_result.scores,
        feedback_summary=score_result.feedback_summary,
        provider=runtime.client.config.provider,
        model=runtime.client.config.model,
    )
    runtime.artifact_writer.write_round(round_result)

    next_best_round_index = best_round_index
    next_best_total_score = best_total_score
    if next_best_round_index == -1 or round_result.total_score > next_best_total_score:
        next_best_round_index = round_index
        next_best_total_score = round_result.total_score

    should_stop = round_index == rounds - 1
    stop_reason = "max_rounds_reached" if should_stop else "continue"
    result = StageRoundFinalizationResult(
        round_result=round_result,
        should_stop=should_stop,
        stop_reason=stop_reason,
        best_round_index=next_best_round_index,
        best_total_score=next_best_total_score,
    )
    record = _build_success_record(
        stage_name="finalize_scored_round_stage",
        round_index=round_index,
        started_at=started_at,
        summary={
            "total_score": round_result.total_score,
            "best_round_index": next_best_round_index,
            "best_total_score": next_best_total_score,
            "stop_reason": stop_reason,
        },
    )
    return result, record


def dispatch_finalize_scored_round_stage(
    state: RoundState,
    score_result: StageScoreResult,
    runtime: OpenClawWorkflowRuntime,
    *,
    round_index: int,
    rounds: int,
    best_round_index: int,
    best_total_score: int,
) -> tuple[StageRoundFinalizationResult, StageRecord]:
    result, record = finalize_scored_round_stage(
        state,
        score_result,
        runtime,
        round_index=round_index,
        rounds=rounds,
        best_round_index=best_round_index,
        best_total_score=best_total_score,
    )
    record.summary = _merge_stage_summary(record.summary, StageExecutionMetadata.local_only())
    return result, record


def advance_or_stop_stage(
    current_round_result: RoundResult,
    revision_result: StageRevisionResult,
    *,
    round_index: int,
    best_round_index: int,
    best_total_score: int,
) -> tuple[StageAdvanceDecisionResult, StageRecord]:
    started_at = _timestamp()
    if revision_result.best_candidate_total_score <= current_round_result.total_score:
        result = StageAdvanceDecisionResult(
            should_continue=False,
            stop_reason="no_improving_revision_candidate",
            next_round_state=None,
            best_round_index=best_round_index,
            best_total_score=best_total_score,
        )
        record = _build_success_record(
            stage_name="advance_or_stop_stage",
            round_index=round_index,
            started_at=started_at,
            summary={
                "should_continue": False,
                "stop_reason": result.stop_reason,
            },
        )
        return result, record

    result = StageAdvanceDecisionResult(
        should_continue=True,
        stop_reason="continue",
        next_round_state=RoundState(
            draft=revision_result.best_candidate_draft,
            cached_scores=revision_result.best_candidate_scores,
            cached_feedback_summary=revision_result.best_candidate_feedback_summary,
            cached_score_metadata=revision_result.best_candidate_score_metadata,
        ),
        best_round_index=best_round_index,
        best_total_score=best_total_score,
    )
    record = _build_success_record(
        stage_name="advance_or_stop_stage",
        round_index=round_index,
        started_at=started_at,
        summary={
            "should_continue": True,
            "stop_reason": result.stop_reason,
            "next_total_score": revision_result.best_candidate_total_score,
        },
    )
    return result, record


def dispatch_advance_or_stop_stage(
    current_round_result: RoundResult,
    revision_result: StageRevisionResult,
    *,
    round_index: int,
    best_round_index: int,
    best_total_score: int,
) -> tuple[StageAdvanceDecisionResult, StageRecord]:
    result, record = advance_or_stop_stage(
        current_round_result,
        revision_result,
        round_index=round_index,
        best_round_index=best_round_index,
        best_total_score=best_total_score,
    )
    record.summary = _merge_stage_summary(record.summary, StageExecutionMetadata.local_only())
    return result, record


def run_local_openclaw_workflow(
    request: OpenClawWorkflowRequest,
    runtime: OpenClawWorkflowRuntime,
) -> OpenClawWorkflowResult:
    run_result = RunResult(
        run_id=request.run_id,
        mode=request.mode,  # type: ignore[arg-type]
        topic=request.topic,
        output_dir=request.run_dir,
    )
    stage_records: list[StageRecord] = []
    failed_stage: str | None = None

    try:
        probe_openclaw_agent(runtime.execution_runtime)
        generation_started = _timestamp()
        try:
            initial_draft, record = dispatch_generate_initial_draft_stage(request, runtime)
            stage_records.append(record)
        except Exception as exc:
            failed_stage = "generate_initial_draft_stage"
            stage_records.append(
                _build_error_record(
                    stage_name=failed_stage,
                    round_index=0,
                    started_at=generation_started,
                    error=exc,
                )
            )
            raise

        current_state = RoundState(draft=initial_draft)

        for round_index in range(request.rounds):
            score_started = _timestamp()
            try:
                score_result, record = dispatch_score_draft_stage(current_state, runtime, round_index=round_index)
                stage_records.append(record)
            except Exception as exc:
                failed_stage = "score_draft_stage"
                stage_records.append(
                    _build_error_record(
                        stage_name=failed_stage,
                        round_index=round_index,
                        started_at=score_started,
                        error=exc,
                    )
                )
                raise

            finalize_started = _timestamp()
            try:
                finalization, record = dispatch_finalize_scored_round_stage(
                    current_state,
                    score_result,
                    runtime,
                    round_index=round_index,
                    rounds=request.rounds,
                    best_round_index=run_result.best_round_index,
                    best_total_score=run_result.best_total_score,
                )
                stage_records.append(record)
            except Exception as exc:
                failed_stage = "finalize_scored_round_stage"
                stage_records.append(
                    _build_error_record(
                        stage_name=failed_stage,
                        round_index=round_index,
                        started_at=finalize_started,
                        error=exc,
                    )
                )
                raise

            run_result.rounds.append(finalization.round_result)
            run_result.best_round_index = finalization.best_round_index
            run_result.best_total_score = finalization.best_total_score

            if finalization.should_stop:
                run_result.stop_reason = finalization.stop_reason
                break

            revision_started = _timestamp()
            try:
                revision_result, record = dispatch_revise_draft_stage(
                    current_state,
                    score_result,
                    runtime,
                    round_index=round_index,
                    revision_candidates=request.revision_candidates,
                )
                stage_records.append(record)
            except Exception as exc:
                failed_stage = "revise_draft_stage"
                stage_records.append(
                    _build_error_record(
                        stage_name=failed_stage,
                        round_index=round_index,
                        started_at=revision_started,
                        error=exc,
                    )
                )
                raise

            advance_started = _timestamp()
            try:
                advance_result, record = dispatch_advance_or_stop_stage(
                    finalization.round_result,
                    revision_result,
                    round_index=round_index,
                    best_round_index=run_result.best_round_index,
                    best_total_score=run_result.best_total_score,
                )
                stage_records.append(record)
            except Exception as exc:
                failed_stage = "advance_or_stop_stage"
                stage_records.append(
                    _build_error_record(
                        stage_name=failed_stage,
                        round_index=round_index,
                        started_at=advance_started,
                        error=exc,
                    )
                )
                raise

            if not advance_result.should_continue or advance_result.next_round_state is None:
                run_result.stopped_early = True
                run_result.stop_reason = advance_result.stop_reason
                break

            current_state = advance_result.next_round_state

        if run_result.stop_reason == "not_started":
            run_result.stop_reason = "completed"

        workflow_trace = _build_workflow_trace(
            request,
            stage_records=stage_records,
            failed_stage=failed_stage,
            stop_reason=run_result.stop_reason,
            workflow_runtime=runtime,
        )
        runtime.artifact_writer.write_summary(run_result)
        runtime.artifact_writer.write_workflow_trace(workflow_trace)
        workflow_trace_path = str(Path(request.run_dir) / "workflow_trace.json")
        return OpenClawWorkflowResult(
            run_result=run_result,
            stage_records=stage_records,
            openclaw_backend=request.openclaw_backend,
            failed_stage=failed_stage,
            workflow_trace_path=workflow_trace_path,
        )
    except Exception as exc:
        run_result.error = str(exc)
        run_result.stopped_early = True
        run_result.stop_reason = "error"
        workflow_trace = _build_workflow_trace(
            request,
            stage_records=stage_records,
            failed_stage=failed_stage,
            stop_reason=run_result.stop_reason,
            workflow_runtime=runtime,
        )
        runtime.artifact_writer.write_summary(run_result)
        runtime.artifact_writer.write_workflow_trace(workflow_trace)
        raise
    finally:
        runtime.execution_runtime.close_client()
