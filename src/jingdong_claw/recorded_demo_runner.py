from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Callable

from .artifacts import ArtifactWriter, create_run_dir
from .config import (
    ConfigError,
    load_settings,
    resolve_provider_config,
    resolve_research_config,
)
from .llm import LLMError, build_client
from .models import DraftDocument, ProviderConfig
from .openclaw_bench import OpenClawBenchError, evaluate_manifest, export_manifest, render_report
from .openclaw_cli_backend import OpenClawCliBackendError, run_openclaw_local_agent
from .openclaw_setup import (
    DEFAULT_OPENCLAW_AGENT_ID,
    DEFAULT_OPENCLAW_PROFILE,
    DEFAULT_OPENCLAW_THINKING,
    OpenClawSetupError,
    OpenClawWorkspaceSetup,
    RECORDED_DEMO_REQUIRED_SKILLS,
    prepare_openclaw_workspace,
)
from .parser import ParseError, parse_document_output
from .research import DeepResearchDraftGenerator, ResearchError
from .scoring import build_feedback_summary, build_round_result, evaluate_draft

DEFAULT_MAX_ROUNDS = 3
DEFAULT_REVISION_CANDIDATES = 2
DEFAULT_AGENT_TIMEOUT_SECONDS = 600


class RecordedDemoRunnerError(RuntimeError):
    """Raised when the recorded OpenClaw-root demo cannot complete."""


@dataclass(slots=True)
class RecordedDemoRequest:
    topic: str
    run_dir: str | None = None
    max_rounds: int | None = None
    provider: str | None = None
    model: str | None = None
    research_breadth: int | None = None
    research_depth: int | None = None
    research_concurrency: int | None = None
    revision_candidates: int | None = None
    skills_dir: str | None = None
    profile_name: str = DEFAULT_OPENCLAW_PROFILE
    agent_id: str = DEFAULT_OPENCLAW_AGENT_ID
    agent_timeout_seconds: int | None = None
    openclaw_thinking: str = DEFAULT_OPENCLAW_THINKING
    sync_skills: bool = False


@dataclass(slots=True)
class RecordedDemoResult:
    run_dir: str
    summary_path: str
    best_draft_path: str
    bench_results_path: str
    bench_report_path: str
    bench_backend: str
    stop_reason: str
    baseline_total_score: int | None
    candidate_total_score: int | None
    delta_total_score: int | None


@dataclass(slots=True)
class RevisionAgentRequest:
    profile_name: str
    agent_id: str
    workspace_root: Path
    env: dict[str, str]
    session_id: str
    draft_path: Path
    scores_path: Path
    feedback_path: Path
    output_markdown_path: Path
    output_json_path: Path
    stdout_path: Path
    stderr_path: Path
    timeout_seconds: int
    thinking_level: str
    revision_candidates: int


@dataclass(slots=True)
class RevisionAgentResult:
    response_text: str
    output_markdown_path: Path
    output_json_path: Path


@dataclass(slots=True)
class RecordedDemoRuntime:
    topic_draft_generator: Callable[[str], DraftDocument] | None = None
    evaluator_client_factory: Callable[[ProviderConfig], object] | None = None
    workspace_setup_factory: Callable[..., OpenClawWorkspaceSetup] | None = None
    revision_agent_runner: Callable[[RevisionAgentRequest], RevisionAgentResult] | None = None
    export_manifest_fn: Callable[..., Path] | None = None
    evaluate_manifest_fn: Callable[..., Path] | None = None
    render_report_fn: Callable[..., tuple[Path, Path]] | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministic recorded demo runner for the OpenClaw-root self-evolution flow."
    )
    parser.add_argument("--topic", required=True)
    parser.add_argument("--run-dir", default=None, help="Exact run directory path for all artifacts.")
    parser.add_argument("--max-rounds", type=int, default=None, help="Maximum evaluated drafts, including round 0.")
    parser.add_argument("--provider", choices=["localapi", "deepseek"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--research-breadth", type=int, default=None)
    parser.add_argument("--research-depth", type=int, default=None)
    parser.add_argument("--research-concurrency", type=int, default=None)
    parser.add_argument("--revision-candidates", type=int, default=None)
    parser.add_argument("--skills-dir", default=None)
    parser.add_argument("--profile-name", default=DEFAULT_OPENCLAW_PROFILE)
    parser.add_argument("--agent-id", default=DEFAULT_OPENCLAW_AGENT_ID)
    parser.add_argument("--agent-timeout-seconds", type=int, default=None)
    parser.add_argument("--openclaw-thinking", default=DEFAULT_OPENCLAW_THINKING)
    parser.add_argument(
        "--sync-skills",
        action="store_true",
        help="Sync repo-owned .cmdop skills into workspace/skills before validating OpenClaw.",
    )
    return parser


def _clean_optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _normalize_positive_int(name: str, value: int | None, default: int) -> int:
    normalized = default if value is None else value
    if normalized < 1:
        raise RecordedDemoRunnerError(f"{name} must be at least 1.")
    return normalized


def normalize_request(request: RecordedDemoRequest) -> RecordedDemoRequest:
    topic = _clean_optional_str(request.topic)
    if topic is None:
        raise RecordedDemoRunnerError("topic must not be empty.")

    return RecordedDemoRequest(
        topic=topic,
        run_dir=_clean_optional_str(request.run_dir),
        max_rounds=_normalize_positive_int("max_rounds", request.max_rounds, DEFAULT_MAX_ROUNDS),
        provider=_clean_optional_str(request.provider),
        model=_clean_optional_str(request.model),
        research_breadth=request.research_breadth,
        research_depth=request.research_depth,
        research_concurrency=request.research_concurrency,
        revision_candidates=_normalize_positive_int(
            "revision_candidates",
            request.revision_candidates,
            DEFAULT_REVISION_CANDIDATES,
        ),
        skills_dir=_clean_optional_str(request.skills_dir),
        profile_name=_clean_optional_str(request.profile_name) or DEFAULT_OPENCLAW_PROFILE,
        agent_id=_clean_optional_str(request.agent_id) or DEFAULT_OPENCLAW_AGENT_ID,
        agent_timeout_seconds=_normalize_positive_int(
            "agent_timeout_seconds",
            request.agent_timeout_seconds,
            DEFAULT_AGENT_TIMEOUT_SECONDS,
        ),
        openclaw_thinking=_clean_optional_str(request.openclaw_thinking) or DEFAULT_OPENCLAW_THINKING,
        sync_skills=bool(request.sync_skills),
    )


def _resolve_run_dir(settings_output_root: Path, request: RecordedDemoRequest) -> Path:
    if request.run_dir:
        run_dir = Path(request.run_dir).resolve()
        if run_dir.exists() and any(run_dir.iterdir()):
            raise RecordedDemoRunnerError(f"run_dir already exists and is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    return create_run_dir(settings_output_root, request.topic)


def _write_markdown(path: Path, draft: DraftDocument) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {draft.title}\n\n{draft.body}\n", encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_markdown_draft(path: Path) -> DraftDocument:
    return parse_document_output(path.read_text(encoding="utf-8"), fallback_title=path.stem.replace("_", " ").strip())


def _build_revision_prompt(request: RevisionAgentRequest) -> str:
    return dedent(
        f"""
        Use the workspace skill `survey-revise-worker` to revise one survey draft.

        Required inputs:
        - Draft markdown: "{request.draft_path}"
        - Scores JSON: "{request.scores_path}"
        - Feedback file: "{request.feedback_path}"

        Run exactly one revision pass through the repo script exposed by that skill.
        Required outputs:
        - Revised markdown: "{request.output_markdown_path}"
        - Worker JSON payload: "{request.output_json_path}"

        Constraints:
        - Use revision_candidates={request.revision_candidates}
        - Do not change unrelated files
        - Do not start a second workflow or additional rounds
        - Reply with one short sentence after the files are written
        """
    ).strip()


def _run_openclaw_revision_agent(request: RevisionAgentRequest) -> RevisionAgentResult:
    try:
        result = run_openclaw_local_agent(
            profile_name=request.profile_name,
            agent_id=request.agent_id,
            session_id=request.session_id,
            workspace_root=request.workspace_root,
            prompt=_build_revision_prompt(request),
            env=request.env,
            timeout_seconds=request.timeout_seconds,
            thinking_level=request.thinking_level,
            stdout_path=request.stdout_path,
            stderr_path=request.stderr_path,
        )
    except OpenClawCliBackendError as exc:
        raise RecordedDemoRunnerError(str(exc)) from exc

    if not result.success:
        error_text = result.error or result.text or "OpenClaw revision agent failed."
        raise RecordedDemoRunnerError(error_text)

    if not request.output_markdown_path.exists():
        raise RecordedDemoRunnerError(
            f"OpenClaw revision agent did not write revised markdown: {request.output_markdown_path}"
        )
    if not request.output_json_path.exists():
        raise RecordedDemoRunnerError(
            f"OpenClaw revision agent did not write revision payload JSON: {request.output_json_path}"
        )

    return RevisionAgentResult(
        response_text=result.text,
        output_markdown_path=request.output_markdown_path,
        output_json_path=request.output_json_path,
    )


def _build_revision_rationale(
    *,
    round_index: int,
    source_draft: DraftDocument,
    revised_draft: DraftDocument,
    previous_total_score: int,
    revised_total_score: int,
    revision_payload: dict[str, object],
    agent_result: RevisionAgentResult,
) -> str:
    worker_best_total = revision_payload.get("best_total_score")
    return "\n".join(
        [
            f"OpenClaw revision round {round_index} updated '{source_draft.title}' into '{revised_draft.title}'.",
            f"The outer runner re-scored the revised draft after the OpenClaw worker finished.",
            f"Previous total score: {previous_total_score}. Revised total score: {revised_total_score}.",
            f"Worker-selected candidate total score: {worker_best_total}.",
            f"OpenClaw agent reply: {agent_result.response_text.strip() or 'n/a'}",
        ]
    )


def _int_or_none(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_bench_results(results_path: Path) -> tuple[str, int | None, int | None, int | None]:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    backend = str(payload.get("backend", "") or "unknown")
    baseline_total = _int_or_none(payload.get("baseline", {}).get("total_score") if isinstance(payload.get("baseline"), dict) else None)
    candidate_total = _int_or_none(payload.get("candidate", {}).get("total_score") if isinstance(payload.get("candidate"), dict) else None)
    delta_total = _int_or_none(payload.get("delta_total_score"))
    return backend, baseline_total, candidate_total, delta_total


def _base_summary(*, run_dir: Path, request: RecordedDemoRequest) -> dict[str, object]:
    return {
        "run_id": run_dir.name,
        "topic": request.topic,
        "output_dir": str(run_dir),
        "entrypoint": "openclaw_root",
        "demo_phase": "phase1_root_openclaw",
        "revision_backend": "openclaw",
        "initial_draft_path": str(run_dir / "initial_draft.md"),
        "best_draft_path": str(run_dir / "best_draft.md"),
        "bench_manifest_path": str(run_dir / "bench_manifest.json"),
        "bench_results_path": str(run_dir / "bench_results.json"),
        "bench_report_path": str(run_dir / "bench_report.md"),
        "bench_backend": "",
        "stop_reason": "not_started",
        "baseline_total_score": None,
        "candidate_total_score": None,
        "delta_total_score": None,
    }


def run_recorded_demo(
    request: RecordedDemoRequest,
    *,
    runtime: RecordedDemoRuntime | None = None,
) -> RecordedDemoResult:
    runtime = runtime or RecordedDemoRuntime()
    normalized_request = normalize_request(request)
    settings = load_settings()
    provider_config = resolve_provider_config(
        settings,
        provider=normalized_request.provider,
        model=normalized_request.model,
    )
    research_config = resolve_research_config(
        settings,
        breadth=normalized_request.research_breadth,
        depth=normalized_request.research_depth,
        concurrency=normalized_request.research_concurrency,
    )

    run_dir = _resolve_run_dir(settings.output_root, normalized_request)
    summary = _base_summary(run_dir=run_dir, request=normalized_request)
    summary_path = run_dir / "summary.json"
    artifact_writer = ArtifactWriter(run_dir)

    workspace_setup_factory = runtime.workspace_setup_factory or prepare_openclaw_workspace
    revision_agent_runner = runtime.revision_agent_runner or _run_openclaw_revision_agent
    evaluator_client_factory = runtime.evaluator_client_factory or build_client
    export_manifest_fn = runtime.export_manifest_fn or export_manifest
    evaluate_manifest_fn = runtime.evaluate_manifest_fn or evaluate_manifest
    render_report_fn = runtime.render_report_fn or render_report

    current_draft: DraftDocument | None = None
    best_draft: DraftDocument | None = None
    stop_reason = "error"
    client = None

    try:
        workspace_setup = workspace_setup_factory(
            provider_config=provider_config,
            required_skills=RECORDED_DEMO_REQUIRED_SKILLS,
            skills_dir=normalized_request.skills_dir,
            workspace_root=Path.cwd().resolve(),
            profile_name=normalized_request.profile_name,
            sync_skills=normalized_request.sync_skills,
        )
        client = evaluator_client_factory(provider_config)
        topic_draft_generator = runtime.topic_draft_generator or DeepResearchDraftGenerator(
            provider_config=provider_config,
            research_config=research_config,
        ).generate

        current_draft = topic_draft_generator(normalized_request.topic)
        best_draft = current_draft
        _write_markdown(run_dir / "initial_draft.md", current_draft)

        best_total_score: int | None = None
        for round_index in range(normalized_request.max_rounds or DEFAULT_MAX_ROUNDS):
            scores = evaluate_draft(client, current_draft)
            feedback_summary = build_feedback_summary(scores)
            round_result = build_round_result(
                round_index=round_index,
                draft=current_draft,
                scores=scores,
                feedback_summary=feedback_summary,
                provider=provider_config.provider,
                model=provider_config.model,
            )
            artifact_writer.write_round(round_result)

            if best_total_score is None or round_result.total_score > best_total_score:
                best_total_score = round_result.total_score
                best_draft = current_draft

            if round_index == (normalized_request.max_rounds or DEFAULT_MAX_ROUNDS) - 1:
                stop_reason = "max_rounds_reached"
                break

            revision_request = RevisionAgentRequest(
                profile_name=workspace_setup.profile_name,
                agent_id=normalized_request.agent_id,
                workspace_root=workspace_setup.workspace_root,
                env=workspace_setup.env,
                session_id=f"{run_dir.name}-revision-{round_index + 1}",
                draft_path=run_dir / f"draft_round_{round_index}.md",
                scores_path=run_dir / f"scores_round_{round_index}.json",
                feedback_path=run_dir / f"feedback_round_{round_index}.md",
                output_markdown_path=run_dir / f"draft_round_{round_index + 1}.md",
                output_json_path=run_dir / f"openclaw_revision_round_{round_index + 1}.json",
                stdout_path=run_dir / f"openclaw_revision_round_{round_index + 1}_stdout.log",
                stderr_path=run_dir / f"openclaw_revision_round_{round_index + 1}_stderr.log",
                timeout_seconds=normalized_request.agent_timeout_seconds or DEFAULT_AGENT_TIMEOUT_SECONDS,
                thinking_level=normalized_request.openclaw_thinking,
                revision_candidates=normalized_request.revision_candidates or DEFAULT_REVISION_CANDIDATES,
            )
            agent_result = revision_agent_runner(revision_request)
            revision_payload = json.loads(agent_result.output_json_path.read_text(encoding="utf-8"))
            revised_draft = _parse_markdown_draft(agent_result.output_markdown_path)
            revised_scores = evaluate_draft(client, revised_draft)
            revised_total_score = sum(item.score for item in revised_scores)

            rationale = _build_revision_rationale(
                round_index=round_index + 1,
                source_draft=current_draft,
                revised_draft=revised_draft,
                previous_total_score=round_result.total_score,
                revised_total_score=revised_total_score,
                revision_payload=revision_payload,
                agent_result=agent_result,
            )
            _write_text(run_dir / f"revision_rationale_round_{round_index + 1}.md", rationale)

            if revised_total_score <= round_result.total_score:
                stop_reason = "no_improving_revision"
                break

            current_draft = revised_draft

        if best_draft is None:
            raise RecordedDemoRunnerError("No draft was selected as best_draft.")

        _write_markdown(run_dir / "best_draft.md", best_draft)

        manifest_path = export_manifest_fn(
            candidate_run_dir=str(run_dir),
            baseline_draft_path=str(run_dir / "initial_draft.md"),
            output_path=str(run_dir / "bench_manifest.json"),
            candidate_label="final_draft",
            baseline_label="initial_draft",
        )
        results_path = evaluate_manifest_fn(
            manifest_path=str(manifest_path),
            output_path=str(run_dir / "bench_results.json"),
            provider=provider_config.provider,
            model=provider_config.model,
        )
        report_path, chart_path = render_report_fn(
            results_path=str(results_path),
            output_dir=str(run_dir),
        )
        bench_backend, baseline_total, candidate_total, delta_total = _read_bench_results(Path(results_path))

        summary.update(
            {
                "stop_reason": stop_reason,
                "initial_draft_path": str(run_dir / "initial_draft.md"),
                "best_draft_path": str(run_dir / "best_draft.md"),
                "bench_manifest_path": str(manifest_path),
                "bench_results_path": str(results_path),
                "bench_report_path": str(report_path),
                "bench_chart_path": str(chart_path),
                "bench_backend": bench_backend,
                "baseline_total_score": baseline_total,
                "candidate_total_score": candidate_total,
                "delta_total_score": delta_total,
                "provider": provider_config.provider,
                "model": provider_config.model,
                "profile_name": workspace_setup.profile_name,
            }
        )
        _write_json(summary_path, summary)

        return RecordedDemoResult(
            run_dir=str(run_dir),
            summary_path=str(summary_path),
            best_draft_path=str(run_dir / "best_draft.md"),
            bench_results_path=str(results_path),
            bench_report_path=str(report_path),
            bench_backend=bench_backend,
            stop_reason=stop_reason,
            baseline_total_score=baseline_total,
            candidate_total_score=candidate_total,
            delta_total_score=delta_total,
        )
    except Exception as exc:
        summary["stop_reason"] = "error"
        summary["error"] = str(exc)
        if (run_dir / "best_draft.md").exists():
            summary["best_draft_path"] = str(run_dir / "best_draft.md")
        _write_json(summary_path, summary)
        if isinstance(exc, RecordedDemoRunnerError):
            raise
        raise RecordedDemoRunnerError(str(exc)) from exc
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def _build_request(args: argparse.Namespace) -> RecordedDemoRequest:
    return RecordedDemoRequest(
        topic=args.topic,
        run_dir=args.run_dir,
        max_rounds=args.max_rounds,
        provider=args.provider,
        model=args.model,
        research_breadth=args.research_breadth,
        research_depth=args.research_depth,
        research_concurrency=args.research_concurrency,
        revision_candidates=args.revision_candidates,
        skills_dir=args.skills_dir,
        profile_name=args.profile_name,
        agent_id=args.agent_id,
        agent_timeout_seconds=args.agent_timeout_seconds,
        openclaw_thinking=args.openclaw_thinking,
        sync_skills=args.sync_skills,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = run_recorded_demo(_build_request(args))
        summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
        print(f"Run directory: {result.run_dir}")
        print(f"Stop reason: {result.stop_reason}")
        print(f"Best draft: {result.best_draft_path}")
        print(f"Bench results: {result.bench_results_path}")
        if summary.get("delta_total_score") is not None:
            print(f"Delta total score: {summary['delta_total_score']}")
        print(f"Summary: {result.summary_path}")
        return 0
    except (
        ConfigError,
        LLMError,
        OSError,
        OpenClawBenchError,
        OpenClawSetupError,
        ParseError,
        ResearchError,
        RecordedDemoRunnerError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
