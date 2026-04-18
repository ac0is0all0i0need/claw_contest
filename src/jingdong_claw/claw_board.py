from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import ConfigError
from .openclaw_bench import OpenClawBenchError, evaluate_manifest, export_manifest, render_report
from .openclaw_bridge import (
    OpenClawDemoRequest,
    OpenClawDemoResult,
    OpenClawBridgeError,
    run_openclaw_demo,
)


class BoardDemoError(RuntimeError):
    """Raised when the dashboard-shell demo cannot finish its wrapper steps."""


@dataclass(slots=True)
class BoardDemoRequest:
    topic: str
    run_dir: str
    max_rounds: int | None = None
    provider: str | None = None
    model: str | None = None
    research_breadth: int | None = None
    research_depth: int | None = None
    research_concurrency: int | None = None
    revision_candidates: int | None = None
    openclaw_backend: str = "openclaw_dashboard"


@dataclass(slots=True)
class BoardDemoResult:
    run_dir: str
    summary_path: str
    workflow_trace_path: str
    board_summary_path: str
    board_report_path: str
    board_timeline_path: str
    initial_draft_path: str
    best_draft_path: str
    bench_manifest_path: str
    bench_results_path: str
    bench_report_path: str
    bench_chart_path: str
    stop_reason: str
    best_total_score: int
    delta_total_score: int | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Board-shell wrapper for the stable Python self-evolution workflow."
    )
    parser.add_argument("--topic", required=True)
    parser.add_argument("--run-dir", required=True, help="Exact run directory for all board demo artifacts.")
    parser.add_argument("--output-json", default=None, help="Optional path for the board summary payload.")
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--provider", choices=["localapi", "deepseek"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--research-breadth", type=int, default=None)
    parser.add_argument("--research-depth", type=int, default=None)
    parser.add_argument("--research-concurrency", type=int, default=None)
    parser.add_argument("--revision-candidates", type=int, default=None)
    parser.add_argument("--openclaw-backend", default="openclaw_dashboard")
    return parser


def _clean_optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _normalize_request(request: BoardDemoRequest) -> BoardDemoRequest:
    topic = _clean_optional_str(request.topic)
    run_dir = _clean_optional_str(request.run_dir)
    openclaw_backend = _clean_optional_str(request.openclaw_backend) or "openclaw_dashboard"
    if topic is None:
        raise BoardDemoError("topic must not be empty.")
    if run_dir is None:
        raise BoardDemoError("run_dir must not be empty.")
    if request.max_rounds is not None and request.max_rounds < 1:
        raise BoardDemoError("max_rounds must be at least 1.")
    if request.revision_candidates is not None and request.revision_candidates < 1:
        raise BoardDemoError("revision_candidates must be at least 1.")

    return BoardDemoRequest(
        topic=topic,
        run_dir=run_dir,
        max_rounds=request.max_rounds,
        provider=_clean_optional_str(request.provider),
        model=_clean_optional_str(request.model),
        research_breadth=request.research_breadth,
        research_depth=request.research_depth,
        research_concurrency=request.research_concurrency,
        revision_candidates=request.revision_candidates,
        openclaw_backend=openclaw_backend,
    )


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _int_or_none(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _copy_text_file(source: Path, target: Path) -> Path:
    if not source.exists():
        raise BoardDemoError(f"Required artifact is missing: {source}")
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def _build_board_timeline(workflow_trace: dict[str, object]) -> list[dict[str, object]]:
    stage_records = workflow_trace.get("stage_records")
    if not isinstance(stage_records, list):
        return []

    timeline: list[dict[str, object]] = []
    for index, item in enumerate(stage_records, start=1):
        if not isinstance(item, dict):
            continue
        timeline.append(
            {
                "position": index,
                "stage_name": item.get("stage_name"),
                "round_index": item.get("round_index"),
                "status": item.get("status"),
                "started_at": item.get("started_at"),
                "ended_at": item.get("ended_at"),
                "summary": item.get("summary"),
            }
        )
    return timeline


def _write_json(path: Path, payload: dict[str, object] | list[dict[str, object]]) -> Path:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _write_board_report(path: Path, payload: dict[str, object]) -> Path:
    timeline = payload.get("timeline", [])
    lines = [
        "# Claw Board Summary",
        "",
        f"- Entrypoint: {payload.get('entrypoint')}",
        f"- Backend: {payload.get('backend')}",
        f"- Topic: {payload.get('topic')}",
        f"- Run directory: {payload.get('run_dir')}",
        f"- Stop reason: {payload.get('stop_reason')}",
        f"- Round count: {payload.get('round_count')}",
        f"- Best round index: {payload.get('best_round_index')}",
        f"- Best total score: {payload.get('best_total_score')}",
        f"- Delta total score: {payload.get('delta_total_score')}",
        "",
        "## Workflow Timeline",
        "",
    ]

    if isinstance(timeline, list) and timeline:
        for item in timeline:
            if not isinstance(item, dict):
                continue
            lines.append(
                "- "
                + f"{item.get('position')}. {item.get('stage_name')} "
                + f"(round={item.get('round_index')}, status={item.get('status')})"
            )
    else:
        lines.append("- No workflow timeline was recorded.")

    lines.extend(
        [
            "",
            "## Key Artifacts",
            "",
            f"- Initial draft: {payload.get('initial_draft_path')}",
            f"- Best draft: {payload.get('best_draft_path')}",
            f"- Workflow trace: {payload.get('workflow_trace_path')}",
            f"- Bench manifest: {payload.get('bench_manifest_path')}",
            f"- Bench results: {payload.get('bench_results_path')}",
            f"- Bench report: {payload.get('bench_report_path')}",
            f"- Board timeline: {payload.get('board_timeline_path')}",
            f"- Board summary: {payload.get('board_summary_path')}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _prepare_named_drafts(run_dir: Path, summary: dict[str, object]) -> tuple[Path, Path]:
    initial_draft_path = _copy_text_file(run_dir / "draft_round_0.md", run_dir / "initial_draft.md")
    best_round_index = _int_or_none(summary.get("best_round_index"))
    if best_round_index is None or best_round_index < 0:
        raise BoardDemoError("summary.json does not contain a valid best_round_index.")
    best_source = run_dir / f"draft_round_{best_round_index}.md"
    best_draft_path = _copy_text_file(best_source, run_dir / "best_draft.md")
    return initial_draft_path, best_draft_path


def _build_bridge_request(request: BoardDemoRequest) -> OpenClawDemoRequest:
    return OpenClawDemoRequest(
        mode="topic",
        topic=request.topic,
        rounds=request.max_rounds,
        provider=request.provider,
        model=request.model,
        run_dir=request.run_dir,
        research_breadth=request.research_breadth,
        research_depth=request.research_depth,
        research_concurrency=request.research_concurrency,
        revision_candidates=request.revision_candidates,
        force_local_only=True,
    )


def run_board_demo(request: BoardDemoRequest) -> BoardDemoResult:
    normalized_request = _normalize_request(request)
    bridge_result = run_openclaw_demo(
        _build_bridge_request(normalized_request),
        openclaw_backend=normalized_request.openclaw_backend,
    )

    return _finalize_board_run(normalized_request, bridge_result)


def _finalize_board_run(request: BoardDemoRequest, bridge_result: OpenClawDemoResult) -> BoardDemoResult:
    run_dir = Path(bridge_result.output_dir).resolve()
    summary_path = Path(bridge_result.summary_path).resolve()
    workflow_trace_path = (run_dir / "workflow_trace.json").resolve()

    summary = _read_json(summary_path)
    workflow_trace = _read_json(workflow_trace_path)
    initial_draft_path, best_draft_path = _prepare_named_drafts(run_dir, summary)

    manifest_path = export_manifest(
        candidate_run_dir=str(run_dir),
        baseline_draft_path=str(initial_draft_path),
        output_path=str(run_dir / "bench_manifest.json"),
        candidate_label="final_draft",
        baseline_label="initial_draft",
    )
    results_path = evaluate_manifest(
        manifest_path=str(manifest_path),
        output_path=str(run_dir / "bench_results.json"),
        provider=request.provider,
        model=request.model,
    )
    bench_report_path, bench_chart_path = render_report(
        results_path=str(results_path),
        output_dir=str(run_dir),
    )

    bench_results = _read_json(results_path)
    baseline = bench_results.get("baseline", {})
    candidate = bench_results.get("candidate", {})
    baseline_total_score = _int_or_none(baseline.get("total_score") if isinstance(baseline, dict) else None)
    candidate_total_score = _int_or_none(candidate.get("total_score") if isinstance(candidate, dict) else None)
    delta_total_score = _int_or_none(bench_results.get("delta_total_score"))
    timeline = _build_board_timeline(workflow_trace)
    board_timeline_path = _write_json(run_dir / "board_timeline.json", timeline)

    board_summary_path = run_dir / "board_summary.json"
    board_report_path = run_dir / "board_summary.md"
    board_summary = {
        "entrypoint": "openclaw_dashboard",
        "backend": "python_pipeline",
        "topic": request.topic,
        "run_dir": str(run_dir),
        "openclaw_backend": request.openclaw_backend,
        "summary_path": str(summary_path),
        "workflow_trace_path": str(workflow_trace_path),
        "board_summary_path": str(board_summary_path),
        "board_report_path": str(board_report_path),
        "board_timeline_path": str(board_timeline_path),
        "initial_draft_path": str(initial_draft_path),
        "best_draft_path": str(best_draft_path),
        "bench_manifest_path": str(manifest_path),
        "bench_results_path": str(results_path),
        "bench_report_path": str(bench_report_path),
        "bench_chart_path": str(bench_chart_path),
        "bench_backend": bench_results.get("backend"),
        "round_count": bridge_result.round_count,
        "best_round_index": bridge_result.best_round_index,
        "best_total_score": bridge_result.best_total_score,
        "stop_reason": bridge_result.stop_reason,
        "baseline_total_score": baseline_total_score,
        "candidate_total_score": candidate_total_score,
        "delta_total_score": delta_total_score,
        "timeline": timeline,
    }
    _write_json(board_summary_path, board_summary)
    _write_board_report(board_report_path, board_summary)

    return BoardDemoResult(
        run_dir=str(run_dir),
        summary_path=str(summary_path),
        workflow_trace_path=str(workflow_trace_path),
        board_summary_path=str(board_summary_path),
        board_report_path=str(board_report_path),
        board_timeline_path=str(board_timeline_path),
        initial_draft_path=str(initial_draft_path),
        best_draft_path=str(best_draft_path),
        bench_manifest_path=str(manifest_path),
        bench_results_path=str(results_path),
        bench_report_path=str(bench_report_path),
        bench_chart_path=str(bench_chart_path),
        stop_reason=bridge_result.stop_reason,
        best_total_score=bridge_result.best_total_score,
        delta_total_score=delta_total_score,
    )


def _build_request(args: argparse.Namespace) -> BoardDemoRequest:
    return BoardDemoRequest(
        topic=args.topic,
        run_dir=args.run_dir,
        max_rounds=args.max_rounds,
        provider=args.provider,
        model=args.model,
        research_breadth=args.research_breadth,
        research_depth=args.research_depth,
        research_concurrency=args.research_concurrency,
        revision_candidates=args.revision_candidates,
        openclaw_backend=args.openclaw_backend,
    )


def _build_stdout_payload(result: BoardDemoResult) -> dict[str, object]:
    payload = _read_json(Path(result.board_summary_path))
    payload["output_json"] = None
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = run_board_demo(_build_request(args))
        payload = _build_stdout_payload(result)
        payload["output_json"] = str(Path(args.output_json).resolve()) if args.output_json else None
        if args.output_json:
            Path(args.output_json).resolve().write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    except (
        BoardDemoError,
        ConfigError,
        OSError,
        OpenClawBenchError,
        OpenClawBridgeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
