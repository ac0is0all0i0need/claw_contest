from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SHARED_DIR = Path(__file__).resolve().parents[2] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from worker_common import bootstrap_repo_imports, write_json

bootstrap_repo_imports(__file__)

from jingdong_claw.claw_board import BoardDemoError, BoardDemoRequest, run_board_demo
from jingdong_claw.config import ConfigError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the repository's dashboard-shell survey demo through the repo-owned skill."
    )
    parser.add_argument("--topic", required=True)
    parser.add_argument("--run-dir", required=True, help="Exact run directory for the board demo.")
    parser.add_argument("--output-json", default=None, help="Optional path for the board summary payload.")
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--provider", choices=["localapi", "deepseek"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--research-breadth", type=int, default=None)
    parser.add_argument("--research-depth", type=int, default=None)
    parser.add_argument("--research-concurrency", type=int, default=None)
    parser.add_argument("--revision-candidates", type=int, default=None)
    return parser


def _build_payload(args: argparse.Namespace) -> dict[str, object]:
    result = run_board_demo(
        BoardDemoRequest(
            topic=args.topic,
            run_dir=args.run_dir,
            max_rounds=args.max_rounds,
            provider=args.provider,
            model=args.model,
            research_breadth=args.research_breadth,
            research_depth=args.research_depth,
            research_concurrency=args.research_concurrency,
            revision_candidates=args.revision_candidates,
        )
    )
    summary = json.loads(Path(result.board_summary_path).read_text(encoding="utf-8"))

    return {
        "topic": args.topic,
        "entrypoint": summary.get("entrypoint"),
        "backend": summary.get("backend"),
        "openclaw_backend": summary.get("openclaw_backend"),
        "bench_backend": summary.get("bench_backend"),
        "stop_reason": result.stop_reason,
        "output_dir": result.run_dir,
        "initial_draft_path": summary.get("initial_draft_path"),
        "best_draft_path": result.best_draft_path,
        "workflow_trace_path": result.workflow_trace_path,
        "board_summary_path": result.board_summary_path,
        "board_report_path": result.board_report_path,
        "bench_results_path": result.bench_results_path,
        "bench_report_path": result.bench_report_path,
        "summary_path": result.summary_path,
        "baseline_total_score": summary.get("baseline_total_score"),
        "candidate_total_score": summary.get("candidate_total_score"),
        "delta_total_score": result.delta_total_score,
        "output_json": str(Path(args.output_json)) if args.output_json else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        payload = _build_payload(args)
        write_json(args.output_json, payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    except (
        BoardDemoError,
        ConfigError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
