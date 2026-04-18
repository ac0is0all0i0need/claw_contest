from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .artifacts import slugify
from .claw_board import BoardDemoError, BoardDemoRequest, run_board_demo
from .config import ConfigError, load_settings, resolve_provider_config
from .openclaw_conductor import (
    OpenClawConductorError,
    OpenClawNativeRequest,
    run_openclaw_native_demo,
)
from .openclaw_setup import (
    DEFAULT_OPENCLAW_PROFILE,
    ROOT_LAUNCH_SKILLS,
    OpenClawSetupError,
    build_root_launch_message,
    build_root_launch_command,
    prepare_openclaw_workspace,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Developer helper for preparing the OpenClaw dashboard demo or debugging its local runners."
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--rounds", type=int, default=None, help="Hard cap for the recorded or native demo.")
    common.add_argument("--provider", choices=["localapi", "deepseek"], default=None)
    common.add_argument("--model", default=None)
    common.add_argument("--output-dir", default=None, help="Base directory for run artifacts.")
    common.add_argument("--run-dir", default=None, help="Exact run directory for the recorded demo runner.")
    common.add_argument("--revision-candidates", type=int, default=None)
    common.add_argument(
        "--execution-mode",
        choices=["hybrid-recorded", "openclaw", "openclaw-strict"],
        default="hybrid-recorded",
    )
    common.add_argument("--skills-dir", default=None, help="Repo-owned source skill directory.")
    common.add_argument("--profile-name", default=DEFAULT_OPENCLAW_PROFILE)
    common.add_argument("--session-id", default=None, help="Optional session id for the printed root launch command.")
    common.add_argument(
        "--run-local-runner",
        action="store_true",
        help="After preparing OpenClaw root launch, run the recorded Python runner locally for debugging.",
    )
    common.add_argument("--agent-max-turns", type=int, default=None)
    common.add_argument("--agent-timeout-seconds", type=int, default=None)
    common.add_argument("--agent-max-retries", type=int, default=None)
    common.add_argument("--openclaw-thinking", default="low")

    subparsers = parser.add_subparsers(dest="command", required=True)

    topic_parser = subparsers.add_parser("topic", parents=[common], help="Prepare or run from a research topic.")
    topic_parser.add_argument("--topic", required=True)
    topic_parser.add_argument("--research-breadth", type=int, default=None)
    topic_parser.add_argument("--research-depth", type=int, default=None)
    topic_parser.add_argument("--research-concurrency", type=int, default=None)

    paper_parser = subparsers.add_parser("paper", parents=[common], help="Run the native conductor from an existing draft.")
    paper_parser.add_argument("--title", required=True)
    body_group = paper_parser.add_mutually_exclusive_group(required=True)
    body_group.add_argument("--paper-file")
    body_group.add_argument("--paper-text")

    return parser


def _load_paper_body(args: argparse.Namespace) -> str:
    if args.paper_text:
        return args.paper_text
    return Path(args.paper_file).read_text(encoding="utf-8")


def _build_native_request(args: argparse.Namespace) -> OpenClawNativeRequest:
    if args.command == "topic":
        return OpenClawNativeRequest(
            mode="topic",
            topic=args.topic,
            rounds=args.rounds,
            provider=args.provider,
            model=args.model,
            output_dir=args.output_dir,
            revision_candidates=args.revision_candidates,
            research_breadth=args.research_breadth,
            research_depth=args.research_depth,
            research_concurrency=args.research_concurrency,
            execution_mode=args.execution_mode,
            skills_dir=args.skills_dir,
            agent_max_turns=args.agent_max_turns,
            agent_timeout_seconds=args.agent_timeout_seconds,
            agent_max_retries=args.agent_max_retries,
        )

    return OpenClawNativeRequest(
        mode="paper",
        title=args.title,
        body=_load_paper_body(args),
        rounds=args.rounds,
        provider=args.provider,
        model=args.model,
        output_dir=args.output_dir,
        revision_candidates=args.revision_candidates,
        execution_mode=args.execution_mode,
        skills_dir=args.skills_dir,
        agent_max_turns=args.agent_max_turns,
        agent_timeout_seconds=args.agent_timeout_seconds,
        agent_max_retries=args.agent_max_retries,
    )


def _build_board_request(args: argparse.Namespace) -> BoardDemoRequest:
    return BoardDemoRequest(
        topic=args.topic,
        run_dir=args.run_dir,
        max_rounds=args.rounds,
        provider=args.provider,
        model=args.model,
        research_breadth=args.research_breadth,
        research_depth=args.research_depth,
        research_concurrency=args.research_concurrency,
        revision_candidates=args.revision_candidates,
    )


def _suggest_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        return Path(args.run_dir).resolve()

    settings = load_settings()
    output_root = Path(args.output_dir) if args.output_dir else settings.output_root
    topic_slug = slugify(args.topic)
    return (output_root / f"openclaw-root-{topic_slug}").resolve()


def _prepare_dashboard_launch(args: argparse.Namespace) -> tuple[Path, str, str]:
    settings = load_settings()
    provider_config = resolve_provider_config(
        settings,
        provider=args.provider,
        model=args.model,
    )
    setup = prepare_openclaw_workspace(
        provider_config=provider_config,
        required_skills=ROOT_LAUNCH_SKILLS,
        skills_dir=args.skills_dir,
        workspace_root=Path.cwd().resolve(),
        profile_name=args.profile_name,
        sync_skills=True,
    )
    run_dir = _suggest_run_dir(args)
    message = build_root_launch_message(
        topic=args.topic,
        run_dir=run_dir,
    )
    command = build_root_launch_command(
        profile_name=setup.profile_name,
        topic=args.topic,
        run_dir=run_dir,
        session_id=args.session_id,
    )
    return run_dir, message, command


def _run_hybrid_recorded(args: argparse.Namespace) -> int:
    if args.command != "topic":
        raise BoardDemoError("hybrid-recorded mode currently supports topic runs only.")

    run_dir, message, command = _prepare_dashboard_launch(args)
    print("OpenClaw frontend: dashboard")
    print(f"Profile: {args.profile_name}")
    print(f"Workspace skills: {(Path.cwd().resolve() / 'skills')}")
    print("Dashboard command: openclaw dashboard")
    print(f"Suggested dashboard message: {message}")
    print(f"Direct agent fallback: {command}")

    if not args.run_local_runner:
        print("Local backend runner: skipped")
        return 0

    request = _build_board_request(args)
    request.run_dir = str(run_dir)
    result = run_board_demo(request)
    summary = json.loads(Path(result.board_summary_path).read_text(encoding="utf-8"))
    print(f"Run directory: {result.run_dir}")
    print(f"Stop reason: {result.stop_reason}")
    print(f"Best draft: {result.best_draft_path}")
    if summary.get("delta_total_score") is not None:
        print(f"Delta total score: {summary['delta_total_score']}")
    print(f"Board summary: {result.board_summary_path}")
    return 0


def _read_json(path: str) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _run_native_mode(args: argparse.Namespace) -> int:
    request = _build_native_request(args)
    result = run_openclaw_native_demo(request)
    summary = _read_json(result.summary_path)
    self_eval = _read_json(result.self_eval_path)

    print(f"OpenClaw backend: {result.openclaw_backend}")
    print(f"Execution mode: {result.execution_mode}")
    print(f"Run directory: {result.output_dir}")
    print(f"Strict compliant: {result.strict_compliant}")
    if isinstance(self_eval.get("total_score"), int):
        print(f"Best total score: {self_eval['total_score']}")
    print(f"Best draft: {result.best_draft_path}")
    print(f"Stop reason: {result.stop_reason}")
    if summary.get("process_report_path"):
        print(f"Process report: {summary['process_report_path']}")
    print(f"Summary: {result.summary_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.execution_mode == "hybrid-recorded":
            return _run_hybrid_recorded(args)
        return _run_native_mode(args)
    except (
        ConfigError,
        OpenClawConductorError,
        OpenClawSetupError,
        OSError,
        BoardDemoError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
