from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .artifacts import ArtifactWriter, create_run_dir
from .config import (
    ConfigError,
    load_settings,
    resolve_provider_config,
    resolve_research_config,
)
from .llm import LLMError, build_client
from .parser import ParseError
from .pipeline import SelfEvolutionPipeline
from .research import DeepResearchDraftGenerator, ResearchError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI self-evolution prototype for research survey drafts.")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--rounds", type=int, default=None, help="Number of evaluated drafts to keep.")
    common.add_argument("--provider", choices=["localapi", "deepseek"], default=None)
    common.add_argument("--model", default=None)
    common.add_argument("--output-dir", default=None, help="Base directory for run artifacts.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    topic_parser = subparsers.add_parser("topic", parents=[common], help="Generate from a topic and iterate.")
    topic_parser.add_argument("--topic", required=True)
    topic_parser.add_argument(
        "--research-breadth",
        type=int,
        default=None,
        help="Number of search branches per deep-research step.",
    )
    topic_parser.add_argument(
        "--research-depth",
        type=int,
        default=None,
        help="Recursive depth for deep-research topic exploration.",
    )
    topic_parser.add_argument(
        "--research-concurrency",
        type=int,
        default=None,
        help="Maximum parallel deep-research search tasks.",
    )

    paper_parser = subparsers.add_parser("paper", parents=[common], help="Start from an existing draft.")
    paper_parser.add_argument("--title", required=True)
    body_group = paper_parser.add_mutually_exclusive_group(required=True)
    body_group.add_argument("--paper-file")
    body_group.add_argument("--paper-text")

    return parser


def _load_paper_body(args: argparse.Namespace) -> str:
    if args.paper_text:
        return args.paper_text
    return Path(args.paper_file).read_text(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
        provider_config = resolve_provider_config(
            settings,
            provider=args.provider,
            model=args.model,
        )
        rounds = args.rounds if args.rounds is not None else settings.default_rounds
        if rounds < 1:
            raise ConfigError("--rounds must be at least 1.")

        label = args.topic if args.command == "topic" else args.title
        output_root = Path(args.output_dir) if args.output_dir else settings.output_root
        run_dir = create_run_dir(output_root, label)
        topic_draft_generator = None

        if args.command == "topic":
            research_config = resolve_research_config(
                settings,
                breadth=args.research_breadth,
                depth=args.research_depth,
                concurrency=args.research_concurrency,
            )
            topic_draft_generator = DeepResearchDraftGenerator(
                provider_config=provider_config,
                research_config=research_config,
            ).generate

        pipeline = SelfEvolutionPipeline(
            client=build_client(provider_config),
            artifact_writer=ArtifactWriter(run_dir),
            topic_draft_generator=topic_draft_generator,
        )

        print(f"Provider: {provider_config.provider}")
        print(f"Model: {provider_config.model}")
        print(f"Run directory: {run_dir}")

        if args.command == "topic":
            result = pipeline.run_topic(topic=args.topic, rounds=rounds)
        else:
            result = pipeline.run_paper(
                title=args.title,
                body=_load_paper_body(args),
                rounds=rounds,
            )

        print(f"Best round: {result.best_round_index}")
        print(f"Best total score: {result.best_total_score}")
        print(f"Stop reason: {result.stop_reason}")
        print(f"Artifacts: {result.output_dir}")
        return 0
    except (ConfigError, ParseError, LLMError, OSError, ResearchError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
