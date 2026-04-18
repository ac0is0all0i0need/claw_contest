from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SHARED_DIR = Path(__file__).resolve().parents[2] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from worker_common import bootstrap_repo_imports, write_json, write_markdown

bootstrap_repo_imports(__file__)

from jingdong_claw.config import ConfigError, load_settings, resolve_provider_config, resolve_research_config
from jingdong_claw.research import DeepResearchDraftGenerator, ResearchError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one survey draft from a topic using the repo's deep-research generator."
    )
    parser.add_argument("--topic", required=True, help="Survey topic to research.")
    parser.add_argument("--provider", choices=["localapi", "deepseek"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--research-breadth", type=int, default=None)
    parser.add_argument("--research-depth", type=int, default=None)
    parser.add_argument("--research-concurrency", type=int, default=None)
    parser.add_argument("--output-json", default=None, help="Optional path for the full JSON result payload.")
    parser.add_argument(
        "--output-markdown",
        default=None,
        help="Optional path for the generated draft as a markdown file.",
    )
    return parser


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
        research_config = resolve_research_config(
            settings,
            breadth=args.research_breadth,
            depth=args.research_depth,
            concurrency=args.research_concurrency,
        )
        draft = DeepResearchDraftGenerator(
            provider_config=provider_config,
            research_config=research_config,
        ).generate(args.topic)

        output_markdown = str(Path(args.output_markdown)) if args.output_markdown else None
        output_json = str(Path(args.output_json)) if args.output_json else None
        payload = {
            "topic": args.topic,
            "title": draft.title,
            "body": draft.body,
            "provider": provider_config.provider,
            "model": provider_config.model,
            "research_config": {
                "breadth": research_config.breadth,
                "depth": research_config.depth,
                "concurrency": research_config.concurrency,
                "firecrawl_base_url": research_config.firecrawl_base_url,
            },
            "output_markdown": output_markdown,
            "output_json": output_json,
        }
        write_markdown(output_markdown, title=draft.title, body=draft.body)
        write_json(output_json, payload)

        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    except (ConfigError, OSError, ResearchError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
