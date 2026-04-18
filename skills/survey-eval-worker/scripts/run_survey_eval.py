from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SHARED_DIR = Path(__file__).resolve().parents[2] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from worker_common import bootstrap_repo_imports, load_draft_document, serialize_scores, write_json, write_text

bootstrap_repo_imports(__file__)

from prompt_templates import VALID_DIMENSIONS

from jingdong_claw.config import ConfigError, load_settings, resolve_provider_config
from jingdong_claw.llm import LLMError, build_client
from jingdong_claw.parser import ParseError
from jingdong_claw.scoring import build_feedback_summary, evaluate_draft


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score one survey draft with the repo's evaluator-only scoring path."
    )
    parser.add_argument("--title", default=None, help="Optional title when draft input is body-only.")
    draft_group = parser.add_mutually_exclusive_group(required=True)
    draft_group.add_argument("--draft-file", default=None)
    draft_group.add_argument("--draft-text", default=None)
    parser.add_argument("--provider", choices=["localapi", "deepseek"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-json", default=None, help="Optional path for the structured evaluation payload.")
    parser.add_argument("--feedback-file", default=None, help="Optional path for the feedback summary.")
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
        draft = load_draft_document(
            title=args.title,
            draft_file=args.draft_file,
            draft_text=args.draft_text,
        )
        client = build_client(provider_config)
        scores = evaluate_draft(client, draft)
        feedback_summary = build_feedback_summary(scores)
        total_score = sum(item.score for item in scores)
        average_score = total_score / len(scores)

        feedback_file = str(Path(args.feedback_file)) if args.feedback_file else None
        output_json = str(Path(args.output_json)) if args.output_json else None
        payload = {
            "title": draft.title,
            "provider": provider_config.provider,
            "model": provider_config.model,
            "dimension_order": list(VALID_DIMENSIONS),
            "total_score": total_score,
            "average_score": average_score,
            "scores": serialize_scores(scores),
            "feedback_summary": feedback_summary,
            "feedback_file": feedback_file,
            "output_json": output_json,
        }
        write_text(feedback_file, feedback_summary)
        write_json(output_json, payload)

        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    except (ConfigError, LLMError, OSError, ParseError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
