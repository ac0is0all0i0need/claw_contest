from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SHARED_DIR = Path(__file__).resolve().parents[2] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from worker_common import (
    bootstrap_repo_imports,
    load_draft_document,
    load_score_results,
    serialize_scores,
    write_json,
    write_markdown,
)

bootstrap_repo_imports(__file__)

from jingdong_claw.config import ConfigError, load_settings, resolve_provider_config
from jingdong_claw.llm import LLMError, build_client
from jingdong_claw.parser import ParseError
from jingdong_claw.pipeline import select_best_revision_candidate
from jingdong_claw.scoring import build_feedback_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select one improved survey revision from existing draft and evaluation evidence."
    )
    parser.add_argument("--title", default=None, help="Optional title when draft input is body-only.")
    draft_group = parser.add_mutually_exclusive_group(required=True)
    draft_group.add_argument("--draft-file", default=None)
    draft_group.add_argument("--draft-text", default=None)
    parser.add_argument("--eval-json", default=None, help="Evaluation worker JSON payload.")
    parser.add_argument("--scores-json", default=None, help="Score list or evaluation payload JSON.")
    parser.add_argument("--feedback-file", default=None, help="Optional plain-text feedback override.")
    parser.add_argument("--feedback-text", default=None, help="Optional inline feedback override.")
    parser.add_argument("--revision-candidates", type=int, default=2)
    parser.add_argument("--provider", choices=["localapi", "deepseek"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-json", default=None, help="Optional path for the revised JSON payload.")
    parser.add_argument(
        "--output-markdown",
        default=None,
        help="Optional path for the selected revised draft as markdown.",
    )
    return parser


def _resolve_feedback_summary(
    *,
    feedback_file: str | None,
    feedback_text: str | None,
    embedded_feedback: str | None,
    scores: list[object],
) -> str:
    if feedback_text is not None:
        text = feedback_text.strip()
        if not text:
            raise ValueError("--feedback-text must not be empty.")
        return text
    if feedback_file is not None:
        text = Path(feedback_file).read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Feedback file '{feedback_file}' is empty.")
        return text
    if embedded_feedback:
        return embedded_feedback
    return build_feedback_summary(scores)  # type: ignore[arg-type]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.eval_json and args.scores_json:
            raise ValueError("Use only one of --eval-json or --scores-json.")
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
        scores, embedded_feedback = load_score_results(
            eval_json=args.eval_json,
            scores_json=args.scores_json,
        )
        feedback_summary = _resolve_feedback_summary(
            feedback_file=args.feedback_file,
            feedback_text=args.feedback_text,
            embedded_feedback=embedded_feedback,
            scores=scores,
        )
        client = build_client(provider_config)
        revised_draft, revised_scores, revised_feedback_summary = select_best_revision_candidate(
            client,
            draft=draft,
            scores=scores,
            feedback_summary=feedback_summary,
            revision_candidates=args.revision_candidates,
        )
        best_total_score = sum(item.score for item in revised_scores)
        best_average_score = best_total_score / len(revised_scores)

        output_markdown = str(Path(args.output_markdown)) if args.output_markdown else None
        output_json = str(Path(args.output_json)) if args.output_json else None
        payload = {
            "source_title": draft.title,
            "title": revised_draft.title,
            "body": revised_draft.body,
            "provider": provider_config.provider,
            "model": provider_config.model,
            "revision_candidates": args.revision_candidates,
            "best_total_score": best_total_score,
            "best_average_score": best_average_score,
            "scores": serialize_scores(revised_scores),
            "feedback_summary": revised_feedback_summary,
            "output_markdown": output_markdown,
            "output_json": output_json,
        }
        write_markdown(output_markdown, title=revised_draft.title, body=revised_draft.body)
        write_json(output_json, payload)

        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    except (ConfigError, LLMError, OSError, ParseError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
