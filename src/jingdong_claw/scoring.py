from __future__ import annotations

from prompt_templates import VALID_DIMENSIONS

from .models import DraftDocument, RoundResult, ScoreResult
from .parser import parse_score_output
from .prompts import build_evaluation_messages


def evaluate_draft(client: object, draft: DraftDocument) -> list[ScoreResult]:
    results: list[ScoreResult] = []
    for dimension in VALID_DIMENSIONS:
        messages = build_evaluation_messages(dimension, draft.title, draft.body)
        response = client.generate_text(messages, temperature=0.0)
        result = parse_score_output(response.text, dimension=dimension)
        results.append(result)
    return results


def build_feedback_summary(scores: list[ScoreResult]) -> str:
    ordered = sorted(scores, key=lambda item: (item.score, item.dimension))
    strongest = sorted(scores, key=lambda item: (-item.score, item.dimension))

    lines = ["Priority fixes:"]
    for item in ordered[:2]:
        lines.append(f"- {item.dimension} ({item.score}): {item.reason}")

    strong_dimensions = [item for item in strongest if item.score > 0][:2]
    if strong_dimensions:
        lines.append("Preserve these strengths:")
        for item in strong_dimensions:
            lines.append(f"- {item.dimension} ({item.score}): {item.reason}")

    lines.append(
        "Revision guidance: strengthen missing representative prior work, improve section flow, "
        "and make the synthesis more comparative where scores indicate weakness."
    )
    return "\n".join(lines)


def build_round_result(
    *,
    round_index: int,
    draft: DraftDocument,
    scores: list[ScoreResult],
    feedback_summary: str,
    provider: str,
    model: str,
) -> RoundResult:
    total_score = sum(score.score for score in scores)
    average_score = total_score / len(scores)
    return RoundResult(
        round_index=round_index,
        draft_title=draft.title,
        draft_body=draft.body,
        scores=scores,
        total_score=total_score,
        average_score=average_score,
        feedback_summary=feedback_summary,
        provider=provider,
        model=model,
    )
