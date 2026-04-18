from __future__ import annotations

from textwrap import dedent

from prompt_templates import VALID_DIMENSIONS, build_review_messages_from_preset

from .models import ChatMessage, ScoreResult


def _format_score_list(items: list[ScoreResult]) -> str:
    return "\n".join(f"- {item.dimension} ({item.score}): {item.reason}" for item in items)


def build_generation_messages(topic: str) -> list[ChatMessage]:
    system_prompt = dedent(
        """
        You write strong, citation-aware literature surveys and review drafts.
        Produce a structured draft with a clear thesis, organized sections, representative prior work,
        comparison or synthesis across methods, limitations, and future directions.
        If exact citations are unavailable, use placeholder citations such as [Author, Year] rather than
        omitting references entirely.

        Return only the following XML-like format:
        <title>Your title</title>
        <body>
        Full markdown body here
        </body>
        """
    ).strip()
    user_prompt = dedent(
        f"""
        Research topic: {topic}

        Write an initial survey or review draft for this topic.
        """
    ).strip()
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_evaluation_messages(dimension_name: str, paper_title: str, paper_content: str) -> list[ChatMessage]:
    if dimension_name not in VALID_DIMENSIONS:
        valid = ", ".join(VALID_DIMENSIONS)
        raise ValueError(f"Unknown dimension '{dimension_name}'. Valid dimensions: {valid}")
    return build_review_messages_from_preset(
        dimension_name=dimension_name,
        paper_title=paper_title,
        paper_content=paper_content,
    )


def format_score_report(scores: list[ScoreResult]) -> str:
    return "\n".join(
        f"- {score.dimension}: {score.score}\n  Reason: {score.reason}"
        for score in scores
    )


def build_revision_objectives(scores: list[ScoreResult]) -> str:
    weakest = sorted(scores, key=lambda item: (item.score, item.dimension))
    strengths = [item for item in sorted(scores, key=lambda item: (-item.score, item.dimension)) if item.score > 0]
    lines: list[str] = []

    lines.append("Fix these weak dimensions with concrete edits:")
    for item in weakest[:2]:
        if item.dimension == "Comprehensiveness":
            lines.append(
                "- Comprehensiveness: broaden representative primary literature coverage using concrete named papers, "
                "surveys, or benchmarks already supported by the evidence. Remove weak blog/news style citations and "
                "do not replace grounded references with placeholders."
            )
        elif item.dimension == "Criticalness":
            lines.append(
                "- Criticalness: add explicit comparisons, trade-offs, limitations, and sharper synthesis. "
                "Do not only list papers; explain what differs across approaches and why it matters."
            )
        elif item.dimension == "Structure":
            lines.append(
                "- Structure: tighten section ordering, remove overlap, and make transitions explicit so the survey "
                "progresses from definition to methods to comparison to open problems."
            )
        elif item.dimension == "Readability":
            lines.append(
                "- Readability: shorten dense sentences, reduce repetition, and keep claims precise and easy to follow."
            )

    if strengths:
        lines.append("Preserve these strengths:")
        lines.extend(f"- Keep {item.dimension} strong: {item.reason}" for item in strengths[:2])

    lines.append(
        "Source-quality rules: prefer named papers, surveys, benchmarks, and primary research sources. "
        "Avoid adding weak ecosystem/blog/news references unless they are necessary and clearly secondary."
    )
    return "\n".join(lines)


def build_revision_strategy(candidate_index: int, candidate_count: int) -> str:
    if candidate_count <= 1:
        return (
            "Produce one strong revision that improves the evaluator's weakest dimensions while preserving grounded evidence."
        )
    if candidate_index == 1:
        return (
            "Strategy A: conservative evidence-preserving revision. Keep the strongest grounded claims, remove weak or "
            "speculative references, tighten structure, and make the draft cleaner without changing its core thesis."
        )
    if candidate_index == 2:
        return (
            "Strategy B: comparative synthesis revision. Add sharper cross-method comparisons, limitations, and higher-value "
            "analysis while preserving grounded references and concrete evidence."
        )
    return (
        "Strategy C: coverage-balancing revision. Improve representative coverage and critical synthesis together, but do not "
        "pad the draft with low-value or weakly grounded citations."
    )


def build_revision_messages(
    *,
    current_title: str,
    current_body: str,
    scores: list[ScoreResult],
    feedback_summary: str,
    candidate_index: int = 1,
    candidate_count: int = 1,
) -> list[ChatMessage]:
    system_prompt = dedent(
        """
        You revise research surveys and review drafts from structured evaluator feedback.
        Your goal is to produce a revised draft that should score higher than the current draft.
        Improve the weakest dimensions first, but do not damage dimensions that already scored well.
        Preserve grounded references and concrete evidence already present in the draft.
        Do not replace named papers or concrete sources with generic placeholders.
        Prefer primary research papers, surveys, and benchmarks over blogs, news articles, or weak ecosystem links.
        Add critical comparison and synthesis, not just more citations.
        Make substantive edits rather than surface paraphrases.

        Return only the following XML-like format:
        <title>Your revised title</title>
        <body>
        Full revised markdown body here
        </body>
        """
    ).strip()
    user_prompt = dedent(
        f"""
        Revise the following survey or review draft.

        Current Title:
        {current_title}

        Current Draft:
        {current_body}

        Structured Scores:
        {format_score_report(scores)}

        Weakness Summary:
        {_format_score_list(sorted(scores, key=lambda item: (item.score, item.dimension))[:2])}

        Revision Objectives:
        {build_revision_objectives(scores)}

        Revision Brief:
        {feedback_summary}

        Revision candidate {candidate_index} of {candidate_count}
        {build_revision_strategy(candidate_index, candidate_count)}

        Improve low-scoring dimensions, especially missing representative work, weak structure,
        and shallow synthesis when they are present. Preserve clarity where already strong.
        """
    ).strip()
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
