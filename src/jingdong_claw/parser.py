from __future__ import annotations

import re

from .models import DraftDocument, ScoreResult


class ParseError(RuntimeError):
    """Raised when model output cannot be parsed into the required structure."""


_STRICT_REASON_RE = re.compile(r"<reason>\s*(.*?)\s*</reason>", re.IGNORECASE | re.DOTALL)
_STRICT_SCORE_RE = re.compile(r"<score>\s*(-2|-1|1|2)\s*</score>", re.IGNORECASE | re.DOTALL)
_FALLBACK_SCORE_RE = re.compile(r"(?i)\bscore\b\s*[:=]\s*(-2|-1|1|2)\b")
_TITLE_RE = re.compile(r"<title>\s*(.*?)\s*</title>", re.IGNORECASE | re.DOTALL)
_BODY_RE = re.compile(r"<body>\s*(.*?)\s*</body>", re.IGNORECASE | re.DOTALL)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_score_output(text: str, *, dimension: str) -> ScoreResult:
    reason_match = _STRICT_REASON_RE.search(text)
    score_match = _STRICT_SCORE_RE.search(text)

    if score_match:
        score = int(score_match.group(1))
    else:
        fallback_score_match = _FALLBACK_SCORE_RE.search(text)
        if not fallback_score_match:
            raise ParseError(f"Could not parse a valid score for dimension {dimension}.")
        score = int(fallback_score_match.group(1))

    if score not in {-2, -1, 1, 2}:
        raise ParseError(f"Parsed invalid score {score} for dimension {dimension}.")

    if reason_match:
        reason = _normalize_whitespace(reason_match.group(1))
    else:
        without_tags = re.sub(r"<[^>]+>", " ", text)
        without_score = _FALLBACK_SCORE_RE.sub(" ", without_tags)
        reason = _normalize_whitespace(without_score)

    if not reason:
        raise ParseError(f"Could not parse a reason for dimension {dimension}.")

    return ScoreResult(
        dimension=dimension,
        score=score,
        reason=reason,
        raw_text=text,
    )


def parse_document_output(text: str, *, fallback_title: str | None = None) -> DraftDocument:
    title_match = _TITLE_RE.search(text)
    body_match = _BODY_RE.search(text)

    if title_match and body_match:
        title = _normalize_whitespace(title_match.group(1))
        body = body_match.group(1).strip()
        if title and body:
            return DraftDocument(title=title, body=body)

    lines = [line.rstrip() for line in text.strip().splitlines() if line.strip()]
    if lines and lines[0].startswith("#"):
        title = lines[0].lstrip("#").strip()
        body = "\n".join(lines[1:]).strip()
        if title and body:
            return DraftDocument(title=title, body=body)

    title_line_match = re.search(r"(?im)^\s*title\s*:\s*(.+)$", text)
    if title_line_match:
        title = title_line_match.group(1).strip()
        body = re.sub(r"(?im)^\s*title\s*:\s*.+$", "", text, count=1).strip()
        if title and body:
            return DraftDocument(title=title, body=body)

    if fallback_title and text.strip():
        return DraftDocument(title=fallback_title, body=text.strip())

    raise ParseError("Could not parse document output into title and body.")
