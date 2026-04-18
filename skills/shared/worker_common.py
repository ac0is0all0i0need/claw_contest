from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


def bootstrap_repo_imports(script_file: str) -> Path:
    script_path = Path(script_file).resolve()
    repo_root = script_path.parents[4]
    import_roots = (
        repo_root,
        repo_root / "src",
    )
    for candidate in import_roots:
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
    return repo_root


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: str | None, payload: dict[str, Any]) -> str | None:
    if not path:
        return None
    output_path = Path(path)
    ensure_parent_dir(output_path)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return str(output_path)


def write_markdown(path: str | None, *, title: str, body: str) -> str | None:
    if not path:
        return None
    output_path = Path(path)
    ensure_parent_dir(output_path)
    output_path.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")
    return str(output_path)


def write_text(path: str | None, content: str) -> str | None:
    if not path:
        return None
    output_path = Path(path)
    ensure_parent_dir(output_path)
    output_path.write_text(content, encoding="utf-8")
    return str(output_path)


def load_text(*, file_path: str | None, inline_text: str | None, label: str) -> str:
    if inline_text is not None:
        text = inline_text.strip()
        if not text:
            raise ValueError(f"{label} text must not be empty.")
        return text
    if file_path is None:
        raise ValueError(f"Provide either inline {label} text or a {label} file.")
    text = Path(file_path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{label} file '{file_path}' is empty.")
    return text


def load_draft_document(
    *,
    title: str | None,
    draft_file: str | None,
    draft_text: str | None,
) -> Any:
    from jingdong_claw.models import DraftDocument
    from jingdong_claw.parser import parse_document_output

    draft_source = load_text(file_path=draft_file, inline_text=draft_text, label="draft")
    if title:
        return DraftDocument(title=title.strip(), body=draft_source)
    return parse_document_output(draft_source)


def load_score_results(
    *,
    eval_json: str | None,
    scores_json: str | None,
) -> tuple[list[Any], str | None]:
    from jingdong_claw.models import ScoreResult

    payload_path = eval_json or scores_json
    if payload_path is None:
        raise ValueError("Provide either --eval-json or --scores-json.")

    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    feedback_summary: str | None = None
    raw_scores: Any = payload

    if isinstance(payload, dict):
        if "feedback_summary" in payload:
            candidate_feedback = payload.get("feedback_summary")
            if isinstance(candidate_feedback, str) and candidate_feedback.strip():
                feedback_summary = candidate_feedback
        if "scores" in payload:
            raw_scores = payload["scores"]

    if not isinstance(raw_scores, list) or not raw_scores:
        raise ValueError(f"Score payload '{payload_path}' does not contain a non-empty score list.")

    scores: list[ScoreResult] = []
    for index, item in enumerate(raw_scores):
        if not isinstance(item, dict):
            raise ValueError(f"Score item {index} in '{payload_path}' is not an object.")
        try:
            dimension = str(item["dimension"])
            score = int(item["score"])
            reason = str(item["reason"])
        except KeyError as exc:
            raise ValueError(f"Score item {index} in '{payload_path}' is missing {exc.args[0]!r}.") from exc
        raw_text = str(item.get("raw_text", ""))
        scores.append(
            ScoreResult(
                dimension=dimension,
                score=score,
                reason=reason,
                raw_text=raw_text,
            )
        )

    return scores, feedback_summary


def serialize_scores(scores: list[Any]) -> list[dict[str, Any]]:
    return [asdict(score) for score in scores]
