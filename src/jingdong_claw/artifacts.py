from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .models import RoundResult, RunResult


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug or "run"


def create_run_dir(base_dir: Path, label: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base_dir / f"{timestamp}-{slugify(label)[:48]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


class ArtifactWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_round(self, round_result: RoundResult) -> None:
        draft_path = self.output_dir / f"draft_round_{round_result.round_index}.md"
        draft_path.write_text(
            f"# {round_result.draft_title}\n\n{round_result.draft_body}\n",
            encoding="utf-8",
        )

        scores_path = self.output_dir / f"scores_round_{round_result.round_index}.json"
        scores_path.write_text(
            json.dumps([asdict(score) for score in round_result.scores], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        feedback_path = self.output_dir / f"feedback_round_{round_result.round_index}.md"
        feedback_path.write_text(round_result.feedback_summary, encoding="utf-8")

    def write_summary(self, run_result: RunResult) -> None:
        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(run_result.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def write_workflow_trace(self, trace: dict[str, object]) -> None:
        trace_path = self.output_dir / "workflow_trace.json"
        trace_path.write_text(
            json.dumps(trace, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
