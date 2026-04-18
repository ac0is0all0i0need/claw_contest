from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from .config import ConfigError, load_settings, resolve_provider_config
from .llm import LLMError, build_client
from .models import DraftDocument
from .scoring import build_feedback_summary, evaluate_draft

LOCAL_BENCH_BACKEND = "prompt_templates_local_benchmark"
LOCAL_BENCH_RUBRIC_SOURCE = "prompt_templates.py"


class OpenClawBenchError(RuntimeError):
    """Raised when offline benchmark preparation or evaluation fails."""


def _clean_optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _read_draft_markdown(path: Path) -> DraftDocument:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise OpenClawBenchError(f"Draft file is empty: {path}")

    if text.startswith("# "):
        title, _, remainder = text.partition("\n")
        return DraftDocument(title=title[2:].strip(), body=remainder.strip())
    return DraftDocument(title=path.stem.replace("_", " ").strip(), body=text)


def _resolve_best_draft(run_dir: Path) -> Path:
    preferred = run_dir / "best_draft.md"
    if preferred.exists():
        return preferred

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        required_evidence = summary.get("required_evidence")
        if isinstance(required_evidence, dict):
            best_draft = required_evidence.get("best_draft")
            if isinstance(best_draft, str) and Path(best_draft).exists():
                return Path(best_draft)

    round_drafts = sorted(run_dir.glob("draft_round_*.md"))
    if round_drafts:
        return round_drafts[-1]

    raise OpenClawBenchError(f"Unable to resolve a draft from run directory: {run_dir}")


def export_manifest(
    *,
    candidate_run_dir: str,
    baseline_run_dir: str | None = None,
    baseline_draft_path: str | None = None,
    output_path: str,
    candidate_label: str = "strict_agent",
    baseline_label: str = "baseline",
) -> Path:
    candidate_run = Path(candidate_run_dir).resolve()
    baseline_run = Path(baseline_run_dir).resolve() if baseline_run_dir else None
    baseline_draft = Path(baseline_draft_path).resolve() if baseline_draft_path else None

    candidate_draft = _resolve_best_draft(candidate_run)
    if baseline_draft is None:
        if baseline_run is None:
            raise OpenClawBenchError("Provide either --baseline-run-dir or --baseline-draft.")
        baseline_draft = _resolve_best_draft(baseline_run)

    manifest = {
        "candidate": {
            "label": candidate_label,
            "run_dir": str(candidate_run),
            "draft_path": str(candidate_draft),
        },
        "baseline": {
            "label": baseline_label,
            "run_dir": str(baseline_run) if baseline_run is not None else "",
            "draft_path": str(baseline_draft),
        },
    }

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def _score_draft(path: Path, *, provider: str | None, model: str | None) -> dict[str, object]:
    settings = load_settings()
    provider_config = resolve_provider_config(settings, provider=provider, model=model)
    client = build_client(provider_config)
    draft = _read_draft_markdown(path)
    scores = evaluate_draft(client, draft)
    return {
        "draft_path": str(path),
        "title": draft.title,
        "total_score": sum(item.score for item in scores),
        "feedback_summary": build_feedback_summary(scores),
        "scores": [
            {
                "dimension": item.dimension,
                "score": item.score,
                "reason": item.reason,
            }
            for item in scores
        ],
        "backend": LOCAL_BENCH_BACKEND,
        "rubric_source": LOCAL_BENCH_RUBRIC_SOURCE,
        "provider": provider_config.provider,
        "model": provider_config.model,
    }


def _evaluate_with_external_bench(command_template: str, manifest_path: Path, output_path: Path) -> dict[str, object]:
    command = command_template.format(manifest=str(manifest_path), output=str(output_path))
    completed = subprocess.run(command, shell=True, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise OpenClawBenchError(
            f"DeepResearch Bench command failed with exit code {completed.returncode}: {completed.stderr.strip()}"
        )
    if not output_path.exists():
        raise OpenClawBenchError("DeepResearch Bench command completed without producing output.")
    return json.loads(output_path.read_text(encoding="utf-8"))


def evaluate_manifest(
    *,
    manifest_path: str,
    output_path: str,
    provider: str | None = None,
    model: str | None = None,
) -> Path:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    candidate = manifest.get("candidate", {})
    baseline = manifest.get("baseline", {})
    candidate_draft = Path(str(candidate.get("draft_path", ""))).resolve()
    baseline_draft = Path(str(baseline.get("draft_path", ""))).resolve()
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    external_command = _clean_optional_str(os.getenv("JDCLAW_DEEPRESEARCH_BENCH_COMMAND"))
    if external_command:
        results = _evaluate_with_external_bench(external_command, Path(manifest_path).resolve(), output)
    else:
        baseline_result = _score_draft(baseline_draft, provider=provider, model=model)
        candidate_result = _score_draft(candidate_draft, provider=provider, model=model)
        results = {
            "backend": LOCAL_BENCH_BACKEND,
            "rubric_source": LOCAL_BENCH_RUBRIC_SOURCE,
            "baseline": baseline_result,
            "candidate": candidate_result,
            "delta_total_score": candidate_result["total_score"] - baseline_result["total_score"],
        }

    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def render_report(*, results_path: str, output_dir: str) -> tuple[Path, Path]:
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = results.get("baseline", {})
    candidate = results.get("candidate", {})
    baseline_total = int(baseline.get("total_score", 0))
    candidate_total = int(candidate.get("total_score", 0))
    labels = [str(baseline.get("title", "baseline")), str(candidate.get("title", "candidate"))]
    values = [baseline_total, candidate_total]

    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values, color=["#6b7280", "#0f766e"])
    plt.ylabel("Total Score")
    plt.title("Offline Benchmark Comparison")
    chart_path = out_dir / "bench_scores.png"
    plt.tight_layout()
    plt.savefig(chart_path, dpi=180)
    plt.close()

    backend = str(results.get("backend", "unknown"))
    report_title = (
        "# Prompt Templates Benchmark Report"
        if backend == LOCAL_BENCH_BACKEND
        else "# DeepResearch Bench Report"
    )
    rubric_source = results.get("rubric_source")
    report_path = out_dir / "bench_report.md"
    report_path.write_text(
        "\n".join(
            [
                report_title,
                "",
                f"- Backend: {backend}",
                (
                    f"- Rubric source: {rubric_source}"
                    if isinstance(rubric_source, str) and rubric_source.strip()
                    else ""
                ),
                f"- Baseline total score: {baseline_total}",
                f"- Candidate total score: {candidate_total}",
                f"- Delta total score: {results.get('delta_total_score', candidate_total - baseline_total)}",
                "",
                "## Candidate Feedback",
                "",
                str(candidate.get("feedback_summary", "")),
                "",
                f"![Benchmark chart]({chart_path})",
            ]
        ),
        encoding="utf-8",
    )
    return report_path, chart_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline benchmark utilities for initial-vs-final draft comparison."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export baseline/candidate manifest.")
    export_parser.add_argument("--candidate-run-dir", required=True)
    export_parser.add_argument("--baseline-run-dir")
    export_parser.add_argument("--baseline-draft")
    export_parser.add_argument("--output", required=True)
    export_parser.add_argument("--candidate-label", default="strict_agent")
    export_parser.add_argument("--baseline-label", default="baseline")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a manifest offline.")
    evaluate_parser.add_argument("--manifest", required=True)
    evaluate_parser.add_argument("--output", required=True)
    evaluate_parser.add_argument("--provider", choices=["localapi", "deepseek"], default=None)
    evaluate_parser.add_argument("--model", default=None)

    report_parser = subparsers.add_parser("report", help="Render an offline benchmark report.")
    report_parser.add_argument("--results", required=True)
    report_parser.add_argument("--output-dir", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "export":
            output = export_manifest(
                candidate_run_dir=args.candidate_run_dir,
                baseline_run_dir=args.baseline_run_dir,
                baseline_draft_path=args.baseline_draft,
                output_path=args.output,
                candidate_label=args.candidate_label,
                baseline_label=args.baseline_label,
            )
            print(f"Manifest: {output}")
            return 0

        if args.command == "evaluate":
            output = evaluate_manifest(
                manifest_path=args.manifest,
                output_path=args.output,
                provider=args.provider,
                model=args.model,
            )
            print(f"Results: {output}")
            return 0

        report_path, chart_path = render_report(
            results_path=args.results,
            output_dir=args.output_dir,
        )
        print(f"Report: {report_path}")
        print(f"Chart: {chart_path}")
        return 0
    except (ConfigError, LLMError, OSError, OpenClawBenchError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
