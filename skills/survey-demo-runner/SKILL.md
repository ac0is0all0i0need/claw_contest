---
name: survey-demo-runner
description: Dashboard-shell skill for running the repository's self-evolution demo through one repo-owned Python board wrapper. Use when the OpenClaw dashboard should trigger the full competition demo from a topic and write all required evidence files to one run directory.
---

# Task Context

Operate as the repo-owned dashboard entry for the recorded survey demo.
Use the Python board wrapper in this skill to execute the full demo.
Do not re-implement orchestration logic in the skill body.
Do not manually chain worker skills one by one unless the runner fails and the user explicitly asks for a lower-level fallback.

# Goals

Run the recorded competition demo from a topic while keeping the board shell as the visible frontend.
Preserve the repo's stable Python backend:
- `deep-research` for the initial draft
- prompt scoring for every evaluated draft
- the repo's Python self-evolution loop for revision and early stop
- a final initial-vs-final comparison that defaults to the repo's `prompt_templates.py` rubric
Return a machine-readable summary that points at the run directory, board summary, and the main output files.

# Scripts And Tools

Primary script:
- `uv run python .cmdop/skills/survey-demo-runner/scripts/run_survey_demo.py --topic "<topic>" --run-dir "<run-dir>"`

Optional outputs:
- `--output-json <path>` writes the runner summary payload
- stdout prints the same summary payload as JSON

Key summary fields:
- `entrypoint`
- `backend`
- `openclaw_backend`
- `output_dir`
- `initial_draft_path`
- `best_draft_path`
- `workflow_trace_path`
- `board_summary_path`
- `board_report_path`
- `bench_results_path`
- `bench_report_path`
- `bench_backend`
- `summary_path`
- `baseline_total_score`
- `candidate_total_score`
- `delta_total_score`
- `stop_reason`

# Recommended Path

1. Confirm the repository root is the current working directory.
2. Supply `--topic` and an explicit `--run-dir`.
3. Let the runner own orchestration. Do not replace it with ad hoc shell pipelines.
4. Capture the JSON payload from stdout or `--output-json` and treat `board_summary.json` plus `summary.json` as the artifact index for the run.
5. Treat `openclaw dashboard` as the public launch shell, and `uv run python -m jingdong_claw.openclaw_demo ...` as a development helper only.

# Failure Recovery

If configuration fails, fix the missing provider, model, Firecrawl, or OpenClaw environment values and rerun the same command.
If the runner reports a bench error, preserve `board_summary.json`, `summary.json`, and the evolution artifacts.
Check `bench_backend` in the final summary to see whether the run used an external benchmark wrapper or the repo's local `prompt_templates.py` rubric path.
If the Python workflow stops early or errors, preserve the run directory for inspection instead of reconstructing artifacts by hand.

# Evidence Requirements

Capture the exact `uv run python ...run_survey_demo.py ...` command.
Capture stdout JSON or the `--output-json` file.
Capture the resolved run directory and the final `board_summary.json` plus `summary.json` paths.
Record the topic plus any non-default provider or model flags.
