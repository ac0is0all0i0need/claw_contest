---
name: survey-revise-worker
description: OpenClaw-native worker for producing one improved survey draft from an existing draft plus evaluation evidence using the repo's revision-candidate selector. Use when an OpenClaw or CMDOP stage needs revision only, without taking over round orchestration.
---

# Task Context

Operate as the revision-stage worker for the repo's survey self-evolution flow.
Consume an existing draft plus prior evaluation evidence.
Produce one selected revised draft and its post-revision evaluation outputs.
Do not run the outer round loop and do not write workflow-level artifacts here.

# Goals

Use the repo's revision-candidate selection logic exactly once for the supplied draft.
Preserve the repo's revision prompts, candidate selection behavior, and evaluator reuse.
Return the best revised draft together with the resulting scores and feedback summary.

# Scripts And Tools

Primary script:
- `uv run python .cmdop/skills/survey-revise-worker/scripts/run_survey_revise.py --draft-file <path> --eval-json <path>`

Repo modules used by the script:
- `jingdong_claw.config.load_settings`
- `jingdong_claw.config.resolve_provider_config`
- `jingdong_claw.llm.build_client`
- `jingdong_claw.scoring.build_feedback_summary`
- `jingdong_claw.pipeline.select_best_revision_candidate`

Input forms:
- Draft input through `--draft-file` or `--draft-text`
- Evaluation evidence through `--eval-json` or `--scores-json`
- Optional feedback override through `--feedback-file` or `--feedback-text`

Optional outputs:
- `--output-json <path>` writes the revised result payload
- `--output-markdown <path>` writes the selected revised draft as markdown

# Recommended Path

1. Reuse the evaluation worker's JSON output via `--eval-json` whenever possible.
2. Provide `--title` only when the draft input is body-only.
3. Keep `--revision-candidates` aligned with the current experiment rather than hard-coding a new orchestration policy in this worker.
4. Pass the resulting revised draft back to evaluation or orchestration outside this skill if another round is needed.

# Failure Recovery

If revision input parsing fails, normalize the draft into `# Title` markdown or `<title>/<body>` format and rerun.
If score evidence is incomplete, regenerate it with the evaluation worker instead of hand-editing score files.
If revision generation fails, preserve the exact CLI command, the source draft, and the evaluation payload before retrying.

# Evidence Requirements

Capture the exact CLI invocation.
Capture stdout JSON or the `--output-json` file.
Capture the source draft path and the evaluation evidence path used for the run.
If markdown output was written, capture the output path and revised title.
