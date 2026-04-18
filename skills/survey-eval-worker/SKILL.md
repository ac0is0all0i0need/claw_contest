---
name: survey-eval-worker
description: OpenClaw-native worker for scoring a survey draft with the repo's evaluator-only path. Use when an OpenClaw or CMDOP stage needs rubric scores and feedback for one draft without triggering revision or multi-round orchestration.
---

# Task Context

Operate as the evaluation-stage worker for the repo's survey self-evolution flow.
Consume one existing draft and return evaluator outputs only.
Do not generate a new draft and do not select revisions here.

# Goals

Score a survey draft with the repo's four existing review dimensions.
Preserve the repo's score ordering, parsing rules, and feedback-summary logic.
Emit structured outputs that the revision worker can reuse directly.

# Scripts And Tools

Primary script:
- `uv run python .cmdop/skills/survey-eval-worker/scripts/run_survey_eval.py --draft-file <path>`

Repo modules used by the script:
- `jingdong_claw.config.load_settings`
- `jingdong_claw.config.resolve_provider_config`
- `jingdong_claw.llm.build_client`
- `jingdong_claw.scoring.evaluate_draft`
- `jingdong_claw.scoring.build_feedback_summary`

Input forms:
- `--draft-file <path>` for a markdown or XML-like draft document
- `--draft-text "<text>"` for inline draft content
- `--title "<title>"` when the provided draft content is body-only

Optional outputs:
- `--output-json <path>` writes the structured evaluation payload
- `--feedback-file <path>` writes the plain-text feedback summary

# Recommended Path

1. Supply the full draft through `--draft-file` whenever possible.
2. Omit `--title` when the draft already includes a parseable title; add `--title` only for body-only inputs.
3. Run the script and keep the JSON payload intact because it is the cleanest handoff into revision.
4. Pass the resulting JSON file to the revision worker through `--eval-json` whenever possible.

# Failure Recovery

If parsing fails, normalize the draft into either `# Title` markdown or the repo's `<title>/<body>` format and rerun.
If model configuration fails, fix the provider environment variables instead of bypassing the repo's client builder.
If evaluator output is malformed, preserve stderr and the draft artifact for inspection before retrying.

# Evidence Requirements

Capture the exact CLI invocation.
Capture stdout JSON or the `--output-json` file.
Capture the feedback summary path when `--feedback-file` is used.
Preserve the scored draft artifact that produced the evaluation result.
