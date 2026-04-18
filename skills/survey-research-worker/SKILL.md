---
name: survey-research-worker
description: OpenClaw-native worker for generating one literature survey draft from a topic using the repo's deep-research integration. Use when an OpenClaw or CMDOP stage needs a black-box research/generation worker without running the full self-evolution loop.
---

# Task Context

Operate as the research-stage worker for the repo's survey self-evolution flow.
Stay inside this repository and treat the existing Python modules as the source of truth.
Generate exactly one draft from a topic. Do not score, revise, or orchestrate rounds.

# Goals

Produce a research-grounded survey draft with the existing deep-research path.
Respect the configured provider, model, and Firecrawl-backed research settings.
Return a machine-readable result and optional artifact files that downstream OpenClaw stages can consume.

# Scripts And Tools

Primary script:
- `uv run python .cmdop/skills/survey-research-worker/scripts/run_survey_research.py --topic "<topic>"`

Repo modules used by the script:
- `jingdong_claw.config.load_settings`
- `jingdong_claw.config.resolve_provider_config`
- `jingdong_claw.config.resolve_research_config`
- `jingdong_claw.research.DeepResearchDraftGenerator`

Optional outputs:
- `--output-json <path>` writes the full result payload
- `--output-markdown <path>` writes `# <title>` plus the generated body

# Recommended Path

1. Confirm the repo root is the current working directory.
2. Confirm provider credentials and Firecrawl credentials are available through the repo's environment variables.
3. Run the script with `--topic` and any provider or research overrides that the current job requires.
4. Prefer `--output-json` when a downstream worker needs structured data.
5. Prefer `--output-markdown` when a human or downstream tool needs the draft body as a standalone document.
6. Pass the resulting draft into the evaluation worker instead of adding custom orchestration here.

# Failure Recovery

If the script fails with configuration errors, fix the missing provider or Firecrawl environment variables and rerun.
If the script fails with `ResearchError`, reduce research breadth, depth, or concurrency only when the runtime is the bottleneck; do not replace the repo's deep-research path with ad hoc generation.
If the output draft is malformed or empty, preserve the stderr message and the exact CLI invocation as evidence before retrying.

# Evidence Requirements

Capture the full `uv run python ...` command.
Capture stdout JSON or the `--output-json` file.
If you wrote markdown output, capture the output path and the generated title.
Record any non-default provider, model, or research override flags used for the run.
