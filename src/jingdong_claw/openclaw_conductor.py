from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .artifacts import create_run_dir
from .config import load_settings, resolve_provider_config
from .models import ProviderConfig
from .openclaw_cli_backend import (
    OpenClawCliBackendError,
    build_openclaw_env,
    ensure_openclaw_profile,
    list_openclaw_skills,
    run_openclaw_local_agent,
    sync_workspace_skills,
)

ExecutionMode = Literal["openclaw", "openclaw-strict"]

CMDOP_NATIVE_BACKEND = "cmdop_native"
OPENCLAW_CLI_BACKEND = "openclaw_cli"
DEFAULT_AGENT_MAX_TURNS = 24
DEFAULT_AGENT_TIMEOUT_SECONDS = 600
DEFAULT_AGENT_MAX_RETRIES = 1
DEFAULT_REVISION_CANDIDATES = 2
DEFAULT_SKILLS_DIR = ".cmdop/skills"
DEFAULT_OPENCLAW_PROFILE = "jdclaw-comp"
DEFAULT_OPENCLAW_AGENT_ID = "main"
DEFAULT_OPENCLAW_THINKING = "low"
VALID_MODES = {"topic", "paper"}
VALID_PROVIDERS = {"localapi", "deepseek"}
SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class OpenClawConductorError(RuntimeError):
    """Raised when the native OpenClaw conductor path cannot complete."""


@dataclass(frozen=True, slots=True)
class WorkerSkillSpec:
    name: str
    directory_name: str
    description: str


WORKER_SKILLS: tuple[WorkerSkillSpec, ...] = (
    WorkerSkillSpec(
        name="survey-research-worker",
        directory_name="survey-research-worker",
        description="Research and initial survey drafting worker.",
    ),
    WorkerSkillSpec(
        name="survey-eval-worker",
        directory_name="survey-eval-worker",
        description="Prompt-based evaluator worker for survey drafts.",
    ),
    WorkerSkillSpec(
        name="survey-revise-worker",
        directory_name="survey-revise-worker",
        description="Revision candidate worker for survey improvement.",
    ),
)

REQUIRED_AGENT_FILES = {
    "best_draft": "best_draft.md",
    "self_eval": "self_eval.json",
    "revision_rationale": "revision_rationale.md",
    "stop_reason": "stop_reason.md",
    "branch_log": "branch_log.json",
    "self_correction_log": "self_correction_log.json",
    "activity_log": "activity_log.json",
}


class ConductorCompletion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stop_reason: str
    summary: str
    strict_compliant: bool = False
    best_draft_title: str = ""
    best_total_score: int | None = None


@dataclass(slots=True)
class OpenClawNativeRequest:
    mode: str
    topic: str | None = None
    title: str | None = None
    body: str | None = None
    rounds: int | None = None
    provider: str | None = None
    model: str | None = None
    output_dir: str | None = None
    revision_candidates: int | None = None
    research_breadth: int | None = None
    research_depth: int | None = None
    research_concurrency: int | None = None
    execution_mode: ExecutionMode = "openclaw-strict"
    skills_dir: str | None = None
    agent_max_turns: int | None = None
    agent_timeout_seconds: int | None = None
    agent_max_retries: int | None = None


@dataclass(slots=True)
class OpenClawNativeResult:
    run_id: str
    mode: str
    output_dir: str
    request_id: str
    openclaw_backend: str
    execution_mode: ExecutionMode
    strict_compliant: bool
    stop_reason: str
    summary_path: str
    session_trace_path: str
    process_report_path: str
    best_draft_path: str
    self_eval_path: str
    prompt_path: str


@dataclass(slots=True)
class OpenClawConductorRuntime:
    client_factory: Callable[[], object] | None = None
    discover_agent_fn: Callable[[], object] | None = None


@dataclass(slots=True)
class SkillAvailability:
    name: str
    description: str
    local_skill_path: str
    local_exists: bool
    found: bool = False
    origin: str = ""
    source: str = ""
    error: str = ""
    strict_ready: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "local_skill_path": self.local_skill_path,
            "local_exists": self.local_exists,
            "found": self.found,
            "origin": self.origin,
            "source": self.source,
            "error": self.error,
            "strict_ready": self.strict_ready,
        }


def _clean_optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _normalize_positive_int(name: str, value: int | None, default: int) -> int:
    normalized = default if value is None else value
    if normalized < 1:
        raise OpenClawConductorError(f"{name} must be at least 1.")
    return normalized


def normalize_request(request: OpenClawNativeRequest) -> OpenClawNativeRequest:
    settings = load_settings()
    mode = _clean_optional_str(request.mode)
    if mode not in VALID_MODES:
        valid_modes = ", ".join(sorted(VALID_MODES))
        raise OpenClawConductorError(f"mode must be one of: {valid_modes}.")

    provider = _clean_optional_str(request.provider)
    if provider is not None and provider not in VALID_PROVIDERS:
        valid_providers = ", ".join(sorted(VALID_PROVIDERS))
        raise OpenClawConductorError(f"provider must be one of: {valid_providers}.")

    topic = _clean_optional_str(request.topic)
    title = _clean_optional_str(request.title)
    body = _clean_optional_str(request.body)
    model = _clean_optional_str(request.model)
    output_dir = _clean_optional_str(request.output_dir)
    skills_dir = _clean_optional_str(request.skills_dir)

    if mode == "topic" and topic is None:
        raise OpenClawConductorError("topic mode requires a non-empty topic.")
    if mode == "paper":
        if title is None:
            raise OpenClawConductorError("paper mode requires a non-empty title.")
        if body is None:
            raise OpenClawConductorError("paper mode requires a non-empty body.")

    if request.execution_mode not in {"openclaw", "openclaw-strict"}:
        raise OpenClawConductorError("execution_mode must be 'openclaw' or 'openclaw-strict'.")

    return OpenClawNativeRequest(
        mode=mode,
        topic=topic,
        title=title,
        body=body,
        rounds=_normalize_positive_int("rounds", request.rounds, settings.default_rounds),
        provider=provider,
        model=model,
        output_dir=output_dir,
        revision_candidates=_normalize_positive_int(
            "revision_candidates",
            request.revision_candidates,
            DEFAULT_REVISION_CANDIDATES,
        ),
        research_breadth=_normalize_positive_int(
            "research_breadth",
            request.research_breadth,
            settings.research_breadth,
        ),
        research_depth=_normalize_positive_int(
            "research_depth",
            request.research_depth,
            settings.research_depth,
        ),
        research_concurrency=_normalize_positive_int(
            "research_concurrency",
            request.research_concurrency,
            settings.research_concurrency,
        ),
        execution_mode=request.execution_mode,
        skills_dir=skills_dir,
        agent_max_turns=_normalize_positive_int(
            "agent_max_turns",
            request.agent_max_turns,
            DEFAULT_AGENT_MAX_TURNS,
        ),
        agent_timeout_seconds=_normalize_positive_int(
            "agent_timeout_seconds",
            request.agent_timeout_seconds,
            DEFAULT_AGENT_TIMEOUT_SECONDS,
        ),
        agent_max_retries=_normalize_positive_int(
            "agent_max_retries",
            request.agent_max_retries,
            DEFAULT_AGENT_MAX_RETRIES,
        ),
    )


def _resolve_skills_dir(request: OpenClawNativeRequest) -> Path:
    if request.skills_dir:
        return Path(request.skills_dir).resolve()
    return (Path.cwd() / DEFAULT_SKILLS_DIR).resolve()


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _parse_path_list_env(name: str) -> list[Path]:
    raw = os.getenv(name, "")
    items = [item.strip() for item in raw.split(os.pathsep) if item.strip()]
    return [Path(item).resolve() for item in items]


def _load_runtime_config() -> dict[str, object]:
    return {
        "backend": (os.getenv("JDCLAW_OPENCLAW_BACKEND", "") or "").strip().lower(),
        "transport": (os.getenv("JDCLAW_OPENCLAW_TRANSPORT", "") or "").strip().lower(),
        "discovery_paths": _parse_path_list_env("JDCLAW_CMDOP_DISCOVERY_PATHS"),
        "use_default_discovery": _parse_bool_env("JDCLAW_CMDOP_USE_DEFAULT_DISCOVERY", True),
        "token_path": (os.getenv("JDCLAW_CMDOP_TOKEN_PATH", "") or "").strip(),
        "agent_info_path": (os.getenv("JDCLAW_CMDOP_AGENT_INFO", "") or "").strip(),
        "profile": (os.getenv("JDCLAW_OPENCLAW_PROFILE", "") or "").strip() or DEFAULT_OPENCLAW_PROFILE,
        "agent_id": (os.getenv("JDCLAW_OPENCLAW_AGENT_ID", "") or "").strip() or DEFAULT_OPENCLAW_AGENT_ID,
        "thinking": (os.getenv("JDCLAW_OPENCLAW_THINKING", "") or "").strip() or DEFAULT_OPENCLAW_THINKING,
    }


def _resolve_cli_thinking(provider_config: ProviderConfig, thinking: str) -> str:
    normalized = (thinking or "").strip().lower() or DEFAULT_OPENCLAW_THINKING
    if provider_config.provider == "localapi" and normalized == "minimal":
        return "low"
    return normalized


def _build_evidence_paths(run_dir: Path) -> dict[str, Path]:
    return {key: run_dir / file_name for key, file_name in REQUIRED_AGENT_FILES.items()}


def _serialize_discovery(result: object) -> dict[str, object]:
    agent_info = getattr(result, "agent_info", None)
    info: dict[str, object] = {
        "found": bool(getattr(result, "found", False)),
        "error": str(getattr(result, "error", "") or ""),
    }
    discovery_path = getattr(result, "discovery_path", None)
    if discovery_path is not None:
        info["discovery_path"] = str(discovery_path)
    if agent_info is not None:
        transport = getattr(getattr(agent_info, "transport", None), "value", "")
        info["agent_info"] = {
            "transport": str(transport),
            "address": str(getattr(agent_info, "address", "") or ""),
            "pid": int(getattr(agent_info, "pid", 0) or 0),
        }
    return info


def _default_discover_agent(runtime_config: dict[str, object]) -> object:
    from cmdop.transport.discovery import discover_agent

    discovery_paths = runtime_config.get("discovery_paths", [])
    use_default_discovery = bool(runtime_config.get("use_default_discovery", True))
    return discover_agent(custom_paths=discovery_paths, use_defaults=use_default_discovery)


def _default_client_factory(runtime_config: dict[str, object]) -> object:
    from cmdop import CMDOPClient

    discovery_paths = runtime_config.get("discovery_paths", [])
    use_default_discovery = bool(runtime_config.get("use_default_discovery", True))
    return CMDOPClient.local(
        discovery_paths=discovery_paths,
        use_defaults=use_default_discovery,
    )


def _select_backend(
    runtime_config: dict[str, object],
    *,
    runtime: OpenClawConductorRuntime,
    discovery_payload: dict[str, object],
) -> str:
    configured_backend = str(runtime_config.get("backend", "") or "").lower()
    if configured_backend in {CMDOP_NATIVE_BACKEND, OPENCLAW_CLI_BACKEND}:
        return configured_backend
    if runtime.client_factory is not None or runtime.discover_agent_fn is not None:
        return CMDOP_NATIVE_BACKEND
    if discovery_payload.get("found", False):
        return CMDOP_NATIVE_BACKEND
    return OPENCLAW_CLI_BACKEND


def _inspect_skills(client: object, skills_dir: Path) -> tuple[list[SkillAvailability], list[str]]:
    strict_issues: list[str] = []
    listed_skills: dict[str, object] = {}
    list_error = ""

    try:
        skills = client.skills.list()
        listed_skills = {getattr(skill, "name", ""): skill for skill in skills}
    except Exception as exc:
        list_error = str(exc)
        strict_issues.append(f"skills.list failed: {exc}")

    results: list[SkillAvailability] = []
    for spec in WORKER_SKILLS:
        if not SKILL_NAME_PATTERN.fullmatch(spec.name):
            strict_issues.append(f"invalid skill name: {spec.name}")
        local_skill_path = (skills_dir / spec.directory_name / "SKILL.md").resolve()
        record = SkillAvailability(
            name=spec.name,
            description=spec.description,
            local_skill_path=str(local_skill_path),
            local_exists=local_skill_path.exists(),
        )

        if not record.local_exists:
            strict_issues.append(f"missing local skill file for {spec.name}: {local_skill_path}")

        listed = listed_skills.get(spec.name)
        if listed is not None:
            record.found = True
            record.origin = str(getattr(listed, "origin", "") or "")
            try:
                detail = client.skills.show(spec.name)
            except Exception as exc:
                record.error = str(exc)
                strict_issues.append(f"skills.show failed for {spec.name}: {exc}")
            else:
                record.source = str(getattr(detail, "source", "") or "")
                record.error = str(getattr(detail, "error", "") or "")
        elif not list_error:
            strict_issues.append(f"workspace skill not discoverable by agent: {spec.name}")

        record.strict_ready = bool(record.local_exists and record.found and record.origin == "workspace")
        if record.found and record.origin != "workspace":
            strict_issues.append(f"{spec.name} origin is '{record.origin}', expected 'workspace'")
        if record.found and record.source:
            source_path = Path(record.source).resolve()
            if source_path.name != "SKILL.md":
                strict_issues.append(f"{spec.name} source is not a SKILL.md file: {record.source}")
            elif skills_dir not in source_path.parents:
                strict_issues.append(f"{spec.name} source is outside configured skill root: {record.source}")
        results.append(record)

    return results, strict_issues


def _inspect_openclaw_cli_skills(
    listed_skills: list[dict[str, object]],
    *,
    skills_dir: Path,
    workspace_skills_dir: Path,
) -> tuple[list[SkillAvailability], list[str]]:
    strict_issues: list[str] = []
    by_name = {
        str(item.get("name", "") or ""): item
        for item in listed_skills
        if isinstance(item, dict)
    }

    results: list[SkillAvailability] = []
    for spec in WORKER_SKILLS:
        if not SKILL_NAME_PATTERN.fullmatch(spec.name):
            strict_issues.append(f"invalid skill name: {spec.name}")
        source_skill_path = (skills_dir / spec.directory_name / "SKILL.md").resolve()
        workspace_skill_path = (workspace_skills_dir / spec.directory_name / "SKILL.md").resolve()
        record = SkillAvailability(
            name=spec.name,
            description=spec.description,
            local_skill_path=str(source_skill_path),
            local_exists=source_skill_path.exists(),
        )

        if not record.local_exists:
            strict_issues.append(f"missing local skill file for {spec.name}: {source_skill_path}")
        if not workspace_skill_path.exists():
            strict_issues.append(f"missing workspace mirror for {spec.name}: {workspace_skill_path}")

        listed = by_name.get(spec.name)
        if listed is None:
            strict_issues.append(f"workspace skill not discoverable by OpenClaw CLI: {spec.name}")
        else:
            record.found = True
            source = str(listed.get("source", "") or "")
            record.origin = "workspace" if source == "openclaw-workspace" else source
            record.source = str(workspace_skill_path)
            if source != "openclaw-workspace":
                strict_issues.append(f"{spec.name} source is '{source}', expected 'openclaw-workspace'")

        record.strict_ready = bool(
            record.local_exists and workspace_skill_path.exists() and record.found and record.origin == "workspace"
        )
        results.append(record)

    return results, strict_issues


def _evaluate_discovery_strictness(
    discovery_payload: dict[str, object],
    runtime_config: dict[str, object],
) -> list[str]:
    issues: list[str] = []
    if not discovery_payload.get("found", False):
        issues.append("cmdop agent is not discoverable")
        return issues

    agent_info = discovery_payload.get("agent_info")
    transport = ""
    if isinstance(agent_info, dict):
        transport = str(agent_info.get("transport", "") or "").lower()
    configured_transport = str(runtime_config.get("transport", "") or "").lower()

    if os.name == "nt" and transport == "pipe":
        issues.append("Windows named-pipe transport is not strict-ready; require TCP")
    if configured_transport and transport and configured_transport != transport:
        issues.append(f"discovered transport '{transport}' does not match configured transport '{configured_transport}'")

    discovery_path = discovery_payload.get("discovery_path")
    configured_paths = runtime_config.get("discovery_paths", [])
    if discovery_path and configured_paths:
        try:
            resolved_discovery_path = Path(str(discovery_path)).resolve()
        except OSError:
            issues.append(f"unable to resolve discovery path: {discovery_path}")
        else:
            if not any(path == resolved_discovery_path for path in configured_paths):
                issues.append(f"discovery path is outside configured allowlist: {resolved_discovery_path}")

    token_path = str(runtime_config.get("token_path", "") or "")
    if os.name == "nt" and transport == "tcp" and token_path and not Path(token_path).exists():
        issues.append(f"required CMDOP token file is missing: {token_path}")

    return issues


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_launcher_context(
    run_dir: Path,
    request: OpenClawNativeRequest,
    provider_config: ProviderConfig,
    skills_dir: Path,
    evidence_paths: dict[str, Path],
    skill_records: list[SkillAvailability],
) -> tuple[Path, dict[str, object]]:
    context: dict[str, object] = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "mode": request.mode,
        "topic": request.topic,
        "title": request.title,
        "seed_draft_path": str(run_dir / "seed_draft.md") if request.mode == "paper" else "",
        "provider": provider_config.provider,
        "model": provider_config.model,
        "execution_mode": request.execution_mode,
        "max_cycles": request.rounds,
        "max_revision_candidates": request.revision_candidates,
        "research_limits": {
            "breadth": request.research_breadth,
            "depth": request.research_depth,
            "concurrency": request.research_concurrency,
        },
        "skills_dir": str(skills_dir),
        "skills": [
            {
                "name": record.name,
                "description": record.description,
                "skill_md": record.local_skill_path,
            }
            for record in skill_records
        ],
        "required_evidence_files": {key: str(path) for key, path in evidence_paths.items()},
    }
    context_path = run_dir / "launcher_context.json"
    _write_json(context_path, context)

    if request.mode == "paper" and request.title and request.body:
        seed_draft_path = run_dir / "seed_draft.md"
        seed_draft_path.write_text(f"# {request.title}\n\n{request.body}\n", encoding="utf-8")

    return context_path, context


def build_agent_prompt(
    context: dict[str, object],
    skill_records: list[SkillAvailability],
    evidence_paths: dict[str, Path],
) -> str:
    skill_lines = "\n".join(
        f"- {record.name}: {record.description} (SKILL.md: {record.local_skill_path})"
        for record in skill_records
    )
    evidence_lines = "\n".join(f"- {key}: {path}" for key, path in evidence_paths.items())
    context_json = json.dumps(context, indent=2, ensure_ascii=False)
    run_dir = str(context.get("run_dir", "") or "")
    provider = str(context.get("provider", "") or "")
    model = str(context.get("model", "") or "")
    research_topic = str(context.get("topic", "") or "")
    seed_draft_path = str(context.get("seed_draft_path", "") or "")
    initial_draft_path = Path(run_dir) / "initial_draft.md"
    initial_eval_path = Path(run_dir) / "initial_evaluation.json"
    initial_feedback_path = Path(run_dir) / "initial_feedback.txt"

    return dedent(
        f"""
        You are the single OpenClaw-native conductor for a self-evolving survey workflow.
        You are the only business-level controller for this run. No external Python workflow will decide
        when to research, evaluate, revise, branch, continue, or stop.

        Hard guardrails:
        - Treat this as an Agent-first, Prompt-first, Skill-backed run.
        - Use the workspace worker skills below as your preferred capability surface.
        - Do not assume any external stage fallback exists.
        - Respect the hard limits in the session context JSON.
        - In strict mode, if a required skill/tool is unavailable, record the failure in the evidence files and stop.
        - Do not explore the repository root, do not read `.`, and do not inspect unrelated docs such as README.md or AGENTS.md.
        - If you need a directory listing, use `exec` with `dir <path>` instead of `read`.
        - Do not hand-author long draft, evaluation, or revision content in tool arguments.
        - Use the worker scripts to generate drafts, scores, and revision candidates as black-box outputs written to files.
        - Do not invent evaluation scores or revision candidates from scratch.

        Available workspace worker skills:
        {skill_lines}

        Required agent-written evidence files:
        {evidence_lines}

        Required execution discipline:
        - Prefer these black-box worker commands over manual editing:
          1. Research worker:
             `uv run python .cmdop/skills/survey-research-worker/scripts/run_survey_research.py --topic "{research_topic}" --provider {provider} --model {model} --output-markdown "{initial_draft_path}" --output-json "{Path(run_dir) / "initial_research.json"}"`
          2. Evaluation worker:
             `uv run python .cmdop/skills/survey-eval-worker/scripts/run_survey_eval.py --draft-file "{initial_draft_path if research_topic else seed_draft_path}" --provider {provider} --model {model} --output-json "{initial_eval_path}" --feedback-file "{initial_feedback_path}"`
          3. Revision worker template:
             `uv run python .cmdop/skills/survey-revise-worker/scripts/run_survey_revise.py --draft-file "<current draft path>" --eval-json "<current eval path>" --provider {provider} --model {model} --revision-candidates 1 --output-markdown "{Path(run_dir) / "revision_candidate_<n>.md"}" --output-json "{Path(run_dir) / "revision_candidate_<n>.json"}"`
        - For paper mode, the seed draft already exists at `{seed_draft_path}`.
        - After selecting the strongest draft, copy or write it to `{evidence_paths["best_draft"]}`.
        - Re-run the evaluation worker on the chosen best draft and write the final evaluator output to `{evidence_paths["self_eval"]}`.
        - Keep large content file-backed; use `read` only on specific files, not directories.

        Evidence format requirements:
        - activity_log.json: JSON array of objects with fields `step`, `kind`, `label`, `status`, `notes`.
          Use `kind` values such as `decision`, `skill_call`, `tool_call`, `revision_branch`, `self_correction`.
        - branch_log.json: JSON object with a `branches` array. Each branch record should include
          `candidate_id`, `strategy`, `selected`, `score`, `notes`.
        - self_correction_log.json: JSON object with an `attempts` array. Each attempt should include
          `issue`, `action`, `resolved`, `notes`.
        - self_eval.json: JSON object with `total_score`, `summary`, and `scores`.
        - best_draft.md: final best draft in markdown.
        - revision_rationale.md: why the chosen revision path won.
        - stop_reason.md: why you stopped.

        Recommended operating strategy:
        1. Decide whether you need fresh research or whether the seed draft is sufficient.
        2. Run the research worker only when there is no usable seed draft.
        3. Use the evaluator worker to score the current best draft.
        4. If improvement is warranted, create one or more revision candidates through the revise worker.
        5. Re-evaluate candidates, keep the strongest branch, and decide whether to continue.
        6. Stop when the draft is good enough, the revision frontier is exhausted, or a strict-mode blocker is hit.

        Final response requirements:
        - Return exactly one JSON object and no extra prose.
        - Use this exact schema:
          {{
            "stop_reason": "<short reason>",
            "summary": "<one paragraph summary>",
            "strict_compliant": true,
            "best_draft_title": "<title>",
            "best_total_score": 0
          }}
        - Do not claim strict compliance unless every required evidence file has been written.

        SESSION_CONTEXT_JSON
        {context_json}
        """
    ).strip()


def _build_agent_options(provider_config: ProviderConfig, request: OpenClawNativeRequest) -> object:
    from cmdop.models.agent import AgentRunOptions

    return AgentRunOptions(
        model=provider_config.model,
        max_turns=request.agent_max_turns or DEFAULT_AGENT_MAX_TURNS,
        max_retries=request.agent_max_retries or DEFAULT_AGENT_MAX_RETRIES,
        timeout_seconds=request.agent_timeout_seconds or DEFAULT_AGENT_TIMEOUT_SECONDS,
    )


def _run_agent(
    client: object,
    *,
    prompt: str,
    provider_config: ProviderConfig,
    request: OpenClawNativeRequest,
) -> object:
    from cmdop.models.agent import AgentType

    run = getattr(getattr(client, "agent", None), "run", None)
    if not callable(run):
        raise OpenClawConductorError("OpenClaw client does not expose agent.run(...).")

    kwargs = {
        "prompt": prompt,
        "agent_type": AgentType.PLANNER,
        "options": _build_agent_options(provider_config, request),
        "output_model": ConductorCompletion,
    }
    try:
        result = run(**kwargs)
    except TypeError:
        kwargs.pop("agent_type", None)
        result = run(**kwargs)

    if not getattr(result, "success", False):
        error = str(getattr(result, "error", "") or "agent run failed")
        raise OpenClawConductorError(error)
    return result


def _load_json(path: Path) -> object:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _normalize_completion_json(raw_text: str) -> str:
    candidate = raw_text.strip()
    if candidate.startswith("```"):
        fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", candidate, re.DOTALL)
        if fenced_match:
            candidate = fenced_match.group(1).strip()

    if candidate.startswith("{") and candidate.endswith("}"):
        return candidate

    object_match = re.search(r"(\{.*\})", candidate, re.DOTALL)
    if object_match:
        return object_match.group(1).strip()
    return candidate


def _extract_completion(result: object) -> ConductorCompletion:
    data = getattr(result, "data", None)
    if isinstance(data, ConductorCompletion):
        return data

    output_json = getattr(result, "output_json", "")
    if isinstance(output_json, str) and output_json.strip():
        return ConductorCompletion.model_validate_json(_normalize_completion_json(output_json))

    text = getattr(result, "text", "")
    if isinstance(text, str) and text.strip():
        return ConductorCompletion.model_validate_json(_normalize_completion_json(text))

    raise OpenClawConductorError("agent completion could not be parsed")


def _verify_evidence(evidence_paths: dict[str, Path]) -> list[str]:
    missing: list[str] = []
    for name, path in evidence_paths.items():
        if not path.exists():
            missing.append(f"missing evidence file: {path}")
            continue
        if not path.read_text(encoding="utf-8").strip():
            missing.append(f"empty evidence file: {path}")
    return missing


def _tool_result_to_dict(tool_result: object) -> dict[str, object]:
    payload = {
        "tool_name": str(getattr(tool_result, "tool_name", "") or ""),
        "tool_call_id": str(getattr(tool_result, "tool_call_id", "") or ""),
        "success": bool(getattr(tool_result, "success", False)),
        "result": str(getattr(tool_result, "result", "") or ""),
        "error": str(getattr(tool_result, "error", "") or ""),
        "duration_ms": int(getattr(tool_result, "duration_ms", 0) or 0),
    }
    return payload


def _build_process_report(
    run_dir: Path,
    evidence_paths: dict[str, Path],
    result: object,
    strict_compliant: bool,
    strict_issues: list[str],
) -> dict[str, object]:
    activity_log = _load_json(evidence_paths["activity_log"])
    branch_log = _load_json(evidence_paths["branch_log"])
    self_correction_log = _load_json(evidence_paths["self_correction_log"])
    self_eval = _load_json(evidence_paths["self_eval"])

    activity_items = activity_log if isinstance(activity_log, list) else []
    branches = branch_log.get("branches", []) if isinstance(branch_log, dict) else []
    correction_attempts = (
        self_correction_log.get("attempts", [])
        if isinstance(self_correction_log, dict)
        else []
    )

    tool_results = [_tool_result_to_dict(item) for item in getattr(result, "tool_results", [])]
    tool_call_count = len(tool_results)
    tool_success_count = sum(1 for item in tool_results if item["success"])
    correction_total = len(correction_attempts)
    correction_successes = sum(1 for item in correction_attempts if item.get("resolved"))

    best_total_score = 0
    if isinstance(self_eval, dict):
        raw_score = self_eval.get("total_score", 0)
        if isinstance(raw_score, int):
            best_total_score = raw_score

    report = {
        "run_id": run_dir.name,
        "strict_compliant": strict_compliant,
        "strict_issues": strict_issues,
        "fallback_count": 0,
        "autonomous_decision_steps": sum(
            1 for item in activity_items if isinstance(item, dict) and item.get("kind") == "decision"
        ),
        "worker_skill_call_count": sum(
            1 for item in activity_items if isinstance(item, dict) and item.get("kind") == "skill_call"
        ),
        "tool_call_count": tool_call_count,
        "tool_call_success_count": tool_success_count,
        "tool_call_success_rate": (tool_success_count / tool_call_count) if tool_call_count else 0.0,
        "self_correction_attempts": correction_total,
        "self_correction_success_count": correction_successes,
        "self_correction_success_rate": (
            correction_successes / correction_total if correction_total else 0.0
        ),
        "revision_branch_count": len(branches) if isinstance(branches, list) else 0,
        "best_total_score": best_total_score,
        "tool_results": tool_results,
    }
    _write_json(run_dir / "process_report.json", report)
    return report


def _build_session_trace(
    backend: str,
    request: OpenClawNativeRequest,
    provider_config: ProviderConfig,
    prompt_path: Path,
    context_path: Path,
    discovery_payload: dict[str, object],
    runtime_config: dict[str, object],
    skill_records: list[SkillAvailability],
    result: object | None,
    strict_issues: list[str],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "backend": backend,
        "execution_mode": request.execution_mode,
        "mode": request.mode,
        "topic": request.topic,
        "title": request.title,
        "provider": provider_config.provider,
        "model": provider_config.model,
        "prompt_path": str(prompt_path),
        "launcher_context_path": str(context_path),
        "agent_discovery": discovery_payload,
        "runtime_config": {
            "backend": str(runtime_config.get("backend", "") or ""),
            "transport": runtime_config.get("transport", ""),
            "discovery_paths": [str(path) for path in runtime_config.get("discovery_paths", [])],
            "use_default_discovery": bool(runtime_config.get("use_default_discovery", True)),
            "token_path": str(runtime_config.get("token_path", "") or ""),
            "agent_info_path": str(runtime_config.get("agent_info_path", "") or ""),
            "profile": str(runtime_config.get("profile", "") or ""),
            "agent_id": str(runtime_config.get("agent_id", "") or ""),
            "thinking": str(runtime_config.get("thinking", "") or ""),
        },
        "skills": [record.to_dict() for record in skill_records],
        "strict_issues": strict_issues,
    }
    if result is not None:
        usage = getattr(result, "usage", None)
        if hasattr(usage, "model_dump"):
            usage_payload = usage.model_dump()
        elif isinstance(usage, dict):
            usage_payload = usage
        else:
            usage_payload = {}
        payload["agent_result"] = {
            "request_id": str(getattr(result, "request_id", "") or ""),
            "success": bool(getattr(result, "success", False)),
            "text": str(getattr(result, "text", "") or ""),
            "error": str(getattr(result, "error", "") or ""),
            "duration_ms": int(getattr(result, "duration_ms", 0) or 0),
            "usage": usage_payload,
            "tool_results": [_tool_result_to_dict(item) for item in getattr(result, "tool_results", [])],
        }
    return payload


def _build_summary(
    backend: str,
    run_dir: Path,
    request: OpenClawNativeRequest,
    completion: ConductorCompletion | None,
    strict_compliant: bool,
    strict_issues: list[str],
    process_report: dict[str, object] | None,
    request_id: str,
) -> dict[str, object]:
    return {
        "run_id": run_dir.name,
        "mode": request.mode,
        "topic": request.topic,
        "title": request.title,
        "output_dir": str(run_dir),
        "backend": backend,
        "execution_mode": request.execution_mode,
        "request_id": request_id,
        "strict_compliant": strict_compliant,
        "strict_issues": strict_issues,
        "stop_reason": completion.stop_reason if completion is not None else "error",
        "best_total_score": completion.best_total_score if completion is not None else None,
        "process_report_path": str(run_dir / "process_report.json") if process_report else "",
        "required_evidence": {key: str(run_dir / file_name) for key, file_name in REQUIRED_AGENT_FILES.items()},
    }


def run_openclaw_native_demo(
    request: OpenClawNativeRequest,
    *,
    runtime: OpenClawConductorRuntime | None = None,
) -> OpenClawNativeResult:
    runtime = runtime or OpenClawConductorRuntime()
    normalized_request = normalize_request(request)
    settings = load_settings()
    provider_config = resolve_provider_config(
        settings,
        provider=normalized_request.provider,
        model=normalized_request.model,
    )

    label = normalized_request.topic if normalized_request.mode == "topic" else normalized_request.title
    if label is None:
        raise OpenClawConductorError("normalized request is missing its label")

    output_root = Path(normalized_request.output_dir) if normalized_request.output_dir else settings.output_root
    run_dir = create_run_dir(output_root, label)
    skills_dir = _resolve_skills_dir(normalized_request)
    workspace_root = Path.cwd().resolve()
    evidence_paths = _build_evidence_paths(run_dir)

    client = None
    agent_result = None
    completion: ConductorCompletion | None = None
    request_id = ""
    strict_issues: list[str] = []
    skill_records: list[SkillAvailability] = []
    runtime_config = _load_runtime_config()
    runtime_config["backend"] = str(runtime_config.get("backend", "") or "")
    openclaw_backend = CMDOP_NATIVE_BACKEND
    prompt_path = run_dir / "agent_prompt.md"
    context_path = run_dir / "launcher_context.json"
    cli_env: dict[str, str] | None = None

    discovery_fn = runtime.discover_agent_fn or (lambda: _default_discover_agent(runtime_config))
    client_factory = runtime.client_factory or (lambda: _default_client_factory(runtime_config))
    discovery_payload: dict[str, object] = {
        "found": False,
        "error": "cmdop discovery skipped for OpenClaw CLI backend",
    }

    configured_backend = str(runtime_config.get("backend", "") or "").lower()
    should_probe_discovery = configured_backend != OPENCLAW_CLI_BACKEND
    if should_probe_discovery:
        discovery_result = discovery_fn()
        discovery_payload = _serialize_discovery(discovery_result)

    openclaw_backend = _select_backend(runtime_config, runtime=runtime, discovery_payload=discovery_payload)
    runtime_config["backend"] = openclaw_backend

    try:
        if openclaw_backend == CMDOP_NATIVE_BACKEND:
            strict_issues.extend(_evaluate_discovery_strictness(discovery_payload, runtime_config))
            client = client_factory()
            skill_records, skill_issues = _inspect_skills(client, skills_dir)
        else:
            cli_env = build_openclaw_env(provider_config=provider_config)
            workspace_skills_dir = sync_workspace_skills(
                source_root=skills_dir,
                workspace_root=workspace_root,
                worker_skills=WORKER_SKILLS,
            )
            ensure_openclaw_profile(
                profile_name=str(runtime_config.get("profile", "") or DEFAULT_OPENCLAW_PROFILE),
                workspace_root=workspace_root,
                provider_config=provider_config,
            )
            listed_skills = list_openclaw_skills(
                profile_name=str(runtime_config.get("profile", "") or DEFAULT_OPENCLAW_PROFILE),
                workspace_root=workspace_root,
                env=cli_env,
            )
            skill_records, skill_issues = _inspect_openclaw_cli_skills(
                listed_skills,
                skills_dir=skills_dir,
                workspace_skills_dir=workspace_skills_dir,
            )
        strict_issues.extend(skill_issues)
        context_path, context = _write_launcher_context(
            run_dir,
            normalized_request,
            provider_config,
            skills_dir,
            evidence_paths,
            skill_records,
        )
        prompt = build_agent_prompt(context, skill_records, evidence_paths)
        prompt_path.write_text(prompt, encoding="utf-8")

        if normalized_request.execution_mode == "openclaw-strict" and strict_issues:
            raise OpenClawConductorError("; ".join(strict_issues))

        if openclaw_backend == CMDOP_NATIVE_BACKEND:
            agent_result = _run_agent(
                client,
                prompt=prompt,
                provider_config=provider_config,
                request=normalized_request,
            )
        else:
            if cli_env is None:
                raise OpenClawConductorError("OpenClaw CLI environment was not initialized.")
            bootstrap_message = dedent(
                f"""
                Read and follow the task spec at "{prompt_path}".
                Use that file as the complete instruction source for this run.
                Do not explore unrelated files or directories.
                Return the final JSON object required by that spec after writing all required evidence files.
                """
            ).strip()
            try:
                agent_result = run_openclaw_local_agent(
                    profile_name=str(runtime_config.get("profile", "") or DEFAULT_OPENCLAW_PROFILE),
                    agent_id=str(runtime_config.get("agent_id", "") or DEFAULT_OPENCLAW_AGENT_ID),
                    session_id=run_dir.name,
                    workspace_root=workspace_root,
                    prompt=bootstrap_message,
                    env=cli_env,
                    timeout_seconds=normalized_request.agent_timeout_seconds or DEFAULT_AGENT_TIMEOUT_SECONDS,
                    thinking_level=_resolve_cli_thinking(
                        provider_config,
                        str(runtime_config.get("thinking", "") or DEFAULT_OPENCLAW_THINKING),
                    ),
                    stdout_path=run_dir / "openclaw_stdout.log",
                    stderr_path=run_dir / "openclaw_stderr.log",
                )
            except OpenClawCliBackendError as exc:
                raise OpenClawConductorError(str(exc)) from exc
        request_id = str(getattr(agent_result, "request_id", "") or "")
        completion = _extract_completion(agent_result)

        evidence_issues = _verify_evidence(evidence_paths)
        strict_issues.extend(evidence_issues)
        if not completion.strict_compliant:
            strict_issues.append("agent reported strict_compliant=false")

        strict_compliant = not strict_issues
        process_report = _build_process_report(
            run_dir,
            evidence_paths,
            agent_result,
            strict_compliant,
            strict_issues,
        )
        session_trace = _build_session_trace(
            openclaw_backend,
            normalized_request,
            provider_config,
            prompt_path,
            context_path,
            discovery_payload,
            runtime_config,
            skill_records,
            agent_result,
            strict_issues,
        )
        _write_json(run_dir / "session_trace.json", session_trace)
        summary = _build_summary(
            openclaw_backend,
            run_dir,
            normalized_request,
            completion,
            strict_compliant,
            strict_issues,
            process_report,
            request_id,
        )
        _write_json(run_dir / "summary.json", summary)

        if normalized_request.execution_mode == "openclaw-strict" and strict_issues:
            raise OpenClawConductorError("; ".join(strict_issues))

        return OpenClawNativeResult(
            run_id=run_dir.name,
            mode=normalized_request.mode,
            output_dir=str(run_dir),
            request_id=request_id,
            openclaw_backend=openclaw_backend,
            execution_mode=normalized_request.execution_mode,
            strict_compliant=strict_compliant,
            stop_reason=completion.stop_reason,
            summary_path=str(run_dir / "summary.json"),
            session_trace_path=str(run_dir / "session_trace.json"),
            process_report_path=str(run_dir / "process_report.json"),
            best_draft_path=str(evidence_paths["best_draft"]),
            self_eval_path=str(evidence_paths["self_eval"]),
            prompt_path=str(prompt_path),
        )
    except Exception as exc:
        if run_dir.exists():
            session_trace = _build_session_trace(
                openclaw_backend,
                normalized_request,
                provider_config,
                prompt_path,
                context_path,
                discovery_payload,
                runtime_config,
                skill_records,
                agent_result,
                strict_issues + [str(exc)],
            )
            _write_json(run_dir / "session_trace.json", session_trace)
            summary = _build_summary(
                openclaw_backend,
                run_dir,
                normalized_request,
                completion,
                False,
                strict_issues + [str(exc)],
                None,
                request_id,
            )
            summary["error"] = str(exc)
            _write_json(run_dir / "summary.json", summary)
        if isinstance(exc, OpenClawConductorError):
            raise
        raise OpenClawConductorError(str(exc)) from exc
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
