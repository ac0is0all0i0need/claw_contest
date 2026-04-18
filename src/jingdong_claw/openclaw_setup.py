from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .artifacts import slugify
from .models import ProviderConfig
from .openclaw_cli_backend import (
    OpenClawCliBackendError,
    build_openclaw_env,
    ensure_openclaw_profile,
    list_openclaw_skills,
    sync_workspace_skills,
)

DEFAULT_OPENCLAW_PROFILE = "jdclaw-comp"
DEFAULT_OPENCLAW_AGENT_ID = "main"
DEFAULT_OPENCLAW_THINKING = "low"


class OpenClawSetupError(RuntimeError):
    """Raised when repo-owned OpenClaw setup cannot be prepared safely."""


@dataclass(frozen=True, slots=True)
class RepoSkillSpec:
    name: str
    directory_name: str
    description: str


SURVEY_RESEARCH_WORKER = RepoSkillSpec(
    name="survey-research-worker",
    directory_name="survey-research-worker",
    description="Deep-research backed initial survey generation worker.",
)
SURVEY_EVAL_WORKER = RepoSkillSpec(
    name="survey-eval-worker",
    directory_name="survey-eval-worker",
    description="Prompt-based evaluator worker for one survey draft.",
)
SURVEY_REVISE_WORKER = RepoSkillSpec(
    name="survey-revise-worker",
    directory_name="survey-revise-worker",
    description="Revision candidate selector worker for one survey draft.",
)
SURVEY_DEMO_RUNNER = RepoSkillSpec(
    name="survey-demo-runner",
    directory_name="survey-demo-runner",
    description="Root OpenClaw skill that launches the recorded competition demo runner.",
)

ROOT_LAUNCH_SKILLS: tuple[RepoSkillSpec, ...] = (
    SURVEY_DEMO_RUNNER,
    SURVEY_RESEARCH_WORKER,
    SURVEY_EVAL_WORKER,
    SURVEY_REVISE_WORKER,
)

RECORDED_DEMO_REQUIRED_SKILLS: tuple[RepoSkillSpec, ...] = (
    SURVEY_REVISE_WORKER,
)


@dataclass(slots=True)
class OpenClawWorkspaceSetup:
    profile_name: str
    workspace_root: Path
    source_skills_dir: Path
    workspace_skills_dir: Path
    env: dict[str, str]
    listed_skills: list[dict[str, object]]


def resolve_source_skills_dir(
    skills_dir: str | Path | None,
    *,
    workspace_root: Path | None = None,
) -> Path:
    if skills_dir is not None:
        return Path(skills_dir).resolve()
    root = workspace_root.resolve() if workspace_root is not None else Path.cwd().resolve()
    return (root / ".cmdop" / "skills").resolve()


def prepare_openclaw_workspace(
    *,
    provider_config: ProviderConfig,
    required_skills: tuple[RepoSkillSpec, ...],
    skills_dir: str | Path | None = None,
    workspace_root: Path | None = None,
    profile_name: str = DEFAULT_OPENCLAW_PROFILE,
    sync_skills: bool = True,
) -> OpenClawWorkspaceSetup:
    resolved_workspace_root = workspace_root.resolve() if workspace_root is not None else Path.cwd().resolve()
    source_skills_dir = resolve_source_skills_dir(skills_dir, workspace_root=resolved_workspace_root)
    env = build_openclaw_env(provider_config=provider_config)

    if sync_skills:
        workspace_skills_dir = sync_workspace_skills(
            source_root=source_skills_dir,
            workspace_root=resolved_workspace_root,
            worker_skills=required_skills,
        )
    else:
        workspace_skills_dir = (resolved_workspace_root / "skills").resolve()
        if not workspace_skills_dir.exists():
            raise OpenClawSetupError(
                f"Workspace skills directory is missing: {workspace_skills_dir}. "
                "Run the OpenClaw demo helper first to sync repo-owned skills."
            )

    try:
        ensure_openclaw_profile(
            profile_name=profile_name,
            workspace_root=resolved_workspace_root,
            provider_config=provider_config,
        )
        listed_skills = list_openclaw_skills(
            profile_name=profile_name,
            workspace_root=resolved_workspace_root,
            env=env,
        )
    except OpenClawCliBackendError as exc:
        raise OpenClawSetupError(str(exc)) from exc

    listed_names = {
        str(item.get("name", "")).strip()
        for item in listed_skills
        if isinstance(item, dict)
    }
    missing = [spec.name for spec in required_skills if spec.name not in listed_names]
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise OpenClawSetupError(
            f"OpenClaw profile '{profile_name}' is missing required workspace skills: {missing_text}."
        )

    return OpenClawWorkspaceSetup(
        profile_name=profile_name,
        workspace_root=resolved_workspace_root,
        source_skills_dir=source_skills_dir,
        workspace_skills_dir=workspace_skills_dir,
        env=env,
        listed_skills=listed_skills,
    )


def build_root_launch_message(*, topic: str, run_dir: str | Path) -> str:
    return (
        "Use survey-demo-runner to run the self-evolution demo for topic "
        f"{topic} and write outputs under {Path(run_dir).resolve()}."
    )


def build_root_launch_command(
    *,
    profile_name: str,
    topic: str,
    run_dir: str | Path,
    session_id: str | None = None,
) -> str:
    normalized_session_id = session_id or f"{slugify(topic)[:32] or 'demo'}-demo"
    message = build_root_launch_message(topic=topic, run_dir=run_dir)
    escaped_message = message.replace("'", "''")
    return (
        f"openclaw --profile {profile_name} agent --local "
        f"--session-id '{normalized_session_id}' "
        f"--message '{escaped_message}'"
    )
