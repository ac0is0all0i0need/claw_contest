from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


class OpenClawCompatibilityError(RuntimeError):
    """Raised when OpenClaw cannot be loaded through the compatibility bootstrap."""


@dataclass(frozen=True, slots=True)
class OpenClawLoadResult:
    OpenClaw: type[Any]
    AsyncOpenClaw: type[Any]
    backend: str
    details: dict[str, object]


def _import_cmdop_classes() -> tuple[type[Any], type[Any]]:
    try:
        cmdop = importlib.import_module("cmdop")
    except Exception as exc:  # pragma: no cover - defensive wrapper
        raise OpenClawCompatibilityError(f"Unable to import cmdop fallback: {exc}") from exc

    try:
        return cmdop.CMDOPClient, cmdop.AsyncCMDOPClient
    except AttributeError as exc:
        raise OpenClawCompatibilityError("cmdop fallback does not expose CMDOPClient classes.") from exc


def load_openclaw(*, allow_cmdop_fallback: bool = True) -> OpenClawLoadResult:
    """Load OpenClaw without requiring vendor package edits.

    The current OpenClaw package imports ``TimeoutError`` from ``cmdop.exceptions``.
    Some installed CMDOP versions expose that class as ``ConnectionTimeoutError``
    instead. This function applies an in-memory alias only when needed, then imports
    OpenClaw lazily.
    """

    try:
        cmdop_exceptions = importlib.import_module("cmdop.exceptions")
    except Exception as exc:
        raise OpenClawCompatibilityError(f"Unable to import cmdop.exceptions: {exc}") from exc

    alias_applied = False
    if not hasattr(cmdop_exceptions, "TimeoutError"):
        connection_timeout = getattr(cmdop_exceptions, "ConnectionTimeoutError", None)
        if connection_timeout is not None:
            setattr(cmdop_exceptions, "TimeoutError", connection_timeout)
            alias_applied = True

    try:
        openclaw = importlib.import_module("openclaw")
    except Exception as exc:
        if not allow_cmdop_fallback:
            raise OpenClawCompatibilityError(f"Unable to import openclaw: {exc}") from exc
        OpenClaw, AsyncOpenClaw = _import_cmdop_classes()
        return OpenClawLoadResult(
            OpenClaw=OpenClaw,
            AsyncOpenClaw=AsyncOpenClaw,
            backend="cmdop_fallback",
            details={
                "alias_used": alias_applied,
                "fallback_used": True,
                "openclaw_error": str(exc),
            },
        )

    try:
        OpenClaw = openclaw.OpenClaw
        AsyncOpenClaw = openclaw.AsyncOpenClaw
    except AttributeError as exc:
        if not allow_cmdop_fallback:
            raise OpenClawCompatibilityError("openclaw does not expose expected client classes.") from exc
        OpenClaw, AsyncOpenClaw = _import_cmdop_classes()
        return OpenClawLoadResult(
            OpenClaw=OpenClaw,
            AsyncOpenClaw=AsyncOpenClaw,
            backend="cmdop_fallback",
            details={
                "alias_used": alias_applied,
                "fallback_used": True,
                "openclaw_error": "OpenClaw imported but expected classes were missing.",
            },
        )

    if alias_applied:
        return OpenClawLoadResult(
            OpenClaw=OpenClaw,
            AsyncOpenClaw=AsyncOpenClaw,
            backend="cmdop_alias_compat",
            details={
                "alias_used": True,
                "fallback_used": False,
                "openclaw_error": None,
            },
        )

    return OpenClawLoadResult(
        OpenClaw=OpenClaw,
        AsyncOpenClaw=AsyncOpenClaw,
        backend="openclaw_imported",
        details={
            "alias_used": False,
            "fallback_used": False,
            "openclaw_error": None,
        },
    )
