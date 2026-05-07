"""Workflow-mode constants shared by CLI, Web UI, and project storage."""

from __future__ import annotations


WORKFLOW_MODE_CLASSIC = "classic"
WORKFLOW_MODE_AGENTIC = "agentic"
DEFAULT_WORKFLOW_MODE = WORKFLOW_MODE_CLASSIC
WORKFLOW_MODES = {
    WORKFLOW_MODE_CLASSIC,
    WORKFLOW_MODE_AGENTIC,
}


def normalize_workflow_mode(mode: object, default: str = DEFAULT_WORKFLOW_MODE) -> str:
    normalized = str(mode or "").strip().lower()
    if normalized in WORKFLOW_MODES:
        return normalized
    return default if default in WORKFLOW_MODES else DEFAULT_WORKFLOW_MODE
