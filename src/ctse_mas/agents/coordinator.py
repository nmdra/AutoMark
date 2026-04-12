"""Coordinator Agent – validates file paths and sets session metadata."""

from __future__ import annotations

import uuid

from ctse_mas.state import AgentState
from ctse_mas.tools.file_validator import validate_submission_files
from ctse_mas.tools.logger import log_agent_action


def coordinator_agent(state: AgentState) -> dict:
    """Validate submission and rubric file paths.

    Accepts ``AgentState`` and returns only the fields it changes:
    ``validation_status``, ``validation_error``, ``session_id``,
    and an updated ``agent_logs`` list.

    The coordinator *never* reads file contents.
    """
    session_id: str = state.get("session_id") or str(uuid.uuid4())
    submission_path: str = state.get("submission_path", "")
    rubric_path: str = state.get("rubric_path", "")

    inputs = {
        "submission_path": submission_path,
        "rubric_path": rubric_path,
    }

    try:
        result = validate_submission_files(submission_path, rubric_path)
        validation_status = result["status"]
        validation_error = ""
    except (ValueError, FileNotFoundError) as exc:
        validation_status = "failed"
        validation_error = str(exc)

    outputs = {
        "validation_status": validation_status,
        "validation_error": validation_error,
    }

    log_entry = log_agent_action(
        session_id=session_id,
        agent="coordinator",
        action="validate_submission_files",
        inputs=inputs,
        outputs=outputs,
    )

    existing_logs: list = list(state.get("agent_logs") or [])
    existing_logs.append(log_entry)

    return {
        "session_id": session_id,
        "validation_status": validation_status,
        "validation_error": validation_error,
        "agent_logs": existing_logs,
    }
