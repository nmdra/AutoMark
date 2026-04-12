"""Ingestion Agent – validates files, reads their content, and extracts student_id."""

from __future__ import annotations

import re
import uuid

from ctse_mas.state import AgentState
from ctse_mas.tools.file_ops import read_json_file, read_text_file, validate_submission_files
from ctse_mas.tools.logger import log_agent_action

# Matches "Student ID: <value>" or "Student ID : <value>" (case-insensitive)
_STUDENT_ID_RE = re.compile(r"student\s+id\s*:\s*(\S+)", re.IGNORECASE)


def _extract_student_id(text: str) -> str:
    """Extract the student ID from the submission text.

    Looks for a line matching ``Student ID: <value>``.  Returns an empty
    string when no match is found.
    """
    match = _STUDENT_ID_RE.search(text)
    return match.group(1) if match else ""


def ingestion_agent(state: AgentState) -> dict:
    """Validate and load submission files then extract the student's ID.

    Responsibilities:
    * Validate that submission and rubric file paths are non-empty, the files
      exist, are non-empty, and have the correct extensions.
    * Read the submission text and parse the rubric JSON.
    * Extract ``student_id`` from the submission text.

    Sets ``ingestion_status`` to ``"success"`` or ``"failed"``.  On failure
    the graph routes directly to the Report Agent for graceful degradation.
    Returns only the fields it changes.
    """
    session_id: str = state.get("session_id") or str(uuid.uuid4())
    submission_path: str = state.get("submission_path", "")
    rubric_path: str = state.get("rubric_path", "")

    inputs = {
        "submission_path": submission_path,
        "rubric_path": rubric_path,
    }

    submission_text = ""
    rubric_data: dict = {}
    student_id = ""
    ingestion_status = "failed"
    error = ""

    try:
        validate_submission_files(submission_path, rubric_path)
        submission_text = read_text_file(submission_path)
        rubric_data = read_json_file(rubric_path)
        student_id = _extract_student_id(submission_text)
        ingestion_status = "success"
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        error = str(exc)

    outputs = {
        "ingestion_status": ingestion_status,
        "student_id": student_id,
        "submission_text_length": len(submission_text),
        "rubric_criteria_count": len(rubric_data.get("criteria", [])),
    }

    log_entry = log_agent_action(
        session_id=session_id,
        agent="ingestion",
        action="ingest_submission",
        inputs=inputs,
        outputs=outputs,
    )

    existing_logs: list = list(state.get("agent_logs") or [])
    existing_logs.append(log_entry)

    updates: dict = {
        "session_id": session_id,
        "ingestion_status": ingestion_status,
        "student_id": student_id,
        "submission_text": submission_text,
        "rubric_data": rubric_data,
        "agent_logs": existing_logs,
    }
    if error:
        updates["error"] = error

    return updates
