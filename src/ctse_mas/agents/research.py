"""Research Agent – reads files and loads them into state."""

from __future__ import annotations

from ctse_mas.state import AgentState
from ctse_mas.tools.file_reader import read_json_file, read_text_file
from ctse_mas.tools.logger import log_agent_action


def research_agent(state: AgentState) -> dict:
    """Read the submission and rubric files and store them in state.

    The research agent *never* scores or evaluates content – it only
    loads data.  Returns only the fields it changes.
    """
    session_id: str = state.get("session_id", "")
    submission_path: str = state.get("submission_path", "")
    rubric_path: str = state.get("rubric_path", "")

    inputs = {
        "submission_path": submission_path,
        "rubric_path": rubric_path,
    }

    try:
        submission_text = read_text_file(submission_path)
        rubric_data = read_json_file(rubric_path)
        research_status = "success"
        error = ""
    except RuntimeError as exc:
        submission_text = ""
        rubric_data = {}
        research_status = "failed"
        error = str(exc)

    outputs = {
        "research_status": research_status,
        "submission_text_length": len(submission_text),
        "rubric_criteria_count": len(rubric_data.get("criteria", [])),
    }

    log_entry = log_agent_action(
        session_id=session_id,
        agent="research",
        action="read_files",
        inputs=inputs,
        outputs=outputs,
    )

    existing_logs: list = list(state.get("agent_logs") or [])
    existing_logs.append(log_entry)

    updates: dict = {
        "submission_text": submission_text,
        "rubric_data": rubric_data,
        "research_status": research_status,
        "agent_logs": existing_logs,
    }
    if error:
        updates["error"] = error

    return updates
