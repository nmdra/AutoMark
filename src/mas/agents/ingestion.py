"""Ingestion Agent – validates files, reads content, and extracts student metadata."""

from __future__ import annotations

import uuid

from langchain_core.messages import HumanMessage, SystemMessage

from mas.agents.metadata_extraction import (
    METADATA_SYSTEM_PROMPT,
    StudentDetails,
    _build_extraction_prompt,
    _build_metadata_context,
    _extract_metadata_regex,
)
from mas.config import settings
from mas.llm import get_metadata_json_llm
from mas.state import AgentState
from mas.tools.file_ops import read_json_file, read_text_file, validate_submission_files
from mas.tools.logger import log_agent_action, timed_model_call

def ingestion_agent(state: AgentState) -> dict:
    """Validate and load submission files then extract student metadata.

    Responsibilities:
    * Validate that submission and rubric file paths are non-empty, the files
      exist, are non-empty, and have the correct extensions.
    * Read the submission text and parse the rubric JSON.
    * Extract ``student_id``, ``student_name``, and ``assignment_number`` using
      the metadata extractor model.

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
    student_name = ""
    assignment_number = ""
    ingestion_status = "failed"
    error = ""

    try:
        validate_submission_files(submission_path, rubric_path)
        submission_text = read_text_file(submission_path)
        rubric_data = read_json_file(rubric_path)
        (
            regex_student_id,
            regex_student_name,
            regex_assignment_number,
        ) = _extract_metadata_regex(submission_text)
        try:
            llm = get_metadata_json_llm(schema=StudentDetails)
            metadata_context = _build_metadata_context(submission_text)
            messages = [
                SystemMessage(content=METADATA_SYSTEM_PROMPT),
                HumanMessage(content=_build_extraction_prompt(metadata_context)),
            ]
            result: StudentDetails = timed_model_call(
                llm=llm,
                messages=messages,
                session_id=session_id,
                service="ingestion",
                task_type="student_details_extraction",
                model=settings.metadata_extractor_model_name,
            )
            student_id = (result.student_number or "").strip() or regex_student_id
            student_name = (result.student_name or "").strip() or regex_student_name
            assignment_number = (
                (result.assignment_number or "").strip() or regex_assignment_number
            )
        except Exception as exc:  # noqa: BLE001
            error = f"LLM extraction failed: {exc}"
            student_id = regex_student_id
            student_name = regex_student_name
            assignment_number = regex_assignment_number
        ingestion_status = "success"
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        error = str(exc)

    outputs = {
        "ingestion_status": ingestion_status,
        "student_id": student_id,
        "student_name": student_name,
        "assignment_number": assignment_number,
        "submission_text_length": len(submission_text),
        "rubric_criteria_count": len(rubric_data.get("criteria", [])),
    }
    if error:
        outputs["error"] = error

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
        "student_name": student_name,
        "assignment_number": assignment_number,
        "submission_text": submission_text,
        "rubric_data": rubric_data,
        "agent_logs": existing_logs,
    }
    if error:
        updates["error"] = error

    return updates
