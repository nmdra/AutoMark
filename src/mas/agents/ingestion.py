"""Ingestion Agent – validates files, reads content, and extracts student metadata."""

from __future__ import annotations

import re
import uuid

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from mas.config import settings
from mas.llm import get_metadata_json_llm
from mas.state import AgentState
from mas.tools.file_ops import read_json_file, read_text_file, validate_submission_files
from mas.tools.logger import log_agent_action, timed_model_call

# Matches "Student ID: <value>" or "Student ID : <value>" (case-insensitive)
_STUDENT_ID_RE = re.compile(r"student\s+id\s*:\s*(\S+)", re.IGNORECASE)

# Matches "Student Name: <value>" or "Student Name - <value>" (case-insensitive)
_STUDENT_NAME_RE = re.compile(r"student\s+name\s*[:=-]\s*([^\n\r]+)", re.IGNORECASE)
# Matches "Assignment No: <value>", "HW-03", etc. (case-insensitive)
_ASSIGNMENT_NUMBER_RE = re.compile(
    r"(?:"
    r"(?:assignment|assg?n?\.?|homework)\s*(?:no\.?|number|#)?\s*[:=.\-]\s*"
    r"|hw\s*[-:#]?\s*"
    r")([A-Za-z0-9-]+)",
    re.IGNORECASE,
)
_METADATA_CONTEXT_MAX_CHARS = 2500
_IDENTITY_LINE_RE = re.compile(
    r"\b(student\s*)?(id|name|registration|reg\s*no|index|roll\s*no|assignment|hw|homework)\b",
    re.IGNORECASE,
)


class StudentDetails(BaseModel):
    """Structured extractor output for text submissions.

    Field names mirror the SmolLM2 extractor schema. ``student_number`` is
    mapped to the internal ``student_id`` state key.
    """

    student_number: str = Field(
        description="The student's unique number/ID. Empty string if not found."
    )
    student_name: str = Field(
        description="The student's full name. Empty string if not found."
    )
    assignment_number: str = Field(
        description="The assignment number/identifier. Empty string if not found."
    )


def _extract_student_id(text: str) -> str:
    """Extract the student ID from the submission text.

    Looks for a line matching ``Student ID: <value>``.  Returns an empty
    string when no match is found.
    """
    match = _STUDENT_ID_RE.search(text)
    return match.group(1) if match else ""


def _extract_student_name(text: str) -> str:
    """Extract the student name from the submission text.

    Looks for a line matching ``Student Name: <value>`` (or ``-``/``=`` separators).
    Returns an empty string when no match is found.
    """
    return _extract_with_pattern(text, _STUDENT_NAME_RE)


def _extract_assignment_number(text: str) -> str:
    """Extract the assignment number from submission text."""
    return _extract_with_pattern(text, _ASSIGNMENT_NUMBER_RE)


def _extract_with_pattern(text: str, pattern: re.Pattern[str]) -> str:
    """Extract and normalize regex group-1 content, returning empty when missing."""
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def _build_metadata_context(submission_text: str) -> str:
    """Build a compact context for identity-field extraction."""
    normalized = submission_text.strip()
    if not normalized:
        return ""

    top_chunk = normalized[:_METADATA_CONTEXT_MAX_CHARS]
    identity_lines: list[str] = []
    seen: set[str] = set()
    for line in normalized.splitlines():
        clean_line = line.strip()
        if not clean_line:
            continue
        if not _IDENTITY_LINE_RE.search(clean_line):
            continue
        if clean_line in seen:
            continue
        seen.add(clean_line)
        identity_lines.append(clean_line)
        if len(identity_lines) >= 20:
            break

    if not identity_lines:
        return top_chunk

    return (
        f"{top_chunk}\n\n"
        "## Candidate Identity Lines\n"
        "\n".join(identity_lines)
    )


def _build_extraction_prompt(metadata_context: str) -> str:
    return (
        "### Instruction:\n"
        "Extract student info as JSON from the following text.\n\n"
        "### Input:\n"
        f"{metadata_context}\n\n"
        "### Response:\n"
    )


def ingestion_agent(state: AgentState) -> dict:
    """Validate and load submission files then extract student metadata.

    Responsibilities:
    * Validate that submission and rubric file paths are non-empty, the files
      exist, are non-empty, and have the correct extensions.
    * Read the submission text and parse the rubric JSON.
    * Extract ``student_id`` and ``student_name`` from the submission text.

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
        regex_id = _extract_student_id(submission_text)
        regex_name = _extract_student_name(submission_text)
        regex_assignment_number = _extract_assignment_number(submission_text)
        student_id = regex_id
        student_name = regex_name
        assignment_number = regex_assignment_number
        use_llm_extraction = (
            not settings.pdf_regex_fast_path_enabled
            or not regex_id
            or not regex_name
            or not regex_assignment_number
        )
        if use_llm_extraction:
            try:
                llm = get_metadata_json_llm(schema=StudentDetails)
                metadata_context = _build_metadata_context(submission_text)
                messages = [
                    SystemMessage(
                        content=(
                            "You are a precise student assignment data extractor.\n"
                            "Output ONLY a valid JSON object. No explanation. No extra text. No markdown.\n"
                            'Always output exactly: {"student_number":"...","student_name":"...","assignment_number":"..."}'
                        )
                    ),
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
                student_id = (result.student_number or regex_id).strip()
                student_name = (result.student_name or regex_name).strip()
                assignment_number = (
                    result.assignment_number or regex_assignment_number
                ).strip()
            except Exception as exc:  # noqa: BLE001
                student_id = regex_id
                student_name = regex_name
                assignment_number = regex_assignment_number
                error = f"LLM extraction failed: {exc}"
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
