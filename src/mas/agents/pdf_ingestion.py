"""PDF Ingestion Agent – converts a PDF submission to text and extracts student details."""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from mas.config import settings
from mas.llm import get_metadata_json_llm
from mas.state import AgentState
from mas.tools.file_ops import read_json_file
from mas.tools.logger import log_agent_action, timed_model_call
from mas.tools.pdf_processor import convert_pdf_to_markdown

# ── Regex patterns reused from the text ingestion path ────────────────────────

# Matches "Student ID: <value>" or "ID: <value>" (case-insensitive)
_STUDENT_ID_RE = re.compile(r"(?:student\s+)?id\s*:\s*(\S+)", re.IGNORECASE)

# Matches "Student Name: <value>" or "Name: <value>" (case-insensitive)
# Captures to end-of-line (strips trailing whitespace)
_STUDENT_NAME_RE = re.compile(
    r"(?:student\s+)?name\s*:\s*(.+?)(?:\s*$)",
    re.IGNORECASE | re.MULTILINE,
)
_ASSIGNMENT_NUMBER_RE = re.compile(
    r"(?:"
    r"(?:assignment|assg?n?\.?|homework)\s*(?:no\.?|number|#)?\s*[:=.\-]\s*"
    r"|hw\s*[-:#]?\s*"
    r")([A-Za-z0-9-]+)",
    re.IGNORECASE,
)


class StudentDetails(BaseModel):
    """Structured extractor output for PDF submissions.

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


def _regex_extract_student_id(text: str) -> str:
    """Extract student ID using a regex pattern.

    Returns an empty string when no match is found.
    """
    match = _STUDENT_ID_RE.search(text)
    return match.group(1).strip() if match else ""


def _regex_extract_student_name(text: str) -> str:
    """Extract student name using a regex pattern.

    Returns an empty string when no match is found.
    """
    match = _STUDENT_NAME_RE.search(text)
    if not match:
        return ""
    name = match.group(1).strip()
    name = re.sub(r"^\*+\s*", "", name)
    name = re.sub(r"\s*\*+$", "", name)
    return name.strip()


def _regex_extract_assignment_number(text: str) -> str:
    """Extract assignment number using a regex pattern."""
    match = _ASSIGNMENT_NUMBER_RE.search(text)
    return match.group(1).strip() if match else ""


_METADATA_CONTEXT_MAX_CHARS = 2500
_IDENTITY_LINE_RE = re.compile(
    r"\b(student\s*)?(id|name|registration|reg\s*no|index|roll\s*no|assignment|hw|homework)\b",
    re.IGNORECASE,
)


def _build_metadata_context(markdown_text: str) -> str:
    """Build a compact context for identity-field extraction."""
    normalized = markdown_text.strip()
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
        f"{chr(10).join(identity_lines)}"
    )


def _build_extraction_prompt(metadata_context: str) -> str:
    return (
        "### Instruction:\n"
        "Extract student info as JSON from the following text.\n\n"
        "### Input:\n"
        f"{metadata_context}\n\n"
        "### Response:\n"
    )


def pdf_ingestion_agent(state: AgentState) -> dict:
    """Convert a PDF submission to text and extract structured student data.

    Responsibilities:
    * Validate that the submission path points to an existing ``.pdf`` file.
    * Convert the PDF to Markdown text using ``pymupdf4llm``.
    * Use the LLM to extract ``student_id``, ``student_name``, and
      ``submission_text`` from the Markdown.
    * Read and parse the JSON rubric.

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
    student_id = ""
    student_name = ""
    assignment_number = ""
    rubric_data: dict[str, Any] = {}
    ingestion_status = "failed"
    error = ""
    markdown_text = ""
    stage1_ok = False

    # --- Stage 1: Validate paths and convert PDF → Markdown ---
    try:
        if not submission_path:
            raise ValueError("submission_path must not be empty")
        if not rubric_path:
            raise ValueError("rubric_path must not be empty")

        pdf_path = Path(submission_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {submission_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Expected a .pdf submission file, got: {pdf_path.suffix!r}"
            )
        if pdf_path.stat().st_size == 0:
            raise ValueError(f"PDF submission file is empty: {submission_path}")

        markdown_text = convert_pdf_to_markdown(submission_path)
        submission_text = markdown_text
        rubric_data = read_json_file(rubric_path)
        stage1_ok = True
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        error = str(exc)

    # --- Stage 2: Optional regex fast-path; otherwise extract via LLM ---
    if stage1_ok:
        regex_id = ""
        regex_name = ""
        regex_assignment_number = ""
        student_id = ""
        student_name = ""
        assignment_number = ""
        if settings.pdf_regex_fast_path_enabled:
            # Fast path: try regex extraction first – no LLM call needed when all
            # metadata fields are found deterministically.
            regex_id = _regex_extract_student_id(markdown_text)
            regex_name = _regex_extract_student_name(markdown_text)
            regex_assignment_number = _regex_extract_assignment_number(markdown_text)
            if regex_id and regex_name and regex_assignment_number:
                student_id = regex_id
                student_name = regex_name
                assignment_number = regex_assignment_number

        if not (student_id and student_name and assignment_number):
            # Either regex is disabled, regex did not fully match, or fast-path
            # extraction was intentionally bypassed; use the LLM extractor.
            try:
                llm = get_metadata_json_llm(schema=StudentDetails)
                metadata_context = _build_metadata_context(markdown_text)
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
                    service="pdf_ingestion",
                    task_type="student_details_extraction",
                    model=settings.metadata_extractor_model_name,
                )
                student_id = (result.student_number or regex_id).strip()
                student_name = (result.student_name or regex_name).strip()
                assignment_number = (
                    result.assignment_number or regex_assignment_number
                ).strip()
            except Exception as exc:  # noqa: BLE001
                # Fall back to regex results (when available) + raw markdown.
                student_id = regex_id
                student_name = regex_name
                assignment_number = regex_assignment_number
                error = str(exc)

        if rubric_data:
            ingestion_status = "success"

    outputs = {
        "ingestion_status": ingestion_status,
        "student_id": student_id,
        "student_name": student_name,
        "assignment_number": assignment_number,
        "submission_text_length": len(submission_text),
        "rubric_criteria_count": len(rubric_data.get("criteria", [])),
    }

    log_entry = log_agent_action(
        session_id=session_id,
        agent="pdf_ingestion",
        action="ingest_pdf_submission",
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
