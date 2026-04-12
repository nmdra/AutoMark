"""PDF Ingestion Agent – converts a PDF submission to text and extracts student details."""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from mas.config import settings
from mas.llm import get_light_json_llm
from mas.state import AgentState
from mas.tools.file_ops import read_json_file
from mas.tools.logger import log_agent_action
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


class StudentDetails(BaseModel):
    """Structured LLM output for student details extracted from a PDF submission."""

    student_id: str = Field(
        description="The student's unique ID (e.g. IT21000001). Empty string if not found."
    )
    student_name: str = Field(
        description="The student's full name. Empty string if not found."
    )
    submission_text: str = Field(
        description="The complete submission content, cleaned of cover-page boilerplate."
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


def _build_extraction_prompt(markdown_text: str) -> str:
    return (
        "The following is a student submission extracted from a PDF. "
        "Extract the student's ID, full name, and submission body text.\n\n"
        "Rules:\n"
        "- student_id: look for patterns like 'Student ID:', 'ID:', or a numeric/alphanumeric code.\n"
        "- student_name: look for patterns like 'Name:', 'Student Name:', or a proper name near the top.\n"
        "- submission_text: the substantive academic content (exclude cover pages, headers, footers).\n"
        "- Return empty strings for fields that cannot be found.\n\n"
        f"## PDF Content\n\n{markdown_text}"
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
        rubric_data = read_json_file(rubric_path)
        stage1_ok = True
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        error = str(exc)

    # --- Stage 2: Optional regex fast-path; otherwise extract via LLM ---
    if stage1_ok:
        regex_id = ""
        regex_name = ""
        if settings.pdf_regex_fast_path_enabled:
            # Fast path: try regex extraction first – no LLM call needed when both
            # student_id and student_name are found deterministically.
            regex_id = _regex_extract_student_id(markdown_text)
            regex_name = _regex_extract_student_name(markdown_text)
            if regex_id and regex_name:
                student_id = regex_id
                student_name = regex_name
                submission_text = markdown_text

        if not submission_text:
            # Either regex is disabled, regex did not fully match, or fast-path
            # extraction was intentionally bypassed; use the LLM extractor.
            try:
                llm = get_light_json_llm(schema=StudentDetails)
                messages = [
                    SystemMessage(
                        content=(
                            "You are a data extraction assistant. "
                            "Extract student details from the provided PDF text. "
                            "Respond ONLY with valid JSON matching the required schema."
                        )
                    ),
                    HumanMessage(content=_build_extraction_prompt(markdown_text)),
                ]
                result: StudentDetails = llm.invoke(messages)
                student_id = result.student_id
                student_name = result.student_name
                submission_text = result.submission_text or markdown_text
            except Exception as exc:  # noqa: BLE001
                # Fall back to regex results (when available) + raw markdown.
                student_id = regex_id
                student_name = regex_name
                submission_text = markdown_text
                error = str(exc)

        if rubric_data:
            ingestion_status = "success"

    outputs = {
        "ingestion_status": ingestion_status,
        "student_id": student_id,
        "student_name": student_name,
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
        "submission_text": submission_text,
        "rubric_data": rubric_data,
        "agent_logs": existing_logs,
    }
    if error:
        updates["error"] = error

    return updates
