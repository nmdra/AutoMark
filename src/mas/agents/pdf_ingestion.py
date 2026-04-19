"""PDF Ingestion Agent – converts a PDF submission to text and extracts student details."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from mas.agents.metadata_extraction import (
    METADATA_SYSTEM_PROMPT,
    StudentDetails,
    _build_extraction_prompt,
    _build_metadata_context,
)
from mas.config import settings
from mas.llm import get_metadata_json_llm
from mas.state import AgentState
from mas.tools.file_ops import read_json_file
from mas.tools.logger import log_agent_action, timed_model_call
from mas.tools.pdf_processor import convert_pdf_to_markdown


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

    # --- Stage 2: Extract metadata via LLM ---
    if stage1_ok:
        try:
            llm = get_metadata_json_llm(schema=StudentDetails)
            metadata_context = _build_metadata_context(markdown_text)
            messages = [
                SystemMessage(content=METADATA_SYSTEM_PROMPT),
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
            student_id = (result.student_number or "").strip()
            student_name = (result.student_name or "").strip()
            assignment_number = (result.assignment_number or "").strip()
        except Exception as exc:  # noqa: BLE001
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
