"""Shared helpers for ingestion metadata extraction."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

METADATA_SYSTEM_PROMPT = (
    "You are a precise student assignment data extractor.\n"
    "Output ONLY a valid JSON object. No explanation. No extra text. No markdown.\n"
    'Always output exactly: {"student_number":"...","student_name":"...","assignment_number":"..."}'
)

_METADATA_CONTEXT_MAX_CHARS = 2500
_IDENTITY_LINE_RE = re.compile(
    r"\b(student\s*)?(id|name|registration|reg\s*no|index|roll\s*no|assignment|hw|homework)\b",
    re.IGNORECASE,
)
_STUDENT_ID_RE = re.compile(
    r"\b(?:student\s*(?:id|number)|registration|reg\s*no|index|roll\s*no)\s*[:#-]?\s*([A-Za-z]{0,4}\d{5,12})\b",
    re.IGNORECASE,
)
_STUDENT_NAME_RE = re.compile(
    r"\bstudent\s*name\s*[:#-]?\s*([^\n\r,;:]{2,80})",
    re.IGNORECASE,
)
_ASSIGNMENT_NUMBER_RE = re.compile(
    r"\b(?:assignment(?:\s*(?:no|number))?|hw|homework)\s*[:#-]?\s*([A-Za-z0-9_-]{1,16})\b",
    re.IGNORECASE,
)


class StudentDetails(BaseModel):
    """Structured extractor output for assignment submissions."""

    student_number: str = Field(
        description="The student's unique number/ID. Empty string if not found."
    )
    student_name: str = Field(
        description="The student's full name. Empty string if not found."
    )
    assignment_number: str = Field(
        description="The assignment number/identifier. Empty string if not found."
    )


def _build_metadata_context(raw_text: str) -> str:
    """Build a compact context for identity-field extraction."""
    normalized = raw_text.strip()
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


def _extract_metadata_regex(raw_text: str) -> tuple[str, str, str]:
    """Extract metadata deterministically from text using lightweight regexes."""
    student_id = ""
    student_name = ""
    assignment_number = ""

    student_id_match = _STUDENT_ID_RE.search(raw_text)
    if student_id_match:
        student_id = (student_id_match.group(1) or "").strip()

    student_name_match = _STUDENT_NAME_RE.search(raw_text)
    if student_name_match:
        student_name = (student_name_match.group(1) or "").strip()

    assignment_match = _ASSIGNMENT_NUMBER_RE.search(raw_text)
    if assignment_match:
        assignment_number = (assignment_match.group(1) or "").strip()

    return student_id, student_name, assignment_number
