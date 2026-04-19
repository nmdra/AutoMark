"""Shared helpers for ingestion metadata extraction."""

from __future__ import annotations

from pydantic import BaseModel, Field

METADATA_SYSTEM_PROMPT = (
    "You are a precise student assignment data extractor.\n"
    "Output ONLY a valid JSON object. No explanation. No extra text. No markdown.\n"
    'Always output exactly: {"student_number":"...","student_name":"...","assignment_number":"..."}'
)

_METADATA_CONTEXT_MAX_CHARS = 2500
_IDENTITY_HINTS = (
    "student",
    "id",
    "name",
    "registration",
    "reg no",
    "index",
    "roll no",
    "assignment",
    "hw",
    "homework",
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
    candidate_lines: list[str] = []
    seen: set[str] = set()
    for line in normalized.splitlines():
        clean_line = line.strip()
        if not clean_line:
            continue
        normalized_line = clean_line.lower()
        if not any(hint in normalized_line for hint in _IDENTITY_HINTS):
            continue
        if clean_line in seen:
            continue
        seen.add(clean_line)
        candidate_lines.append(clean_line)
        if len(candidate_lines) >= 20:
            break

    if not candidate_lines:
        return top_chunk

    return (
        f"{top_chunk}\n\n"
        "## Candidate Identity Lines\n"
        "\n".join(candidate_lines)
    )


def _build_extraction_prompt(metadata_context: str) -> str:
    return (
        "### Instruction:\n"
        "Extract student info as JSON from the following text.\n\n"
        "### Input:\n"
        f"{metadata_context}\n\n"
        "### Response:\n"
    )
