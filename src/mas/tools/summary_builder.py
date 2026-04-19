"""Deterministic summary builder for API responses."""

from __future__ import annotations

from typing import Any


def _format_number(value: Any) -> str:
    """Return a compact deterministic numeric representation."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "0"
    if number.is_integer():
        return str(int(number))
    return f"{number:.2f}".rstrip("0").rstrip(".")


def build_deterministic_summary(
    scored_criteria: list[dict[str, Any]],
    total_score: float,
    total_marks: int,
    grade: str,
) -> str:
    """Build a JSON-safe deterministic plain-text summary from scoring outputs."""
    percentage = (float(total_score) / float(total_marks) * 100.0) if total_marks > 0 else 0.0
    headline = (
        f"Total score: {_format_number(total_score)}/{_format_number(total_marks)} "
        f"({percentage:.2f}%), grade {grade}."
    )

    if not scored_criteria:
        return f"{headline} No criterion scores available."

    breakdown_parts: list[str] = []
    for criterion in scored_criteria:
        name = str(
            criterion.get("name")
            or criterion.get("criterion_id")
            or "Unknown criterion"
        ).strip()
        score = _format_number(criterion.get("score", 0))
        max_score = _format_number(criterion.get("max_score", 0))
        breakdown_parts.append(f"{name}: {score}/{max_score}")

    return f"{headline} Breakdown: {'; '.join(breakdown_parts)}."
