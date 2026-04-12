"""Score calculation tool used by the Analysis Agent."""

from __future__ import annotations

from typing import Any


def calculate_total_score(
    scored_criteria: list[dict[str, Any]],
    total_marks: int,
) -> dict[str, Any]:
    """Sum criterion scores, compute percentage, and assign a letter grade.

    Parameters
    ----------
    scored_criteria:
        List of dicts, each containing at least ``"score"`` (int) and
        ``"max_score"`` (int).
    total_marks:
        Maximum possible marks for the assignment (from the rubric).

    Returns
    -------
    dict
        ``{"total_score": int, "percentage": float, "grade": str}``
    """
    total_score: int = sum(int(c["score"]) for c in scored_criteria)
    percentage: float = (total_score / total_marks * 100) if total_marks > 0 else 0.0

    if percentage >= 90:
        grade = "A"
    elif percentage >= 75:
        grade = "B"
    elif percentage >= 60:
        grade = "C"
    elif percentage >= 50:
        grade = "D"
    else:
        grade = "F"

    return {
        "total_score": total_score,
        "percentage": round(percentage, 2),
        "grade": grade,
    }
