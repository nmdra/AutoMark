"""Global AgentState TypedDict shared across all agents."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────────────────────
    submission_path: str
    rubric_path: str
    session_id: str

    # ── Coordinator ────────────────────────────────────────────────────────
    validation_status: str   # "success" | "failed"
    validation_error: str

    # ── Research ───────────────────────────────────────────────────────────
    submission_text: str
    rubric_data: dict[str, Any]
    research_status: str     # "success" | "failed"

    # ── Analysis ───────────────────────────────────────────────────────────
    scored_criteria: list[dict[str, Any]]
    total_score: float
    grade: str

    # ── Report ─────────────────────────────────────────────────────────────
    final_report: str
    output_filepath: str

    # ── Observability ──────────────────────────────────────────────────────
    agent_logs: list[dict[str, Any]]   # append-only trace entries
    error: str
