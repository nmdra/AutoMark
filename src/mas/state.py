"""Global AgentState TypedDict shared across all agents."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────────────────────
    submission_path: str
    rubric_path: str
    session_id: str
    db_path: str

    # ── Ingestion ──────────────────────────────────────────────────────────
    student_id: str
    ingestion_status: str    # "success" | "failed"
    submission_text: str
    rubric_data: dict[str, Any]

    # ── Analysis ───────────────────────────────────────────────────────────
    scored_criteria: list[dict[str, Any]]
    total_score: float
    grade: str

    # ── Historical ─────────────────────────────────────────────────────────
    past_reports: list[dict[str, Any]]   # previous scored reports from DB
    progression_insights: str            # LLM-generated trend text

    # ── Report ─────────────────────────────────────────────────────────────
    final_report: str
    output_filepath: str

    # ── Observability ──────────────────────────────────────────────────────
    agent_logs: list[dict[str, Any]]   # append-only trace entries
    error: str
