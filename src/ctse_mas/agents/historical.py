"""Historical Agent – persists results and generates progression insights."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ctse_mas.llm import get_prose_llm
from ctse_mas.state import AgentState
from ctse_mas.tools.db_manager import get_past_reports, save_report
from ctse_mas.tools.logger import log_agent_action

_DEFAULT_DB_PATH = str(
    Path(__file__).parent.parent.parent.parent / "data" / "students.db"
)


def _build_insights_prompt(
    student_id: str,
    past_reports: list[dict[str, Any]],
    current_score: float,
    current_grade: str,
) -> str:
    past_lines = "\n".join(
        f"  - Session {r['session_id']} ({r['timestamp']}): "
        f"{r['total_score']} marks, grade {r['grade']}"
        for r in past_reports
    )
    return (
        f"You are an academic progress advisor reviewing student {student_id!r}.\n\n"
        f"**Current result:** {current_score} marks, grade {current_grade}\n\n"
        f"**Historical results:**\n{past_lines}\n\n"
        "Write 2-3 sentences summarising the student's progression trend and "
        "whether their performance is improving, declining, or stable. "
        "Be concise and constructive."
    )


def historical_agent(state: AgentState) -> dict:
    """Persist the current grading result and generate progression insights.

    Steps:
    1. Save the current scored report to the SQLite database.
    2. Retrieve all previous reports for this student.
    3. If past reports exist, use the LLM to produce ``progression_insights``.

    Returns only the fields it changes.
    """
    session_id: str = state.get("session_id", "")
    student_id: str = state.get("student_id", "unknown")
    scored_criteria: list[dict[str, Any]] = state.get("scored_criteria", [])
    total_score: float = state.get("total_score", 0)
    grade: str = state.get("grade", "F")
    db_path: str = state.get("db_path") or _DEFAULT_DB_PATH
    timestamp: str = datetime.now(tz=timezone.utc).isoformat()

    inputs = {
        "student_id": student_id,
        "total_score": total_score,
        "grade": grade,
        "db_path": db_path,
    }

    progression_insights = ""
    past_reports: list[dict[str, Any]] = []

    try:
        save_report(
            db_path=db_path,
            student_id=student_id,
            session_id=session_id,
            timestamp=timestamp,
            scored_criteria=scored_criteria,
            total_score=total_score,
            grade=grade,
        )
        past_reports = get_past_reports(db_path, student_id)
        # Exclude the record we just saved (last entry) to get truly past reports
        past_reports = past_reports[:-1]
    except Exception as exc:  # noqa: BLE001
        past_reports = []
        progression_insights = ""
        db_error = str(exc)
    else:
        db_error = ""

    if past_reports:
        try:
            llm = get_prose_llm()
            prompt = _build_insights_prompt(
                student_id, past_reports, total_score, grade
            )
            messages = [
                SystemMessage(
                    content=(
                        "You are a helpful academic progress advisor. "
                        "Output concise plain text only – no Markdown formatting."
                    )
                ),
                HumanMessage(content=prompt),
            ]
            response = llm.invoke(messages)
            progression_insights = response.content
        except Exception:  # noqa: BLE001
            progression_insights = ""

    outputs = {
        "past_reports_count": len(past_reports),
        "progression_insights_length": len(progression_insights),
    }

    log_entry = log_agent_action(
        session_id=session_id,
        agent="historical",
        action="save_and_compare",
        inputs=inputs,
        outputs=outputs,
    )

    existing_logs: list = list(state.get("agent_logs") or [])
    existing_logs.append(log_entry)

    updates: dict = {
        "past_reports": past_reports,
        "progression_insights": progression_insights,
        "agent_logs": existing_logs,
    }
    if db_error:
        updates["error"] = db_error

    return updates
