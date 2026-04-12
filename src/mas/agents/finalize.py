"""Finalize Agent – combines historical persistence and report generation.

This agent replaces the sequential ``historical → report`` chain in the
LangGraph pipeline.  It uses a ``ThreadPoolExecutor`` to submit the two
independent LLM calls — historical-progression insights and feedback-report
prose — to Ollama *concurrently*, taking advantage of Ollama's parallel
request processing (``OLLAMA_NUM_PARALLEL >= 2`` on the server side).

Pipeline savings (typical):
    • Sequential: insights_llm_time + report_llm_time  (e.g. 8s + 20s = 28s)
    • Parallel  : max(insights_llm_time, report_llm_time)  (e.g. max(8s, 20s) = 20s)

The progression insights are *appended* to the final report as a dedicated
section rather than being woven into the LLM prompt (which would require the
report call to wait for insights), preserving full parallelism.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from mas.config import settings
from mas.llm import get_light_prose_llm, get_prose_llm
from mas.state import AgentState
from mas.tools.db_manager import get_past_reports, save_report
from mas.tools.file_writer import (
    write_analysis_report,
    write_feedback_report,
    write_marking_sheet,
)
from mas.tools.logger import log_agent_action, timed_model_call

# Import prompt builders and helpers from the original agent modules so that
# those modules remain independently testable.
from mas.agents.historical import _build_insights_prompt  # type: ignore[attr-defined]
from mas.agents.report import _build_report_prompt, _build_fallback_report  # type: ignore[attr-defined]

_DEFAULT_OUTPUT = "output/feedback_report.md"


# ── LLM sub-tasks (run in worker threads) ─────────────────────────────────────


def _task_generate_insights(
    student_id: str,
    past_reports: list[dict[str, Any]],
    total_score: float,
    grade: str,
    session_id: str = "",
) -> str:
    """Call the LLM to produce progression-trend insights.

    Returns an empty string on any failure so the rest of the pipeline is
    unaffected.
    """
    try:
        llm = get_light_prose_llm()
        prompt = _build_insights_prompt(student_id, past_reports, total_score, grade)
        messages = [
            SystemMessage(
                content=(
                    "You are a helpful academic progress advisor. "
                    "Output concise plain text only – no Markdown formatting."
                )
            ),
            HumanMessage(content=prompt),
        ]
        response = timed_model_call(
            llm=llm,
            messages=messages,
            session_id=session_id,
            service="finalize",
            task_type="progression_insights",
            model=settings.light_model_name,
        )
        return response.content
    except Exception:  # noqa: BLE001
        return ""


def _task_generate_report_prose(state: AgentState, session_id: str = "") -> str:
    """Call the LLM (or use the template fallback) to produce the feedback report.

    Note: ``progression_insights`` is intentionally absent from the prompt here
    because insights are being generated in parallel.  They are appended to the
    report text after both futures resolve.
    """
    if not settings.llm_report_enabled:
        return _build_fallback_report(state)
    try:
        llm = get_prose_llm()
        prompt = _build_report_prompt(state)
        messages = [
            SystemMessage(
                content=(
                    "You are a helpful academic feedback writer. "
                    "Output clean, well-structured Markdown only."
                )
            ),
            HumanMessage(content=prompt),
        ]
        response = timed_model_call(
            llm=llm,
            messages=messages,
            session_id=session_id,
            service="finalize",
            task_type="feedback_report_generation",
            model=settings.analysis_model_name,
        )
        return response.content
    except Exception:  # noqa: BLE001
        return _build_fallback_report(state)


# ── Agent ─────────────────────────────────────────────────────────────────────


def finalize_agent(state: AgentState) -> dict:
    """Persist results, then run insights and report LLMs concurrently.

    Steps
    -----
    1. Save the current grading result to SQLite and retrieve past reports.
    2. Submit two LLM tasks to a ``ThreadPoolExecutor``:
       - Task A: generate progression insights (if past reports meet threshold).
       - Task B: generate feedback-report prose (or template when disabled).
    3. Wait for both tasks and append insights as a report section.
    4. Write all output files (analysis report, feedback report, marking sheet).

    Returns only the state fields it sets.
    """
    session_id: str = state.get("session_id", "")
    student_id: str = state.get("student_id", "unknown")
    student_name: str = state.get("student_name", "")
    scored_criteria: list[dict[str, Any]] = state.get("scored_criteria", [])
    total_score: float = state.get("total_score", 0)
    grade: str = state.get("grade", "F")
    db_path: str = state.get("db_path") or settings.db_path
    timestamp: str = datetime.now(tz=timezone.utc).isoformat()

    rubric_data: dict[str, Any] = state.get("rubric_data", {})
    total_marks: int = int(rubric_data.get("total_marks", 0))

    output_path: str = state.get("output_filepath") or _DEFAULT_OUTPUT
    marking_sheet_output: str = (
        state.get("marking_sheet_path") or settings.marking_sheet_path
    )
    analysis_report_path: str = (
        state.get("analysis_report_path") or settings.analysis_report_path
    )

    inputs = {
        "student_id": student_id,
        "total_score": total_score,
        "grade": grade,
        "db_path": db_path,
        "output_path": output_path,
    }

    # ── Step 1: Synchronous DB operations ─────────────────────────────────────
    db_error = ""
    all_reports: list[dict[str, Any]] = []
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
        all_reports = get_past_reports(db_path, student_id)
        # Exclude the record just saved (last entry) to keep truly past reports.
        past_reports = all_reports[:-1]
    except Exception as exc:  # noqa: BLE001
        db_error = str(exc)

    # Inject past_reports into the state snapshot used by the report prompt so
    # _build_report_prompt can include historical context if it chooses.
    state_with_context: AgentState = {**state, "past_reports": past_reports}  # type: ignore[misc]

    # ── Step 2: Parallel LLM calls ────────────────────────────────────────────
    progression_insights = ""
    report_text = ""

    run_insights = (
        len(past_reports) >= settings.min_reports_for_insights and not db_error
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        insights_future: Future[str] | None = None
        if run_insights:
            insights_future = executor.submit(
                _task_generate_insights,
                student_id,
                past_reports,
                total_score,
                grade,
                session_id,
            )

        report_future: Future[str] = executor.submit(
            _task_generate_report_prose,
            state_with_context,
            session_id,
        )

        # Collect results (blocking until both threads complete)
        report_text = report_future.result()
        if insights_future is not None:
            progression_insights = insights_future.result()

    # ── Step 3: Append insights section to report ─────────────────────────────
    if progression_insights:
        report_text = (
            report_text.rstrip()
            + "\n\n## Progression Insights\n\n"
            + progression_insights
            + "\n"
        )

    # ── Step 4: Write all output files ────────────────────────────────────────
    resolved_output = write_feedback_report(report_text, output_path)

    resolved_marking = ""
    try:
        resolved_marking = write_marking_sheet(
            student_id=student_id,
            student_name=student_name,
            module=rubric_data.get("module", "Unknown Module"),
            assignment=rubric_data.get("assignment", "Unknown Assignment"),
            scored_criteria=scored_criteria,
            total_score=total_score,
            total_marks=total_marks,
            grade=grade,
            output_path=marking_sheet_output,
        )
    except Exception:  # noqa: BLE001
        resolved_marking = ""

    resolved_analysis = ""
    try:
        resolved_analysis = write_analysis_report(
            past_reports=all_reports,
            progression_insights=progression_insights,
            student_id=student_id,
            output_path=analysis_report_path,
        )
    except Exception:  # noqa: BLE001
        resolved_analysis = ""

    # ── Summary extraction ────────────────────────────────────────────────────
    summary = ""
    paragraphs = [p.strip() for p in report_text.split("\n\n") if p.strip()]
    for paragraph in paragraphs:
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if not lines:
            continue
        if lines[0].startswith("#"):
            continue
        if all(line.startswith("**") and ":**" in line for line in lines):
            continue
        if all(set(line) <= {"-", "*", "_"} and len(line) >= 3 for line in lines):
            continue
        summary = paragraph
        break
    if not summary and paragraphs:
        summary = paragraphs[0]

    # ── Observability ──────────────────────────────────────────────────────────
    outputs = {
        "past_reports_count": len(past_reports),
        "progression_insights_length": len(progression_insights),
        "output_filepath": resolved_output,
        "marking_sheet_path": resolved_marking,
        "analysis_report_path": resolved_analysis,
    }

    log_entry = log_agent_action(
        session_id=session_id,
        agent="finalize",
        action="persist_and_generate_reports",
        inputs=inputs,
        outputs=outputs,
    )

    existing_logs: list = list(state.get("agent_logs") or [])
    existing_logs.append(log_entry)

    updates: dict = {
        "past_reports": past_reports,
        "progression_insights": progression_insights,
        "analysis_report_path": resolved_analysis,
        "final_report": report_text,
        "summary": summary,
        "output_filepath": resolved_output,
        "marking_sheet_path": resolved_marking,
        "agent_logs": existing_logs,
    }
    if db_error:
        updates["error"] = db_error

    return updates
