"""Report Agent – generates a structured feedback report and writes it to disk."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from mas.config import settings
from mas.llm import get_prose_llm
from mas.state import AgentState
from mas.tools.file_writer import write_feedback_report, write_marking_sheet
from mas.tools.logger import log_agent_action

_DEFAULT_OUTPUT = "output/feedback_report.md"


def _build_report_prompt(state: AgentState) -> str:
    rubric_data: dict[str, Any] = state.get("rubric_data", {})
    scored_criteria: list[dict] = state.get("scored_criteria", [])
    total_score = state.get("total_score", 0)
    grade = state.get("grade", "N/A")
    total_marks = rubric_data.get("total_marks", 0)
    module = rubric_data.get("module", "Unknown Module")
    assignment = rubric_data.get("assignment", "Unknown Assignment")
    progression_insights: str = state.get("progression_insights", "")

    criteria_lines = "\n".join(
        f"- **{c['name']}** ({c['criterion_id']}): {c['score']}/{c['max_score']} – {c['justification']}"
        for c in scored_criteria
    )

    progression_section = (
        f"\n**Progression Insights:**\n{progression_insights}\n"
        if progression_insights
        else ""
    )

    return (
        f"Write a structured Markdown feedback report for the following assessment.\n\n"
        f"**Module:** {module}\n"
        f"**Assignment:** {assignment}\n"
        f"**Total Score:** {total_score}/{total_marks}\n"
        f"**Grade:** {grade}\n\n"
        f"**Criterion Scores:**\n{criteria_lines}\n"
        f"{progression_section}\n"
        "The report must include:\n"
        "1. A brief overall summary (2-3 sentences).\n"
        "2. Per-criterion feedback referencing the score and justification.\n"
        "3. Constructive suggestions for improvement.\n"
        + ("4. A progression section summarising the student's historical trend.\n" if progression_insights else "")
        + "Format the output as clean Markdown."
    )


def _build_fallback_report(state: AgentState) -> str:
    """Produce a plain Markdown report without LLM if the model is unavailable."""
    rubric_data: dict[str, Any] = state.get("rubric_data", {})
    scored_criteria: list[dict] = state.get("scored_criteria", [])
    total_score = state.get("total_score", 0)
    grade = state.get("grade", "N/A")
    total_marks = rubric_data.get("total_marks", 0)
    module = rubric_data.get("module", "Unknown Module")
    assignment = rubric_data.get("assignment", "Unknown Assignment")

    lines = [
        f"# Feedback Report",
        f"",
        f"**Module:** {module}  ",
        f"**Assignment:** {assignment}  ",
        f"**Total Score:** {total_score}/{total_marks}  ",
        f"**Grade:** {grade}",
        f"",
        f"## Criterion Scores",
        f"",
    ]
    for c in scored_criteria:
        lines.append(
            f"### {c['name']} ({c['criterion_id']})"
        )
        lines.append(f"**Score:** {c['score']}/{c['max_score']}")
        lines.append(f"**Justification:** {c['justification']}")
        lines.append("")

    if not scored_criteria:
        lines.append("_No criteria were scored._")

    return "\n".join(lines)


def report_agent(state: AgentState) -> dict:
    """Generate a Markdown feedback report and write it to disk.

    Also produces a separate marking sheet report.  Uses the prose LLM to
    produce a well-formatted feedback report.  Falls back to a template-based
    report when the LLM is unavailable.  Returns only the fields it changes.
    """
    session_id: str = state.get("session_id", "")
    output_path: str = state.get("output_filepath") or _DEFAULT_OUTPUT
    marking_sheet_output: str = (
        state.get("marking_sheet_path") or settings.marking_sheet_path
    )

    rubric_data: dict[str, Any] = state.get("rubric_data", {})
    total_marks: int = int(rubric_data.get("total_marks", 0))

    inputs = {
        "total_score": state.get("total_score"),
        "grade": state.get("grade"),
        "output_path": output_path,
    }

    # Attempt LLM-generated prose report
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
        response = llm.invoke(messages)
        report_text: str = response.content
    except Exception:
        report_text = _build_fallback_report(state)

    # Write feedback report to disk
    resolved_path = write_feedback_report(report_text, output_path)

    # Write separate marking sheet
    try:
        resolved_marking_path = write_marking_sheet(
            student_id=state.get("student_id", ""),
            student_name=state.get("student_name", ""),
            module=rubric_data.get("module", "Unknown Module"),
            assignment=rubric_data.get("assignment", "Unknown Assignment"),
            scored_criteria=state.get("scored_criteria", []),
            total_score=state.get("total_score", 0),
            total_marks=total_marks,
            grade=state.get("grade", "N/A"),
            output_path=marking_sheet_output,
        )
    except Exception:  # noqa: BLE001
        resolved_marking_path = ""

    outputs = {
        "output_filepath": resolved_path,
        "marking_sheet_path": resolved_marking_path,
        "report_length": len(report_text),
    }

    log_entry = log_agent_action(
        session_id=session_id,
        agent="report",
        action="write_feedback_report_and_marking_sheet",
        inputs=inputs,
        outputs=outputs,
    )

    existing_logs: list = list(state.get("agent_logs") or [])
    existing_logs.append(log_entry)

    return {
        "final_report": report_text,
        "output_filepath": resolved_path,
        "marking_sheet_path": resolved_marking_path,
        "agent_logs": existing_logs,
    }
