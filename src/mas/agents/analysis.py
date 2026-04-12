"""Analysis Agent – scores each rubric criterion using the LLM."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from mas.llm import get_json_llm
from mas.state import AgentState
from mas.tools.logger import log_agent_action
from mas.tools.score_calculator import calculate_total_score


class CriterionScore(BaseModel):
    """Structured LLM output for a single rubric criterion."""

    criterion_id: str = Field(description="The criterion ID from the rubric.")
    score: int = Field(description="Integer score between 0 and max_score.")
    justification: str = Field(
        description="Single sentence justification, maximum 20 words."
    )


class RubricScores(BaseModel):
    """Structured LLM output for all rubric criteria."""

    scores: list[CriterionScore]


def _build_system_prompt() -> str:
    return (
        "You are a strict academic grader. "
        "For each criterion provided, assign an integer score between 0 and its "
        "max_score and write a single-sentence justification of at most 20 words. "
        "Respond ONLY with valid JSON matching the required schema. "
        "Do NOT calculate totals."
    )


def _build_user_prompt(submission_text: str, rubric_data: dict[str, Any]) -> str:
    criteria_summary = json.dumps(rubric_data.get("criteria", []), indent=2)
    return (
        f"## Student Submission\n\n{submission_text}\n\n"
        f"## Rubric Criteria\n\n{criteria_summary}\n\n"
        "Score each criterion and provide a one-sentence justification."
    )


def analysis_agent(state: AgentState) -> dict:
    """Score each rubric criterion and compute the total via the score calculator.

    The LLM is NEVER used to compute totals – that is handled deterministically
    by ``calculate_total_score``.  Returns only the fields it changes.
    """
    session_id: str = state.get("session_id", "")
    submission_text: str = state.get("submission_text", "")
    rubric_data: dict[str, Any] = state.get("rubric_data", {})
    total_marks: int = int(rubric_data.get("total_marks", 0))
    criteria: list[dict] = rubric_data.get("criteria", [])

    inputs = {
        "criteria_count": len(criteria),
        "submission_length": len(submission_text),
    }

    llm = get_json_llm(schema=RubricScores)

    messages = [
        SystemMessage(content=_build_system_prompt()),
        HumanMessage(content=_build_user_prompt(submission_text, rubric_data)),
    ]

    try:
        result: RubricScores = llm.invoke(messages)
        llm_scores: list[CriterionScore] = result.scores
    except Exception as exc:
        # Fallback: assign 0 for all criteria
        llm_scores = [
            CriterionScore(
                criterion_id=c["id"],
                score=0,
                justification="LLM evaluation failed; score defaulted to 0.",
            )
            for c in criteria
        ]
        state_error = str(exc)
    else:
        state_error = ""

    # Build scored_criteria by merging LLM scores with rubric max_score values
    max_by_id: dict[str, int] = {c["id"]: c["max_score"] for c in criteria}
    name_by_id: dict[str, str] = {c["id"]: c["name"] for c in criteria}

    scored_criteria: list[dict[str, Any]] = []
    for cs in llm_scores:
        max_score = max_by_id.get(cs.criterion_id, 0)
        clamped_score = max(0, min(int(cs.score), max_score))
        scored_criteria.append(
            {
                "criterion_id": cs.criterion_id,
                "name": name_by_id.get(cs.criterion_id, cs.criterion_id),
                "score": clamped_score,
                "max_score": max_score,
                "justification": cs.justification,
            }
        )

    # Deterministic totalling – LLM is NOT used here
    totals = calculate_total_score(scored_criteria, total_marks)

    outputs = {
        "total_score": totals["total_score"],
        "percentage": totals["percentage"],
        "grade": totals["grade"],
    }

    log_entry = log_agent_action(
        session_id=session_id,
        agent="analysis",
        action="score_criteria",
        inputs=inputs,
        outputs=outputs,
    )

    existing_logs: list = list(state.get("agent_logs") or [])
    existing_logs.append(log_entry)

    updates: dict = {
        "scored_criteria": scored_criteria,
        "total_score": totals["total_score"],
        "percentage": totals["percentage"],
        "grade": totals["grade"],
        "agent_logs": existing_logs,
    }
    if state_error:
        updates["error"] = state_error

    return updates
