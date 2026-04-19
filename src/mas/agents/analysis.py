"""Analysis Agent – scores each rubric criterion using the LLM."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from mas.llm import get_analysis_prefix_cached_json_llm, get_json_llm
from mas.state import AgentState
from mas.tools.logger import log_agent_action, timed_model_call
from mas.tools.prompt_cache import hash_rubric_payload
from mas.tools.score_calculator import calculate_total_score
from mas.config import settings


class CriterionScore(BaseModel):
    """Structured LLM output for a single rubric criterion."""

    criterion_id: str = Field(description="The criterion ID from the rubric.")
    score: int = Field(description="Integer score between 0 and max_score.")
    assignment_mistake: str = Field(
        default="none",
        description=(
            "Common mistake classification for this criterion: "
            "'none', 'missing_answer', or 'out_of_context'."
        ),
    )
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
        "max_score and classify assignment_mistake as one of: "
        "'none', 'missing_answer', 'out_of_context'. "
        "Use 'missing_answer' when the criterion is not addressed. "
        "Use 'out_of_context' when the response is unrelated to the criterion. "
        "Write a single-sentence justification of at most 20 words. "
        "Respond ONLY with valid JSON matching the required schema. "
        "Do NOT calculate totals."
    )


def _compact_criteria(rubric_data: dict[str, Any]) -> list[dict[str, Any]]:
    max_descriptor_chars = settings.rubric_descriptor_max_chars
    compact: list[dict[str, Any]] = []
    for criterion in rubric_data.get("criteria", []):
        descriptor = str(criterion.get("description") or "").strip()
        if len(descriptor) > max_descriptor_chars:
            descriptor = descriptor[:max_descriptor_chars].rstrip() + "…"
        compact.append(
            {
                "id": str(criterion.get("id") or ""),
                "name": str(criterion.get("name") or ""),
                "max_score": int(criterion.get("max_score") or 0),
                "descriptor": descriptor,
            }
        )
    return compact


def _build_prefix_context(rubric_data: dict[str, Any]) -> str:
    compact_payload = {
        "total_marks": int(rubric_data.get("total_marks") or 0),
        "criteria": _compact_criteria(rubric_data),
    }
    return (
        "## Rubric Criteria\n\n"
        f"{json.dumps(compact_payload['criteria'], indent=2)}\n\n"
        "Use these criteria exactly as provided."
    )


def _build_submission_prompt(submission_text: str) -> str:
    # Truncate to avoid bloating the context window with very long submissions.
    max_chars = settings.submission_max_chars
    if len(submission_text) > max_chars:
        submission_text = submission_text[:max_chars] + "\n\n[...truncated...]"
    return (
        f"## Student Submission\n\n{submission_text}\n\n"
        "Score each criterion and provide a one-sentence justification."
    )


def _parse_scoring_response(raw_response: Any) -> RubricScores:
    if isinstance(raw_response, RubricScores):
        return raw_response
    if isinstance(raw_response, BaseModel):
        return RubricScores.model_validate(raw_response.model_dump())
    if isinstance(raw_response, dict):
        return RubricScores.model_validate(raw_response)
    content = str(getattr(raw_response, "content", "") or "")
    return RubricScores.model_validate_json(content)


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

    system_prompt = _build_system_prompt()
    prefix_context = _build_prefix_context(rubric_data)
    submission_prompt = _build_submission_prompt(submission_text)
    compact_rubric_payload = {
        "total_marks": int(rubric_data.get("total_marks") or 0),
        "criteria": _compact_criteria(rubric_data),
    }
    rubric_hash = hash_rubric_payload(compact_rubric_payload)

    try:
        if settings.prompt_cache_enabled:
            llm = get_analysis_prefix_cached_json_llm(
                system_prompt=system_prompt,
                system_prompt_version=settings.analysis_system_prompt_version,
                rubric_hash=rubric_hash,
                prefix_context=prefix_context,
                submission_prompt=submission_prompt,
            )
            messages: list[Any] = []
        else:
            llm = get_json_llm(schema=RubricScores)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prefix_context + "\n\n" + submission_prompt),
            ]

        result = timed_model_call(
            llm=llm,
            messages=messages,
            session_id=session_id,
            service="analysis",
            task_type="rubric_scoring",
            model=settings.analysis_model_name,
        )
        llm_scores = _parse_scoring_response(result).scores
    except Exception as exc:
        # Fallback: assign 0 for all criteria
        if settings.prompt_cache_enabled:
            try:
                fallback_llm = get_json_llm(schema=RubricScores)
                fallback_messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=prefix_context + "\n\n" + submission_prompt),
                ]
                fallback_result = timed_model_call(
                    llm=fallback_llm,
                    messages=fallback_messages,
                    session_id=session_id,
                    service="analysis",
                    task_type="rubric_scoring_fallback",
                    model=settings.analysis_model_name,
                )
                llm_scores = _parse_scoring_response(fallback_result).scores
                state_error = ""
            except Exception as fallback_exc:
                llm_scores = [
                    CriterionScore(
                        criterion_id=c["id"],
                        score=0,
                        justification="LLM evaluation failed; score defaulted to 0.",
                    )
                    for c in criteria
                ]
                state_error = f"{exc} | fallback failed: {fallback_exc}"
        else:
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
    allowed_mistakes = {"none", "missing_answer", "out_of_context"}
    for cs in llm_scores:
        max_score = max_by_id.get(cs.criterion_id, 0)
        clamped_score = max(0, min(int(cs.score), max_score))
        mistake = str(cs.assignment_mistake or "none").strip().lower()
        if mistake not in allowed_mistakes:
            mistake = "none"
        scored_criteria.append(
            {
                "criterion_id": cs.criterion_id,
                "name": name_by_id.get(cs.criterion_id, cs.criterion_id),
                "score": clamped_score,
                "max_score": max_score,
                "assignment_mistake": mistake,
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
