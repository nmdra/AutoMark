"""Tests for the Analysis Agent."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from mas.agents.analysis import CriterionScore, RubricScores, analysis_agent
from mas.state import AgentState

SAMPLE_RUBRIC = {
    "module": "CTSE",
    "total_marks": 10,
    "criteria": [
        {"id": "C1", "name": "Accuracy", "description": "Correct info.", "max_score": 5},
        {"id": "C2", "name": "Clarity", "description": "Clear writing.", "max_score": 5},
    ],
}

SAMPLE_SUBMISSION = "Containerisation is an OS-level virtualisation technique."


def _make_state(**kwargs) -> AgentState:
    base: AgentState = {
        "session_id": "test-session",
        "submission_text": SAMPLE_SUBMISSION,
        "rubric_data": SAMPLE_RUBRIC,
        "agent_logs": [],
    }
    base.update(kwargs)
    return base


def _mock_llm_output() -> RubricScores:
    return RubricScores(
        scores=[
            CriterionScore(
                criterion_id="C1",
                score=4,
                justification="Good definition provided.",
            ),
            CriterionScore(
                criterion_id="C2",
                score=3,
                justification="Writing is mostly clear.",
            ),
        ]
    )


class TestAnalysisAgent:
    @patch("mas.agents.analysis.get_json_llm")
    def test_scores_returned_correctly(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_llm_output()
        mock_get_llm.return_value = mock_llm

        result = analysis_agent(_make_state())

        assert result["total_score"] == 7
        assert result["grade"] == "C"  # 70%
        assert len(result["scored_criteria"]) == 2

    @patch("mas.agents.analysis.get_json_llm")
    def test_score_clamped_to_max(self, mock_get_llm):
        """Scores above max_score must be clamped."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = RubricScores(
            scores=[
                CriterionScore(criterion_id="C1", score=99, justification="Over max."),
                CriterionScore(criterion_id="C2", score=5, justification="Full marks."),
            ]
        )
        mock_get_llm.return_value = mock_llm

        result = analysis_agent(_make_state())

        c1 = next(c for c in result["scored_criteria"] if c["criterion_id"] == "C1")
        assert c1["score"] == 5  # clamped to max_score

    @patch("mas.agents.analysis.get_json_llm")
    def test_score_clamped_to_zero(self, mock_get_llm):
        """Negative scores must be clamped to 0."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = RubricScores(
            scores=[
                CriterionScore(criterion_id="C1", score=-3, justification="Negative."),
                CriterionScore(criterion_id="C2", score=2, justification="Some marks."),
            ]
        )
        mock_get_llm.return_value = mock_llm

        result = analysis_agent(_make_state())

        c1 = next(c for c in result["scored_criteria"] if c["criterion_id"] == "C1")
        assert c1["score"] == 0

    @patch("mas.agents.analysis.get_json_llm")
    def test_llm_failure_defaults_to_zero(self, mock_get_llm):
        """When the LLM raises an exception, all scores fall back to 0."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("Ollama not reachable")
        mock_get_llm.return_value = mock_llm

        result = analysis_agent(_make_state())

        assert result["total_score"] == 0
        assert result["grade"] == "F"
        assert "error" in result

    @patch("mas.agents.analysis.get_json_llm")
    def test_total_score_is_not_calculated_by_llm(self, mock_get_llm):
        """Total score must be deterministic – not from the LLM response."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_llm_output()
        mock_get_llm.return_value = mock_llm

        result = analysis_agent(_make_state())

        # 4 + 3 = 7  (deterministic, not provided by mock)
        assert result["total_score"] == 7

    @patch("mas.agents.analysis.get_json_llm")
    def test_log_entry_appended(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_llm_output()
        mock_get_llm.return_value = mock_llm

        result = analysis_agent(_make_state())

        assert len(result["agent_logs"]) == 1
        assert result["agent_logs"][0]["agent"] == "analysis"

    @patch("mas.agents.analysis.get_json_llm")
    def test_grade_a(self, mock_get_llm):
        """Score ≥ 90% should yield grade A."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = RubricScores(
            scores=[
                CriterionScore(criterion_id="C1", score=5, justification="Excellent."),
                CriterionScore(criterion_id="C2", score=5, justification="Perfect."),
            ]
        )
        mock_get_llm.return_value = mock_llm

        result = analysis_agent(_make_state())

        assert result["grade"] == "A"
        assert result["total_score"] == 10

    @patch("mas.agents.analysis.get_json_llm")
    def test_returns_only_changed_fields(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_llm_output()
        mock_get_llm.return_value = mock_llm

        result = analysis_agent(_make_state())

        expected = {"scored_criteria", "total_score", "grade", "agent_logs"}
        assert set(result.keys()) == expected
