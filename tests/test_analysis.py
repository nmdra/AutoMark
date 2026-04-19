"""Tests for the Analysis Agent."""

from __future__ import annotations

import json
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

import mas.agents.analysis as analysis_module
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
                assignment_mistake="none",
                justification="Good definition provided.",
            ),
            CriterionScore(
                criterion_id="C2",
                score=3,
                assignment_mistake="none",
                justification="Writing is mostly clear.",
            ),
        ]
    )


@pytest.fixture(autouse=True)
def _disable_prompt_cache_for_existing_tests(monkeypatch):
    monkeypatch.setattr(
        analysis_module,
        "settings",
        replace(analysis_module.settings, prompt_cache_enabled=False),
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

        expected = {"scored_criteria", "total_score", "percentage", "grade", "agent_logs"}
        assert set(result.keys()) == expected

    @patch("mas.agents.analysis.get_json_llm")
    def test_assignment_mistake_is_preserved(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = RubricScores(
            scores=[
                CriterionScore(
                    criterion_id="C1",
                    score=0,
                    assignment_mistake="missing_answer",
                    justification="Criterion not addressed.",
                ),
                CriterionScore(
                    criterion_id="C2",
                    score=0,
                    assignment_mistake="out_of_context",
                    justification="Response unrelated to criterion.",
                ),
            ]
        )
        mock_get_llm.return_value = mock_llm

        result = analysis_agent(_make_state())

        c1 = next(c for c in result["scored_criteria"] if c["criterion_id"] == "C1")
        c2 = next(c for c in result["scored_criteria"] if c["criterion_id"] == "C2")
        assert c1["assignment_mistake"] == "missing_answer"
        assert c2["assignment_mistake"] == "out_of_context"

    @patch("mas.agents.analysis.get_json_llm")
    def test_invalid_assignment_mistake_defaults_to_none(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = RubricScores(
            scores=[
                CriterionScore(
                    criterion_id="C1",
                    score=2,
                    assignment_mistake="unknown",
                    justification="Some relevant content.",
                ),
                CriterionScore(
                    criterion_id="C2",
                    score=2,
                    assignment_mistake="none",
                    justification="Some relevant content.",
                ),
            ]
        )
        mock_get_llm.return_value = mock_llm

        result = analysis_agent(_make_state())

        c1 = next(c for c in result["scored_criteria"] if c["criterion_id"] == "C1")
        assert c1["assignment_mistake"] == "none"

    def test_repeated_grading_same_rubric_uses_cached_prefix_and_same_scoring(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            analysis_module,
            "settings",
            replace(analysis_module.settings, prompt_cache_enabled=True),
        )

        class FakePrefixResponse:
            def __init__(self, call_index: int) -> None:
                self.content = json.dumps(
                    {
                        "scores": [
                            {
                                "criterion_id": "C1",
                                "score": 4,
                                "assignment_mistake": "none",
                                "justification": "Good definition provided.",
                            },
                            {
                                "criterion_id": "C2",
                                "score": 3,
                                "assignment_mistake": "none",
                                "justification": "Writing is mostly clear.",
                            },
                        ]
                    }
                )
                self.usage_metadata = {"input_tokens": 20, "output_tokens": 12}
                self.response_metadata = {"model": "phi4-mini"}
                self.model_call_metadata = {
                    "cache_hit": call_index > 0,
                    "cache_status": "hit" if call_index > 0 else "miss",
                    "warmup_ms": 0.0 if call_index > 0 else 15.0,
                    "analysis_latency_ms": 22.0,
                }

        class FakePrefixClient:
            def __init__(self):
                self.calls = 0

            def invoke(self, messages):
                response = FakePrefixResponse(self.calls)
                self.calls += 1
                return response

        fake_client = FakePrefixClient()
        monkeypatch.setattr(
            analysis_module,
            "get_analysis_prefix_cached_json_llm",
            lambda **_: fake_client,
        )

        first = analysis_agent(_make_state())
        second = analysis_agent(_make_state())

        assert first["total_score"] == second["total_score"] == 7
        assert first["scored_criteria"] == second["scored_criteria"]
        assert fake_client.calls == 2
