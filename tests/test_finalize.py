"""Tests for the Finalize Agent (combined historical persistence + parallel LLM reports)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mas.agents.finalize import finalize_agent, _task_generate_insights, _task_generate_report_prose
from mas.state import AgentState
from mas.tools.db_manager import get_past_reports, init_db, save_report


# ── Fixtures & helpers ────────────────────────────────────────────────────────

SCORED_CRITERIA = [
    {"criterion_id": "C1", "name": "Accuracy", "score": 4, "max_score": 5,
     "justification": "Good definition."},
    {"criterion_id": "C2", "name": "Clarity", "score": 3, "max_score": 5,
     "justification": "Mostly clear."},
]

SAMPLE_RUBRIC = {
    "module": "CTSE",
    "assignment": "Cloud Assignment",
    "total_marks": 10,
    "criteria": [
        {"id": "C1", "name": "Accuracy", "max_score": 5},
        {"id": "C2", "name": "Clarity", "max_score": 5},
    ],
}


def _make_state(db_path: str, output_path: str = "", **kwargs) -> AgentState:
    base: AgentState = {
        "session_id": "test-session",
        "student_id": "IT21000001",
        "student_name": "Alice",
        "scored_criteria": SCORED_CRITERIA,
        "total_score": 7,
        "grade": "C",
        "rubric_data": SAMPLE_RUBRIC,
        "db_path": db_path,
        "output_filepath": output_path,
        "agent_logs": [],
    }
    base.update(kwargs)
    return base


# ── finalize_agent core behaviour ─────────────────────────────────────────────

class TestFinalizeAgent:
    def test_saves_report_to_db(self, tmp_path):
        state = _make_state(str(tmp_path / "students.db"),
                            output_path=str(tmp_path / "report.md"))
        finalize_agent(state)

        reports = get_past_reports(str(tmp_path / "students.db"), "IT21000001")
        assert len(reports) == 1
        assert reports[0]["total_score"] == 7

    def test_feedback_report_written_to_disk(self, tmp_path):
        out = str(tmp_path / "report.md")
        state = _make_state(str(tmp_path / "students.db"), output_path=out)

        finalize_agent(state)

        assert Path(out).exists()

    def test_marking_sheet_written_to_disk(self, tmp_path):
        out = str(tmp_path / "report.md")
        sheet = str(tmp_path / "sheet.md")
        state = _make_state(
            str(tmp_path / "students.db"),
            output_path=out,
            marking_sheet_path=sheet,
        )

        finalize_agent(state)

        assert Path(sheet).exists()

    def test_no_progression_insights_for_first_submission(self, tmp_path):
        state = _make_state(str(tmp_path / "students.db"),
                            output_path=str(tmp_path / "report.md"))
        result = finalize_agent(state)

        assert result["past_reports"] == []
        assert result["progression_insights"] == ""

    def test_returns_all_expected_fields(self, tmp_path):
        state = _make_state(str(tmp_path / "students.db"),
                            output_path=str(tmp_path / "report.md"))
        result = finalize_agent(state)

        expected = {
            "past_reports", "progression_insights", "analysis_report_path",
            "final_report", "summary", "output_filepath", "marking_sheet_path",
            "agent_logs",
        }
        assert expected.issubset(set(result.keys()))

    def test_log_entry_appended(self, tmp_path):
        state = _make_state(str(tmp_path / "students.db"),
                            output_path=str(tmp_path / "report.md"))
        result = finalize_agent(state)

        assert len(result["agent_logs"]) == 1
        assert result["agent_logs"][0]["agent"] == "finalize"
        assert result["agent_logs"][0]["action"] == "persist_and_generate_reports"

    def test_existing_logs_preserved(self, tmp_path):
        prior = {"agent": "analysis", "action": "score_criteria"}
        state = _make_state(str(tmp_path / "students.db"),
                            output_path=str(tmp_path / "report.md"),
                            agent_logs=[prior])
        result = finalize_agent(state)

        assert len(result["agent_logs"]) == 2
        assert result["agent_logs"][0] == prior

    def test_past_reports_populated_for_returning_student(self, tmp_path):
        db = str(tmp_path / "students.db")
        save_report(db, "IT21000001", "old-session", "2025-01-01T00:00:00+00:00",
                    SCORED_CRITERIA, 5, "D")

        state = _make_state(db, output_path=str(tmp_path / "report.md"))
        result = finalize_agent(state)

        assert len(result["past_reports"]) == 1
        assert result["past_reports"][0]["grade"] == "D"

    @patch("mas.agents.finalize.get_prose_llm")
    @patch("mas.agents.finalize.get_light_prose_llm")
    def test_insights_generated_for_returning_student(
        self, mock_get_light_llm, mock_get_prose_llm, tmp_path
    ):
        db = str(tmp_path / "students.db")
        save_report(db, "IT21000001", "old-session", "2025-01-01T00:00:00+00:00",
                    SCORED_CRITERIA, 5, "D")

        # Light LLM handles insights
        mock_light_llm = MagicMock()
        mock_light_llm.invoke.return_value = MagicMock(content="Student is improving.")
        mock_get_light_llm.return_value = mock_light_llm

        # Analysis LLM handles report prose
        mock_prose_llm = MagicMock()
        mock_prose_llm.invoke.return_value = MagicMock(content="# Report\n\nGood work.")
        mock_get_prose_llm.return_value = mock_prose_llm

        state = _make_state(db, output_path=str(tmp_path / "report.md"))
        result = finalize_agent(state)

        # Insights should appear in both the dedicated field and the final report.
        assert "improving" in result["progression_insights"]
        assert "## Progression Insights" in result["final_report"]

    @patch("mas.agents.finalize.get_prose_llm")
    @patch("mas.agents.finalize.get_light_prose_llm")
    def test_insights_llm_failure_gives_empty_insights(
        self, mock_get_light_llm, mock_get_prose_llm, tmp_path
    ):
        db = str(tmp_path / "students.db")
        save_report(db, "IT21000001", "old-session", "2025-01-01T00:00:00+00:00",
                    SCORED_CRITERIA, 5, "D")

        # Light LLM fails
        mock_light_llm = MagicMock()
        mock_light_llm.invoke.side_effect = RuntimeError("LLM unavailable")
        mock_get_light_llm.return_value = mock_light_llm

        # Analysis LLM works fine (report prose still generated)
        mock_prose_llm = MagicMock()
        mock_prose_llm.invoke.return_value = MagicMock(content="# Report\n\nGood work.")
        mock_get_prose_llm.return_value = mock_prose_llm

        state = _make_state(db, output_path=str(tmp_path / "report.md"))
        result = finalize_agent(state)

        assert result["progression_insights"] == ""

    @patch("mas.agents.finalize.settings")
    def test_report_uses_template_when_llm_disabled(self, mock_settings, tmp_path):
        mock_settings.llm_report_enabled = False
        mock_settings.min_reports_for_insights = 1
        mock_settings.db_path = str(tmp_path / "students.db")
        mock_settings.marking_sheet_path = str(tmp_path / "sheet.md")
        mock_settings.analysis_report_path = str(tmp_path / "analysis.md")

        state = _make_state(str(tmp_path / "students.db"),
                            output_path=str(tmp_path / "report.md"))
        result = finalize_agent(state)

        # Template report must still contain core grading info.
        assert len(result["final_report"]) > 0
        assert result["output_filepath"]

    def test_summary_extracted(self, tmp_path):
        state = _make_state(str(tmp_path / "students.db"),
                            output_path=str(tmp_path / "report.md"))
        result = finalize_agent(state)

        # Summary should be a non-empty string (first real paragraph).
        assert isinstance(result["summary"], str)

    @patch("mas.agents.finalize.get_prose_llm")
    def test_summary_is_deterministic_and_plain_text(self, mock_get_prose_llm, tmp_path):
        mock_prose = MagicMock()
        mock_prose.invoke.return_value = MagicMock(
            content="# Feedback Report\n\n---\n\nThis is model prose that should not be used."
        )
        mock_get_prose_llm.return_value = mock_prose

        state = _make_state(
            str(tmp_path / "students.db"),
            output_path=str(tmp_path / "report.md"),
        )
        result = finalize_agent(state)

        assert result["summary"] == (
            "Total score: 7/10 (70.00%), grade C. "
            "Breakdown: Accuracy: 4/5; Clarity: 3/5."
        )
        assert "\n" not in result["summary"]

    def test_analysis_report_written(self, tmp_path):
        db = str(tmp_path / "students.db")
        analysis_path = str(tmp_path / "analysis.md")
        state = _make_state(db,
                            output_path=str(tmp_path / "report.md"),
                            analysis_report_path=analysis_path)
        finalize_agent(state)

        assert Path(analysis_path).exists()


# ── _task_generate_insights unit tests ────────────────────────────────────────

class TestTaskGenerateInsights:
    def test_returns_empty_below_threshold(self):
        # min_reports_for_insights is 1 by default; with 0 past reports, nothing.
        result = _task_generate_insights("S001", [], 7.0, "C")
        assert result == ""

    @patch("mas.agents.finalize.get_light_prose_llm")
    def test_calls_llm_with_past_reports(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Good trend.")
        mock_get_llm.return_value = mock_llm

        past = [{"session_id": "s1", "timestamp": "2025-01-01", "total_score": 5, "grade": "D"}]
        result = _task_generate_insights("S001", past, 7.0, "C")

        assert result == "Good trend."
        mock_llm.invoke.assert_called_once()

    @patch("mas.agents.finalize.get_light_prose_llm")
    def test_returns_empty_on_llm_failure(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("fail")
        mock_get_llm.return_value = mock_llm

        past = [{"session_id": "s1", "timestamp": "2025-01-01", "total_score": 5, "grade": "D"}]
        result = _task_generate_insights("S001", past, 7.0, "C")

        assert result == ""


# ── _task_generate_report_prose unit tests ────────────────────────────────────

class TestTaskGenerateReportProse:
    @patch("mas.agents.finalize.get_prose_llm")
    def test_uses_llm_when_enabled(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="# Report\n\nGood.")
        mock_get_llm.return_value = mock_llm

        state: AgentState = {
            "session_id": "s",
            "rubric_data": SAMPLE_RUBRIC,
            "scored_criteria": SCORED_CRITERIA,
            "total_score": 7,
            "grade": "C",
            "agent_logs": [],
        }
        result = _task_generate_report_prose(state)

        assert result == "# Report\n\nGood."

    @patch("mas.agents.finalize.get_prose_llm")
    def test_falls_back_to_template_on_llm_failure(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("fail")
        mock_get_llm.return_value = mock_llm

        state: AgentState = {
            "session_id": "s",
            "rubric_data": SAMPLE_RUBRIC,
            "scored_criteria": SCORED_CRITERIA,
            "total_score": 7,
            "grade": "C",
            "agent_logs": [],
        }
        result = _task_generate_report_prose(state)

        assert len(result) > 0  # fallback template always produces output

    @patch("mas.agents.finalize.settings")
    def test_uses_template_when_llm_disabled(self, mock_settings):
        mock_settings.llm_report_enabled = False

        state: AgentState = {
            "session_id": "s",
            "rubric_data": SAMPLE_RUBRIC,
            "scored_criteria": SCORED_CRITERIA,
            "total_score": 7,
            "grade": "C",
            "agent_logs": [],
        }
        result = _task_generate_report_prose(state)

        assert len(result) > 0
