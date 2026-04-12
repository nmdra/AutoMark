"""Tests for the Historical Agent."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mas.agents.historical import historical_agent
from mas.state import AgentState
from mas.tools.db_manager import get_past_reports, init_db, save_report


# ── Helpers ───────────────────────────────────────────────────────────────────

SCORED_CRITERIA = [
    {"criterion_id": "C1", "name": "Accuracy", "score": 4, "max_score": 5,
     "justification": "Good definition."},
    {"criterion_id": "C2", "name": "Clarity", "score": 3, "max_score": 5,
     "justification": "Mostly clear."},
]


def _make_state(db_path: str, **kwargs) -> AgentState:
    base: AgentState = {
        "session_id": "test-session",
        "student_id": "IT21000001",
        "scored_criteria": SCORED_CRITERIA,
        "total_score": 7,
        "grade": "C",
        "db_path": db_path,
        "agent_logs": [],
    }
    base.update(kwargs)
    return base


# ── db_manager unit tests ─────────────────────────────────────────────────────

class TestDbManager:
    def test_init_db_creates_file(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(db_path)
        assert Path(db_path).exists()

    def test_init_db_idempotent(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(db_path)
        init_db(db_path)  # should not raise
        assert Path(db_path).exists()

    def test_save_and_retrieve_report(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(db_path)
        save_report(
            db_path=db_path,
            student_id="S001",
            session_id="sess-1",
            timestamp="2025-01-01T00:00:00+00:00",
            scored_criteria=SCORED_CRITERIA,
            total_score=7,
            grade="C",
        )
        reports = get_past_reports(db_path, "S001")
        assert len(reports) == 1
        assert reports[0]["total_score"] == 7
        assert reports[0]["grade"] == "C"
        assert reports[0]["session_id"] == "sess-1"

    def test_get_past_reports_empty_for_unknown_student(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(db_path)
        reports = get_past_reports(db_path, "UNKNOWN")
        assert reports == []

    def test_get_past_reports_returns_empty_when_db_missing(self, tmp_path):
        db_path = str(tmp_path / "nonexistent.db")
        reports = get_past_reports(db_path, "S001")
        assert reports == []

    def test_multiple_reports_ordered_chronologically(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(db_path)
        for i in range(3):
            save_report(
                db_path=db_path,
                student_id="S002",
                session_id=f"sess-{i}",
                timestamp=f"2025-01-0{i + 1}T00:00:00+00:00",
                scored_criteria=SCORED_CRITERIA,
                total_score=i * 2,
                grade="F",
            )
        reports = get_past_reports(db_path, "S002")
        assert len(reports) == 3
        assert reports[0]["session_id"] == "sess-0"
        assert reports[2]["session_id"] == "sess-2"

    def test_reports_isolated_by_student(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(db_path)
        save_report(db_path, "S001", "s1", "2025-01-01T00:00:00+00:00",
                    SCORED_CRITERIA, 7, "C")
        save_report(db_path, "S002", "s2", "2025-01-01T00:00:00+00:00",
                    SCORED_CRITERIA, 5, "D")
        assert len(get_past_reports(db_path, "S001")) == 1
        assert len(get_past_reports(db_path, "S002")) == 1

    def test_save_report_creates_db_if_missing(self, tmp_path):
        db_path = str(tmp_path / "new_dir" / "test.db")
        save_report(db_path, "S001", "s1", "2025-01-01T00:00:00+00:00",
                    SCORED_CRITERIA, 7, "C")
        assert Path(db_path).exists()
        reports = get_past_reports(db_path, "S001")
        assert len(reports) == 1


# ── historical_agent tests ────────────────────────────────────────────────────

class TestHistoricalAgent:
    def test_saves_report_to_db(self, tmp_path):
        db_path = str(tmp_path / "students.db")
        result = historical_agent(_make_state(db_path))

        reports = get_past_reports(db_path, "IT21000001")
        assert len(reports) == 1
        assert reports[0]["total_score"] == 7

    def test_no_progression_insights_for_first_submission(self, tmp_path):
        db_path = str(tmp_path / "students.db")
        result = historical_agent(_make_state(db_path))

        assert result["past_reports"] == []
        assert result["progression_insights"] == ""

    def test_past_reports_populated_for_returning_student(self, tmp_path):
        db_path = str(tmp_path / "students.db")
        # Pre-seed a previous report
        save_report(db_path, "IT21000001", "old-session", "2025-01-01T00:00:00+00:00",
                    SCORED_CRITERIA, 5, "D")

        result = historical_agent(_make_state(db_path))

        assert len(result["past_reports"]) == 1
        assert result["past_reports"][0]["grade"] == "D"

    @patch("mas.agents.historical.get_prose_llm")
    def test_progression_insights_generated_for_returning_student(
        self, mock_get_llm, tmp_path
    ):
        db_path = str(tmp_path / "students.db")
        save_report(db_path, "IT21000001", "old-session", "2025-01-01T00:00:00+00:00",
                    SCORED_CRITERIA, 5, "D")

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Student is improving steadily.")
        mock_get_llm.return_value = mock_llm

        result = historical_agent(_make_state(db_path))

        assert result["progression_insights"] == "Student is improving steadily."

    @patch("mas.agents.historical.get_prose_llm")
    def test_llm_failure_gives_empty_insights(self, mock_get_llm, tmp_path):
        db_path = str(tmp_path / "students.db")
        save_report(db_path, "IT21000001", "old-session", "2025-01-01T00:00:00+00:00",
                    SCORED_CRITERIA, 5, "D")

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM unavailable")
        mock_get_llm.return_value = mock_llm

        result = historical_agent(_make_state(db_path))

        assert result["progression_insights"] == ""

    def test_log_entry_appended(self, tmp_path):
        db_path = str(tmp_path / "students.db")
        result = historical_agent(_make_state(db_path))

        assert len(result["agent_logs"]) == 1
        assert result["agent_logs"][0]["agent"] == "historical"
        assert result["agent_logs"][0]["action"] == "save_and_compare"

    def test_existing_logs_preserved(self, tmp_path):
        db_path = str(tmp_path / "students.db")
        prior = {"agent": "analysis", "action": "score_criteria"}
        result = historical_agent(_make_state(db_path, agent_logs=[prior]))

        assert len(result["agent_logs"]) == 2
        assert result["agent_logs"][0] == prior

    def test_returns_expected_fields(self, tmp_path):
        db_path = str(tmp_path / "students.db")
        result = historical_agent(_make_state(db_path))

        expected = {"past_reports", "progression_insights", "analysis_report_path", "agent_logs"}
        assert set(result.keys()) == expected
