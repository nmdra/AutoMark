"""Tests for the Report Agent."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mas.agents.report import report_agent
from mas.state import AgentState

SAMPLE_RUBRIC = {
    "module": "CTSE",
    "assignment": "Cloud Fundamentals",
    "total_marks": 10,
    "criteria": [
        {"id": "C1", "name": "Accuracy", "max_score": 5},
        {"id": "C2", "name": "Clarity", "max_score": 5},
    ],
}

SCORED_CRITERIA = [
    {
        "criterion_id": "C1",
        "name": "Accuracy",
        "score": 4,
        "max_score": 5,
        "justification": "Good definition.",
    },
    {
        "criterion_id": "C2",
        "name": "Clarity",
        "score": 3,
        "max_score": 5,
        "justification": "Mostly clear.",
    },
]


def _make_state(output_path: str = "", **kwargs) -> AgentState:
    base: AgentState = {
        "session_id": "test-session",
        "rubric_data": SAMPLE_RUBRIC,
        "scored_criteria": SCORED_CRITERIA,
        "total_score": 7,
        "grade": "C",
        "output_filepath": output_path,
        "agent_logs": [],
    }
    base.update(kwargs)
    return base


class TestReportAgent:
    @patch("mas.agents.report.get_prose_llm")
    def test_report_written_to_disk(self, mock_get_llm, tmp_path):
        output = str(tmp_path / "out" / "report.md")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="# Feedback Report\n\nGood job.")
        mock_get_llm.return_value = mock_llm

        result = report_agent(_make_state(output_path=output))

        assert Path(output).exists()
        assert result["final_report"] == "# Feedback Report\n\nGood job."

    @patch("mas.agents.report.get_prose_llm")
    def test_output_filepath_in_result(self, mock_get_llm, tmp_path):
        output = str(tmp_path / "report.md")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Report content.")
        mock_get_llm.return_value = mock_llm

        result = report_agent(_make_state(output_path=output))

        assert "output_filepath" in result
        assert result["output_filepath"] == str(Path(output).resolve())

    @patch("mas.agents.report.get_prose_llm")
    def test_parent_dirs_created(self, mock_get_llm, tmp_path):
        deep_path = str(tmp_path / "a" / "b" / "c" / "report.md")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Deep report.")
        mock_get_llm.return_value = mock_llm

        report_agent(_make_state(output_path=deep_path))

        assert Path(deep_path).exists()

    @patch("mas.agents.report.get_prose_llm")
    def test_fallback_report_when_llm_fails(self, mock_get_llm, tmp_path):
        output = str(tmp_path / "report.md")
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM unavailable")
        mock_get_llm.return_value = mock_llm

        result = report_agent(_make_state(output_path=output))

        assert Path(output).exists()
        assert len(result["final_report"]) > 0

    @patch("mas.agents.report.get_prose_llm")
    def test_overwrite_existing_file(self, mock_get_llm, tmp_path):
        output = tmp_path / "report.md"
        output.write_text("old content", encoding="utf-8")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="new content")
        mock_get_llm.return_value = mock_llm

        report_agent(_make_state(output_path=str(output)))

        assert output.read_text(encoding="utf-8") == "new content"

    @patch("mas.agents.report.get_prose_llm")
    def test_log_entry_appended(self, mock_get_llm, tmp_path):
        output = str(tmp_path / "report.md")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Report.")
        mock_get_llm.return_value = mock_llm

        result = report_agent(_make_state(output_path=output))

        assert len(result["agent_logs"]) == 1
        assert result["agent_logs"][0]["agent"] == "report"
        assert result["agent_logs"][0]["action"] == "write_feedback_report"

    @patch("mas.agents.report.get_prose_llm")
    def test_returns_only_changed_fields(self, mock_get_llm, tmp_path):
        output = str(tmp_path / "report.md")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Report.")
        mock_get_llm.return_value = mock_llm

        result = report_agent(_make_state(output_path=output))

        expected = {"final_report", "summary", "output_filepath", "marking_sheet_path", "agent_logs"}
        assert set(result.keys()) == expected
