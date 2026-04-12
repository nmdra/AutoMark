"""Tests for the Research Agent."""

from __future__ import annotations

import json

import pytest

from ctse_mas.agents.research import research_agent
from ctse_mas.state import AgentState


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_state(submission_path: str = "", rubric_path: str = "", **kwargs) -> AgentState:
    state: AgentState = {
        "submission_path": submission_path,
        "rubric_path": rubric_path,
        "session_id": "test-session",
        "agent_logs": [],
    }
    state.update(kwargs)
    return state


SAMPLE_RUBRIC = {
    "module": "CTSE",
    "total_marks": 10,
    "criteria": [
        {"id": "C1", "name": "Accuracy", "max_score": 5},
        {"id": "C2", "name": "Clarity", "max_score": 5},
    ],
}


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestResearchAgent:
    def test_success_reads_files(self, tmp_path):
        sub = tmp_path / "submission.txt"
        rub = tmp_path / "rubric.json"
        sub.write_text("My answer here.", encoding="utf-8")
        rub.write_text(json.dumps(SAMPLE_RUBRIC), encoding="utf-8")

        result = research_agent(_make_state(str(sub), str(rub)))

        assert result["research_status"] == "success"
        assert result["submission_text"] == "My answer here."
        assert result["rubric_data"]["total_marks"] == 10

    def test_returns_only_changed_fields(self, tmp_path):
        sub = tmp_path / "submission.txt"
        rub = tmp_path / "rubric.json"
        sub.write_text("text", encoding="utf-8")
        rub.write_text(json.dumps(SAMPLE_RUBRIC), encoding="utf-8")

        result = research_agent(_make_state(str(sub), str(rub)))

        expected = {"submission_text", "rubric_data", "research_status", "agent_logs"}
        assert set(result.keys()) == expected

    def test_missing_submission_sets_failed_status(self, tmp_path):
        rub = tmp_path / "rubric.json"
        rub.write_text(json.dumps(SAMPLE_RUBRIC), encoding="utf-8")

        result = research_agent(_make_state(str(tmp_path / "nope.txt"), str(rub)))

        assert result["research_status"] == "failed"
        assert "error" in result

    def test_missing_rubric_sets_failed_status(self, tmp_path):
        sub = tmp_path / "submission.txt"
        sub.write_text("text", encoding="utf-8")

        result = research_agent(_make_state(str(sub), str(tmp_path / "nope.json")))

        assert result["research_status"] == "failed"
        assert "error" in result

    def test_invalid_json_rubric_sets_failed_status(self, tmp_path):
        sub = tmp_path / "submission.txt"
        rub = tmp_path / "rubric.json"
        sub.write_text("text", encoding="utf-8")
        rub.write_text("not-valid-json{{{", encoding="utf-8")

        result = research_agent(_make_state(str(sub), str(rub)))

        assert result["research_status"] == "failed"

    def test_log_entry_appended(self, tmp_path):
        sub = tmp_path / "submission.txt"
        rub = tmp_path / "rubric.json"
        sub.write_text("text", encoding="utf-8")
        rub.write_text(json.dumps(SAMPLE_RUBRIC), encoding="utf-8")

        result = research_agent(_make_state(str(sub), str(rub)))

        assert len(result["agent_logs"]) == 1
        log = result["agent_logs"][0]
        assert log["agent"] == "research"
        assert log["action"] == "read_files"

    def test_existing_logs_preserved(self, tmp_path):
        sub = tmp_path / "submission.txt"
        rub = tmp_path / "rubric.json"
        sub.write_text("text", encoding="utf-8")
        rub.write_text(json.dumps(SAMPLE_RUBRIC), encoding="utf-8")

        prior_log = {"agent": "coordinator", "action": "validate"}
        result = research_agent(
            _make_state(str(sub), str(rub), agent_logs=[prior_log])
        )

        assert len(result["agent_logs"]) == 2
        assert result["agent_logs"][0] == prior_log
