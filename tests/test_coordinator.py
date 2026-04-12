"""Tests for the Coordinator Agent."""

from __future__ import annotations

import pytest

from ctse_mas.agents.coordinator import coordinator_agent
from ctse_mas.state import AgentState


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_state(submission_path: str = "", rubric_path: str = "", **kwargs) -> AgentState:
    state: AgentState = {
        "submission_path": submission_path,
        "rubric_path": rubric_path,
        "agent_logs": [],
    }
    state.update(kwargs)
    return state


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestCoordinatorAgent:
    def test_success_with_valid_files(self, tmp_path):
        sub = tmp_path / "submission.txt"
        rub = tmp_path / "rubric.json"
        sub.write_text("content", encoding="utf-8")
        rub.write_text('{"criteria": []}', encoding="utf-8")

        state = _make_state(str(sub), str(rub))
        result = coordinator_agent(state)

        assert result["validation_status"] == "success"
        assert result["validation_error"] == ""
        assert "session_id" in result
        assert len(result["agent_logs"]) == 1

    def test_missing_submission_file(self, tmp_path):
        rub = tmp_path / "rubric.json"
        rub.write_text('{"criteria": []}', encoding="utf-8")

        state = _make_state(str(tmp_path / "missing.txt"), str(rub))
        result = coordinator_agent(state)

        assert result["validation_status"] == "failed"
        assert "not found" in result["validation_error"].lower()

    def test_missing_rubric_file(self, tmp_path):
        sub = tmp_path / "submission.txt"
        sub.write_text("content", encoding="utf-8")

        state = _make_state(str(sub), str(tmp_path / "missing.json"))
        result = coordinator_agent(state)

        assert result["validation_status"] == "failed"

    def test_empty_submission_path(self, tmp_path):
        rub = tmp_path / "rubric.json"
        rub.write_text('{"criteria": []}', encoding="utf-8")

        state = _make_state("", str(rub))
        result = coordinator_agent(state)

        assert result["validation_status"] == "failed"
        assert "empty" in result["validation_error"].lower()

    def test_empty_rubric_path(self, tmp_path):
        sub = tmp_path / "submission.txt"
        sub.write_text("content", encoding="utf-8")

        state = _make_state(str(sub), "")
        result = coordinator_agent(state)

        assert result["validation_status"] == "failed"

    def test_wrong_extension_submission(self, tmp_path):
        sub = tmp_path / "submission.pdf"
        rub = tmp_path / "rubric.json"
        sub.write_text("content", encoding="utf-8")
        rub.write_text('{"criteria": []}', encoding="utf-8")

        state = _make_state(str(sub), str(rub))
        result = coordinator_agent(state)

        assert result["validation_status"] == "failed"

    def test_wrong_extension_rubric(self, tmp_path):
        sub = tmp_path / "submission.txt"
        rub = tmp_path / "rubric.txt"
        sub.write_text("content", encoding="utf-8")
        rub.write_text('{"criteria": []}', encoding="utf-8")

        state = _make_state(str(sub), str(rub))
        result = coordinator_agent(state)

        assert result["validation_status"] == "failed"

    def test_session_id_preserved(self, tmp_path):
        sub = tmp_path / "submission.txt"
        rub = tmp_path / "rubric.json"
        sub.write_text("content", encoding="utf-8")
        rub.write_text('{"criteria": []}', encoding="utf-8")

        state = _make_state(str(sub), str(rub), session_id="my-session-123")
        result = coordinator_agent(state)

        assert result["session_id"] == "my-session-123"

    def test_session_id_generated_when_missing(self, tmp_path):
        sub = tmp_path / "submission.txt"
        rub = tmp_path / "rubric.json"
        sub.write_text("content", encoding="utf-8")
        rub.write_text('{"criteria": []}', encoding="utf-8")

        state = _make_state(str(sub), str(rub))
        result = coordinator_agent(state)

        assert result["session_id"]
        assert len(result["session_id"]) > 0

    def test_log_entry_fields(self, tmp_path):
        sub = tmp_path / "submission.txt"
        rub = tmp_path / "rubric.json"
        sub.write_text("content", encoding="utf-8")
        rub.write_text('{"criteria": []}', encoding="utf-8")

        state = _make_state(str(sub), str(rub))
        result = coordinator_agent(state)

        log = result["agent_logs"][0]
        for field in ("timestamp", "session_id", "agent", "action", "inputs", "outputs"):
            assert field in log

    def test_returns_only_changed_fields(self, tmp_path):
        sub = tmp_path / "submission.txt"
        rub = tmp_path / "rubric.json"
        sub.write_text("content", encoding="utf-8")
        rub.write_text('{"criteria": []}', encoding="utf-8")

        state = _make_state(str(sub), str(rub))
        result = coordinator_agent(state)

        expected_keys = {"session_id", "validation_status", "validation_error", "agent_logs"}
        assert set(result.keys()) == expected_keys
