"""Tests for the Ingestion Agent."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from mas.agents.ingestion import _extract_student_id, _extract_student_name, ingestion_agent
from mas.state import AgentState


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_state(submission_path: str = "", rubric_path: str = "", **kwargs) -> AgentState:
    state: AgentState = {
        "submission_path": submission_path,
        "rubric_path": rubric_path,
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

SAMPLE_SUBMISSION = "Student Name: Jane Smith\nStudent ID: IT21000001\n\nMy answer here."


# ── _extract_student_id unit tests ────────────────────────────────────────────

class TestExtractStudentId:
    def test_extracts_standard_format(self):
        assert _extract_student_id("Student ID: IT21000001") == "IT21000001"

    def test_case_insensitive(self):
        assert _extract_student_id("student id: ABC123") == "ABC123"

    def test_extra_whitespace_between_words(self):
        assert _extract_student_id("Student  ID : XYZ999") == "XYZ999"

    def test_returns_empty_when_not_found(self):
        assert _extract_student_id("No ID here") == ""

    def test_multiline_submission(self):
        text = "Student Name: Jane\nStudent ID: IT21000099\nDate: 2025-01-01"
        assert _extract_student_id(text) == "IT21000099"




# ── _extract_student_name unit tests ──────────────────────────────────────────

class TestExtractStudentName:
    def test_extracts_standard_format(self):
        assert _extract_student_name("Student Name: Jane Smith") == "Jane Smith"

    def test_case_insensitive(self):
        assert _extract_student_name("student name: alice") == "alice"

    def test_allows_alternate_separator(self):
        assert _extract_student_name("Student Name - Bob Brown") == "Bob Brown"

    def test_returns_empty_when_not_found(self):
        assert _extract_student_name("No name here") == ""


# ── ingestion_agent tests ─────────────────────────────────────────────────────

class TestIngestionAgent:
    def test_success_with_valid_files(self, tmp_path):
        sub = tmp_path / "submission.txt"
        rub = tmp_path / "rubric.json"
        sub.write_text(SAMPLE_SUBMISSION, encoding="utf-8")
        rub.write_text(json.dumps(SAMPLE_RUBRIC), encoding="utf-8")

        result = ingestion_agent(_make_state(str(sub), str(rub)))

        assert result["ingestion_status"] == "success"
        assert result["student_id"] == "IT21000001"
        assert result["student_name"] == "Jane Smith"
        assert result["submission_text"] == SAMPLE_SUBMISSION
        assert result["rubric_data"]["total_marks"] == 10

    def test_session_id_generated_when_missing(self, tmp_path):
        sub = tmp_path / "submission.txt"
        rub = tmp_path / "rubric.json"
        sub.write_text(SAMPLE_SUBMISSION, encoding="utf-8")
        rub.write_text(json.dumps(SAMPLE_RUBRIC), encoding="utf-8")

        result = ingestion_agent(_make_state(str(sub), str(rub)))

        assert result.get("session_id")
        assert len(result["session_id"]) > 0

    def test_session_id_preserved(self, tmp_path):
        sub = tmp_path / "submission.txt"
        rub = tmp_path / "rubric.json"
        sub.write_text(SAMPLE_SUBMISSION, encoding="utf-8")
        rub.write_text(json.dumps(SAMPLE_RUBRIC), encoding="utf-8")

        result = ingestion_agent(_make_state(str(sub), str(rub), session_id="my-session"))

        assert result["session_id"] == "my-session"

    def test_missing_submission_sets_failed_status(self, tmp_path):
        rub = tmp_path / "rubric.json"
        rub.write_text(json.dumps(SAMPLE_RUBRIC), encoding="utf-8")

        result = ingestion_agent(_make_state(str(tmp_path / "nope.txt"), str(rub)))

        assert result["ingestion_status"] == "failed"
        assert "error" in result

    def test_missing_rubric_sets_failed_status(self, tmp_path):
        sub = tmp_path / "submission.txt"
        sub.write_text(SAMPLE_SUBMISSION, encoding="utf-8")

        result = ingestion_agent(_make_state(str(sub), str(tmp_path / "nope.json")))

        assert result["ingestion_status"] == "failed"
        assert "error" in result

    def test_empty_submission_path_sets_failed_status(self, tmp_path):
        rub = tmp_path / "rubric.json"
        rub.write_text(json.dumps(SAMPLE_RUBRIC), encoding="utf-8")

        result = ingestion_agent(_make_state("", str(rub)))

        assert result["ingestion_status"] == "failed"
        assert "empty" in result["error"].lower()

    def test_wrong_extension_sets_failed_status(self, tmp_path):
        sub = tmp_path / "submission.pdf"
        rub = tmp_path / "rubric.json"
        sub.write_text(SAMPLE_SUBMISSION, encoding="utf-8")
        rub.write_text(json.dumps(SAMPLE_RUBRIC), encoding="utf-8")

        result = ingestion_agent(_make_state(str(sub), str(rub)))

        assert result["ingestion_status"] == "failed"

    def test_invalid_json_rubric_sets_failed_status(self, tmp_path):
        sub = tmp_path / "submission.txt"
        rub = tmp_path / "rubric.json"
        sub.write_text(SAMPLE_SUBMISSION, encoding="utf-8")
        rub.write_text("not-valid-json{{{", encoding="utf-8")

        result = ingestion_agent(_make_state(str(sub), str(rub)))

        assert result["ingestion_status"] == "failed"

    def test_llm_returns_empty_when_metadata_not_found(self, tmp_path):
        sub = tmp_path / "submission.txt"
        rub = tmp_path / "rubric.json"
        sub.write_text("No ID in this text.", encoding="utf-8")
        rub.write_text(json.dumps(SAMPLE_RUBRIC), encoding="utf-8")
        with patch("mas.agents.ingestion.get_light_json_llm") as mock_llm:
            mock_details = MagicMock()
            mock_details.student_id = ""
            mock_details.student_name = ""
            mock_llm.return_value.invoke.return_value = mock_details
            result = ingestion_agent(_make_state(str(sub), str(rub)))

        assert result["ingestion_status"] == "success"
        assert result["student_id"] == ""
        assert result["student_name"] == ""

    @patch("mas.agents.ingestion.get_light_json_llm")
    @patch("mas.agents.ingestion.settings")
    def test_llm_extraction_used_when_fast_path_disabled(
        self, mock_settings, mock_llm, tmp_path
    ):
        sub = tmp_path / "submission.txt"
        rub = tmp_path / "rubric.json"
        sub.write_text(SAMPLE_SUBMISSION, encoding="utf-8")
        rub.write_text(json.dumps(SAMPLE_RUBRIC), encoding="utf-8")
        mock_settings.pdf_regex_fast_path_enabled = False
        mock_settings.light_model_name = "test-light-model"

        mock_details = MagicMock()
        mock_details.student_id = "IT21000001"
        mock_details.student_name = "Jane Smith"
        mock_llm.return_value.invoke.return_value = mock_details

        result = ingestion_agent(_make_state(str(sub), str(rub)))

        assert result["ingestion_status"] == "success"
        assert result["student_id"] == "IT21000001"
        assert result["student_name"] == "Jane Smith"
        assert mock_llm.called

    @patch("mas.agents.ingestion.get_light_json_llm")
    def test_llm_fallback_used_when_regex_missing_fields(self, mock_llm, tmp_path):
        sub = tmp_path / "submission.txt"
        rub = tmp_path / "rubric.json"
        sub.write_text("Answer text without explicit metadata.", encoding="utf-8")
        rub.write_text(json.dumps(SAMPLE_RUBRIC), encoding="utf-8")

        mock_details = MagicMock()
        mock_details.student_id = "IT21009999"
        mock_details.student_name = "Alex Lee"
        mock_llm.return_value.invoke.return_value = mock_details

        result = ingestion_agent(_make_state(str(sub), str(rub)))

        assert result["ingestion_status"] == "success"
        assert result["student_id"] == "IT21009999"
        assert result["student_name"] == "Alex Lee"
        assert mock_llm.called

    @patch("mas.agents.ingestion.get_light_json_llm")
    def test_llm_failure_falls_back_to_regex(self, mock_llm, tmp_path):
        sub = tmp_path / "submission.txt"
        rub = tmp_path / "rubric.json"
        sub.write_text("Student ID: IT21000077\nContent body.", encoding="utf-8")
        rub.write_text(json.dumps(SAMPLE_RUBRIC), encoding="utf-8")
        mock_llm.return_value.invoke.side_effect = RuntimeError("llm unavailable")

        result = ingestion_agent(_make_state(str(sub), str(rub)))

        assert result["ingestion_status"] == "success"
        assert result["student_id"] == "IT21000077"
        assert result["student_name"] == ""
        assert "error" in result
        assert "llm unavailable" in result["error"]

    def test_log_entry_appended(self, tmp_path):
        sub = tmp_path / "submission.txt"
        rub = tmp_path / "rubric.json"
        sub.write_text(SAMPLE_SUBMISSION, encoding="utf-8")
        rub.write_text(json.dumps(SAMPLE_RUBRIC), encoding="utf-8")

        result = ingestion_agent(_make_state(str(sub), str(rub)))

        assert len(result["agent_logs"]) == 1
        log = result["agent_logs"][0]
        assert log["agent"] == "ingestion"
        assert log["action"] == "ingest_submission"

    def test_existing_logs_preserved(self, tmp_path):
        sub = tmp_path / "submission.txt"
        rub = tmp_path / "rubric.json"
        sub.write_text(SAMPLE_SUBMISSION, encoding="utf-8")
        rub.write_text(json.dumps(SAMPLE_RUBRIC), encoding="utf-8")

        prior_log = {"agent": "other", "action": "prior"}
        result = ingestion_agent(_make_state(str(sub), str(rub), agent_logs=[prior_log]))

        assert len(result["agent_logs"]) == 2
        assert result["agent_logs"][0] == prior_log

    def test_log_entry_fields(self, tmp_path):
        sub = tmp_path / "submission.txt"
        rub = tmp_path / "rubric.json"
        sub.write_text(SAMPLE_SUBMISSION, encoding="utf-8")
        rub.write_text(json.dumps(SAMPLE_RUBRIC), encoding="utf-8")

        result = ingestion_agent(_make_state(str(sub), str(rub)))

        log = result["agent_logs"][0]
        for field in ("timestamp", "session_id", "agent", "action", "inputs", "outputs"):
            assert field in log

    def test_returns_expected_fields_on_success(self, tmp_path):
        sub = tmp_path / "submission.txt"
        rub = tmp_path / "rubric.json"
        sub.write_text(SAMPLE_SUBMISSION, encoding="utf-8")
        rub.write_text(json.dumps(SAMPLE_RUBRIC), encoding="utf-8")

        result = ingestion_agent(_make_state(str(sub), str(rub)))

        expected = {
            "session_id", "ingestion_status", "student_id", "student_name",
            "submission_text", "rubric_data", "agent_logs",
        }
        assert set(result.keys()) == expected
