"""Tests for the Ingestion Agent."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from mas.agents.ingestion import ingestion_agent
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

SAMPLE_SUBMISSION = (
    "Student Name: Jane Smith\n"
    "Student ID: IT21000001\n"
    "Assignment No: 03\n\n"
    "My answer here."
)

def _mock_metadata_llm(
    mock_llm,
    *,
    student_number: str = "IT21000001",
    student_name: str = "Jane Smith",
    assignment_number: str = "03",
):
    details = MagicMock()
    details.student_number = student_number
    details.student_name = student_name
    details.assignment_number = assignment_number
    mock_llm.return_value.invoke.return_value = details


# ── ingestion_agent tests ─────────────────────────────────────────────────────

class TestIngestionAgent:
    @patch("mas.agents.ingestion.get_metadata_json_llm")
    def test_success_with_valid_files(self, mock_llm, tmp_path):
        sub = tmp_path / "submission.txt"
        rub = tmp_path / "rubric.json"
        sub.write_text(SAMPLE_SUBMISSION, encoding="utf-8")
        rub.write_text(json.dumps(SAMPLE_RUBRIC), encoding="utf-8")
        _mock_metadata_llm(mock_llm)

        result = ingestion_agent(_make_state(str(sub), str(rub)))

        assert result["ingestion_status"] == "success"
        assert result["student_id"] == "IT21000001"
        assert result["student_name"] == "Jane Smith"
        assert result["assignment_number"] == "03"
        assert result["submission_text"] == SAMPLE_SUBMISSION
        assert result["rubric_data"]["total_marks"] == 10

    @patch("mas.agents.ingestion.get_metadata_json_llm")
    def test_session_id_generated_when_missing(self, mock_llm, tmp_path):
        sub = tmp_path / "submission.txt"
        rub = tmp_path / "rubric.json"
        sub.write_text(SAMPLE_SUBMISSION, encoding="utf-8")
        rub.write_text(json.dumps(SAMPLE_RUBRIC), encoding="utf-8")
        _mock_metadata_llm(mock_llm)

        result = ingestion_agent(_make_state(str(sub), str(rub)))

        assert result.get("session_id")
        assert len(result["session_id"]) > 0

    @patch("mas.agents.ingestion.get_metadata_json_llm")
    def test_session_id_preserved(self, mock_llm, tmp_path):
        sub = tmp_path / "submission.txt"
        rub = tmp_path / "rubric.json"
        sub.write_text(SAMPLE_SUBMISSION, encoding="utf-8")
        rub.write_text(json.dumps(SAMPLE_RUBRIC), encoding="utf-8")
        _mock_metadata_llm(mock_llm)

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
        with patch("mas.agents.ingestion.get_metadata_json_llm") as mock_llm:
            mock_details = MagicMock()
            mock_details.student_number = ""
            mock_details.student_name = ""
            mock_details.assignment_number = ""
            mock_llm.return_value.invoke.return_value = mock_details
            result = ingestion_agent(_make_state(str(sub), str(rub)))

        assert result["ingestion_status"] == "success"
        assert result["student_id"] == ""
        assert result["student_name"] == ""
        assert result["assignment_number"] == ""

    @patch("mas.agents.ingestion.get_metadata_json_llm")
    @patch("mas.agents.ingestion.settings")
    def test_llm_extraction_used_for_metadata(
        self, mock_settings, mock_llm, tmp_path
    ):
        sub = tmp_path / "submission.txt"
        rub = tmp_path / "rubric.json"
        sub.write_text(SAMPLE_SUBMISSION, encoding="utf-8")
        rub.write_text(json.dumps(SAMPLE_RUBRIC), encoding="utf-8")
        mock_settings.metadata_extractor_model_name = "test-extractor-model"
        _mock_metadata_llm(mock_llm)

        result = ingestion_agent(_make_state(str(sub), str(rub)))

        assert result["ingestion_status"] == "success"
        assert result["student_id"] == "IT21000001"
        assert result["student_name"] == "Jane Smith"
        assert result["assignment_number"] == "03"
        assert mock_llm.called

    @patch("mas.agents.ingestion.get_metadata_json_llm")
    def test_llm_extraction_succeeds_with_unstructured_text(self, mock_llm, tmp_path):
        sub = tmp_path / "submission.txt"
        rub = tmp_path / "rubric.json"
        sub.write_text("Answer text without explicit metadata.", encoding="utf-8")
        rub.write_text(json.dumps(SAMPLE_RUBRIC), encoding="utf-8")

        _mock_metadata_llm(
            mock_llm,
            student_number="IT21009999",
            student_name="Alex Lee",
            assignment_number="2",
        )

        result = ingestion_agent(_make_state(str(sub), str(rub)))

        assert result["ingestion_status"] == "success"
        assert result["student_id"] == "IT21009999"
        assert result["student_name"] == "Alex Lee"
        assert result["assignment_number"] == "2"
        assert mock_llm.called

    @patch("mas.agents.ingestion.get_metadata_json_llm")
    def test_llm_failure_uses_regex_fallback_metadata(self, mock_llm, tmp_path):
        sub = tmp_path / "submission.txt"
        rub = tmp_path / "rubric.json"
        sub.write_text("Student ID: IT21000077\nAssignment No: 01\nContent body.", encoding="utf-8")
        rub.write_text(json.dumps(SAMPLE_RUBRIC), encoding="utf-8")
        mock_llm.return_value.invoke.side_effect = RuntimeError("llm unavailable")

        result = ingestion_agent(_make_state(str(sub), str(rub)))

        assert result["ingestion_status"] == "success"
        assert result["student_id"] == "IT21000077"
        assert result["student_name"] == ""
        assert result["assignment_number"] == "01"
        assert "error" in result
        assert "llm unavailable" in result["error"]

    @patch("mas.agents.ingestion.get_metadata_json_llm")
    def test_log_entry_appended(self, mock_llm, tmp_path):
        sub = tmp_path / "submission.txt"
        rub = tmp_path / "rubric.json"
        sub.write_text(SAMPLE_SUBMISSION, encoding="utf-8")
        rub.write_text(json.dumps(SAMPLE_RUBRIC), encoding="utf-8")
        _mock_metadata_llm(mock_llm)

        result = ingestion_agent(_make_state(str(sub), str(rub)))

        assert len(result["agent_logs"]) == 1
        log = result["agent_logs"][0]
        assert log["agent"] == "ingestion"
        assert log["action"] == "ingest_submission"

    @patch("mas.agents.ingestion.get_metadata_json_llm")
    def test_existing_logs_preserved(self, mock_llm, tmp_path):
        sub = tmp_path / "submission.txt"
        rub = tmp_path / "rubric.json"
        sub.write_text(SAMPLE_SUBMISSION, encoding="utf-8")
        rub.write_text(json.dumps(SAMPLE_RUBRIC), encoding="utf-8")
        _mock_metadata_llm(mock_llm)

        prior_log = {"agent": "other", "action": "prior"}
        result = ingestion_agent(_make_state(str(sub), str(rub), agent_logs=[prior_log]))

        assert len(result["agent_logs"]) == 2
        assert result["agent_logs"][0] == prior_log

    @patch("mas.agents.ingestion.get_metadata_json_llm")
    def test_log_entry_fields(self, mock_llm, tmp_path):
        sub = tmp_path / "submission.txt"
        rub = tmp_path / "rubric.json"
        sub.write_text(SAMPLE_SUBMISSION, encoding="utf-8")
        rub.write_text(json.dumps(SAMPLE_RUBRIC), encoding="utf-8")
        _mock_metadata_llm(mock_llm)

        result = ingestion_agent(_make_state(str(sub), str(rub)))

        log = result["agent_logs"][0]
        for field in ("timestamp", "session_id", "agent", "action", "inputs", "outputs"):
            assert field in log

    @patch("mas.agents.ingestion.get_metadata_json_llm")
    def test_returns_expected_fields_on_success(self, mock_llm, tmp_path):
        sub = tmp_path / "submission.txt"
        rub = tmp_path / "rubric.json"
        sub.write_text(SAMPLE_SUBMISSION, encoding="utf-8")
        rub.write_text(json.dumps(SAMPLE_RUBRIC), encoding="utf-8")
        _mock_metadata_llm(mock_llm)

        result = ingestion_agent(_make_state(str(sub), str(rub)))

        expected = {
            "session_id", "ingestion_status", "student_id", "student_name", "assignment_number",
            "submission_text", "rubric_data", "agent_logs",
        }
        assert set(result.keys()) == expected
