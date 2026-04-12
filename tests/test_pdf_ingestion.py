"""Tests for the PDF Ingestion Agent (mocked LLM and PDF processor)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mas.agents.pdf_ingestion import _build_extraction_prompt, pdf_ingestion_agent
from mas.state import AgentState

SAMPLE_RUBRIC = {
    "module": "CTSE",
    "total_marks": 10,
    "criteria": [
        {"id": "C1", "name": "Accuracy", "max_score": 5},
        {"id": "C2", "name": "Clarity", "max_score": 5},
    ],
}


def _make_state(submission_path: str = "", rubric_path: str = "", **kwargs) -> AgentState:
    state: AgentState = {
        "submission_path": submission_path,
        "rubric_path": rubric_path,
        "agent_logs": [],
    }
    state.update(kwargs)
    return state


def _write_rubric(tmp_path: Path) -> Path:
    rub = tmp_path / "rubric.json"
    rub.write_text(json.dumps(SAMPLE_RUBRIC), encoding="utf-8")
    return rub


def _fake_pdf(tmp_path: Path, name: str = "submission.pdf") -> Path:
    """Create a minimal non-empty placeholder file with .pdf extension."""
    pdf = tmp_path / name
    pdf.write_bytes(b"%PDF-1.4 placeholder")
    return pdf


# ── _build_extraction_prompt ──────────────────────────────────────────────────


class TestBuildExtractionPrompt:
    def test_contains_student_id_instruction(self):
        prompt = _build_extraction_prompt("some text")
        assert "student_id" in prompt.lower()

    def test_contains_student_name_instruction(self):
        prompt = _build_extraction_prompt("some text")
        assert "student_name" in prompt.lower()

    def test_contains_markdown_text(self):
        prompt = _build_extraction_prompt("my submission here")
        assert "my submission here" in prompt


# ── pdf_ingestion_agent ───────────────────────────────────────────────────────


class TestPdfIngestionAgent:
    @patch("mas.agents.pdf_ingestion.get_json_llm")
    @patch("mas.agents.pdf_ingestion.convert_pdf_to_markdown")
    def test_success_sets_ingestion_status(self, mock_convert, mock_llm, tmp_path):
        pdf = _fake_pdf(tmp_path)
        rub = _write_rubric(tmp_path)

        mock_convert.return_value = "Student ID: S001\nStudent Name: Alice\nContent here."
        mock_details = MagicMock()
        mock_details.student_id = "S001"
        mock_details.student_name = "Alice"
        mock_details.submission_text = "Content here."
        mock_llm.return_value.invoke.return_value = mock_details

        result = pdf_ingestion_agent(_make_state(str(pdf), str(rub)))

        assert result["ingestion_status"] == "success"

    @patch("mas.agents.pdf_ingestion.get_json_llm")
    @patch("mas.agents.pdf_ingestion.convert_pdf_to_markdown")
    def test_extracts_student_id_and_name(self, mock_convert, mock_llm, tmp_path):
        pdf = _fake_pdf(tmp_path)
        rub = _write_rubric(tmp_path)

        mock_convert.return_value = "Student ID: IT21000042\nContent."
        mock_details = MagicMock()
        mock_details.student_id = "IT21000042"
        mock_details.student_name = "Bob Smith"
        mock_details.submission_text = "Content."
        mock_llm.return_value.invoke.return_value = mock_details

        result = pdf_ingestion_agent(_make_state(str(pdf), str(rub)))

        assert result["student_id"] == "IT21000042"
        assert result["student_name"] == "Bob Smith"

    @patch("mas.agents.pdf_ingestion.get_json_llm")
    @patch("mas.agents.pdf_ingestion.convert_pdf_to_markdown")
    def test_submission_text_populated(self, mock_convert, mock_llm, tmp_path):
        pdf = _fake_pdf(tmp_path)
        rub = _write_rubric(tmp_path)

        mock_convert.return_value = "Raw markdown."
        mock_details = MagicMock()
        mock_details.student_id = ""
        mock_details.student_name = ""
        mock_details.submission_text = "Cleaned submission body."
        mock_llm.return_value.invoke.return_value = mock_details

        result = pdf_ingestion_agent(_make_state(str(pdf), str(rub)))

        assert result["submission_text"] == "Cleaned submission body."

    @patch("mas.agents.pdf_ingestion.get_json_llm")
    @patch("mas.agents.pdf_ingestion.convert_pdf_to_markdown")
    def test_rubric_data_loaded(self, mock_convert, mock_llm, tmp_path):
        pdf = _fake_pdf(tmp_path)
        rub = _write_rubric(tmp_path)

        mock_convert.return_value = "Content."
        mock_details = MagicMock()
        mock_details.student_id = ""
        mock_details.student_name = ""
        mock_details.submission_text = "Content."
        mock_llm.return_value.invoke.return_value = mock_details

        result = pdf_ingestion_agent(_make_state(str(pdf), str(rub)))

        assert result["rubric_data"]["total_marks"] == 10

    def test_missing_pdf_sets_failed_status(self, tmp_path):
        rub = _write_rubric(tmp_path)
        result = pdf_ingestion_agent(
            _make_state(str(tmp_path / "missing.pdf"), str(rub))
        )
        assert result["ingestion_status"] == "failed"
        assert "error" in result

    def test_non_pdf_extension_sets_failed_status(self, tmp_path):
        txt = tmp_path / "sub.txt"
        txt.write_text("content", encoding="utf-8")
        rub = _write_rubric(tmp_path)

        result = pdf_ingestion_agent(_make_state(str(txt), str(rub)))

        assert result["ingestion_status"] == "failed"

    def test_empty_submission_path_sets_failed_status(self, tmp_path):
        rub = _write_rubric(tmp_path)
        result = pdf_ingestion_agent(_make_state("", str(rub)))
        assert result["ingestion_status"] == "failed"
        assert "error" in result

    def test_empty_rubric_path_sets_failed_status(self, tmp_path):
        pdf = _fake_pdf(tmp_path)
        result = pdf_ingestion_agent(_make_state(str(pdf), ""))
        assert result["ingestion_status"] == "failed"

    def test_session_id_generated_when_missing(self, tmp_path):
        pdf = _fake_pdf(tmp_path)
        rub = _write_rubric(tmp_path)

        with patch("mas.agents.pdf_ingestion.convert_pdf_to_markdown") as mc, \
             patch("mas.agents.pdf_ingestion.get_json_llm") as ml:
            mc.return_value = "Text."
            det = MagicMock()
            det.student_id = ""
            det.student_name = ""
            det.submission_text = "Text."
            ml.return_value.invoke.return_value = det
            result = pdf_ingestion_agent(_make_state(str(pdf), str(rub)))

        assert result.get("session_id")
        assert len(result["session_id"]) > 0

    def test_session_id_preserved(self, tmp_path):
        pdf = _fake_pdf(tmp_path)
        rub = _write_rubric(tmp_path)

        with patch("mas.agents.pdf_ingestion.convert_pdf_to_markdown") as mc, \
             patch("mas.agents.pdf_ingestion.get_json_llm") as ml:
            mc.return_value = "Text."
            det = MagicMock()
            det.student_id = ""
            det.student_name = ""
            det.submission_text = "Text."
            ml.return_value.invoke.return_value = det
            result = pdf_ingestion_agent(
                _make_state(str(pdf), str(rub), session_id="my-session")
            )

        assert result["session_id"] == "my-session"

    @patch("mas.agents.pdf_ingestion.get_json_llm")
    @patch("mas.agents.pdf_ingestion.convert_pdf_to_markdown")
    def test_log_entry_appended(self, mock_convert, mock_llm, tmp_path):
        pdf = _fake_pdf(tmp_path)
        rub = _write_rubric(tmp_path)
        mock_convert.return_value = "Text."
        det = MagicMock()
        det.student_id = ""
        det.student_name = ""
        det.submission_text = "Text."
        mock_llm.return_value.invoke.return_value = det

        result = pdf_ingestion_agent(_make_state(str(pdf), str(rub)))

        assert len(result["agent_logs"]) == 1
        log = result["agent_logs"][0]
        assert log["agent"] == "pdf_ingestion"
        assert log["action"] == "ingest_pdf_submission"

    @patch("mas.agents.pdf_ingestion.get_json_llm")
    @patch("mas.agents.pdf_ingestion.convert_pdf_to_markdown")
    def test_existing_logs_preserved(self, mock_convert, mock_llm, tmp_path):
        pdf = _fake_pdf(tmp_path)
        rub = _write_rubric(tmp_path)
        mock_convert.return_value = "Text."
        det = MagicMock()
        det.student_id = ""
        det.student_name = ""
        det.submission_text = "Text."
        mock_llm.return_value.invoke.return_value = det

        prior = {"agent": "other", "action": "prior"}
        result = pdf_ingestion_agent(
            _make_state(str(pdf), str(rub), agent_logs=[prior])
        )

        assert len(result["agent_logs"]) == 2
        assert result["agent_logs"][0] == prior

    @patch("mas.agents.pdf_ingestion.get_json_llm")
    @patch("mas.agents.pdf_ingestion.convert_pdf_to_markdown")
    def test_returns_expected_fields_on_success(self, mock_convert, mock_llm, tmp_path):
        pdf = _fake_pdf(tmp_path)
        rub = _write_rubric(tmp_path)
        mock_convert.return_value = "Text."
        det = MagicMock()
        det.student_id = "S1"
        det.student_name = "Alice"
        det.submission_text = "Text."
        mock_llm.return_value.invoke.return_value = det

        result = pdf_ingestion_agent(_make_state(str(pdf), str(rub)))

        expected = {
            "session_id", "ingestion_status", "student_id", "student_name",
            "submission_text", "rubric_data", "agent_logs",
        }
        assert set(result.keys()) == expected
