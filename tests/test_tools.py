"""Unit tests for all tool modules – no LLM required."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from mas.tools.file_ops import read_json_file, read_text_file, validate_submission_files
from mas.tools.file_writer import (
    write_analysis_report,
    write_feedback_report,
    write_marking_sheet,
)
from mas.tools.logger import log_agent_action
from mas.tools.pdf_processor import convert_pdf_to_markdown
from mas.tools.score_calculator import calculate_total_score


# ── file_reader ───────────────────────────────────────────────────────────────


class TestReadTextFile:
    def test_reads_content(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("hello world", encoding="utf-8")
        assert read_text_file(str(f)) == "hello world"

    def test_reads_multiline(self, tmp_path):
        f = tmp_path / "multi.txt"
        f.write_text("line1\nline2\nline3", encoding="utf-8")
        assert read_text_file(str(f)) == "line1\nline2\nline3"

    def test_reads_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        assert read_text_file(str(f)) == ""

    def test_missing_file_raises_runtime_error(self, tmp_path):
        with pytest.raises(RuntimeError, match="Failed to read text file"):
            read_text_file(str(tmp_path / "missing.txt"))

    def test_error_message_contains_path(self, tmp_path):
        bad_path = str(tmp_path / "no_such_file.txt")
        with pytest.raises(RuntimeError, match=bad_path):
            read_text_file(bad_path)


class TestReadJsonFile:
    def test_reads_dict(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text('{"key": "value", "num": 42}', encoding="utf-8")
        result = read_json_file(str(f))
        assert result == {"key": "value", "num": 42}

    def test_reads_nested_structure(self, tmp_path):
        data = {"criteria": [{"id": "C1", "max_score": 5}], "total_marks": 5}
        f = tmp_path / "rubric.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        assert read_json_file(str(f)) == data

    def test_missing_file_raises_runtime_error(self, tmp_path):
        with pytest.raises(RuntimeError, match="Failed to read JSON file"):
            read_json_file(str(tmp_path / "missing.json"))

    def test_invalid_json_raises_runtime_error(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("not { valid json }", encoding="utf-8")
        with pytest.raises(RuntimeError, match="Failed to parse JSON file"):
            read_json_file(str(f))

    def test_invalid_json_error_contains_path(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("{{{", encoding="utf-8")
        with pytest.raises(RuntimeError, match=str(f)):
            read_json_file(str(f))


# ── file_validator ────────────────────────────────────────────────────────────


class TestValidateSubmissionFiles:
    def test_success_returns_status_dict(self, tmp_path):
        sub = tmp_path / "sub.txt"
        rub = tmp_path / "rub.json"
        sub.write_text("content", encoding="utf-8")
        rub.write_text('{"criteria":[]}', encoding="utf-8")
        result = validate_submission_files(str(sub), str(rub))
        assert result == {"status": "success"}

    def test_empty_submission_path_raises_value_error(self, tmp_path):
        rub = tmp_path / "rub.json"
        rub.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="submission_path must not be empty"):
            validate_submission_files("", str(rub))

    def test_empty_rubric_path_raises_value_error(self, tmp_path):
        sub = tmp_path / "sub.txt"
        sub.write_text("content", encoding="utf-8")
        with pytest.raises(ValueError, match="rubric_path must not be empty"):
            validate_submission_files(str(sub), "")

    def test_missing_submission_raises_file_not_found(self, tmp_path):
        rub = tmp_path / "rub.json"
        rub.write_text("{}", encoding="utf-8")
        with pytest.raises(FileNotFoundError, match="Submission file not found"):
            validate_submission_files(str(tmp_path / "nope.txt"), str(rub))

    def test_missing_rubric_raises_file_not_found(self, tmp_path):
        sub = tmp_path / "sub.txt"
        sub.write_text("content", encoding="utf-8")
        with pytest.raises(FileNotFoundError, match="Rubric file not found"):
            validate_submission_files(str(sub), str(tmp_path / "nope.json"))

    def test_empty_submission_file_raises_value_error(self, tmp_path):
        sub = tmp_path / "sub.txt"
        rub = tmp_path / "rub.json"
        sub.write_text("", encoding="utf-8")
        rub.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            validate_submission_files(str(sub), str(rub))

    def test_empty_rubric_file_raises_value_error(self, tmp_path):
        sub = tmp_path / "sub.txt"
        rub = tmp_path / "rub.json"
        sub.write_text("content", encoding="utf-8")
        rub.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            validate_submission_files(str(sub), str(rub))

    def test_wrong_submission_extension_raises_value_error(self, tmp_path):
        sub = tmp_path / "sub.pdf"
        rub = tmp_path / "rub.json"
        sub.write_text("content", encoding="utf-8")
        rub.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match=r"\.txt"):
            validate_submission_files(str(sub), str(rub))

    def test_wrong_rubric_extension_raises_value_error(self, tmp_path):
        sub = tmp_path / "sub.txt"
        rub = tmp_path / "rub.csv"
        sub.write_text("content", encoding="utf-8")
        rub.write_text("a,b", encoding="utf-8")
        with pytest.raises(ValueError, match=r"\.json"):
            validate_submission_files(str(sub), str(rub))

    def test_case_insensitive_extensions(self, tmp_path):
        sub = tmp_path / "sub.TXT"
        rub = tmp_path / "rub.JSON"
        sub.write_text("content", encoding="utf-8")
        rub.write_text('{"criteria":[]}', encoding="utf-8")
        result = validate_submission_files(str(sub), str(rub))
        assert result == {"status": "success"}


# ── file_writer ───────────────────────────────────────────────────────────────


class TestWriteFeedbackReport:
    def test_creates_file(self, tmp_path):
        dest = tmp_path / "report.md"
        write_feedback_report("# Report", str(dest))
        assert dest.exists()

    def test_writes_content(self, tmp_path):
        dest = tmp_path / "report.md"
        write_feedback_report("# Hello\n\nWorld", str(dest))
        assert dest.read_text(encoding="utf-8") == "# Hello\n\nWorld"

    def test_creates_parent_directories(self, tmp_path):
        dest = tmp_path / "a" / "b" / "c" / "report.md"
        write_feedback_report("content", str(dest))
        assert dest.exists()

    def test_overwrites_existing_file(self, tmp_path):
        dest = tmp_path / "report.md"
        dest.write_text("old", encoding="utf-8")
        write_feedback_report("new content", str(dest))
        assert dest.read_text(encoding="utf-8") == "new content"

    def test_returns_resolved_absolute_path(self, tmp_path):
        dest = tmp_path / "report.md"
        returned = write_feedback_report("content", str(dest))
        assert returned == str(dest.resolve())

    def test_writes_empty_content(self, tmp_path):
        dest = tmp_path / "report.md"
        write_feedback_report("", str(dest))
        assert dest.read_text(encoding="utf-8") == ""


# ── score_calculator ──────────────────────────────────────────────────────────


class TestCalculateTotalScore:
    def _criteria(self, *scores_and_maxes):
        return [{"score": s, "max_score": m} for s, m in scores_and_maxes]

    def test_correct_sum(self):
        result = calculate_total_score(self._criteria((3, 5), (4, 5)), 10)
        assert result["total_score"] == 7

    def test_percentage_calculation(self):
        result = calculate_total_score(self._criteria((9, 10)), 10)
        assert result["percentage"] == 90.0

    def test_percentage_rounded_to_two_decimals(self):
        result = calculate_total_score(self._criteria((1, 3)), 3)
        assert result["percentage"] == 33.33

    def test_grade_a_at_90_percent(self):
        result = calculate_total_score(self._criteria((9, 10)), 10)
        assert result["grade"] == "A"

    def test_grade_a_at_100_percent(self):
        result = calculate_total_score(self._criteria((10, 10)), 10)
        assert result["grade"] == "A"

    def test_grade_b_at_75_percent(self):
        result = calculate_total_score(self._criteria((75, 100)), 100)
        assert result["grade"] == "B"

    def test_grade_b_below_90(self):
        result = calculate_total_score(self._criteria((80, 100)), 100)
        assert result["grade"] == "B"

    def test_grade_c_at_60_percent(self):
        result = calculate_total_score(self._criteria((60, 100)), 100)
        assert result["grade"] == "C"

    def test_grade_d_at_50_percent(self):
        result = calculate_total_score(self._criteria((50, 100)), 100)
        assert result["grade"] == "D"

    def test_grade_f_below_50(self):
        result = calculate_total_score(self._criteria((49, 100)), 100)
        assert result["grade"] == "F"

    def test_grade_f_at_zero(self):
        result = calculate_total_score(self._criteria((0, 10)), 10)
        assert result["grade"] == "F"

    def test_zero_total_marks_gives_zero_percent(self):
        result = calculate_total_score(self._criteria((5, 10)), 0)
        assert result["percentage"] == 0.0
        assert result["grade"] == "F"

    def test_empty_criteria_gives_zero_score(self):
        result = calculate_total_score([], 10)
        assert result["total_score"] == 0
        assert result["percentage"] == 0.0

    def test_result_keys(self):
        result = calculate_total_score(self._criteria((5, 10)), 10)
        assert set(result.keys()) == {"total_score", "percentage", "grade"}

    def test_multiple_criteria_summed(self):
        criteria = self._criteria((4, 5), (3, 5), (2, 5), (5, 5))
        result = calculate_total_score(criteria, 20)
        assert result["total_score"] == 14
        assert result["percentage"] == 70.0
        assert result["grade"] == "C"


# ── logger ────────────────────────────────────────────────────────────────────


class TestLogAgentAction:
    def test_returns_entry_dict(self, tmp_path):
        log_file = tmp_path / "trace.log"
        with patch("mas.tools.logger._LOG_FILE", log_file):
            entry = log_agent_action(
                session_id="s1",
                agent="coordinator",
                action="validate",
                inputs={"path": "/tmp/sub.txt"},
                outputs={"status": "success"},
            )
        assert isinstance(entry, dict)

    def test_entry_contains_required_fields(self, tmp_path):
        log_file = tmp_path / "trace.log"
        with patch("mas.tools.logger._LOG_FILE", log_file):
            entry = log_agent_action(
                session_id="s1",
                agent="research",
                action="read_files",
                inputs={"sub": "a.txt"},
                outputs={"status": "ok"},
            )
        for field in ("timestamp", "session_id", "agent", "action", "inputs", "outputs"):
            assert field in entry

    def test_entry_values_match_args(self, tmp_path):
        log_file = tmp_path / "trace.log"
        with patch("mas.tools.logger._LOG_FILE", log_file):
            entry = log_agent_action(
                session_id="abc-123",
                agent="analysis",
                action="score_criteria",
                inputs={"count": 3},
                outputs={"total": 15},
            )
        assert entry["session_id"] == "abc-123"
        assert entry["agent"] == "analysis"
        assert entry["action"] == "score_criteria"
        assert entry["inputs"] == {"count": 3}
        assert entry["outputs"] == {"total": 15}

    def test_writes_to_log_file(self, tmp_path):
        log_file = tmp_path / "trace.log"
        with patch("mas.tools.logger._LOG_FILE", log_file):
            log_agent_action(
                session_id="s1",
                agent="report",
                action="write",
                inputs={},
                outputs={},
            )
        assert log_file.exists()
        assert log_file.stat().st_size > 0

    def test_appends_multiple_entries(self, tmp_path):
        log_file = tmp_path / "trace.log"
        with patch("mas.tools.logger._LOG_FILE", log_file):
            for i in range(3):
                log_agent_action(
                    session_id=f"s{i}",
                    agent="coordinator",
                    action="validate",
                    inputs={},
                    outputs={},
                )
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3

    def test_each_line_is_valid_json(self, tmp_path):
        log_file = tmp_path / "trace.log"
        with patch("mas.tools.logger._LOG_FILE", log_file):
            for i in range(2):
                log_agent_action(
                    session_id=f"s{i}",
                    agent="coordinator",
                    action="validate",
                    inputs={"i": i},
                    outputs={},
                )
        for line in log_file.read_text(encoding="utf-8").strip().splitlines():
            parsed = json.loads(line)
            assert "timestamp" in parsed

    def test_timestamp_is_iso_format(self, tmp_path):
        from datetime import datetime

        log_file = tmp_path / "trace.log"
        with patch("mas.tools.logger._LOG_FILE", log_file):
            entry = log_agent_action(
                session_id="s1",
                agent="coordinator",
                action="validate",
                inputs={},
                outputs={},
            )
        # Should parse without error
        datetime.fromisoformat(entry["timestamp"])

    def test_prints_to_stdout(self, tmp_path, capsys):
        log_file = tmp_path / "trace.log"
        with patch("mas.tools.logger._LOG_FILE", log_file):
            log_agent_action(
                session_id="s1",
                agent="analysis",
                action="score_criteria",
                inputs={},
                outputs={},
            )
        captured = capsys.readouterr()
        assert "analysis" in captured.out.lower()
        assert "score_criteria" in captured.out.lower()


# ── write_analysis_report ─────────────────────────────────────────────────────


class TestWriteAnalysisReport:
    _PAST = [
        {"session_id": "s1", "timestamp": "2025-01-01T00:00:00+00:00", "total_score": 15, "grade": "B"},
        {"session_id": "s2", "timestamp": "2025-02-01T00:00:00+00:00", "total_score": 18, "grade": "A"},
    ]

    def test_creates_file(self, tmp_path):
        dest = tmp_path / "analysis.md"
        write_analysis_report(self._PAST, "Improving.", "S001", str(dest))
        assert dest.exists()

    def test_returns_resolved_path(self, tmp_path):
        dest = tmp_path / "analysis.md"
        returned = write_analysis_report(self._PAST, "Improving.", "S001", str(dest))
        assert returned == str(dest.resolve())

    def test_contains_student_id(self, tmp_path):
        dest = tmp_path / "analysis.md"
        write_analysis_report(self._PAST, "", "IT21000001", str(dest))
        content = dest.read_text(encoding="utf-8")
        assert "IT21000001" in content

    def test_contains_historical_scores(self, tmp_path):
        dest = tmp_path / "analysis.md"
        write_analysis_report(self._PAST, "", "S001", str(dest))
        content = dest.read_text(encoding="utf-8")
        assert "15" in content
        assert "18" in content

    def test_contains_progression_insights(self, tmp_path):
        dest = tmp_path / "analysis.md"
        write_analysis_report(self._PAST, "Student is improving.", "S001", str(dest))
        content = dest.read_text(encoding="utf-8")
        assert "Student is improving." in content

    def test_no_history_shows_placeholder(self, tmp_path):
        dest = tmp_path / "analysis.md"
        write_analysis_report([], "", "S001", str(dest))
        content = dest.read_text(encoding="utf-8")
        assert "No historical records" in content

    def test_no_insights_section_omitted(self, tmp_path):
        dest = tmp_path / "analysis.md"
        write_analysis_report(self._PAST, "", "S001", str(dest))
        content = dest.read_text(encoding="utf-8")
        assert "Progression Insights" not in content

    def test_creates_parent_directories(self, tmp_path):
        dest = tmp_path / "a" / "b" / "analysis.md"
        write_analysis_report([], "", "S001", str(dest))
        assert dest.exists()


# ── write_marking_sheet ───────────────────────────────────────────────────────


class TestWriteMarkingSheet:
    _CRITERIA = [
        {"name": "Accuracy", "criterion_id": "C1", "score": 4, "max_score": 5,
         "justification": "Good definition."},
        {"name": "Clarity", "criterion_id": "C2", "score": 3, "max_score": 5,
         "justification": "Mostly clear."},
    ]

    def _write(self, tmp_path, **kwargs):
        defaults = dict(
            student_id="S001",
            student_name="Alice",
            module="CTSE",
            assignment="Cloud Basics",
            scored_criteria=self._CRITERIA,
            total_score=7,
            total_marks=10,
            grade="C",
            output_path=str(tmp_path / "sheet.md"),
        )
        defaults.update(kwargs)
        return write_marking_sheet(**defaults)

    def test_creates_file(self, tmp_path):
        dest = tmp_path / "sheet.md"
        write_marking_sheet(
            student_id="S001", student_name="Alice", module="CTSE",
            assignment="Cloud", scored_criteria=self._CRITERIA,
            total_score=7, total_marks=10, grade="C", output_path=str(dest),
        )
        assert dest.exists()

    def test_returns_resolved_path(self, tmp_path):
        dest = tmp_path / "sheet.md"
        returned = write_marking_sheet(
            student_id="S001", student_name="", module="CTSE",
            assignment="Cloud", scored_criteria=self._CRITERIA,
            total_score=7, total_marks=10, grade="C", output_path=str(dest),
        )
        assert returned == str(dest.resolve())

    def test_contains_student_id(self, tmp_path):
        path = self._write(tmp_path)
        content = Path(path).read_text(encoding="utf-8")
        assert "S001" in content

    def test_contains_student_name(self, tmp_path):
        path = self._write(tmp_path)
        content = Path(path).read_text(encoding="utf-8")
        assert "Alice" in content

    def test_name_omitted_when_empty(self, tmp_path):
        dest = tmp_path / "sheet.md"
        write_marking_sheet(
            student_id="S001", student_name="", module="CTSE",
            assignment="Cloud", scored_criteria=self._CRITERIA,
            total_score=7, total_marks=10, grade="C", output_path=str(dest),
        )
        content = dest.read_text(encoding="utf-8")
        assert "Student Name" not in content

    def test_contains_criterion_scores(self, tmp_path):
        path = self._write(tmp_path)
        content = Path(path).read_text(encoding="utf-8")
        assert "Accuracy" in content
        assert "Clarity" in content

    def test_contains_grade(self, tmp_path):
        path = self._write(tmp_path)
        content = Path(path).read_text(encoding="utf-8")
        assert "C" in content

    def test_contains_percentage(self, tmp_path):
        path = self._write(tmp_path)
        content = Path(path).read_text(encoding="utf-8")
        assert "70.00%" in content

    def test_creates_parent_directories(self, tmp_path):
        dest = tmp_path / "deep" / "sheet.md"
        write_marking_sheet(
            student_id="S001", student_name="", module="CTSE",
            assignment="Cloud", scored_criteria=[], total_score=0,
            total_marks=10, grade="F", output_path=str(dest),
        )
        assert dest.exists()


# ── pdf_processor ─────────────────────────────────────────────────────────────


class TestConvertPdfToMarkdown:
    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            convert_pdf_to_markdown(str(tmp_path / "missing.pdf"))

    def test_non_pdf_extension_raises_value_error(self, tmp_path):
        txt = tmp_path / "doc.txt"
        txt.write_text("content", encoding="utf-8")
        with pytest.raises(ValueError, match=r"Expected a \.pdf file"):
            convert_pdf_to_markdown(str(txt))

    def test_successful_conversion(self, tmp_path):
        import pymupdf  # type: ignore[import]

        pdf_path = tmp_path / "test.pdf"
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((50, 100), "Student ID: IT21000001\nHello World")
        doc.save(str(pdf_path))
        doc.close()

        result = convert_pdf_to_markdown(str(pdf_path))
        assert isinstance(result, str)
        assert len(result) > 0

    def test_conversion_contains_text(self, tmp_path):
        import pymupdf  # type: ignore[import]

        pdf_path = tmp_path / "test.pdf"
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((50, 100), "Student ID: IT21000001")
        doc.save(str(pdf_path))
        doc.close()

        result = convert_pdf_to_markdown(str(pdf_path))
        assert "IT21000001" in result

