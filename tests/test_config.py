"""Unit tests for the config module."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch


# ── helpers ───────────────────────────────────────────────────────────────────

def _reload_settings(env: dict[str, str]):
    """Reload settings with the given environment variables patched in."""
    from mas.config import _load_settings

    with patch.dict(os.environ, env, clear=False):
        return _load_settings()


# ── default values ────────────────────────────────────────────────────────────


class TestSettingsDefaults:
    def test_default_model_name(self):
        s = _reload_settings({})
        assert s.model_name == "phi4-mini:3.8b-q4_K_M"

    def test_default_ollama_base_url(self):
        s = _reload_settings({})
        assert s.ollama_base_url == "http://localhost:11434"

    def test_default_db_path_ends_with_students_db(self):
        s = _reload_settings({})
        assert s.db_path.endswith("students.db")

    def test_default_log_file(self):
        s = _reload_settings({})
        assert s.log_file == "agent_trace.log"

    def test_default_submission_path_ends_with_submission_txt(self):
        s = _reload_settings({})
        assert s.submission_path.endswith("submission.txt")

    def test_default_rubric_path_ends_with_rubric_json(self):
        s = _reload_settings({})
        assert s.rubric_path.endswith("rubric.json")

    def test_default_output_path_ends_with_feedback_report_md(self):
        s = _reload_settings({})
        assert s.output_path.endswith("feedback_report.md")

    def test_default_db_path_contains_data_directory(self):
        s = _reload_settings({})
        assert "data" in s.db_path

    def test_default_paths_are_absolute(self):
        s = _reload_settings({})
        assert Path(s.db_path).is_absolute()
        assert Path(s.submission_path).is_absolute()
        assert Path(s.rubric_path).is_absolute()
        assert Path(s.output_path).is_absolute()


# ── env-var overrides ─────────────────────────────────────────────────────────


class TestSettingsEnvOverrides:
    def test_model_name_override(self):
        s = _reload_settings({"AUTOMARK_MODEL_NAME": "llama3:8b"})
        assert s.model_name == "llama3:8b"

    def test_ollama_base_url_override(self):
        s = _reload_settings({"AUTOMARK_OLLAMA_BASE_URL": "http://ollama:11434"})
        assert s.ollama_base_url == "http://ollama:11434"

    def test_db_path_override(self):
        s = _reload_settings({"AUTOMARK_DB_PATH": "/tmp/custom.db"})
        assert s.db_path == "/tmp/custom.db"

    def test_log_file_override(self):
        s = _reload_settings({"AUTOMARK_LOG_FILE": "/var/log/automark.log"})
        assert s.log_file == "/var/log/automark.log"

    def test_submission_path_override(self):
        s = _reload_settings({"AUTOMARK_SUBMISSION_PATH": "/tmp/sub.txt"})
        assert s.submission_path == "/tmp/sub.txt"

    def test_rubric_path_override(self):
        s = _reload_settings({"AUTOMARK_RUBRIC_PATH": "/tmp/rubric.json"})
        assert s.rubric_path == "/tmp/rubric.json"

    def test_output_path_override(self):
        s = _reload_settings({"AUTOMARK_OUTPUT_PATH": "/tmp/report.md"})
        assert s.output_path == "/tmp/report.md"

    def test_empty_env_var_falls_back_to_default(self):
        s = _reload_settings({"AUTOMARK_MODEL_NAME": ""})
        assert s.model_name == "phi4-mini:3.8b-q4_K_M"

    def test_multiple_overrides_applied_together(self):
        s = _reload_settings(
            {
                "AUTOMARK_MODEL_NAME": "mistral:7b",
                "AUTOMARK_DB_PATH": "/data/prod.db",
            }
        )
        assert s.model_name == "mistral:7b"
        assert s.db_path == "/data/prod.db"


# ── immutability ──────────────────────────────────────────────────────────────


class TestSettingsImmutability:
    def test_settings_is_frozen(self):
        import pytest

        s = _reload_settings({})
        with pytest.raises((AttributeError, TypeError)):
            s.model_name = "changed"  # type: ignore[misc]


# ── module-level singleton ────────────────────────────────────────────────────


class TestModuleSingleton:
    def test_settings_singleton_is_settings_instance(self):
        from mas.config import Settings, settings

        assert isinstance(settings, Settings)

    def test_settings_singleton_has_all_fields(self):
        from mas.config import settings

        for field in (
            "model_name",
            "ollama_base_url",
            "db_path",
            "log_file",
            "submission_path",
            "rubric_path",
            "output_path",
        ):
            assert hasattr(settings, field)
