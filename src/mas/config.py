"""Application-wide configuration for AutoMark.

Settings are resolved once at import time.  Each value can be overridden by
setting the corresponding environment variable before starting the process.

Environment variables
---------------------
AUTOMARK_MODEL_NAME
    Ollama model identifier used for all LLM calls.
    Default: ``phi4-mini:3.8b-q4_K_M``

AUTOMARK_OLLAMA_BASE_URL
    Base URL for the Ollama HTTP API.
    Default: ``http://localhost:11434``

AUTOMARK_DB_PATH
    Absolute or relative path to the SQLite database file.
    Default: ``<project_root>/data/students.db``

AUTOMARK_LOG_FILE
    Path to the structured JSON agent trace log.
    Default: ``agent_trace.log``  (relative to the working directory)

AUTOMARK_SUBMISSION_PATH
    Default path to the student submission text file used by the CLI.
    Default: ``<project_root>/data/submission.txt``

AUTOMARK_RUBRIC_PATH
    Default path to the grading rubric JSON file used by the CLI.
    Default: ``<project_root>/data/rubric.json``

AUTOMARK_OUTPUT_PATH
    Default output path for the generated Markdown feedback report.
    Default: ``<project_root>/output/feedback_report.md``

AUTOMARK_ANALYSIS_REPORT_PATH
    Default output path for the performance analysis report.
    Default: ``<project_root>/output/analysis_report.md``

AUTOMARK_MARKING_SHEET_PATH
    Default output path for the marking sheet report.
    Default: ``<project_root>/output/marking_sheet.md``
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Project root: three levels up from src/mas/config.py
_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _env(key: str, default: str) -> str:
    """Return the env-var value, or *default* when the variable is unset or empty."""
    return os.environ.get(key) or default


@dataclass(frozen=True)
class Settings:
    """Immutable application settings resolved from environment variables."""

    # ── LLM / Ollama ──────────────────────────────────────────────────────────
    model_name: str
    ollama_base_url: str

    # ── Storage ───────────────────────────────────────────────────────────────
    db_path: str
    log_file: str

    # ── Default data paths (used by the CLI entry point) ──────────────────────
    submission_path: str
    rubric_path: str
    output_path: str
    analysis_report_path: str
    marking_sheet_path: str


def _load_settings() -> Settings:
    root = _PROJECT_ROOT
    return Settings(
        model_name=_env("AUTOMARK_MODEL_NAME", "phi4-mini:3.8b-q4_K_M"),
        ollama_base_url=_env("AUTOMARK_OLLAMA_BASE_URL", "http://localhost:11434"),
        db_path=_env("AUTOMARK_DB_PATH", str(root / "data" / "students.db")),
        log_file=_env("AUTOMARK_LOG_FILE", "agent_trace.log"),
        submission_path=_env(
            "AUTOMARK_SUBMISSION_PATH", str(root / "data" / "submission.txt")
        ),
        rubric_path=_env("AUTOMARK_RUBRIC_PATH", str(root / "data" / "rubric.json")),
        output_path=_env(
            "AUTOMARK_OUTPUT_PATH", str(root / "output" / "feedback_report.md")
        ),
        analysis_report_path=_env(
            "AUTOMARK_ANALYSIS_REPORT_PATH",
            str(root / "output" / "analysis_report.md"),
        ),
        marking_sheet_path=_env(
            "AUTOMARK_MARKING_SHEET_PATH",
            str(root / "output" / "marking_sheet.md"),
        ),
    )


settings: Settings = _load_settings()
