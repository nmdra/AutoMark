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

AUTOMARK_OUTPUT_PATH
    Default output path for the generated Markdown feedback report.
    Default: ``<project_root>/output/feedback_report.md``

AUTOMARK_ANALYSIS_REPORT_PATH
    Default output path for the performance analysis report.
    Default: ``<project_root>/output/analysis_report.md``

AUTOMARK_MARKING_SHEET_PATH
    Default output path for the marking sheet report.
    Default: ``<project_root>/output/marking_sheet.md``

AUTOMARK_NUM_CTX
    Ollama context window size (tokens).  Smaller values reduce KV-cache
    allocation and speed up prefill; increase only when submissions are long.
    Default: ``4096``

AUTOMARK_NUM_PREDICT
    Maximum number of tokens the model will generate per response.
    Capping this prevents runaway generation latency.
    Default: ``512``

AUTOMARK_LLM_REPORT_ENABLED
    Set to ``false`` (case-insensitive) to skip the prose-report LLM call and
    use the deterministic template instead.  Eliminates one full LLM round-trip.
    Default: ``true``

AUTOMARK_SUBMISSION_MAX_CHARS
    Maximum characters of submission text sent to the analysis LLM.
    Longer text is truncated before the prompt is built, reducing context size.
    Default: ``8000``

AUTOMARK_MIN_REPORTS_FOR_INSIGHTS
    Minimum number of *past* reports required before the historical-insights
    LLM call is made.  Set to ``2`` to require at least two prior sessions
    before generating trend commentary.
    Default: ``1``

Ollama parallel-request note
-----------------------------
The finalize agent runs the historical-insights LLM call and the feedback-
report prose LLM call concurrently using a ``ThreadPoolExecutor``.  To
fully benefit from this, configure Ollama with ``OLLAMA_NUM_PARALLEL >= 2``
on the Ollama server side.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Project root: three levels up from src/mas/config.py
_PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load .env from the project root (no-op when the file is absent)
load_dotenv(_PROJECT_ROOT / ".env")


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

    # ── Default output paths ──────────────────────────────────────────────────
    output_path: str
    analysis_report_path: str
    marking_sheet_path: str

    # ── LLM performance knobs ─────────────────────────────────────────────────
    num_ctx: int
    num_predict: int
    llm_report_enabled: bool
    submission_max_chars: int
    min_reports_for_insights: int


def _load_settings() -> Settings:
    root = _PROJECT_ROOT
    return Settings(
        model_name=_env("AUTOMARK_MODEL_NAME", "phi4-mini:3.8b-q4_K_M"),
        ollama_base_url=_env("AUTOMARK_OLLAMA_BASE_URL", "http://localhost:11434"),
        db_path=_env("AUTOMARK_DB_PATH", str(root / "data" / "students.db")),
        log_file=_env("AUTOMARK_LOG_FILE", "agent_trace.log"),
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
        num_ctx=int(_env("AUTOMARK_NUM_CTX", "4096")),
        num_predict=int(_env("AUTOMARK_NUM_PREDICT", "512")),
        llm_report_enabled=_env(
            "AUTOMARK_LLM_REPORT_ENABLED", "true"
        ).lower() not in ("false", "0", "no"),
        submission_max_chars=int(_env("AUTOMARK_SUBMISSION_MAX_CHARS", "8000")),
        min_reports_for_insights=int(
            _env("AUTOMARK_MIN_REPORTS_FOR_INSIGHTS", "1")
        ),
    )


settings: Settings = _load_settings()
