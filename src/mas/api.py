"""FastAPI REST wrapper for the AutoMark LangGraph grading pipeline.

Endpoints
---------
GET  /health                        – liveness check
POST /grade                         – run the grading pipeline
GET  /sessions/{session_id}/logs    – retrieve trace log entries for a session

Security note
-------------
File paths supplied by clients are resolved to absolute paths and validated
against a configurable base directory (``AUTOMARK_DATA_BASE_DIR``, defaulting
to the project data folder) to prevent path-traversal attacks.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mas.config import settings
from mas.graph import build_graph
from mas.state import AgentState

# Project root: three levels up from src/mas/api.py → src/mas → src → project root
_PROJECT_ROOT = Path(__file__).parent.parent.parent

# Allowed base directory for submission and rubric files.
# Defaults to the project data directory; override with AUTOMARK_DATA_BASE_DIR.
_DATA_BASE_DIR: Path = Path(
    os.environ.get("AUTOMARK_DATA_BASE_DIR", "")
    or _PROJECT_ROOT / "data"
).resolve()

# Output directory for grading results (per-session files are created here).
_OUTPUT_DIR: Path = Path(settings.output_path).parent.resolve()

# ── Pydantic models ────────────────────────────────────────────────────────────


class GradeRequest(BaseModel):
    """Request body for POST /grade."""

    submission_path: str = Field(
        description="Relative path to the student submission file (.txt or .pdf), relative to the server data directory."
    )
    rubric_path: str = Field(
        description="Relative path to the grading rubric JSON file, relative to the server data directory."
    )


class CriterionResult(BaseModel):
    """Score and justification for a single rubric criterion."""

    criterion_id: str
    name: str
    score: float
    max_score: float
    justification: str


class GradeResponse(BaseModel):
    """Response body for POST /grade."""

    session_id: str
    student_id: str
    student_name: str
    total_score: float
    grade: str
    criteria: list[CriterionResult]
    feedback_report: str
    output_filepath: str


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    service: str


class LogEntry(BaseModel):
    """A single structured log entry from agent_trace.log."""

    timestamp: str
    session_id: str
    agent: str
    action: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]


# ── Application ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AutoMark API",
    description="REST API wrapper for the AutoMark LangGraph auto-grading pipeline.",
    version="0.1.0",
)

# Build the graph once at startup and reuse it across all requests.
# LangGraph's CompiledStateGraph.invoke() creates a fresh state machine per
# call and does not share mutable state between invocations, so this is safe
# for concurrent requests.
_graph = build_graph()

# Allowlist pattern: relative paths with alphanumerics, dots, hyphens,
# underscores, and forward slashes only.  Blocks traversal sequences and
# shell-special characters before the path ever touches the filesystem.
_SAFE_PATH_RE = re.compile(r"^[A-Za-z0-9._/\- ]+$")


# ── Path validation helper ─────────────────────────────────────────────────────


def _resolve_safe_path(raw: str) -> Path:
    """Join *raw* (treated as a relative filename) with *_DATA_BASE_DIR* and resolve.

    Only relative paths matching ``_SAFE_PATH_RE`` are accepted.  The resolved
    path is also checked against *_DATA_BASE_DIR* to prevent path-traversal
    attacks.  Raises ``HTTPException(422)`` for invalid or dangerous inputs.
    """
    if not _SAFE_PATH_RE.match(raw):
        raise HTTPException(
            status_code=422,
            detail=(
                "Path contains disallowed characters. "
                "Supply a simple relative filename (letters, digits, dots, hyphens, underscores)."
            ),
        )
    if ".." in Path(raw).parts:
        raise HTTPException(
            status_code=422,
            detail="Path traversal sequences ('..') are not allowed.",
        )
    # Join with the trusted base directory — raw never reaches the FS alone.
    resolved = (_DATA_BASE_DIR / raw).resolve()
    try:
        resolved.relative_to(_DATA_BASE_DIR)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Path '{raw}' escapes the allowed data directory. "
                f"Supply a filename inside the data folder."
            ),
        )
    return resolved


# ── Endpoints ──────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, summary="Health check")
def health() -> HealthResponse:
    """Return a simple liveness status confirming the service is running."""
    return HealthResponse(status="ok", service="automark-api")


@app.post("/grade", response_model=GradeResponse, summary="Grade a submission")
def grade(request: GradeRequest) -> GradeResponse:
    """Validate file paths, run the LangGraph pipeline, and return the graded result.

    Raises
    ------
    422 – if either file path is outside the allowed data directory or does not exist.
    500 – if the pipeline encounters an internal error.
    """
    submission = _resolve_safe_path(request.submission_path)
    rubric = _resolve_safe_path(request.rubric_path)

    if not submission.exists():
        raise HTTPException(
            status_code=422,
            detail=f"Submission file not found: {request.submission_path}",
        )
    if not rubric.exists():
        raise HTTPException(
            status_code=422,
            detail=f"Rubric file not found: {request.rubric_path}",
        )

    session_id = str(uuid.uuid4())

    # Generate session-specific output paths to prevent concurrent-request conflicts.
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    session_output = str(_OUTPUT_DIR / f"feedback_report_{session_id}.md")
    session_marking = str(_OUTPUT_DIR / f"marking_sheet_{session_id}.md")

    initial_state: AgentState = {
        "submission_path": str(submission),
        "rubric_path": str(rubric),
        "db_path": settings.db_path,
        "output_filepath": session_output,
        "marking_sheet_path": session_marking,
        "session_id": session_id,
        "agent_logs": [],
    }

    try:
        final_state: AgentState = _graph.invoke(initial_state)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline execution failed: {exc}",
        ) from exc

    if final_state.get("error"):
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {final_state['error']}",
        )

    criteria: list[CriterionResult] = [
        CriterionResult(
            criterion_id=c["criterion_id"],
            name=c["name"],
            score=c["score"],
            max_score=c["max_score"],
            justification=c["justification"],
        )
        for c in (final_state.get("scored_criteria") or [])
    ]

    return GradeResponse(
        session_id=session_id,
        student_id=final_state.get("student_id") or "",
        student_name=final_state.get("student_name") or "",
        total_score=final_state.get("total_score") or 0.0,
        grade=final_state.get("grade") or "N/A",
        criteria=criteria,
        feedback_report=final_state.get("final_report") or "",
        output_filepath=final_state.get("output_filepath") or "",
    )


@app.get(
    "/sessions/{session_id}/logs",
    response_model=list[LogEntry],
    summary="Retrieve session trace logs",
)
def session_logs(session_id: str) -> list[LogEntry]:
    """Read agent_trace.log and return all entries matching *session_id*.

    Raises
    ------
    404 – if no log entries are found for the given session ID.
    500 – if the log file cannot be read.
    """
    log_path = Path(settings.log_file)

    if not log_path.exists():
        raise HTTPException(
            status_code=500,
            detail="Log file is unavailable; no grading runs have been recorded yet.",
        )

    entries: list[LogEntry] = []
    try:
        with log_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record: dict[str, Any] = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("session_id") == session_id:
                    entries.append(
                        LogEntry(
                            timestamp=record.get("timestamp", ""),
                            session_id=record.get("session_id", ""),
                            agent=record.get("agent", ""),
                            action=record.get("action", ""),
                            inputs=record.get("inputs", {}),
                            outputs=record.get("outputs", {}),
                        )
                    )
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read log file: {exc}",
        ) from exc

    if not entries:
        raise HTTPException(
            status_code=404,
            detail=f"No log entries found for session '{session_id}'.",
        )

    return entries

# ── Development entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    # Bind to localhost only for local development.
    # Use `uvicorn mas.api:app --host 0.0.0.0` (or the Docker CMD) for network access.
    _port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("mas.api:app", host="127.0.0.1", port=_port, reload=True)
