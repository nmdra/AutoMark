"""FastAPI REST wrapper for the AutoMark LangGraph grading pipeline.

Endpoints
---------
GET  /health                        – liveness check
POST /grade                         – run the grading pipeline
GET  /sessions/{session_id}/logs    – retrieve trace log entries for a session
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mas.config import settings
from mas.graph import build_graph
from mas.state import AgentState

# ── Pydantic models ────────────────────────────────────────────────────────────


class GradeRequest(BaseModel):
    """Request body for POST /grade."""

    submission_path: str = Field(
        description="Absolute or relative path to the student submission file (.txt or .pdf)."
    )
    rubric_path: str = Field(
        description="Absolute or relative path to the grading rubric JSON file."
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
    422 – if either file path does not exist on disk.
    500 – if the pipeline encounters an internal error.
    """
    submission = Path(request.submission_path)
    rubric = Path(request.rubric_path)

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

    initial_state: AgentState = {
        "submission_path": str(submission),
        "rubric_path": str(rubric),
        "db_path": settings.db_path,
        "output_filepath": settings.output_path,
        "session_id": session_id,
        "agent_logs": [],
    }

    try:
        graph = build_graph()
        final_state: AgentState = graph.invoke(initial_state)
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
            status_code=404,
            detail=f"No log entries found for session '{session_id}'.",
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

    uvicorn.run("mas.api:app", host="0.0.0.0", port=8000, reload=True)
