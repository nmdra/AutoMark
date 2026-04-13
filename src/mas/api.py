"""FastAPI REST wrapper for the AutoMark LangGraph grading pipeline."""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from queue import Full, Queue
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import fitz
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

from mas.config import settings
from mas.graph import build_graph
from mas.state import AgentState
from mas.tools.db_manager import (
    ITEM_STATUS_FAILED,
    ITEM_STATUS_QUEUED,
    JOB_STATUS_QUEUED,
    JOB_STATUS_RUNNING,
    JOB_STATUS_CANCELLED,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    create_job,
    get_job,
    get_job_artifact,
    get_job_items,
    init_db,
    is_job_cancel_requested,
    list_jobs,
    mark_job_failed,
    mark_job_item_completed,
    mark_job_item_failed,
    mark_job_item_running,
    mark_job_running,
    mark_remaining_items_cancelled,
    refresh_job_progress,
    request_job_cancel,
    save_job_artifact,
)

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
_EXPORT_DIR: Path = (_OUTPUT_DIR / "exports").resolve()

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
    percentage: float
    grade: str
    summary: str
    criteria: list[CriterionResult]
    output_filepath: str
    marking_sheet_path: str


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


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class ExportFormat(str, Enum):
    csv = "csv"
    json = "json"
    pdf = "pdf"


class ApiErrorPayload(BaseModel):
    code: str
    message: str


class BatchGradeItemRequest(BaseModel):
    submission_path: str
    rubric_path: str
    correlation_id: str | None = None


class BatchGradeRequest(BaseModel):
    items: list[BatchGradeItemRequest] = Field(min_length=1)
    max_retries: int | None = Field(default=None, ge=0, le=5)

    @model_validator(mode="after")
    def _validate_batch_size(self) -> "BatchGradeRequest":
        if len(self.items) > settings.batch_max_items:
            raise ValueError(
                f"Batch size {len(self.items)} exceeds limit {settings.batch_max_items}."
            )
        return self


class BatchGradeAcceptedResponse(BaseModel):
    job_id: str
    status: JobStatus
    total_items: int
    accepted_items: int
    rejected_items: int


class JobProgress(BaseModel):
    total: int
    queued: int
    running: int
    completed: int
    failed: int
    cancelled: int


class BatchJobItemResponse(BaseModel):
    item_index: int
    correlation_id: str
    submission_path: str
    rubric_path: str
    status: str
    attempts: int
    session_id: str | None = None
    student_id: str | None = None
    student_name: str | None = None
    total_score: float | None = None
    percentage: float | None = None
    grade: str | None = None
    summary: str | None = None
    criteria: list[dict[str, Any]] = Field(default_factory=list)
    output_filepath: str | None = None
    marking_sheet_path: str | None = None
    error: str | None = None
    started_at: str | None = None
    completed_at: str | None = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str
    updated_at: str
    started_at: str | None = None
    completed_at: str | None = None
    cancel_requested: bool
    max_retries: int
    error: str | None = None
    progress: JobProgress
    items: list[BatchJobItemResponse]


class JobListEntry(BaseModel):
    job_id: str
    status: JobStatus
    total_items: int
    created_at: str
    updated_at: str
    started_at: str | None = None
    completed_at: str | None = None
    cancel_requested: bool
    max_retries: int
    error: str | None = None


class JobListResponse(BaseModel):
    jobs: list[JobListEntry]
    count: int


class JobCancelResponse(BaseModel):
    job_id: str
    status: JobStatus
    cancel_requested: bool


class JobArtifactResponse(BaseModel):
    artifact_id: str
    job_id: str
    format: ExportFormat
    file_path: str
    size_bytes: int
    created_at: str


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

# Initialise the SQLite database once at startup so that every subsequent
# save_report call avoids the CREATE TABLE / PRAGMA overhead.
init_db(settings.db_path)

# Allowlist pattern: relative paths with alphanumerics, dots, hyphens,
# underscores, and forward slashes only.  Blocks traversal sequences and
# shell-special characters before the path ever touches the filesystem.
_SAFE_PATH_RE = re.compile(r"^[A-Za-z0-9._/\- ]+$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _build_unique_paths(student_id: str, student_name: str) -> tuple[str, str]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", student_name) if student_name else "unknown"
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", student_id) if student_id else "unknown"
    file_stem = f"{timestamp}_{safe_name}_{safe_id}"

    def _unique_path(directory: Path, stem: str, suffix: str) -> Path:
        candidate = directory / f"{stem}{suffix}"
        counter = 1
        while candidate.exists():
            candidate = directory / f"{stem}_{counter}{suffix}"
            counter += 1
        return candidate

    final_output = str(_unique_path(_OUTPUT_DIR, f"{file_stem}_feedback_report", ".md"))
    final_marking = str(_unique_path(_OUTPUT_DIR, f"{file_stem}_marking_sheet", ".md"))
    return final_output, final_marking


def _replace_path_references(value: Any, old_path: str, new_path: str) -> Any:
    if not old_path or old_path == new_path:
        return value
    if isinstance(value, dict):
        return {
            key: _replace_path_references(item, old_path, new_path)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_path_references(item, old_path, new_path) for item in value]
    if isinstance(value, str) and value == old_path:
        return new_path
    return value


def _run_pipeline_for_paths(submission: Path, rubric: Path, session_id: str) -> dict[str, Any]:
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

    student_id = final_state.get("student_id") or ""
    student_name = final_state.get("student_name") or ""
    final_output, final_marking = _build_unique_paths(student_id, student_name)
    current_output = final_state.get("output_filepath") or ""
    current_marking = final_state.get("marking_sheet_path") or ""

    if current_output and Path(current_output).exists():
        Path(current_output).rename(final_output)
        final_state = _replace_path_references(final_state, current_output, final_output)
        final_state["output_filepath"] = final_output
    else:
        final_output = current_output

    if current_marking and Path(current_marking).exists():
        Path(current_marking).rename(final_marking)
        final_state = _replace_path_references(final_state, current_marking, final_marking)
        final_state["marking_sheet_path"] = final_marking
    else:
        final_marking = current_marking

    return {
        "session_id": session_id,
        "student_id": student_id,
        "student_name": student_name,
        "total_score": final_state.get("total_score") or 0.0,
        "percentage": final_state.get("percentage") or 0.0,
        "grade": final_state.get("grade") or "N/A",
        "summary": final_state.get("summary") or "",
        "criteria": final_state.get("scored_criteria") or [],
        "output_filepath": final_output,
        "marking_sheet_path": final_marking,
    }


def _execute_grade(submission_path: str, rubric_path: str) -> dict[str, Any]:
    submission = _resolve_safe_path(submission_path)
    rubric = _resolve_safe_path(rubric_path)
    if not submission.exists():
        raise HTTPException(
            status_code=422,
            detail=f"Submission file not found: {submission_path}",
        )
    if not rubric.exists():
        raise HTTPException(
            status_code=422,
            detail=f"Rubric file not found: {rubric_path}",
        )
    return _run_pipeline_for_paths(submission, rubric, str(uuid.uuid4()))


def _as_job_status_response(job_id: str) -> JobStatusResponse:
    job = get_job(settings.db_path, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    items = [
        BatchJobItemResponse(
            item_index=item["item_index"],
            correlation_id=item["correlation_id"],
            submission_path=item["submission_path"],
            rubric_path=item["rubric_path"],
            status=item["status"],
            attempts=item["attempts"],
            session_id=item.get("session_id"),
            student_id=item.get("student_id"),
            student_name=item.get("student_name"),
            total_score=item.get("total_score"),
            percentage=item.get("percentage"),
            grade=item.get("grade"),
            summary=item.get("summary"),
            criteria=item.get("criteria") or [],
            output_filepath=item.get("output_filepath"),
            marking_sheet_path=item.get("marking_sheet_path"),
            error=item.get("error"),
            started_at=item.get("started_at"),
            completed_at=item.get("completed_at"),
        )
        for item in get_job_items(settings.db_path, job_id)
    ]
    return JobStatusResponse(
        job_id=job["job_id"],
        status=JobStatus(job["status"]),
        created_at=job["created_at"],
        updated_at=job["updated_at"],
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        cancel_requested=bool(job.get("cancel_requested")),
        max_retries=int(job.get("max_retries") or 0),
        error=job.get("error"),
        progress=JobProgress(**job["progress"]),
        items=items,
    )


def _build_export_content(job: dict[str, Any], items: list[dict[str, Any]], export_format: str) -> bytes:
    if export_format == ExportFormat.json.value:
        payload = {
            "job": {
                "job_id": job["job_id"],
                "status": job["status"],
                "created_at": job["created_at"],
                "updated_at": job["updated_at"],
                "started_at": job.get("started_at"),
                "completed_at": job.get("completed_at"),
                "cancel_requested": bool(job.get("cancel_requested")),
                "max_retries": int(job.get("max_retries") or 0),
                "error": job.get("error"),
                "progress": job["progress"],
            },
            "items": items,
            "generated_at": _utc_now(),
        }
        return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    if export_format == ExportFormat.csv.value:
        import csv
        import io

        buf = io.StringIO()
        writer = csv.DictWriter(
            buf,
            fieldnames=[
                "item_index",
                "correlation_id",
                "submission_path",
                "rubric_path",
                "status",
                "attempts",
                "session_id",
                "student_id",
                "student_name",
                "total_score",
                "percentage",
                "grade",
                "summary",
                "output_filepath",
                "marking_sheet_path",
                "error",
            ],
        )
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "item_index": item.get("item_index"),
                    "correlation_id": item.get("correlation_id"),
                    "submission_path": item.get("submission_path"),
                    "rubric_path": item.get("rubric_path"),
                    "status": item.get("status"),
                    "attempts": item.get("attempts"),
                    "session_id": item.get("session_id"),
                    "student_id": item.get("student_id"),
                    "student_name": item.get("student_name"),
                    "total_score": item.get("total_score"),
                    "percentage": item.get("percentage"),
                    "grade": item.get("grade"),
                    "summary": item.get("summary"),
                    "output_filepath": item.get("output_filepath"),
                    "marking_sheet_path": item.get("marking_sheet_path"),
                    "error": item.get("error"),
                }
            )
        return buf.getvalue().encode("utf-8")

    if export_format == ExportFormat.pdf.value:
        doc = fitz.open()
        lines = [
            f"AutoMark Batch Job Export",
            f"Job ID: {job['job_id']}",
            f"Status: {job['status']}",
            f"Created: {job['created_at']}",
            f"Completed: {job.get('completed_at') or ''}",
            "",
        ]
        progress = job["progress"]
        lines.extend(
            [
                f"Total: {progress['total']}",
                f"Completed: {progress['completed']}",
                f"Failed: {progress['failed']}",
                f"Cancelled: {progress['cancelled']}",
                "",
                "Items:",
            ]
        )
        for item in items:
            lines.append(
                f"- [{item.get('status')}] #{item.get('item_index')} {item.get('correlation_id')} "
                f"(grade={item.get('grade') or 'N/A'}, score={item.get('total_score')}) "
                f"{item.get('error') or ''}"
            )
        page = doc.new_page()
        page.insert_textbox(
            fitz.Rect(36, 36, 559, 806),
            "\n".join(lines),
            fontsize=10,
            fontname="helv",
        )
        return doc.tobytes()

    raise HTTPException(status_code=422, detail=f"Unsupported export format: {export_format}")


class _JobQueueManager:
    def __init__(self) -> None:
        self._queue: Queue[str] = Queue(maxsize=settings.job_queue_max_size)
        self._workers: list[threading.Thread] = []
        for idx in range(settings.job_worker_concurrency):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"automark-job-worker-{idx}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

    def enqueue(self, job_id: str) -> None:
        self._queue.put_nowait(job_id)

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._process_job(job_id)
            finally:
                self._queue.task_done()

    def _process_job(self, job_id: str) -> None:
        try:
            if not mark_job_running(settings.db_path, job_id):
                refresh_job_progress(settings.db_path, job_id)
                return
            for item in get_job_items(settings.db_path, job_id):
                if item.get("status") != ITEM_STATUS_QUEUED:
                    continue
                if is_job_cancel_requested(settings.db_path, job_id):
                    mark_remaining_items_cancelled(settings.db_path, job_id)
                    refresh_job_progress(settings.db_path, job_id)
                    return
                if not mark_job_item_running(settings.db_path, int(item["id"])):
                    continue
                result: dict[str, Any] | None = None
                last_error = ""
                job_row = get_job(settings.db_path, job_id)
                max_retries = int(
                    (job_row or {}).get("max_retries", settings.job_max_retries)
                )
                for _ in range(max_retries + 1):
                    try:
                        result = _execute_grade(
                            submission_path=item["submission_path"],
                            rubric_path=item["rubric_path"],
                        )
                        break
                    except HTTPException as exc:
                        last_error = str(exc.detail)
                    except (RuntimeError, ValueError, OSError) as exc:
                        last_error = str(exc)
                if result is not None:
                    mark_job_item_completed(settings.db_path, int(item["id"]), result)
                else:
                    mark_job_item_failed(
                        settings.db_path,
                        int(item["id"]),
                        last_error or "Unknown grading failure.",
                    )
                refresh_job_progress(settings.db_path, job_id)
            if is_job_cancel_requested(settings.db_path, job_id):
                mark_remaining_items_cancelled(settings.db_path, job_id)
            refresh_job_progress(settings.db_path, job_id)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            mark_job_failed(settings.db_path, job_id, str(exc))


_job_queue = _JobQueueManager()


# ── Endpoints ──────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, summary="Health check")
def health() -> HealthResponse:
    """Return a simple liveness status confirming the service is running."""
    return HealthResponse(status="ok", service="automark-api")


@app.post("/grade", response_model=GradeResponse, summary="Grade a submission")
async def grade(request: GradeRequest) -> GradeResponse:
    """Validate file paths, run the LangGraph pipeline, and return the graded result.

    The pipeline is executed in a worker thread (via ``asyncio.to_thread``) so
    the event loop remains unblocked while the LLM calls are in flight.

    Raises
    ------
    422 – if either file path is outside the allowed data directory or does not exist.
    500 – if the pipeline encounters an internal error.
    """
    import asyncio

    result = await asyncio.to_thread(_execute_grade, request.submission_path, request.rubric_path)
    criteria: list[CriterionResult] = [
        CriterionResult(
            criterion_id=c["criterion_id"],
            name=c["name"],
            score=c["score"],
            max_score=c["max_score"],
            justification=c["justification"],
        )
        for c in (result.get("criteria") or [])
    ]

    return GradeResponse(
        session_id=result["session_id"],
        student_id=result["student_id"],
        student_name=result["student_name"],
        total_score=result["total_score"],
        percentage=result["percentage"],
        grade=result["grade"],
        summary=result["summary"],
        criteria=criteria,
        output_filepath=result["output_filepath"],
        marking_sheet_path=result["marking_sheet_path"],
    )


@app.post(
    "/grade/batch",
    response_model=BatchGradeAcceptedResponse,
    summary="Submit a batch grading job",
)
async def grade_batch(request: BatchGradeRequest) -> BatchGradeAcceptedResponse:
    """Submit multiple grading items to the async job queue."""
    job_id = str(uuid.uuid4())
    items_for_db: list[dict[str, Any]] = []
    accepted = 0
    rejected = 0
    for index, item in enumerate(request.items):
        correlation_id = item.correlation_id or f"item-{index}"
        status = ITEM_STATUS_QUEUED
        error: str | None = None
        try:
            submission = _resolve_safe_path(item.submission_path)
            rubric = _resolve_safe_path(item.rubric_path)
            if not submission.exists():
                raise ValueError(f"Submission file not found: {item.submission_path}")
            if not rubric.exists():
                raise ValueError(f"Rubric file not found: {item.rubric_path}")
            accepted += 1
        except (HTTPException, ValueError) as exc:
            status = ITEM_STATUS_FAILED
            error = str(exc.detail) if isinstance(exc, HTTPException) else str(exc)
            rejected += 1
        items_for_db.append(
            {
                "item_index": index,
                "correlation_id": correlation_id,
                "submission_path": item.submission_path,
                "rubric_path": item.rubric_path,
                "status": status,
                "error": error,
            }
        )

    max_retries = request.max_retries if request.max_retries is not None else settings.job_max_retries
    create_job(
        db_path=settings.db_path,
        job_id=job_id,
        items=items_for_db,
        max_retries=max_retries,
    )
    if accepted > 0:
        try:
            _job_queue.enqueue(job_id)
        except Full as exc:
            mark_job_failed(
                settings.db_path,
                job_id,
                "Job queue is full. Try again later.",
            )
            raise HTTPException(status_code=503, detail="Job queue is full.") from exc
        status = JobStatus.queued
    else:
        refresh_job_progress(settings.db_path, job_id)
        refreshed = get_job(settings.db_path, job_id)
        status = JobStatus((refreshed or {}).get("status", JOB_STATUS_FAILED))

    return BatchGradeAcceptedResponse(
        job_id=job_id,
        status=status,
        total_items=len(request.items),
        accepted_items=accepted,
        rejected_items=rejected,
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse, summary="Get job status/details")
async def get_job_status(job_id: str) -> JobStatusResponse:
    return _as_job_status_response(job_id)


@app.get("/jobs", response_model=JobListResponse, summary="List async grading jobs")
async def get_jobs(
    status: JobStatus | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> JobListResponse:
    rows = list_jobs(
        settings.db_path,
        status=status.value if status else None,
        limit=limit,
        offset=offset,
    )
    jobs = [
        JobListEntry(
            job_id=row["job_id"],
            status=JobStatus(row["status"]),
            total_items=row["total_items"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row.get("started_at"),
            completed_at=row.get("completed_at"),
            cancel_requested=bool(row.get("cancel_requested")),
            max_retries=int(row.get("max_retries") or 0),
            error=row.get("error"),
        )
        for row in rows
    ]
    return JobListResponse(jobs=jobs, count=len(jobs))


@app.post("/jobs/{job_id}/cancel", response_model=JobCancelResponse, summary="Cancel a queued/running job")
async def cancel_job(job_id: str) -> JobCancelResponse:
    job = get_job(settings.db_path, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job["status"] in {JOB_STATUS_COMPLETED, JOB_STATUS_FAILED, JOB_STATUS_CANCELLED}:
        return JobCancelResponse(
            job_id=job_id,
            status=JobStatus(job["status"]),
            cancel_requested=bool(job.get("cancel_requested")),
        )
    changed = request_job_cancel(settings.db_path, job_id)
    refreshed = get_job(settings.db_path, job_id)
    if not refreshed:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JobCancelResponse(
        job_id=job_id,
        status=JobStatus(refreshed["status"]),
        cancel_requested=changed or bool(refreshed.get("cancel_requested")),
    )


@app.post(
    "/jobs/{job_id}/exports/{export_format}",
    response_model=JobArtifactResponse,
    summary="Generate export artifact for a completed batch job",
)
async def generate_job_export(job_id: str, export_format: ExportFormat) -> JobArtifactResponse:
    job = get_job(settings.db_path, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job["status"] in {JOB_STATUS_QUEUED, JOB_STATUS_RUNNING}:
        raise HTTPException(
            status_code=409,
            detail="Job is still running. Generate exports after completion.",
        )
    items = get_job_items(settings.db_path, job_id)
    content = _build_export_content(job, items, export_format.value)
    if len(content) > settings.export_max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Generated export exceeds size limit "
                f"({len(content)} > {settings.export_max_bytes} bytes)."
            ),
        )
    _EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    extension = "json" if export_format == ExportFormat.json else export_format.value
    artifact_id = str(uuid.uuid4())
    artifact_path = _EXPORT_DIR / f"{artifact_id}.{extension}"
    artifact_path.write_bytes(content)
    save_job_artifact(
        db_path=settings.db_path,
        artifact_id=artifact_id,
        job_id=job_id,
        export_format=export_format.value,
        file_path=str(artifact_path.resolve()),
        size_bytes=len(content),
    )
    artifact = get_job_artifact(settings.db_path, job_id, export_format.value)
    return JobArtifactResponse(
        artifact_id=artifact["artifact_id"],  # type: ignore[index]
        job_id=artifact["job_id"],  # type: ignore[index]
        format=ExportFormat(artifact["format"]),  # type: ignore[index]
        file_path=artifact["file_path"],  # type: ignore[index]
        size_bytes=artifact["size_bytes"],  # type: ignore[index]
        created_at=artifact["created_at"],  # type: ignore[index]
    )


@app.get(
    "/jobs/{job_id}/exports/{export_format}",
    summary="Download an existing export artifact",
)
async def download_job_export(job_id: str, export_format: ExportFormat) -> FileResponse:
    artifact = get_job_artifact(settings.db_path, job_id, export_format.value)
    if not artifact:
        raise HTTPException(
            status_code=404,
            detail="Export artifact not found. Generate it first.",
        )
    path = Path(artifact["file_path"])
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(_EXPORT_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid artifact path.") from exc
    if not resolved_path.exists():
        raise HTTPException(status_code=404, detail="Artifact file does not exist on disk.")
    media_type = {
        ExportFormat.json.value: "application/json",
        ExportFormat.csv.value: "text/csv",
        ExportFormat.pdf.value: "application/pdf",
    }[export_format.value]
    return FileResponse(
        path=str(resolved_path),
        media_type=media_type,
        filename=resolved_path.name,
    )


@app.get(
    "/sessions/{session_id}/logs",
    response_model=list[LogEntry],
    summary="Retrieve session trace logs",
)
async def session_logs(session_id: str) -> list[LogEntry]:
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
