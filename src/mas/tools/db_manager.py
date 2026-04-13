"""SQLite persistence helpers for reports, async jobs, and export artifacts."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


JOB_STATUS_QUEUED = "queued"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"
JOB_STATUS_CANCELLED = "cancelled"

ITEM_STATUS_QUEUED = "queued"
ITEM_STATUS_RUNNING = "running"
ITEM_STATUS_COMPLETED = "completed"
ITEM_STATUS_FAILED = "failed"
ITEM_STATUS_CANCELLED = "cancelled"

_TERMINAL_JOB_STATUSES = {
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_CANCELLED,
}


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _connect(db_path: str) -> sqlite3.Connection:
    """Return a SQLite connection with required PRAGMAs applied."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _dict_conn(db_path: str) -> sqlite3.Connection:
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str) -> None:
    """Initialise the SQLite database and create required tables.

    Safe to call multiple times – uses ``CREATE TABLE IF NOT EXISTS``.

    Applies two one-time performance optimisations:

    * **WAL journal mode** – allows concurrent readers during writes and
      reduces fsync overhead compared to the default DELETE mode.
    * **Index on ``student_id``** – eliminates full-table scans in
      ``get_past_reports`` as the database grows.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Parent directories are created
        automatically.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id   TEXT    NOT NULL,
                session_id   TEXT    NOT NULL,
                timestamp    TEXT    NOT NULL,
                report_json  TEXT    NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_reports_student_id
            ON reports (student_id)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id            TEXT PRIMARY KEY,
                status            TEXT    NOT NULL,
                total_items       INTEGER NOT NULL,
                max_retries       INTEGER NOT NULL DEFAULT 0,
                cancel_requested  INTEGER NOT NULL DEFAULT 0,
                created_at        TEXT    NOT NULL,
                updated_at        TEXT    NOT NULL,
                started_at        TEXT,
                completed_at      TEXT,
                error             TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS job_items (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id            TEXT    NOT NULL,
                item_index        INTEGER NOT NULL,
                correlation_id    TEXT    NOT NULL,
                submission_path   TEXT    NOT NULL,
                rubric_path       TEXT    NOT NULL,
                status            TEXT    NOT NULL,
                attempts          INTEGER NOT NULL DEFAULT 0,
                session_id        TEXT,
                student_id        TEXT,
                student_name      TEXT,
                total_score       REAL,
                percentage        REAL,
                grade             TEXT,
                summary           TEXT,
                output_filepath   TEXT,
                marking_sheet_path TEXT,
                criteria_json     TEXT,
                error             TEXT,
                created_at        TEXT    NOT NULL,
                updated_at        TEXT    NOT NULL,
                started_at        TEXT,
                completed_at      TEXT,
                FOREIGN KEY(job_id) REFERENCES jobs(job_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS job_artifacts (
                artifact_id       TEXT PRIMARY KEY,
                job_id            TEXT    NOT NULL,
                format            TEXT    NOT NULL,
                file_path         TEXT    NOT NULL,
                size_bytes        INTEGER NOT NULL,
                created_at        TEXT    NOT NULL,
                FOREIGN KEY(job_id) REFERENCES jobs(job_id),
                UNIQUE(job_id, format)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_jobs_status
            ON jobs (status, created_at DESC)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_job_items_job_id
            ON job_items (job_id, item_index ASC)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_job_items_status
            ON job_items (job_id, status)
            """
        )
        conn.commit()


def save_report(
    db_path: str,
    student_id: str,
    session_id: str,
    timestamp: str,
    scored_criteria: list[dict[str, Any]],
    total_score: float,
    grade: str,
) -> None:
    """Persist a single grading result to the database.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.
    student_id:
        The student's unique identifier extracted from the submission.
    session_id:
        The current pipeline session identifier.
    timestamp:
        ISO-8601 timestamp string for when the report was generated.
    scored_criteria:
        List of per-criterion score dicts produced by the Analysis Agent.
    total_score:
        Aggregate numeric score.
    grade:
        Letter grade (A–F).
    """
    report_data = {
        "scored_criteria": scored_criteria,
        "total_score": total_score,
        "grade": grade,
    }
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO reports (student_id, session_id, timestamp, report_json) "
            "VALUES (?, ?, ?, ?)",
            (student_id, session_id, timestamp, json.dumps(report_data)),
        )
        conn.commit()


def get_past_reports(db_path: str, student_id: str) -> list[dict[str, Any]]:
    """Retrieve all previously saved reports for a given student.

    Returns an empty list when the database does not exist or the student has
    no prior records.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.
    student_id:
        The student's unique identifier.

    Returns
    -------
    list[dict]
        Chronologically ordered list of report dicts, each containing
        ``session_id``, ``timestamp``, ``scored_criteria``, ``total_score``,
        and ``grade``.
    """
    if not Path(db_path).exists():
        return []

    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT session_id, timestamp, report_json "
            "FROM reports WHERE student_id = ? ORDER BY id ASC",
            (student_id,),
        ).fetchall()

    results: list[dict[str, Any]] = []
    for session_id, timestamp, report_json in rows:
        data = json.loads(report_json)
        results.append(
            {
                "session_id": session_id,
                "timestamp": timestamp,
                **data,
            }
        )
    return results


def create_job(
    db_path: str,
    job_id: str,
    items: list[dict[str, Any]],
    max_retries: int = 0,
) -> None:
    """Create a new async grading job and its item rows."""
    init_db(db_path)
    now = _utc_now()
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO jobs (
                job_id, status, total_items, max_retries, cancel_requested,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, 0, ?, ?)
            """,
            (job_id, JOB_STATUS_QUEUED, len(items), max_retries, now, now),
        )
        conn.executemany(
            """
            INSERT INTO job_items (
                job_id, item_index, correlation_id, submission_path, rubric_path,
                status, attempts, created_at, updated_at, error
            ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
            """,
            [
                (
                    job_id,
                    item["item_index"],
                    item["correlation_id"],
                    item["submission_path"],
                    item["rubric_path"],
                    item.get("status", ITEM_STATUS_QUEUED),
                    now,
                    now,
                    item.get("error"),
                )
                for item in items
            ],
        )
        conn.commit()
    refresh_job_progress(db_path, job_id)


def get_job(db_path: str, job_id: str) -> dict[str, Any] | None:
    """Return a job with aggregate progress metrics."""
    if not Path(db_path).exists():
        return None
    with _dict_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if row is None:
            return None
        progress = conn.execute(
            """
            SELECT
                SUM(CASE WHEN status = ? THEN 1 ELSE 0 END) AS queued,
                SUM(CASE WHEN status = ? THEN 1 ELSE 0 END) AS running,
                SUM(CASE WHEN status = ? THEN 1 ELSE 0 END) AS completed,
                SUM(CASE WHEN status = ? THEN 1 ELSE 0 END) AS failed,
                SUM(CASE WHEN status = ? THEN 1 ELSE 0 END) AS cancelled
            FROM job_items WHERE job_id = ?
            """,
            (
                ITEM_STATUS_QUEUED,
                ITEM_STATUS_RUNNING,
                ITEM_STATUS_COMPLETED,
                ITEM_STATUS_FAILED,
                ITEM_STATUS_CANCELLED,
                job_id,
            ),
        ).fetchone()
        return {
            **dict(row),
            "progress": {
                "total": row["total_items"],
                "queued": int(progress["queued"] or 0),
                "running": int(progress["running"] or 0),
                "completed": int(progress["completed"] or 0),
                "failed": int(progress["failed"] or 0),
                "cancelled": int(progress["cancelled"] or 0),
            },
        }


def list_jobs(
    db_path: str,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """List jobs ordered by creation time descending."""
    if not Path(db_path).exists():
        return []
    query = "SELECT * FROM jobs"
    params: list[Any] = []
    if status:
        query += " WHERE status = ?"
        params.append(status)
    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    with _dict_conn(db_path) as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
    return [dict(r) for r in rows]


def get_job_items(db_path: str, job_id: str) -> list[dict[str, Any]]:
    """Return all items for a job in index order."""
    if not Path(db_path).exists():
        return []
    with _dict_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM job_items WHERE job_id = ? ORDER BY item_index ASC",
            (job_id,),
        ).fetchall()
    items: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if item.get("criteria_json"):
            try:
                item["criteria"] = json.loads(item["criteria_json"])
            except json.JSONDecodeError:
                item["criteria"] = []
        else:
            item["criteria"] = []
        item.pop("criteria_json", None)
        items.append(item)
    return items


def get_queued_job_items(db_path: str, job_id: str) -> list[dict[str, Any]]:
    """Return job items still awaiting execution."""
    if not Path(db_path).exists():
        return []
    with _dict_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM job_items
            WHERE job_id = ? AND status = ?
            ORDER BY item_index ASC
            """,
            (job_id, ITEM_STATUS_QUEUED),
        ).fetchall()
    return [dict(r) for r in rows]


def refresh_job_progress(db_path: str, job_id: str) -> dict[str, int]:
    """Recompute job aggregate counts and adjust terminal status if needed."""
    now = _utc_now()
    with _dict_conn(db_path) as conn:
        row = conn.execute(
            """
            SELECT
                SUM(CASE WHEN status = ? THEN 1 ELSE 0 END) AS queued,
                SUM(CASE WHEN status = ? THEN 1 ELSE 0 END) AS running,
                SUM(CASE WHEN status = ? THEN 1 ELSE 0 END) AS completed,
                SUM(CASE WHEN status = ? THEN 1 ELSE 0 END) AS failed,
                SUM(CASE WHEN status = ? THEN 1 ELSE 0 END) AS cancelled
            FROM job_items WHERE job_id = ?
            """,
            (
                ITEM_STATUS_QUEUED,
                ITEM_STATUS_RUNNING,
                ITEM_STATUS_COMPLETED,
                ITEM_STATUS_FAILED,
                ITEM_STATUS_CANCELLED,
                job_id,
            ),
        ).fetchone()
        counts = {
            "queued": int(row["queued"] or 0),
            "running": int(row["running"] or 0),
            "completed": int(row["completed"] or 0),
            "failed": int(row["failed"] or 0),
            "cancelled": int(row["cancelled"] or 0),
        }
        job = conn.execute(
            "SELECT status, total_items, cancel_requested FROM jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if job is None:
            return counts
        new_status: str | None = None
        done = (
            counts["queued"] == 0
            and counts["running"] == 0
            and (counts["completed"] + counts["failed"] + counts["cancelled"] >= job["total_items"])
        )
        if done and job["status"] not in _TERMINAL_JOB_STATUSES:
            if job["cancel_requested"]:
                new_status = JOB_STATUS_CANCELLED
            elif counts["completed"] == 0 and counts["failed"] > 0:
                new_status = JOB_STATUS_FAILED
            else:
                new_status = JOB_STATUS_COMPLETED
        if new_status:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, completed_at = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (new_status, now, now, job_id),
            )
        else:
            conn.execute(
                "UPDATE jobs SET updated_at = ? WHERE job_id = ?",
                (now, job_id),
            )
        conn.commit()
    return counts


def mark_job_running(db_path: str, job_id: str) -> bool:
    """Transition a queued job to running."""
    now = _utc_now()
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            UPDATE jobs
            SET status = ?, started_at = COALESCE(started_at, ?), updated_at = ?
            WHERE job_id = ? AND status = ?
            """,
            (JOB_STATUS_RUNNING, now, now, job_id, JOB_STATUS_QUEUED),
        )
        conn.commit()
        return row.rowcount > 0


def mark_job_failed(db_path: str, job_id: str, error: str) -> None:
    """Mark job as failed with an error message."""
    now = _utc_now()
    with _connect(db_path) as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status = ?, error = ?, completed_at = ?, updated_at = ?
            WHERE job_id = ?
            """,
            (JOB_STATUS_FAILED, error, now, now, job_id),
        )
        conn.execute(
            """
            UPDATE job_items
            SET
                status = ?,
                error = COALESCE(error, ?),
                completed_at = COALESCE(completed_at, ?),
                updated_at = ?
            WHERE job_id = ? AND status IN (?, ?)
            """,
            (
                ITEM_STATUS_FAILED,
                error,
                now,
                now,
                job_id,
                ITEM_STATUS_QUEUED,
                ITEM_STATUS_RUNNING,
            ),
        )
        conn.commit()
    refresh_job_progress(db_path, job_id)


def request_job_cancel(db_path: str, job_id: str) -> bool:
    """Request cancellation for a queued or running job."""
    now = _utc_now()
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            UPDATE jobs
            SET cancel_requested = 1, updated_at = ?
            WHERE job_id = ? AND status IN (?, ?)
            """,
            (now, job_id, JOB_STATUS_QUEUED, JOB_STATUS_RUNNING),
        )
        conn.commit()
        return row.rowcount > 0


def is_job_cancel_requested(db_path: str, job_id: str) -> bool:
    """Return whether cancellation was requested for a job."""
    if not Path(db_path).exists():
        return False
    with _dict_conn(db_path) as conn:
        row = conn.execute(
            "SELECT cancel_requested FROM jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
    return bool(row and row["cancel_requested"])


def mark_remaining_items_cancelled(db_path: str, job_id: str) -> int:
    """Cancel all queued items for a job."""
    now = _utc_now()
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            UPDATE job_items
            SET status = ?, updated_at = ?, completed_at = ?, error = COALESCE(error, ?)
            WHERE job_id = ? AND status = ?
            """,
            (ITEM_STATUS_CANCELLED, now, now, "Cancelled by request.", job_id, ITEM_STATUS_QUEUED),
        )
        conn.commit()
        return int(row.rowcount)


def mark_job_item_running(db_path: str, item_id: int) -> bool:
    """Mark a queued item as running and increment attempts."""
    now = _utc_now()
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            UPDATE job_items
            SET status = ?, attempts = attempts + 1, started_at = COALESCE(started_at, ?), updated_at = ?
            WHERE id = ? AND status = ?
            """,
            (ITEM_STATUS_RUNNING, now, now, item_id, ITEM_STATUS_QUEUED),
        )
        conn.commit()
        return row.rowcount > 0


def mark_job_item_completed(
    db_path: str,
    item_id: int,
    result: dict[str, Any],
) -> None:
    """Persist a completed item result."""
    now = _utc_now()
    with _connect(db_path) as conn:
        conn.execute(
            """
            UPDATE job_items
            SET
                status = ?,
                session_id = ?,
                student_id = ?,
                student_name = ?,
                total_score = ?,
                percentage = ?,
                grade = ?,
                summary = ?,
                output_filepath = ?,
                marking_sheet_path = ?,
                criteria_json = ?,
                error = NULL,
                updated_at = ?,
                completed_at = ?
            WHERE id = ?
            """,
            (
                ITEM_STATUS_COMPLETED,
                result.get("session_id"),
                result.get("student_id"),
                result.get("student_name"),
                result.get("total_score"),
                result.get("percentage"),
                result.get("grade"),
                result.get("summary"),
                result.get("output_filepath"),
                result.get("marking_sheet_path"),
                json.dumps(result.get("criteria") or []),
                now,
                now,
                item_id,
            ),
        )
        conn.commit()


def mark_job_item_failed(db_path: str, item_id: int, error: str) -> None:
    """Mark an item as failed."""
    now = _utc_now()
    with _connect(db_path) as conn:
        conn.execute(
            """
            UPDATE job_items
            SET status = ?, error = ?, updated_at = ?, completed_at = ?
            WHERE id = ?
            """,
            (ITEM_STATUS_FAILED, error, now, now, item_id),
        )
        conn.commit()


def save_job_artifact(
    db_path: str,
    artifact_id: str,
    job_id: str,
    export_format: str,
    file_path: str,
    size_bytes: int,
) -> None:
    """Insert or replace a generated export artifact row."""
    init_db(db_path)
    now = _utc_now()
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO job_artifacts (artifact_id, job_id, format, file_path, size_bytes, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id, format) DO UPDATE SET
                artifact_id=excluded.artifact_id,
                file_path=excluded.file_path,
                size_bytes=excluded.size_bytes,
                created_at=excluded.created_at
            """,
            (artifact_id, job_id, export_format, file_path, size_bytes, now),
        )
        conn.commit()


def get_job_artifact(db_path: str, job_id: str, export_format: str) -> dict[str, Any] | None:
    """Fetch artifact metadata for a job/format combination."""
    if not Path(db_path).exists():
        return None
    with _dict_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM job_artifacts WHERE job_id = ? AND format = ?",
            (job_id, export_format),
        ).fetchone()
    return dict(row) if row else None
