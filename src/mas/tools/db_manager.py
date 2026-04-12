"""Database manager tool used by the Historical Agent.

Handles SQLite CRUD operations for persisting student report data.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def init_db(db_path: str) -> None:
    """Initialise the SQLite database and create the ``reports`` table.

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
    with sqlite3.connect(db_path) as conn:
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
    with sqlite3.connect(db_path) as conn:
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

    with sqlite3.connect(db_path) as conn:
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
