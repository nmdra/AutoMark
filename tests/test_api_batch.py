"""Tests for batch grading, async jobs, and export APIs."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from mas import api
from mas.tools.db_manager import (
    ITEM_STATUS_FAILED,
    ITEM_STATUS_QUEUED,
    create_job,
    get_job,
    get_job_items,
    init_db,
    mark_job_failed,
    mark_job_item_completed,
    mark_job_item_running,
    mark_job_running,
    refresh_job_progress,
    request_job_cancel,
    save_job_artifact,
)


def _override_frozen_setting(name: str, value):
    object.__setattr__(api.settings, name, value)


def _make_client(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "output"
    db_path = tmp_path / "students.db"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    _override_frozen_setting("db_path", str(db_path))
    _override_frozen_setting("output_path", str(out_dir / "feedback_report.md"))
    _override_frozen_setting("batch_max_items", 100)
    _override_frozen_setting("job_max_retries", 1)
    _override_frozen_setting("export_max_bytes", 10_485_760)

    api._DATA_BASE_DIR = data_dir.resolve()
    api._OUTPUT_DIR = out_dir.resolve()
    api._EXPORT_DIR = (out_dir / "exports").resolve()
    init_db(str(db_path))
    return TestClient(api.app), data_dir, db_path


def test_batch_submit_accepts_valid_and_rejects_invalid(tmp_path, monkeypatch):
    client, data_dir, db_path = _make_client(tmp_path)
    (data_dir / "submission.txt").write_text("Student ID: S1\nAnswer", encoding="utf-8")
    (data_dir / "rubric.json").write_text(
        json.dumps(
            {
                "module": "M",
                "assignment": "A",
                "total_marks": 10,
                "criteria": [{"id": "C1", "name": "N", "description": "D", "max_score": 10}],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(api._job_queue, "enqueue", lambda _job_id: None)

    response = client.post(
        "/grade/batch",
        json={
            "items": [
                {"submission_path": "submission.txt", "rubric_path": "rubric.json", "correlation_id": "ok-1"},
                {"submission_path": "missing.txt", "rubric_path": "rubric.json", "correlation_id": "bad-1"},
            ]
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["accepted_items"] == 1
    assert body["rejected_items"] == 1
    job = get_job(str(db_path), body["job_id"])
    assert job is not None
    assert job["progress"]["failed"] == 1
    assert job["progress"]["queued"] == 1


def test_get_job_status_returns_items(tmp_path):
    client, _data_dir, db_path = _make_client(tmp_path)
    job_id = str(uuid.uuid4())
    create_job(
        str(db_path),
        job_id,
        items=[
            {
                "item_index": 0,
                "correlation_id": "x1",
                "submission_path": "s.txt",
                "rubric_path": "r.json",
                "status": ITEM_STATUS_FAILED,
                "error": "bad file",
            }
        ],
        max_retries=1,
    )

    response = client.get(f"/jobs/{job_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] == job_id
    assert len(body["items"]) == 1
    assert body["items"][0]["status"] == ITEM_STATUS_FAILED


def test_cancel_job_marks_cancel_requested(tmp_path):
    client, _data_dir, db_path = _make_client(tmp_path)
    job_id = str(uuid.uuid4())
    create_job(
        str(db_path),
        job_id,
        items=[
            {
                "item_index": 0,
                "correlation_id": "x1",
                "submission_path": "s.txt",
                "rubric_path": "r.json",
                "status": ITEM_STATUS_QUEUED,
            }
        ],
        max_retries=0,
    )

    response = client.post(f"/jobs/{job_id}/cancel")
    assert response.status_code == 200
    body = response.json()
    assert body["cancel_requested"] is True


def test_cancel_requested_job_finishes_as_cancelled(tmp_path):
    _client, _data_dir, db_path = _make_client(tmp_path)
    job_id = str(uuid.uuid4())
    create_job(
        str(db_path),
        job_id,
        items=[
            {
                "item_index": 0,
                "correlation_id": "x1",
                "submission_path": "s.txt",
                "rubric_path": "r.json",
                "status": ITEM_STATUS_QUEUED,
            }
        ],
        max_retries=0,
    )
    mark_job_running(str(db_path), job_id)
    item = get_job_items(str(db_path), job_id)[0]
    mark_job_item_running(str(db_path), int(item["id"]))
    assert request_job_cancel(str(db_path), job_id) is True
    mark_job_item_completed(
        str(db_path),
        int(item["id"]),
        {
            "session_id": "sess-1",
            "student_id": "S1",
            "student_name": "N",
            "total_score": 8.0,
            "percentage": 80.0,
            "grade": "B",
            "summary": "ok",
            "criteria": [],
            "output_filepath": "/tmp/report.md",
            "marking_sheet_path": "/tmp/sheet.md",
        },
    )
    refresh_job_progress(str(db_path), job_id)
    job = get_job(str(db_path), job_id)
    assert job is not None
    assert job["cancel_requested"] == 1
    assert job["status"] == "cancelled"


def test_list_jobs_supports_status_and_pagination(tmp_path):
    client, _data_dir, db_path = _make_client(tmp_path)
    queued_job_id = str(uuid.uuid4())
    failed_job_id = str(uuid.uuid4())
    completed_job_id = str(uuid.uuid4())

    create_job(
        str(db_path),
        queued_job_id,
        items=[
            {
                "item_index": 0,
                "correlation_id": "queued",
                "submission_path": "q.txt",
                "rubric_path": "r.json",
                "status": ITEM_STATUS_QUEUED,
            }
        ],
        max_retries=0,
    )
    create_job(
        str(db_path),
        failed_job_id,
        items=[
            {
                "item_index": 0,
                "correlation_id": "failed",
                "submission_path": "f.txt",
                "rubric_path": "r.json",
                "status": ITEM_STATUS_QUEUED,
            }
        ],
        max_retries=0,
    )
    mark_job_failed(str(db_path), failed_job_id, "boom")

    create_job(
        str(db_path),
        completed_job_id,
        items=[
            {
                "item_index": 0,
                "correlation_id": "completed",
                "submission_path": "c.txt",
                "rubric_path": "r.json",
                "status": ITEM_STATUS_QUEUED,
            }
        ],
        max_retries=0,
    )
    mark_job_running(str(db_path), completed_job_id)
    completed_item = get_job_items(str(db_path), completed_job_id)[0]
    mark_job_item_running(str(db_path), int(completed_item["id"]))
    mark_job_item_completed(
        str(db_path),
        int(completed_item["id"]),
        {
            "session_id": "sess-2",
            "student_id": "S2",
            "student_name": "N2",
            "total_score": 9.0,
            "percentage": 90.0,
            "grade": "A",
            "summary": "ok",
            "criteria": [],
            "output_filepath": "/tmp/report2.md",
            "marking_sheet_path": "/tmp/sheet2.md",
        },
    )
    refresh_job_progress(str(db_path), completed_job_id)

    all_jobs = client.get("/jobs?limit=2&offset=0")
    assert all_jobs.status_code == 200
    all_payload = all_jobs.json()
    assert all_payload["count"] == 2

    filtered = client.get("/jobs?status=failed&limit=10&offset=0")
    assert filtered.status_code == 200
    filtered_payload = filtered.json()
    assert filtered_payload["count"] == 1
    assert filtered_payload["jobs"][0]["job_id"] == failed_job_id
    assert filtered_payload["jobs"][0]["status"] == "failed"


def _create_completed_job_for_export(db_path: Path) -> str:
    job_id = str(uuid.uuid4())
    create_job(
        str(db_path),
        job_id,
        items=[
            {
                "item_index": 0,
                "correlation_id": "x1",
                "submission_path": "s.txt",
                "rubric_path": "r.json",
                "status": ITEM_STATUS_QUEUED,
            }
        ],
        max_retries=0,
    )
    mark_job_running(str(db_path), job_id)
    item = get_job_items(str(db_path), job_id)[0]
    mark_job_item_running(str(db_path), int(item["id"]))
    mark_job_item_completed(
        str(db_path),
        int(item["id"]),
        {
            "session_id": "sess-1",
            "student_id": "S1",
            "student_name": "N",
            "total_score": 8.0,
            "percentage": 80.0,
            "grade": "B",
            "summary": "ok",
            "criteria": [{"criterion_id": "C1", "name": "N", "score": 8, "max_score": 10, "justification": "Good"}],
            "output_filepath": "/tmp/report.md",
            "marking_sheet_path": "/tmp/sheet.md",
        },
    )
    refresh_job_progress(str(db_path), job_id)
    return job_id


def test_worker_handles_partial_failures(tmp_path, monkeypatch):
    _client, _data_dir, db_path = _make_client(tmp_path)
    job_id = str(uuid.uuid4())
    create_job(
        str(db_path),
        job_id,
        items=[
            {
                "item_index": 0,
                "correlation_id": "ok",
                "submission_path": "good.txt",
                "rubric_path": "rubric.json",
                "status": ITEM_STATUS_QUEUED,
            },
            {
                "item_index": 1,
                "correlation_id": "bad",
                "submission_path": "bad.txt",
                "rubric_path": "rubric.json",
                "status": ITEM_STATUS_QUEUED,
            },
        ],
        max_retries=0,
    )

    def _fake_grade(submission_path: str, rubric_path: str):
        if submission_path == "bad.txt":
            raise RuntimeError("boom")
        return {
            "session_id": "sess-1",
            "student_id": "S1",
            "student_name": "N",
            "total_score": 8.0,
            "percentage": 80.0,
            "grade": "B",
            "summary": "ok",
            "criteria": [],
            "output_filepath": "/tmp/report.md",
            "marking_sheet_path": "/tmp/sheet.md",
        }

    monkeypatch.setattr(api, "_execute_grade", _fake_grade)
    api._job_queue._process_job(job_id)
    job = get_job(str(db_path), job_id)
    assert job is not None
    assert job["progress"]["completed"] == 1
    assert job["progress"]["failed"] == 1
    assert job["status"] == "completed"


def test_generate_and_download_json_export(tmp_path):
    client, _data_dir, db_path = _make_client(tmp_path)
    job_id = _create_completed_job_for_export(Path(db_path))

    created = client.post(f"/jobs/{job_id}/exports/json")
    assert created.status_code == 200
    payload = created.json()
    assert payload["job_id"] == job_id
    assert payload["format"] == "json"

    downloaded = client.get(f"/jobs/{job_id}/exports/json")
    assert downloaded.status_code == 200
    assert downloaded.headers["content-type"].startswith("application/json")
    exported = downloaded.json()
    assert exported["job"]["job_id"] == job_id
    assert len(exported["items"]) == 1
    assert "generated_at" in exported


def test_generate_and_download_csv_export(tmp_path):
    client, _data_dir, db_path = _make_client(tmp_path)
    job_id = _create_completed_job_for_export(Path(db_path))

    created = client.post(f"/jobs/{job_id}/exports/csv")
    assert created.status_code == 200
    assert created.json()["format"] == "csv"

    downloaded = client.get(f"/jobs/{job_id}/exports/csv")
    assert downloaded.status_code == 200
    assert downloaded.headers["content-type"].startswith("text/csv")
    assert "correlation_id" in downloaded.text
    assert "x1" in downloaded.text


def test_generate_and_download_pdf_export(tmp_path):
    client, _data_dir, db_path = _make_client(tmp_path)
    job_id = _create_completed_job_for_export(Path(db_path))

    created = client.post(f"/jobs/{job_id}/exports/pdf")
    assert created.status_code == 200
    assert created.json()["format"] == "pdf"

    downloaded = client.get(f"/jobs/{job_id}/exports/pdf")
    assert downloaded.status_code == 200
    assert downloaded.headers["content-type"].startswith("application/pdf")
    assert downloaded.content.startswith(b"%PDF")


def test_export_generation_rejects_oversized_output(tmp_path):
    client, _data_dir, db_path = _make_client(tmp_path)
    _override_frozen_setting("export_max_bytes", 64)
    job_id = _create_completed_job_for_export(Path(db_path))

    response = client.post(f"/jobs/{job_id}/exports/json")
    assert response.status_code == 413


def test_export_download_rejects_artifact_outside_export_dir(tmp_path):
    client, _data_dir, db_path = _make_client(tmp_path)
    job_id = _create_completed_job_for_export(Path(db_path))
    save_job_artifact(
        db_path=str(db_path),
        artifact_id=str(uuid.uuid4()),
        job_id=job_id,
        export_format="json",
        file_path=str((tmp_path / "outside.json").resolve()),
        size_bytes=10,
    )

    response = client.get(f"/jobs/{job_id}/exports/json")
    assert response.status_code == 400
