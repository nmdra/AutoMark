"""File operations toolset used by the Ingestion Agent.

Combines file validation and file reading into a single module.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


# ── Validation ────────────────────────────────────────────────────────────────


def validate_submission_files(
    submission_path: str,
    rubric_path: str,
) -> dict[str, str]:
    """Validate that the submission and rubric files are ready for processing.

    Checks performed:
    * Path strings are non-empty.
    * Files exist on disk.
    * Files are non-empty (size > 0 bytes).
    * Submission has a ``.txt`` extension.
    * Rubric has a ``.json`` extension.

    Parameters
    ----------
    submission_path:
        Absolute or relative path to the plain-text student submission.
    rubric_path:
        Absolute or relative path to the JSON rubric file.

    Returns
    -------
    dict
        ``{"status": "success"}`` when all checks pass.

    Raises
    ------
    ValueError
        If either path string is empty.
    FileNotFoundError
        If a file does not exist.
    ValueError
        If a file is empty or has an incorrect extension.
    """
    if not submission_path:
        raise ValueError("submission_path must not be empty")
    if not rubric_path:
        raise ValueError("rubric_path must not be empty")

    sub = Path(submission_path)
    rub = Path(rubric_path)

    if not sub.exists():
        raise FileNotFoundError(f"Submission file not found: {submission_path}")
    if not rub.exists():
        raise FileNotFoundError(f"Rubric file not found: {rubric_path}")

    if sub.stat().st_size == 0:
        raise ValueError(f"Submission file is empty: {submission_path}")
    if rub.stat().st_size == 0:
        raise ValueError(f"Rubric file is empty: {rubric_path}")

    if sub.suffix.lower() != ".txt":
        raise ValueError(
            f"Submission file must have a .txt extension, got: {sub.suffix!r}"
        )
    if rub.suffix.lower() != ".json":
        raise ValueError(
            f"Rubric file must have a .json extension, got: {rub.suffix!r}"
        )

    return {"status": "success"}


# ── Reading ───────────────────────────────────────────────────────────────────


def read_text_file(path: str) -> str:
    """Read a plain-text file and return its contents as a string.

    Parameters
    ----------
    path:
        Path to the text file.

    Returns
    -------
    str
        The full file contents.

    Raises
    ------
    RuntimeError
        Wraps any ``OSError`` encountered while reading the file.
    """
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to read text file '{path}': {exc}") from exc


def read_json_file(path: str) -> dict[str, Any]:
    """Read a JSON file and return the parsed object.

    The raw file contents are cached in memory (keyed by absolute path) so
    that repeated reads of the same rubric file – common when grading multiple
    submissions with one rubric – avoid redundant disk I/O.  Each caller
    receives a freshly parsed dict, so mutations do not affect the cache.

    Parameters
    ----------
    path:
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed JSON content.

    Raises
    ------
    RuntimeError
        Wraps ``OSError`` or ``json.JSONDecodeError`` encountered during read.
    """
    raw = _read_json_raw(str(Path(path).resolve()))
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse JSON file '{path}': {exc}"
        ) from exc


@lru_cache(maxsize=32)
def _read_json_raw(absolute_path: str) -> str:
    """Read and cache the raw text of a JSON file.

    Caches the *string* (immutable) rather than a parsed dict so that callers
    always receive independent dict objects and cannot corrupt the cache via
    mutation.

    Parameters
    ----------
    absolute_path:
        Resolved absolute path to the JSON file (used as the cache key).

    Raises
    ------
    RuntimeError
        Wraps any ``OSError`` encountered while reading.
    """
    try:
        return Path(absolute_path).read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            f"Failed to read JSON file '{absolute_path}': {exc}"
        ) from exc
