"""File validation tool used by the Coordinator Agent."""

from __future__ import annotations

from pathlib import Path


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
