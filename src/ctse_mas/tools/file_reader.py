"""File reader tools used by the Research Agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to read JSON file '{path}': {exc}") from exc

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse JSON file '{path}': {exc}"
        ) from exc
