"""File writer tool used by the Report Agent."""

from __future__ import annotations

from pathlib import Path


def write_feedback_report(content: str, output_path: str) -> str:
    """Write the feedback report to disk, creating directories as needed.

    An existing file at ``output_path`` will be overwritten.

    Parameters
    ----------
    content:
        The Markdown-formatted feedback report string.
    output_path:
        Destination file path (absolute or relative).

    Returns
    -------
    str
        The resolved absolute path of the written file.
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    return str(dest.resolve())
