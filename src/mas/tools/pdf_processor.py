"""PDF processing tool used by the PDF Ingestion Agent.

Converts a PDF file into Markdown text using ``pymupdf4llm``, making the
content available for LLM-based extraction of student details.
"""

from __future__ import annotations

from pathlib import Path


def convert_pdf_to_markdown(pdf_path: str) -> str:
    """Convert a PDF file to Markdown text.

    Uses ``pymupdf4llm`` to extract page content while preserving document
    structure (headings, tables, lists) as Markdown.

    Parameters
    ----------
    pdf_path:
        Absolute or relative path to the PDF file.

    Returns
    -------
    str
        The full document content as a Markdown string.

    Raises
    ------
    FileNotFoundError
        If the PDF file does not exist.
    ValueError
        If the path does not point to a ``.pdf`` file.
    RuntimeError
        Wraps any underlying conversion error from ``pymupdf4llm``.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a .pdf file, got: {path.suffix!r}"
        )

    try:
        import pymupdf4llm  # type: ignore[import]

        return pymupdf4llm.to_markdown(str(path))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to convert PDF '{pdf_path}': {exc}"
        ) from exc
