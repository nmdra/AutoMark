"""Tests for deterministic API summary generation."""

from __future__ import annotations

from mas.tools.summary_builder import build_deterministic_summary


def test_summary_contains_deterministic_breakdown():
    scored = [
        {"criterion_id": "C1", "name": "Definition of Containerisation", "score": 0, "max_score": 5},
        {"criterion_id": "C2", "name": "Benefits of Containerisation", "score": 3, "max_score": 5},
        {"criterion_id": "C3", "name": "Role of Kubernetes", "score": 1, "max_score": 5},
        {"criterion_id": "C4", "name": "Technical Accuracy and Depth", "score": 2, "max_score": 5},
    ]

    summary = build_deterministic_summary(scored, total_score=6, total_marks=20, grade="F")

    assert summary == (
        "Total score: 6/20 (30.00%), grade F. Breakdown: "
        "Definition of Containerisation: 0/5; "
        "Benefits of Containerisation: 3/5; "
        "Role of Kubernetes: 1/5; "
        "Technical Accuracy and Depth: 2/5."
    )


def test_summary_when_no_criteria():
    summary = build_deterministic_summary([], total_score=0, total_marks=0, grade="F")
    assert summary == "Total score: 0/0 (0.00%), grade F. No criterion scores available."
