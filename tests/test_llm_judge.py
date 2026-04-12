"""LLM-as-a-Judge test using phi4-mini via the Ollama REST API.

The test is automatically skipped when the Ollama service is not reachable.
When Ollama is available the judge is asked a single yes/no question:
"Are the scores assigned to the student submission fair?"

The expected response from phi4-mini is strictly "YES" or "NO" (case-insensitive
after stripping whitespace).
"""

from __future__ import annotations

import json

import pytest
import requests

_OLLAMA_BASE_URL = "http://localhost:11434"
_MODEL = "phi4-mini:3.8b-q4_0"

# ── Sample data for the judge ─────────────────────────────────────────────────

_SUBMISSION = (
    "Containerisation is a lightweight form of operating-system-level "
    "virtualisation that packages an application together with all its "
    "dependencies into a single portable unit called a container. Unlike "
    "virtual machines, containers share the host OS kernel, making them more "
    "efficient in resource consumption and start-up time.\n\n"
    "Key benefits include: portability across environments, process isolation, "
    "horizontal scalability via orchestration platforms such as Kubernetes, "
    "reduced resource overhead compared to VMs, and faster CI/CD pipelines.\n\n"
    "Kubernetes is an open-source container orchestration system that automates "
    "deployment, scaling, and management of containerised applications. It "
    "provides declarative configuration, self-healing capabilities, rolling "
    "updates, service discovery, and integrates with major cloud providers."
)

_RUBRIC = {
    "module": "CTSE – IT4080",
    "assignment": "Cloud Technology Fundamentals",
    "total_marks": 20,
    "criteria": [
        {
            "id": "C1",
            "name": "Definition of Containerisation",
            "description": "Accurately defines containerisation and distinguishes it from VMs.",
            "max_score": 5,
        },
        {
            "id": "C2",
            "name": "Benefits of Containerisation",
            "description": "Identifies and explains at least four key benefits.",
            "max_score": 5,
        },
        {
            "id": "C3",
            "name": "Role of Kubernetes",
            "description": "Explains Kubernetes purpose and relevance to modern DevOps.",
            "max_score": 5,
        },
        {
            "id": "C4",
            "name": "Technical Accuracy and Depth",
            "description": "Demonstrates technical accuracy and appropriate depth.",
            "max_score": 5,
        },
    ],
}

# Scores that are clearly fair for the above submission
_SCORED_CRITERIA = [
    {"criterion_id": "C1", "name": "Definition of Containerisation", "score": 5, "max_score": 5,
     "justification": "Accurate definition and clear VM distinction provided."},
    {"criterion_id": "C2", "name": "Benefits of Containerisation", "score": 5, "max_score": 5,
     "justification": "Five distinct benefits explained with sufficient detail."},
    {"criterion_id": "C3", "name": "Role of Kubernetes", "score": 4, "max_score": 5,
     "justification": "Good Kubernetes explanation; cloud integrations mentioned."},
    {"criterion_id": "C4", "name": "Technical Accuracy and Depth", "score": 4, "max_score": 5,
     "justification": "Technically accurate with appropriate depth throughout."},
]

# Scores that are clearly unfair (penalised heavily despite a strong submission)
_UNFAIR_SCORED_CRITERIA = [
    {"criterion_id": "C1", "name": "Definition of Containerisation", "score": 0, "max_score": 5,
     "justification": "No definition given."},
    {"criterion_id": "C2", "name": "Benefits of Containerisation", "score": 0, "max_score": 5,
     "justification": "No benefits listed."},
    {"criterion_id": "C3", "name": "Role of Kubernetes", "score": 0, "max_score": 5,
     "justification": "Kubernetes not mentioned."},
    {"criterion_id": "C4", "name": "Technical Accuracy and Depth", "score": 0, "max_score": 5,
     "justification": "No technical content."},
]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _ollama_available() -> bool:
    """Return True if the Ollama REST API is reachable."""
    try:
        resp = requests.get(f"{_OLLAMA_BASE_URL}/api/tags", timeout=3)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


def _model_available() -> bool:
    """Return True if the required phi4-mini model is pulled in Ollama."""
    try:
        resp = requests.get(f"{_OLLAMA_BASE_URL}/api/tags", timeout=3)
        if resp.status_code != 200:
            return False
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        return any(_MODEL in name for name in models)
    except requests.ConnectionError:
        return False


def _ask_judge(submission: str, rubric: dict, scored_criteria: list[dict]) -> str:
    """Ask phi4-mini via the Ollama generate endpoint if the scores are fair.

    Returns the raw model response text with leading/trailing whitespace stripped.
    """
    criteria_lines = "\n".join(
        f"  - {c['name']} ({c['criterion_id']}): {c['score']}/{c['max_score']} "
        f"– {c['justification']}"
        for c in scored_criteria
    )
    total_score = sum(c["score"] for c in scored_criteria)
    total_marks = rubric["total_marks"]

    prompt = (
        "You are an impartial academic grading auditor.\n\n"
        f"## Student Submission\n\n{submission}\n\n"
        f"## Rubric\n\nModule: {rubric['module']}\n"
        f"Assignment: {rubric['assignment']}\n\n"
        "Criteria:\n"
        + "\n".join(
            f"  - {c['name']} ({c['id']}): max {c['max_score']} marks – {c['description']}"
            for c in rubric["criteria"]
        )
        + f"\n\n## Assigned Scores\n\n{criteria_lines}\n"
        f"\nTotal: {total_score}/{total_marks}\n\n"
        "Based solely on the submission content and the rubric criteria above, "
        "are the assigned scores fair?\n\n"
        "Reply with ONLY one word: YES or NO."
    )

    payload = {
        "model": _MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }
    resp = requests.post(
        f"{_OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


# ── Fixtures / marks ──────────────────────────────────────────────────────────

requires_ollama = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama service not reachable at localhost:11434",
)

requires_model = pytest.mark.skipif(
    not _model_available(),
    reason=f"Model {_MODEL!r} is not pulled in Ollama",
)


# ── Tests ─────────────────────────────────────────────────────────────────────


@requires_ollama
@requires_model
class TestLLMJudge:
    def test_fair_scores_judged_yes(self):
        """phi4-mini should confirm that high, well-justified scores are fair."""
        answer = _ask_judge(_SUBMISSION, _RUBRIC, _SCORED_CRITERIA)
        assert answer.upper() in {"YES", "NO"}, (
            f"Judge must respond with exactly YES or NO, got: {answer!r}"
        )
        assert answer.upper() == "YES", (
            f"Expected the scores to be judged as fair (YES), got: {answer!r}"
        )

    def test_unfair_scores_judged_no(self):
        """phi4-mini should flag zero scores for a strong submission as unfair."""
        answer = _ask_judge(_SUBMISSION, _RUBRIC, _UNFAIR_SCORED_CRITERIA)
        assert answer.upper() in {"YES", "NO"}, (
            f"Judge must respond with exactly YES or NO, got: {answer!r}"
        )
        assert answer.upper() == "NO", (
            f"Expected zero scores to be judged as unfair (NO), got: {answer!r}"
        )

    def test_response_is_yes_or_no(self):
        """The model response must be strictly YES or NO (no extra words)."""
        answer = _ask_judge(_SUBMISSION, _RUBRIC, _SCORED_CRITERIA)
        assert answer.upper() in {"YES", "NO"}, (
            f"Judge must respond with exactly YES or NO, got: {answer!r}"
        )
