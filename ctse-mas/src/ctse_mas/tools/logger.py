"""Structured JSON trace logger for agent observability."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOG_FILE = Path("agent_trace.log")


def log_agent_action(
    *,
    session_id: str,
    agent: str,
    action: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
) -> dict[str, Any]:
    """Record a single agent action as a structured JSON log entry.

    Writes one JSON object per line to ``agent_trace.log`` and returns the
    entry so callers can append it to ``AgentState["agent_logs"]``.
    """
    entry: dict[str, Any] = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "session_id": session_id,
        "agent": agent,
        "action": action,
        "inputs": inputs,
        "outputs": outputs,
    }
    with _LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry
