"""Structured JSON trace logger for agent observability."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mas.config import settings

_LOG_FILE = Path(settings.log_file)


def log_agent_action(
    *,
    session_id: str,
    agent: str,
    action: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
) -> dict[str, Any]:
    """Record a single agent action as a structured JSON log entry.

    Writes one JSON object per line to ``agent_trace.log``, prints a
    human-readable summary to stdout, and returns the entry so callers can
    append it to ``AgentState["agent_logs"]``.
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

    try:
        print(
            f"[{entry['timestamp']}] [{agent.upper()}] {action}",
            file=sys.stdout,
            flush=True,
        )
    except BrokenPipeError:
        pass

    return entry
