"""Structured JSON logger for agent and model-call observability."""

from __future__ import annotations

import sys
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import structlog

from mas.config import settings

_LOG_FILE = Path(settings.log_file)
_JSON_RENDERER = structlog.processors.JSONRenderer()
_CONSOLE_RENDERER = structlog.dev.ConsoleRenderer(
    colors=True,
    pad_event_to=30,
    sort_keys=False,
)
_METRICS_LOCK = threading.Lock()
_MEDIAN_WINDOW_SIZE = 100
_LATENCY_WINDOWS: dict[tuple[str, str, str], deque[float]] = defaultdict(
    lambda: deque(maxlen=_MEDIAN_WINDOW_SIZE)
)
_TOKEN_WINDOWS: dict[tuple[str, str, str], deque[int]] = defaultdict(
    lambda: deque(maxlen=_MEDIAN_WINDOW_SIZE)
)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _extract_token_usage(response: Any) -> dict[str, int | None]:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    usage_metadata = getattr(response, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        prompt_tokens = _as_int(
            usage_metadata.get("input_tokens") or usage_metadata.get("prompt_tokens")
        )
        completion_tokens = _as_int(
            usage_metadata.get("output_tokens") or usage_metadata.get("completion_tokens")
        )
        total_tokens = _as_int(usage_metadata.get("total_tokens"))

    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, dict):
        prompt_tokens = prompt_tokens or _as_int(
            response_metadata.get("prompt_eval_count")
            or response_metadata.get("input_tokens")
        )
        completion_tokens = completion_tokens or _as_int(
            response_metadata.get("eval_count") or response_metadata.get("output_tokens")
        )
        total_tokens = total_tokens or _as_int(response_metadata.get("total_tokens"))

    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _emit_entry(entry: dict[str, Any]) -> None:
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    rendered_json = _JSON_RENDERER(None, "info", dict(entry))
    with _LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(rendered_json + "\n")

    try:
        console_event: dict[str, Any] = {
            "event": f"{entry.get('event_type', 'event')}:{entry.get('status', 'unknown')}"
        }
        for key in (
            "service",
            "session_id",
            "agent",
            "action",
            "task_type",
            "model",
            "latency_ms",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "error",
        ):
            value = entry.get(key)
            if value not in (None, ""):
                console_event[key] = value
        print(
            _CONSOLE_RENDERER(None, "info", console_event),
            file=sys.stdout,
            flush=True,
        )
    except BrokenPipeError:
        pass


def log_agent_action(
    *,
    session_id: str,
    agent: str,
    action: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
) -> dict[str, Any]:
    """Record one agent action as a structured JSON log entry."""
    status = outputs.get("status", "success")
    if not isinstance(status, str):
        status = "success"

    entry: dict[str, Any] = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "event_type": "agent_action",
        "session_id": session_id,
        "service": agent,
        "status": status,
        "agent": agent,
        "action": action,
        "inputs": inputs,
        "outputs": outputs,
        "details": {
            "inputs": inputs,
            "outputs": outputs,
        },
    }
    _emit_entry(entry)
    return entry


def log_model_call(
    *,
    session_id: str,
    service: str,
    task_type: str,
    model: str,
    latency_ms: float,
    status: str,
    response: Any = None,
    error: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record one model invocation as a structured JSON log entry."""
    tokens = _extract_token_usage(response)
    response_metadata = getattr(response, "response_metadata", None)
    response_model = (
        response_metadata.get("model")
        if isinstance(response_metadata, dict)
        else None
    )
    resolved_model = str(response_model or model)
    metrics_key = (str(service), str(task_type), resolved_model)
    with _METRICS_LOCK:
        _LATENCY_WINDOWS[metrics_key].append(float(latency_ms))
        if isinstance(tokens["total_tokens"], int):
            _TOKEN_WINDOWS[metrics_key].append(tokens["total_tokens"])
        median_latency_ms = float(median(_LATENCY_WINDOWS[metrics_key]))
        median_total_tokens = (
            int(median(_TOKEN_WINDOWS[metrics_key]))
            if _TOKEN_WINDOWS[metrics_key]
            else None
        )

    entry: dict[str, Any] = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "event_type": "model_call",
        "session_id": session_id,
        "service": service,
        "status": status,
        "model": resolved_model,
        "task_type": str(task_type),
        "latency_ms": round(latency_ms, 2),
        "prompt_tokens": tokens["prompt_tokens"],
        "completion_tokens": tokens["completion_tokens"],
        "total_tokens": tokens["total_tokens"],
        "details": {
            "model": resolved_model,
            "task_type": str(task_type),
            "latency_ms": round(latency_ms, 2),
            "tokens": tokens,
            "median_latency_ms": round(median_latency_ms, 2),
            "median_total_tokens": median_total_tokens,
        },
    }
    if metadata:
        entry["details"]["metadata"] = metadata
        for key, value in metadata.items():
            if isinstance(value, bool | int | float | str):
                entry[f"meta_{key}"] = value
    if error:
        entry["error"] = error
        entry["details"]["error"] = error

    _emit_entry(entry)
    return entry


def timed_model_call(
    *,
    llm: Any,
    messages: list[Any],
    session_id: str,
    service: str,
    task_type: str,
    model: str,
) -> Any:
    """Invoke an LLM and log model, task type, tokens, latency, and status."""
    started = perf_counter()
    response = None
    try:
        response = llm.invoke(messages)
    except Exception as exc:
        log_model_call(
            session_id=session_id,
            service=service,
            task_type=task_type,
            model=model,
            latency_ms=(perf_counter() - started) * 1000,
            status="error",
            response=response,
            error=str(exc),
            metadata=None,
        )
        raise

    response_metadata = getattr(response, "model_call_metadata", None)
    log_model_call(
        session_id=session_id,
        service=service,
        task_type=task_type,
        model=model,
        latency_ms=(perf_counter() - started) * 1000,
        status="success",
        response=response,
        metadata=response_metadata if isinstance(response_metadata, dict) else None,
    )
    return response
