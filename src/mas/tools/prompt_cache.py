"""Prompt-prefix cache primitives for Ollama context reuse."""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter
from typing import Callable


@dataclass(frozen=True)
class PromptCacheKey:
    """Cache key for a static prompt prefix context."""

    model: str
    system_prompt_version: str
    rubric_hash: str


@dataclass
class PromptCacheEntry:
    """Cached prefix context returned by Ollama."""

    context_tokens: list[int]
    created_at: float
    expires_at: float


class PromptPrefixCache:
    """Thread-safe bounded LRU+TTL cache with warmup de-duplication."""

    def __init__(self, *, max_entries: int, ttl_seconds: int) -> None:
        self._max_entries = max(1, int(max_entries))
        self._ttl_seconds = max(1, int(ttl_seconds))
        self._entries: OrderedDict[PromptCacheKey, PromptCacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._inflight: dict[PromptCacheKey, threading.Event] = {}

    def get_or_warm(
        self,
        *,
        key: PromptCacheKey,
        warmup: Callable[[], list[int]],
        now: Callable[[], float],
    ) -> tuple[PromptCacheEntry, bool, float]:
        """Get cached entry or warm one and cache it.

        Returns (entry, cache_hit, warmup_ms).
        """
        while True:
            with self._lock:
                entry = self._entries.get(key)
                current = now()
                if entry is not None and entry.expires_at > current:
                    self._entries.move_to_end(key)
                    return entry, True, 0.0

                if entry is not None:
                    del self._entries[key]

                in_flight = self._inflight.get(key)
                if in_flight is not None:
                    wait_event = in_flight
                else:
                    wait_event = threading.Event()
                    self._inflight[key] = wait_event
                    break

            wait_event.wait()

        started = perf_counter()
        try:
            context_tokens = warmup()
            current = now()
            warmed = PromptCacheEntry(
                context_tokens=context_tokens,
                created_at=current,
                expires_at=current + self._ttl_seconds,
            )
            with self._lock:
                self._entries[key] = warmed
                self._entries.move_to_end(key)
                while len(self._entries) > self._max_entries:
                    self._entries.popitem(last=False)
            return warmed, False, (perf_counter() - started) * 1000
        finally:
            with self._lock:
                event = self._inflight.pop(key, None)
                if event is not None:
                    event.set()


def hash_rubric_payload(payload: dict) -> str:
    """Create a stable rubric hash for cache keying."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
