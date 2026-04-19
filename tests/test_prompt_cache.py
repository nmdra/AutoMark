"""Tests for prompt-prefix caching primitives."""

from __future__ import annotations

import time

from mas.tools.prompt_cache import (
    PromptCacheKey,
    PromptPrefixCache,
    hash_rubric_payload,
)


def test_prompt_cache_key_changes_with_model_version_and_rubric_hash():
    k1 = PromptCacheKey(model="m1", system_prompt_version="v1", rubric_hash="r1")
    k2 = PromptCacheKey(model="m2", system_prompt_version="v1", rubric_hash="r1")
    k3 = PromptCacheKey(model="m1", system_prompt_version="v2", rubric_hash="r1")
    k4 = PromptCacheKey(model="m1", system_prompt_version="v1", rubric_hash="r2")

    assert k1 != k2
    assert k1 != k3
    assert k1 != k4


def test_rubric_hash_changes_when_content_changes():
    base = {"criteria": [{"id": "C1", "name": "Accuracy", "max_score": 5}]}
    changed = {"criteria": [{"id": "C1", "name": "Accuracy+", "max_score": 5}]}
    assert hash_rubric_payload(base) != hash_rubric_payload(changed)


def test_prompt_cache_ttl_invalidation():
    cache = PromptPrefixCache(max_entries=4, ttl_seconds=1)
    key = PromptCacheKey(model="m", system_prompt_version="v1", rubric_hash="r")
    now_value = 100.0

    def now() -> float:
        return now_value

    calls = {"count": 0}

    def warmup() -> list[int]:
        calls["count"] += 1
        return [1, 2, 3]

    _, hit1, _ = cache.get_or_warm(key=key, warmup=warmup, now=now)
    assert hit1 is False
    _, hit2, _ = cache.get_or_warm(key=key, warmup=warmup, now=now)
    assert hit2 is True
    now_value = 102.0
    _, hit3, _ = cache.get_or_warm(key=key, warmup=warmup, now=now)
    assert hit3 is False
    assert calls["count"] == 2


def test_prompt_cache_lru_eviction():
    cache = PromptPrefixCache(max_entries=1, ttl_seconds=60)
    now = time.time

    warmed: list[str] = []

    def warmup_for(value: str):
        def _warm() -> list[int]:
            warmed.append(value)
            return [1]

        return _warm

    k1 = PromptCacheKey(model="m1", system_prompt_version="v1", rubric_hash="r1")
    k2 = PromptCacheKey(model="m1", system_prompt_version="v1", rubric_hash="r2")

    cache.get_or_warm(key=k1, warmup=warmup_for("k1"), now=now)
    cache.get_or_warm(key=k2, warmup=warmup_for("k2"), now=now)
    _, hit, _ = cache.get_or_warm(key=k1, warmup=warmup_for("k1b"), now=now)

    assert hit is False
    assert warmed == ["k1", "k2", "k1b"]


def test_repeated_same_rubric_reuses_prefix_and_keeps_scoring_stable():
    cache = PromptPrefixCache(max_entries=8, ttl_seconds=300)
    rubric_hash = hash_rubric_payload(
        {"criteria": [{"id": "C1", "name": "Accuracy", "max_score": 5}]}
    )
    key = PromptCacheKey(model="phi4-mini", system_prompt_version="v1", rubric_hash=rubric_hash)
    submission = "Containerisation is OS-level virtualization."

    warmup_calls = {"count": 0}

    def warmup() -> list[int]:
        warmup_calls["count"] += 1
        time.sleep(0.01)
        return [42, 43]

    def grade_once() -> tuple[dict[str, int], bool, float]:
        _, cache_hit, warmup_ms = cache.get_or_warm(
            key=key,
            warmup=warmup,
            now=time.time,
        )
        score = 4 if "virtualization" in submission.lower() else 0
        return {"C1": score}, cache_hit, warmup_ms

    score1, hit1, warmup_ms1 = grade_once()
    score2, hit2, warmup_ms2 = grade_once()

    assert score1 == score2
    assert hit1 is False
    assert hit2 is True
    assert warmup_ms1 > 0
    assert warmup_ms2 == 0
    assert warmup_calls["count"] == 1
