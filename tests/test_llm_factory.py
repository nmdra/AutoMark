"""Tests for LLM factory structured-output behavior."""

from __future__ import annotations

from types import SimpleNamespace

from pydantic import BaseModel

import mas.llm as llm_module
from mas.tools.prompt_cache import PromptPrefixCache


class _DummySchema(BaseModel):
    value: str


def _reset_llm_caches() -> None:
    """Clear cached LLM clients/wrappers so each test starts from a clean slate."""
    llm_module._json_llm_instances.clear()
    llm_module._light_json_llm_instances.clear()
    llm_module._metadata_json_llm_instances.clear()
    llm_module._plain_json_llm_instance = None
    llm_module._plain_light_json_llm_instance = None
    llm_module._plain_metadata_json_llm_instance = None


def test_get_json_llm_uses_langchain_structured_wrapper(monkeypatch):
    _reset_llm_caches()

    class FakeChatOllama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def with_structured_output(self, schema):
            return {"structured": True, "schema": schema}

    monkeypatch.setattr(llm_module, "ChatOllama", FakeChatOllama)

    client = llm_module.get_json_llm(schema=_DummySchema)

    assert client["structured"] is True
    assert client["schema"] is _DummySchema


def test_prefix_cached_ollama_warmup_payload_omits_raw(monkeypatch):
    llm_module._analysis_prefix_prompt_cache = PromptPrefixCache(
        max_entries=8,
        ttl_seconds=300,
    )
    captured_payloads: list[dict] = []

    def fake_generate(payload):
        captured_payloads.append(payload)
        if len(captured_payloads) == 1:
            return {"context": [10, 11], "prompt_eval_count": 8, "eval_count": 1}
        return {
            "response": '{"scores":[]}',
            "prompt_eval_count": 4,
            "eval_count": 2,
            "model": "phi4-mini:3.8b-q4_K_M",
        }

    monkeypatch.setattr(llm_module, "_ollama_generate", fake_generate)

    client = llm_module.get_analysis_prefix_cached_json_llm(
        system_prompt="sys",
        system_prompt_version="v1",
        rubric_hash="rubric-hash",
        prefix_context="rubric only",
        submission_prompt="submission only",
    )
    response = client.invoke([SimpleNamespace(content="ignored")])

    assert len(captured_payloads) == 2
    warmup_payload = captured_payloads[0]
    assert "raw" not in warmup_payload
    assert warmup_payload["prompt"] == "rubric only"
    assert warmup_payload["system"] == "sys"
    assert response.content == '{"scores":[]}'
