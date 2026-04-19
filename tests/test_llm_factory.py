"""Tests for LLM factory structured-output behavior."""

from __future__ import annotations

from pydantic import BaseModel

import mas.llm as llm_module


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
