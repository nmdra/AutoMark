"""Tests for LLM factory structured-output behavior."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

from pydantic import BaseModel

import mas.llm as llm_module


class _DummySchema(BaseModel):
    value: str


def _reset_llm_caches() -> None:
    """Clear cached LLM clients/wrappers so each test starts from a clean slate."""
    llm_module._json_llm_instances.clear()
    llm_module._light_json_llm_instances.clear()
    llm_module._metadata_json_llm_instances.clear()
    llm_module._outlines_json_llm_instances.clear()
    llm_module._plain_json_llm_instance = None
    llm_module._plain_light_json_llm_instance = None
    llm_module._plain_metadata_json_llm_instance = None


def test_get_json_llm_prefers_outlines_wrapper(monkeypatch):
    _reset_llm_caches()

    created: list[dict] = []

    class FakeOutlinesWrapper:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(llm_module, "_OutlinesStructuredJSONLLM", FakeOutlinesWrapper)

    client = llm_module.get_json_llm(schema=_DummySchema)

    assert isinstance(client, FakeOutlinesWrapper)
    assert created and created[0]["schema"] is _DummySchema


def test_get_json_llm_falls_back_when_outlines_fails(monkeypatch):
    _reset_llm_caches()

    class FakeChatOllama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def with_structured_output(self, schema):
            return {"fallback": True, "schema": schema}

    def _raise(**kwargs):
        raise ImportError("outlines unavailable")

    monkeypatch.setattr(llm_module, "_OutlinesStructuredJSONLLM", _raise)
    monkeypatch.setattr(llm_module, "ChatOllama", FakeChatOllama)

    client = llm_module.get_json_llm(schema=_DummySchema)

    assert client["fallback"] is True
    assert client["schema"] is _DummySchema


def test_outlines_wrapper_invoke_parses_json_string():
    wrapper = llm_module._OutlinesStructuredJSONLLM.__new__(
        llm_module._OutlinesStructuredJSONLLM
    )
    wrapper._schema = _DummySchema
    wrapper._generator = MagicMock(return_value='{"value":"ok"}')
    wrapper._options = {}
    wrapper._keep_alive = "10m"

    response = wrapper.invoke([SimpleNamespace(type="human", content="hello")])

    assert isinstance(response, _DummySchema)
    assert response.value == "ok"
    wrapper._generator.assert_called_once()


def test_outlines_wrapper_init_sets_generator_and_options(monkeypatch):
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, host: str):
            captured["host"] = host

    fake_outlines = types.ModuleType("outlines")

    def _from_ollama(client, model_name):
        captured["model_name"] = model_name
        return {"client": client, "model_name": model_name}

    fake_outlines.from_ollama = _from_ollama

    fake_outlines_generator = types.ModuleType("outlines.generator")

    def _generator(model, output_type):
        captured["generator_model"] = model
        captured["output_type"] = output_type
        return MagicMock()

    fake_outlines_generator.Generator = _generator

    fake_ollama = types.ModuleType("ollama")
    fake_ollama.Client = FakeClient

    monkeypatch.setitem(sys.modules, "outlines", fake_outlines)
    monkeypatch.setitem(sys.modules, "outlines.generator", fake_outlines_generator)
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)

    wrapper = llm_module._OutlinesStructuredJSONLLM(
        model_name="demo-model",
        base_url="http://localhost:11434",
        schema=_DummySchema,
        temperature=0.0,
        num_ctx=2048,
        num_predict=512,
        keep_alive="10m",
    )

    assert captured["host"] == "http://localhost:11434"
    assert captured["model_name"] == "demo-model"
    assert captured["output_type"] is _DummySchema
    assert wrapper._options == {"temperature": 0.0, "num_ctx": 2048, "num_predict": 512}
    assert wrapper._keep_alive == "10m"
