"""LLM factory functions for the ctse-mas system.

All inference runs locally via Ollama – no external API calls are made.

Instances are cached at module level so that the same ``ChatOllama`` object
(and its underlying HTTP session) is reused across all agent calls within a
process, avoiding repeated object construction overhead.

For parallel LLM requests (e.g. the finalize agent running insights + report
concurrently), configure ``OLLAMA_NUM_PARALLEL >= 2`` on the Ollama server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_ollama import ChatOllama

if TYPE_CHECKING:
    from pydantic import BaseModel

from mas.config import settings

# Keep the model loaded in Ollama memory between consecutive requests.
_KEEP_ALIVE = "10m"

# ── Cached singleton instances ─────────────────────────────────────────────────
# Keyed by schema to support both structured-output and prose variants.
_prose_llm_instance: ChatOllama | None = None
_json_llm_instances: dict[type, Any] = {}
_plain_json_llm_instance: ChatOllama | None = None


def get_json_llm(schema: type[Any] | None = None) -> ChatOllama:
    """Return a ChatOllama instance configured for deterministic JSON output.

    Parameters
    ----------
    schema:
        Optional Pydantic ``BaseModel`` subclass.  When provided the model is
        instructed to return JSON that conforms to the schema's JSON Schema,
        leveraging Ollama's token-sampling ``format`` parameter.

    The returned instance (or its wrapped structured-output counterpart) is
    cached globally so that the underlying HTTP session is reused.
    """
    global _plain_json_llm_instance  # noqa: PLW0603

    kwargs: dict[str, Any] = {
        "model": settings.model_name,
        "base_url": settings.ollama_base_url,
        "temperature": 0.0,
        "format": "json",
        "num_ctx": settings.num_ctx,
        "num_predict": settings.num_predict,
        "keep_alive": _KEEP_ALIVE,
    }

    if schema is not None:
        # Structured-output wrappers are lightweight; cache by schema type.
        if schema not in _json_llm_instances:
            if _plain_json_llm_instance is None:
                _plain_json_llm_instance = ChatOllama(**kwargs)
            _json_llm_instances[schema] = _plain_json_llm_instance.with_structured_output(schema)
        return _json_llm_instances[schema]

    if _plain_json_llm_instance is None:
        _plain_json_llm_instance = ChatOllama(**kwargs)
    return _plain_json_llm_instance


def get_prose_llm() -> ChatOllama:
    """Return a ChatOllama instance configured for natural-language prose output.

    Intended *exclusively* for the Report Agent and Historical Agent.
    The instance is cached globally to reuse the underlying HTTP session.
    """
    global _prose_llm_instance  # noqa: PLW0603

    if _prose_llm_instance is None:
        _prose_llm_instance = ChatOllama(
            model=settings.model_name,
            base_url=settings.ollama_base_url,
            temperature=0.3,
            num_ctx=settings.num_ctx,
            num_predict=settings.num_predict,
            keep_alive=_KEEP_ALIVE,
        )
    return _prose_llm_instance
