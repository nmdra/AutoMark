"""LLM factory functions for the ctse-mas system.

All inference runs locally via Ollama – no external API calls are made.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_ollama import ChatOllama

if TYPE_CHECKING:
    from pydantic import BaseModel

from mas.config import settings


def get_json_llm(schema: type[Any] | None = None) -> ChatOllama:
    """Return a ChatOllama instance configured for deterministic JSON output.

    Parameters
    ----------
    schema:
        Optional Pydantic ``BaseModel`` subclass.  When provided the model is
        instructed to return JSON that conforms to the schema's JSON Schema,
        leveraging Ollama's token-sampling ``format`` parameter.
    """
    kwargs: dict[str, Any] = {
        "model": settings.model_name,
        "base_url": settings.ollama_base_url,
        "temperature": 0.0,
        "format": "json",
    }
    llm = ChatOllama(**kwargs)
    if schema is not None:
        return llm.with_structured_output(schema)
    return llm


def get_prose_llm() -> ChatOllama:
    """Return a ChatOllama instance configured for natural-language prose output.

    Intended *exclusively* for the Report Agent.
    """
    return ChatOllama(
        model=settings.model_name,
        base_url=settings.ollama_base_url,
        temperature=0.3,
    )
