"""LLM factory functions for the ctse-mas system.

All inference runs locally via Ollama – no external API calls are made.

Two model tiers are used to balance quality and speed:

* **Analysis model** (``settings.analysis_model_name``, default
  ``phi4-mini:3.8b-q4_K_M``) – used for tasks that require deep reading
  comprehension or long coherent generation:

  - :func:`get_json_llm` – structured rubric scoring (Analysis Agent)
  - :func:`get_prose_llm` – feedback-report prose generation (Report Agent /
    Finalize Agent report task)

* **Light model** (``settings.light_model_name``, default
  ``gemma3:1b-it-q4_K_M``) – used for lightweight prose generation:

  - :func:`get_light_prose_llm` – short progression-insight paragraph
    (Historical Agent / Finalize Agent insights task)

* **Metadata extractor model**
  (``settings.metadata_extractor_model_name``, default
  ``hf.co/nimendraai/SmolLM2-360M-Assignment-Metadata-Extractor:Q4_K_M``) –
  used for strict JSON extraction of student metadata from noisy assignment
  text:

  - :func:`get_metadata_json_llm` – student number/name/assignment number
    extraction for text and PDF ingestion agents

All instances are cached at module level so that the same ``ChatOllama``
object (and its underlying HTTP session) is reused across agent calls within
a process, avoiding repeated object-construction overhead.

For schema-constrained JSON extraction/scoring, this module uses LangChain's
structured output wrappers.

For parallel LLM requests (e.g. the finalize agent running the light-model
insights call and the analysis-model report call concurrently), configure
``OLLAMA_NUM_PARALLEL >= 2`` on the Ollama server.
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
_prose_llm_instance: ChatOllama | None = None
_json_llm_instances: dict[type, Any] = {}
_plain_json_llm_instance: ChatOllama | None = None

_light_prose_llm_instance: ChatOllama | None = None
_light_json_llm_instances: dict[type, Any] = {}
_plain_light_json_llm_instance: ChatOllama | None = None

_metadata_json_llm_instances: dict[type, Any] = {}
_plain_metadata_json_llm_instance: ChatOllama | None = None

# ── Analysis-model factory functions ──────────────────────────────────────────

def get_json_llm(schema: type[Any] | None = None) -> ChatOllama:
    """Return a ``ChatOllama`` instance using the *analysis model* for JSON output.

    Use this for computationally demanding tasks that need deep reading
    comprehension – specifically the Analysis Agent's rubric-scoring call.

    Parameters
    ----------
    schema:
        Optional Pydantic ``BaseModel`` subclass.  When provided the model is
        instructed to return JSON conforming to the schema's JSON Schema.

    The returned instance (or its structured-output wrapper) is cached
    globally so that the underlying HTTP session is reused.
    """
    global _plain_json_llm_instance  # noqa: PLW0603

    kwargs: dict[str, Any] = {
        "model": settings.analysis_model_name,
        "base_url": settings.ollama_base_url,
        "temperature": 0.0,
        "format": "json",
        "num_ctx": settings.num_ctx,
        "num_predict": settings.num_predict,
        "keep_alive": _KEEP_ALIVE,
    }

    if schema is not None:
        if schema not in _json_llm_instances:
            if _plain_json_llm_instance is None:
                _plain_json_llm_instance = ChatOllama(**kwargs)
            _json_llm_instances[schema] = _plain_json_llm_instance.with_structured_output(
                schema
            )
        return _json_llm_instances[schema]

    if _plain_json_llm_instance is None:
        _plain_json_llm_instance = ChatOllama(**kwargs)
    return _plain_json_llm_instance


def get_prose_llm() -> ChatOllama:
    """Return a ``ChatOllama`` instance using the *analysis model* for prose output.

    Use this for long-form natural-language generation tasks – specifically
    the Report Agent / Finalize Agent feedback-report prose call.

    The instance is cached globally to reuse the underlying HTTP session.
    """
    global _prose_llm_instance  # noqa: PLW0603

    if _prose_llm_instance is None:
        _prose_llm_instance = ChatOllama(
            model=settings.analysis_model_name,
            base_url=settings.ollama_base_url,
            temperature=0.3,
            num_ctx=settings.num_ctx,
            num_predict=settings.num_predict,
            keep_alive=_KEEP_ALIVE,
        )
    return _prose_llm_instance


# ── Light-model factory functions ──────────────────────────────────────────────

def get_light_json_llm(schema: type[Any] | None = None) -> ChatOllama:
    """Return a ``ChatOllama`` instance using the *light model* for JSON output.

    Use this for lightweight structured-extraction tasks where a small model
    is sufficient – specifically the PDF Ingestion Agent's student-detail
    extraction fallback.

    Parameters
    ----------
    schema:
        Optional Pydantic ``BaseModel`` subclass for structured output.

    The returned instance (or its structured-output wrapper) is cached
    globally so that the underlying HTTP session is reused.
    """
    global _plain_light_json_llm_instance  # noqa: PLW0603

    kwargs: dict[str, Any] = {
        "model": settings.light_model_name,
        "base_url": settings.ollama_base_url,
        "temperature": 0.0,
        "format": "json",
        "num_ctx": settings.num_ctx,
        "num_predict": settings.num_predict,
        "keep_alive": _KEEP_ALIVE,
    }

    if schema is not None:
        if schema not in _light_json_llm_instances:
            if _plain_light_json_llm_instance is None:
                _plain_light_json_llm_instance = ChatOllama(**kwargs)
            _light_json_llm_instances[schema] = (
                _plain_light_json_llm_instance.with_structured_output(schema)
            )
        return _light_json_llm_instances[schema]

    if _plain_light_json_llm_instance is None:
        _plain_light_json_llm_instance = ChatOllama(**kwargs)
    return _plain_light_json_llm_instance


def get_light_prose_llm() -> ChatOllama:
    """Return a ``ChatOllama`` instance using the *light model* for prose output.

    Use this for short natural-language generation tasks where a small model
    is sufficient – specifically the Historical Agent / Finalize Agent
    progression-insights call.

    The instance is cached globally to reuse the underlying HTTP session.
    """
    global _light_prose_llm_instance  # noqa: PLW0603

    if _light_prose_llm_instance is None:
        _light_prose_llm_instance = ChatOllama(
            model=settings.light_model_name,
            base_url=settings.ollama_base_url,
            temperature=0.3,
            num_ctx=settings.num_ctx,
            num_predict=settings.num_predict,
            keep_alive=_KEEP_ALIVE,
        )
    return _light_prose_llm_instance


def get_metadata_json_llm(schema: type[Any] | None = None) -> ChatOllama:
    """Return metadata-extractor JSON client (or structured wrapper when schema is provided).

    Use this for robust extraction of student metadata fields from noisy
    assignment text/PDF snippets.
    """
    global _plain_metadata_json_llm_instance  # noqa: PLW0603

    kwargs: dict[str, Any] = {
        "model": settings.metadata_extractor_model_name,
        "base_url": settings.ollama_base_url,
        "temperature": 0.0,
        "format": "json",
        "num_ctx": settings.num_ctx,
        "num_predict": settings.num_predict,
        "keep_alive": _KEEP_ALIVE,
    }

    if schema is not None:
        if schema not in _metadata_json_llm_instances:
            if _plain_metadata_json_llm_instance is None:
                _plain_metadata_json_llm_instance = ChatOllama(**kwargs)
            _metadata_json_llm_instances[schema] = (
                _plain_metadata_json_llm_instance.with_structured_output(schema)
            )
        return _metadata_json_llm_instances[schema]

    if _plain_metadata_json_llm_instance is None:
        _plain_metadata_json_llm_instance = ChatOllama(**kwargs)
    return _plain_metadata_json_llm_instance
