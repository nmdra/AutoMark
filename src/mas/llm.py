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

For schema-constrained JSON extraction/scoring, this module uses Outlines when
available, with automatic fallback to LangChain's structured output wrappers.

For parallel LLM requests (e.g. the finalize agent running the light-model
insights call and the analysis-model report call concurrently), configure
``OLLAMA_NUM_PARALLEL >= 2`` on the Ollama server.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
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

_outlines_json_llm_instances: dict[tuple[str, type], Any] = {}


class _OutlinesStructuredJSONLLM:
    """Outlines-backed structured JSON wrapper exposing LangChain-like ``invoke``."""

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str,
        schema: type[Any],
        temperature: float,
        num_ctx: int,
        num_predict: int,
        keep_alive: str,
    ) -> None:
        import outlines
        from outlines.inputs import Chat
        from ollama import Client as OllamaClient
        from outlines.generator import Generator

        self._chat_type = Chat
        self._schema = schema
        self._generator = Generator(
            outlines.from_ollama(
                OllamaClient(host=base_url),
                model_name=model_name,
            ),
            output_type=schema,
        )
        self._options: dict[str, Any] = {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        }
        self._keep_alive = keep_alive

    @staticmethod
    def _to_chat_messages(messages: list[Any]) -> list[dict[str, str]]:
        """Convert LangChain-style messages into role/content chat dictionaries."""
        role_map = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "tool": "tool",
            "function": "tool",
        }
        formatted: list[dict[str, str]] = []
        for message in messages:
            msg_type = str(getattr(message, "type", "human")).lower()
            role = role_map.get(msg_type, "user")
            content = getattr(message, "content", message)
            if isinstance(content, list):
                content = "\n".join(str(item) for item in content)
            formatted.append({"role": role, "content": str(content)})
        return formatted

    def invoke(self, messages: list[Any]) -> Any:
        chat_messages = self._to_chat_messages(messages)
        chat_input = self._chat_type(chat_messages)
        # outlines.models.ollama currently prints formatted input to stdout,
        # so silence it to keep agent logs/prompt content out of console output.
        with redirect_stdout(io.StringIO()):
            raw = self._generator(
                chat_input,
                options=self._options,
                keep_alive=self._keep_alive,
            )

        if isinstance(raw, self._schema):
            return raw
        if isinstance(raw, dict):
            return self._schema.model_validate(raw)
        if isinstance(raw, str):
            return self._schema.model_validate(json.loads(raw))
        raise TypeError(
            f"Unsupported structured output type from Outlines: {type(raw)!r}"
        )


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
            cache_key = ("analysis", schema)
            if cache_key in _outlines_json_llm_instances:
                _json_llm_instances[schema] = _outlines_json_llm_instances[cache_key]
            else:
                try:
                    llm = _OutlinesStructuredJSONLLM(
                        model_name=settings.analysis_model_name,
                        base_url=settings.ollama_base_url,
                        schema=schema,
                        temperature=0.0,
                        num_ctx=settings.num_ctx,
                        num_predict=settings.num_predict,
                        keep_alive=_KEEP_ALIVE,
                    )
                except (ImportError, ModuleNotFoundError, AttributeError, TypeError):
                    if _plain_json_llm_instance is None:
                        _plain_json_llm_instance = ChatOllama(**kwargs)
                    llm = _plain_json_llm_instance.with_structured_output(schema)
                _outlines_json_llm_instances[cache_key] = llm
                _json_llm_instances[schema] = llm
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
            cache_key = ("light", schema)
            if cache_key in _outlines_json_llm_instances:
                _light_json_llm_instances[schema] = _outlines_json_llm_instances[
                    cache_key
                ]
            else:
                try:
                    llm = _OutlinesStructuredJSONLLM(
                        model_name=settings.light_model_name,
                        base_url=settings.ollama_base_url,
                        schema=schema,
                        temperature=0.0,
                        num_ctx=settings.num_ctx,
                        num_predict=settings.num_predict,
                        keep_alive=_KEEP_ALIVE,
                    )
                except (ImportError, ModuleNotFoundError, AttributeError, TypeError):
                    if _plain_light_json_llm_instance is None:
                        _plain_light_json_llm_instance = ChatOllama(**kwargs)
                    llm = _plain_light_json_llm_instance.with_structured_output(schema)
                _outlines_json_llm_instances[cache_key] = llm
                _light_json_llm_instances[schema] = llm
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
            cache_key = ("metadata", schema)
            if cache_key in _outlines_json_llm_instances:
                _metadata_json_llm_instances[schema] = _outlines_json_llm_instances[
                    cache_key
                ]
            else:
                try:
                    llm = _OutlinesStructuredJSONLLM(
                        model_name=settings.metadata_extractor_model_name,
                        base_url=settings.ollama_base_url,
                        schema=schema,
                        temperature=0.0,
                        num_ctx=settings.num_ctx,
                        num_predict=settings.num_predict,
                        keep_alive=_KEEP_ALIVE,
                    )
                except (ImportError, ModuleNotFoundError, AttributeError, TypeError):
                    if _plain_metadata_json_llm_instance is None:
                        _plain_metadata_json_llm_instance = ChatOllama(**kwargs)
                    llm = _plain_metadata_json_llm_instance.with_structured_output(
                        schema
                    )
                _outlines_json_llm_instances[cache_key] = llm
                _metadata_json_llm_instances[schema] = llm
        return _metadata_json_llm_instances[schema]

    if _plain_metadata_json_llm_instance is None:
        _plain_metadata_json_llm_instance = ChatOllama(**kwargs)
    return _plain_metadata_json_llm_instance
