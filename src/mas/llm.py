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

import json
from dataclasses import dataclass
from time import perf_counter, time
from typing import TYPE_CHECKING, Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from langchain_ollama import ChatOllama

if TYPE_CHECKING:
    from pydantic import BaseModel

from mas.config import settings
from mas.tools.prompt_cache import (
    PromptCacheEntry,
    PromptCacheKey,
    PromptPrefixCache,
)

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
_analysis_prefix_prompt_cache = PromptPrefixCache(
    max_entries=settings.prompt_cache_max_entries,
    ttl_seconds=settings.prompt_cache_ttl_seconds,
)

# ── Analysis-model factory functions ──────────────────────────────────────────

def get_json_llm(schema: type[Any] | None = None) -> Any:
    """Return analysis-model JSON client or structured-output wrapper.

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

def get_light_json_llm(schema: type[Any] | None = None) -> Any:
    """Return light-model JSON client or structured-output wrapper.

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


def get_metadata_json_llm(schema: type[Any] | None = None) -> Any:
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


@dataclass
class OllamaRawResponse:
    """Minimal response wrapper consumed by timed_model_call observability."""

    content: str
    usage_metadata: dict[str, int] | None = None
    response_metadata: dict[str, Any] | None = None
    model_call_metadata: dict[str, Any] | None = None


def _ollama_generate(payload: dict[str, Any]) -> dict[str, Any]:
    """Call Ollama /api/generate with JSON payload."""
    base_url = settings.ollama_base_url.rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError(
            "Invalid AUTOMARK_OLLAMA_BASE_URL; expected http(s) URL with host"
        )
    url = f"{base_url}/api/generate"
    req = Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=90) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Ollama prefix-cache request failed: {exc}") from exc


def _extract_usage_from_ollama(payload: dict[str, Any]) -> dict[str, int]:
    prompt_tokens = int(payload.get("prompt_eval_count") or 0)
    completion_tokens = int(payload.get("eval_count") or 0)
    return {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


class OllamaPrefixCachedJsonClient:
    """Provider-specific Ollama JSON client with prefix context reuse."""

    def __init__(
        self,
        *,
        model: str,
        system_prompt: str,
        system_prompt_version: str,
        rubric_hash: str,
        prefix_context: str,
        submission_prompt: str,
    ) -> None:
        self._model = model
        self._system_prompt = system_prompt
        self._system_prompt_version = system_prompt_version
        self._rubric_hash = rubric_hash
        self._prefix_context = prefix_context
        self._submission_prompt = submission_prompt

    def _warmup_prefix(self) -> list[int]:
        warmup_payload = {
            "model": self._model,
            "system": self._system_prompt,
            "prompt": self._prefix_context,
            "stream": False,
            "keep_alive": _KEEP_ALIVE,
            "options": {
                "temperature": 0.0,
                "num_ctx": settings.num_ctx,
                "num_predict": 1,
            },
        }
        warmup = _ollama_generate(warmup_payload)
        context_tokens = warmup.get("context")
        if not isinstance(context_tokens, list) or not context_tokens:
            raise RuntimeError("Ollama prefix warmup did not return reusable context")
        return [int(token) for token in context_tokens]

    def invoke(self, messages: list[Any]) -> OllamaRawResponse:  # noqa: ARG002
        cache_key = PromptCacheKey(
            model=self._model,
            system_prompt_version=self._system_prompt_version,
            rubric_hash=self._rubric_hash,
        )
        prefix_entry: PromptCacheEntry
        prefix_entry, cache_hit, warmup_ms = _analysis_prefix_prompt_cache.get_or_warm(
            key=cache_key,
            warmup=self._warmup_prefix,
            now=time,
        )

        started = perf_counter()
        scoring_payload = {
            "model": self._model,
            "system": self._system_prompt,
            "prompt": self._submission_prompt,
            "context": prefix_entry.context_tokens,
            "format": "json",
            "stream": False,
            "keep_alive": _KEEP_ALIVE,
            "options": {
                "temperature": 0.0,
                "num_ctx": settings.num_ctx,
                "num_predict": settings.num_predict,
            },
        }
        response = _ollama_generate(scoring_payload)
        analysis_latency_ms = (perf_counter() - started) * 1000
        response_metadata = {
            "model": response.get("model") or self._model,
            "prompt_eval_count": response.get("prompt_eval_count"),
            "eval_count": response.get("eval_count"),
            "total_duration": response.get("total_duration"),
            "eval_duration": response.get("eval_duration"),
        }
        return OllamaRawResponse(
            content=str(response.get("response") or ""),
            usage_metadata=_extract_usage_from_ollama(response),
            response_metadata=response_metadata,
            model_call_metadata={
                "cache_hit": cache_hit,
                "warmup_ms": round(float(warmup_ms), 2),
                "analysis_latency_ms": round(float(analysis_latency_ms), 2),
            },
        )


def get_analysis_prefix_cached_json_llm(
    *,
    system_prompt: str,
    system_prompt_version: str,
    rubric_hash: str,
    prefix_context: str,
    submission_prompt: str,
) -> OllamaPrefixCachedJsonClient:
    """Return provider-specific Ollama client that reuses warmed prefix context."""
    return OllamaPrefixCachedJsonClient(
        model=settings.analysis_model_name,
        system_prompt=system_prompt,
        system_prompt_version=system_prompt_version,
        rubric_hash=rubric_hash,
        prefix_context=prefix_context,
        submission_prompt=submission_prompt,
    )
