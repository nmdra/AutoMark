"""LangGraph pipeline definition and CLI entry point.

Pipeline flow:
    _detect_submission_type
        → ingestion (if .txt)     → (conditional) analysis → finalize
        → pdf_ingestion (if .pdf) → (conditional) analysis → finalize
                                          ↓ (ingestion_status != 'success')
                                        finalize  (graceful degradation)

The ``finalize`` node combines historical persistence and report generation,
running the insights LLM and report-prose LLM concurrently via a
``ThreadPoolExecutor`` to leverage Ollama's parallel request processing.
"""

from __future__ import annotations

from pathlib import Path

from langgraph.graph import END, StateGraph

from mas.agents.analysis import analysis_agent
from mas.agents.finalize import finalize_agent
from mas.agents.ingestion import ingestion_agent
from mas.agents.pdf_ingestion import pdf_ingestion_agent
from mas.state import AgentState


# ── Routing functions ─────────────────────────────────────────────────────────

def _route_submission_type(state: AgentState) -> str:
    """Route to 'pdf_ingestion' for .pdf files, or 'ingestion' for everything else."""
    submission_path: str = state.get("submission_path", "")
    if Path(submission_path).suffix.lower() == ".pdf":
        return "pdf_ingestion"
    return "ingestion"


def _route_after_ingestion(state: AgentState) -> str:
    """Route to 'analysis' on success, or 'finalize' for graceful degradation."""
    if state.get("ingestion_status") == "success":
        return "analysis"
    return "finalize"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Construct and compile the LangGraph StateGraph."""
    builder = StateGraph(AgentState)

    # Register agent nodes
    builder.add_node("ingestion", ingestion_agent)
    builder.add_node("pdf_ingestion", pdf_ingestion_agent)
    builder.add_node("analysis", analysis_agent)
    builder.add_node("finalize", finalize_agent)

    # Entry point: route based on submission file type
    builder.set_entry_point("_detect")
    builder.add_node("_detect", lambda s: s)  # pass-through node

    builder.add_conditional_edges(
        "_detect",
        _route_submission_type,
        {
            "ingestion": "ingestion",
            "pdf_ingestion": "pdf_ingestion",
        },
    )

    # Conditional edges: ingestion → analysis | finalize
    builder.add_conditional_edges(
        "ingestion",
        _route_after_ingestion,
        {
            "analysis": "analysis",
            "finalize": "finalize",
        },
    )

    builder.add_conditional_edges(
        "pdf_ingestion",
        _route_after_ingestion,
        {
            "analysis": "analysis",
            "finalize": "finalize",
        },
    )

    builder.add_edge("analysis", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()
