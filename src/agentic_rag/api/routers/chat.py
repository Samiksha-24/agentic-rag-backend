from __future__ import annotations

import asyncio
import logging
import time
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException

from agentic_rag.api.config import Settings, get_settings
from agentic_rag.api.dependencies import (
    get_document_tool,
    get_llm_for_settings,
    get_web_search_tool_factory,
)
from agentic_rag.api.schemas import (
    ChatHistoryMessage,
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    SourceItem,
    TraceStep,
)
from agentic_rag.api.state import app_state
from agentic_rag.crew_builder import build_crew
from agentic_rag.llm_providers import LLMConfigError, active_model_name
from agentic_rag.tools.hybrid_search import HybridDocumentSearchTool

logger = logging.getLogger("agentic_rag.api.chat")

router = APIRouter(tags=["chat"])


def _make_step_recorder():
    """
    Same trace-capture pattern used by the Streamlit UI's step_callback,
    adapted to collect structured steps (with timestamps, for a rough
    retrieval-vs-generation latency split) instead of rendering markdown.
    """
    steps: List[dict] = []

    def _callback(step):
        icon, title, detail = "\U0001F539", "Agent step", ""
        is_tool_step = False
        try:
            tool = getattr(step, "tool", None)
            tool_input = getattr(step, "tool_input", None)
            thought = getattr(step, "thought", None) or getattr(step, "log", "")
            if tool:
                is_tool_step = True
                icon = "\U0001F4C4" if "Document" in str(tool) else "\U0001F310"
                title = f"Using tool: {tool}"
                detail = f"Query: {tool_input}" if tool_input else str(thought)[:180]
            elif thought:
                title = "Reasoning"
                detail = str(thought)[:180]
        except Exception:
            detail = str(step)[:180]

        steps.append({
            "icon": icon,
            "title": title,
            "detail": detail,
            "timestamp": time.time(),
            "is_tool_step": is_tool_step,
        })

    return _callback, steps


def _split_latency(start_ts: float, end_ts: float, steps: List[dict]) -> tuple[Optional[float], Optional[float]]:
    """
    Approximate retrieval vs. generation latency from step timestamps:
    "retrieval" = from start through the last tool-use step,
    "generation" = the remainder. Best-effort -- CrewAI doesn't expose
    per-phase timing natively, and this is clearly an approximation.
    """
    tool_steps = [s for s in steps if s["is_tool_step"]]
    if not tool_steps:
        return None, None
    retrieval_end = tool_steps[-1]["timestamp"]
    retrieval_ms = (retrieval_end - start_ts) * 1000
    generation_ms = (end_ts - retrieval_end) * 1000
    return max(retrieval_ms, 0.0), max(generation_ms, 0.0)


def _run_crew_sync(crew, query: str, history: str) -> str:
    return crew.kickoff(inputs={"query": query, "history": history}).raw


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    settings: Settings = Depends(get_settings),
    tool: HybridDocumentSearchTool = Depends(get_document_tool),
) -> ChatResponse:
    session = app_state.get_or_create_session(request.session_id)
    research_mode = request.research_mode if request.research_mode is not None else settings.research_mode_default

    try:
        llm = get_llm_for_settings(settings)
    except LLMConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    web_search_factory = get_web_search_tool_factory(settings)
    web_search_tool = web_search_factory() if web_search_factory else None

    # Conversation memory: recent turns feed the query-rewriter agent so
    # follow-ups ("when was it released?") resolve against prior context.
    history_text = app_state.recent_history_text(session.id)

    step_callback, steps = _make_step_recorder()
    crew = build_crew(
        tool, web_search_tool, llm, research_mode, step_callback, enable_query_rewriter=True
    )

    app_state.append_turn(session.id, "user", request.message)

    start_ts = time.time()
    try:
        answer = await asyncio.to_thread(_run_crew_sync, crew, request.message, history_text)
    except Exception as exc:
        logger.exception("Crew execution failed for session %s", session.id)
        raise HTTPException(status_code=502, detail=f"Agent execution failed: {exc}") from exc
    end_ts = time.time()

    total_ms = (end_ts - start_ts) * 1000
    retrieval_ms, generation_ms = _split_latency(start_ts, end_ts, steps)
    confidence = tool.last_confidence
    app_state.record_query(total_ms, retrieval_ms, generation_ms, confidence)

    sources = list(tool.last_sources) or [{"type": "none", "label": "No source found", "score": None}]
    app_state.append_turn(session.id, "assistant", answer, sources, confidence)

    return ChatResponse(
        session_id=session.id,
        answer=answer,
        sources=[SourceItem(**s) for s in sources],
        trace=[TraceStep(icon=s["icon"], title=s["title"], detail=s["detail"]) for s in steps],
        research_mode=research_mode,
        confidence=confidence,
        total_latency_ms=int(total_ms),
        retrieval_latency_ms=int(retrieval_ms) if retrieval_ms is not None else None,
        generation_latency_ms=int(generation_ms) if generation_ms is not None else None,
        active_model=active_model_name(settings.llm_provider),
    )


@router.get("/chat/{session_id}/history", response_model=ChatHistoryResponse)
def chat_history(session_id: str) -> ChatHistoryResponse:
    session = app_state.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id '{session_id}'.")
    return ChatHistoryResponse(
        session_id=session.id,
        messages=[
            ChatHistoryMessage(
                role=turn.role,
                content=turn.content,
                sources=[SourceItem(**s) for s in turn.sources] if turn.sources else None,
                confidence=turn.confidence,
            )
            for turn in session.history
        ],
    )
