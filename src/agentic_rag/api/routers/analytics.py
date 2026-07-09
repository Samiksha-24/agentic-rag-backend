from __future__ import annotations

from fastapi import APIRouter, Depends

from agentic_rag.api.config import Settings, get_settings
from agentic_rag.api.dependencies import get_document_tool
from agentic_rag.api.schemas import AnalyticsResponse
from agentic_rag.api.state import app_state
from agentic_rag.llm_providers import active_model_name
from agentic_rag.tools.hybrid_search import HybridDocumentSearchTool

router = APIRouter(tags=["analytics"])


@router.get("/analytics", response_model=AnalyticsResponse)
def get_analytics(
    settings: Settings = Depends(get_settings),
    tool: HybridDocumentSearchTool = Depends(get_document_tool),
) -> AnalyticsResponse:
    docs = tool.list_documents()
    snapshot = app_state.analytics_snapshot()

    return AnalyticsResponse(
        documents_indexed=len(docs),
        total_chunks=sum(d["chunks"] for d in docs),
        total_queries=snapshot["total_queries"],
        active_sessions=snapshot["active_sessions"],
        avg_response_time_ms=snapshot["avg_response_time_ms"],
        avg_retrieval_latency_ms=snapshot["avg_retrieval_latency_ms"],
        avg_generation_latency_ms=snapshot["avg_generation_latency_ms"],
        avg_confidence=snapshot["avg_confidence"],
        active_model=active_model_name(settings.llm_provider),
        active_llm_provider=settings.llm_provider,
        research_mode_default=settings.research_mode_default,
        uptime_seconds=snapshot["uptime_seconds"],
        query_volume_daily=snapshot["query_volume_daily"],
        latency_by_hour=snapshot["latency_by_hour"],
        new_sessions_by_day=snapshot["new_sessions_by_day"],
    )
