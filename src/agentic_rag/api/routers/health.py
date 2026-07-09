from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from agentic_rag.api.config import Settings, get_settings
from agentic_rag.api.dependencies import get_document_tool
from agentic_rag.api.schemas import HealthResponse
from agentic_rag.api.state import app_state
from agentic_rag.llm_providers import active_model_name

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    warnings: list[str] = []

    try:
        tool = get_document_tool()
        vector_store_ready = tool is not None
        documents_indexed = len(tool.list_documents()) if tool else 0
    except Exception as exc:  # pragma: no cover - defensive
        vector_store_ready = False
        documents_indexed = 0
        warnings.append(f"Vector store unavailable: {exc}")

    if settings.llm_provider.lower() not in ("gemini", "groq"):
        warnings.append(
            f"LLM_PROVIDER='{settings.llm_provider}' is not supported — set it to 'gemini' or 'groq'."
        )
    else:
        import os

        key_name = "GEMINI_API_KEY" if settings.llm_provider.lower() == "gemini" else "GROQ_API_KEY"
        if not os.getenv(key_name):
            warnings.append(f"LLM_PROVIDER={settings.llm_provider} but {key_name} is not set.")

    status = "ok" if not warnings else "degraded"

    return HealthResponse(
        status=status,
        version=settings.version,
        environment=settings.environment,
        uptime_seconds=round(time.time() - app_state.started_at, 1),
        vector_store_ready=vector_store_ready,
        documents_indexed=documents_indexed,
        active_llm_provider=settings.llm_provider,
        active_model=active_model_name(settings.llm_provider),
        warnings=warnings,
    )
