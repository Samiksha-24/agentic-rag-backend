"""
dependencies.py
================
FastAPI dependency-injection functions. Routes declare what they need
(`Depends(get_document_tool)`, etc.) instead of reaching into globals
directly, which keeps routers testable and keeps wiring in one place.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from agentic_rag.api.config import Settings, get_settings
from agentic_rag.api.state import app_state
from agentic_rag.tools.hybrid_search import HybridDocumentSearchTool
from agentic_rag.llm_providers import get_llm

logger = logging.getLogger("agentic_rag.api")


def get_document_tool() -> HybridDocumentSearchTool:
    """
    Returns the singleton HybridDocumentSearchTool (dense + BM25 + reranker),
    lazily initialized from settings on first use (also done eagerly at app
    startup — see main.py).
    """
    if app_state.document_tool is None:
        settings = get_settings()
        app_state.init_document_tool(
            settings.qdrant_persist_dir,
            settings.retrieval_top_k,
            settings.retrieval_fetch_k,
            settings.use_reranker,
            settings.reranker_model,
        )
    return app_state.document_tool


def get_web_search_tool_factory(settings: Settings) -> Optional[Callable[[], object]]:
    """
    Returns a zero-arg factory for the configured web-search fallback tool,
    or None if web search is disabled. Mirrors the original apps' behavior
    (SerperDevTool for app.py, FireCrawlWebSearchTool for the local-LLM
    variants) but made provider-selectable via WEB_SEARCH_PROVIDER instead
    of being hardcoded per entry point.
    """
    provider = settings.web_search_provider.lower().strip()

    if provider == "none" or provider == "":
        return None

    if provider == "serper":
        try:
            from crewai_tools import SerperDevTool
        except ImportError:
            logger.warning("WEB_SEARCH_PROVIDER=serper but crewai_tools.SerperDevTool is unavailable.")
            return None
        return SerperDevTool

    if provider == "firecrawl":
        from agentic_rag.tools.custom_tool import FireCrawlWebSearchTool
        return FireCrawlWebSearchTool

    logger.warning("Unknown WEB_SEARCH_PROVIDER '%s'; web search disabled.", provider)
    return None


def get_llm_for_settings(settings: Settings):
    return get_llm(settings.llm_provider)
