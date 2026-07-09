"""
main.py
=======
FastAPI application entrypoint for the Agentic RAG backend.

Run locally:
    uvicorn agentic_rag.api.main:app --reload --port 8000

This wraps the *existing* CrewAI crew (see crew_builder.py, which is the
same logic app_runtime.py's Streamlit UI uses) behind a REST API, without
changing how retrieval, agents, or tasks behave. It does not replace
app.py / app_deep_seek.py / app_llama3.2.py — those still work standalone.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agentic_rag.api.config import get_settings
from agentic_rag.api.dependencies import get_document_tool
from agentic_rag.api.logging_config import configure_logging
from agentic_rag.api.routers import analytics, chat, documents, health, sources

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger("agentic_rag.api")

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description=(
        "REST API for the Agentic RAG platform: multi-document Q&A over a "
        "CrewAI retriever/response-synthesizer pipeline, with citations, "
        "conversation sessions, and analytics."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "detail": str(exc)},
    )


@app.on_event("startup")
def on_startup() -> None:
    logger.info(
        "Starting %s v%s (env=%s, llm_provider=%s)",
        settings.app_name,
        settings.version,
        settings.environment,
        settings.llm_provider,
    )
    # Eagerly initialize the document index so the first real request isn't
    # slowed down by lazy setup, and so /health reports accurately right away.
    get_document_tool()
    logger.info("Document index ready (persist_dir=%s)", settings.qdrant_persist_dir)


@app.get("/", tags=["health"])
def root() -> dict:
    return {"service": settings.app_name, "version": settings.version, "docs": "/docs"}


app.include_router(health.router)
app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(analytics.router)
app.include_router(sources.router)
