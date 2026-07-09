"""
config.py
=========
Centralized, validated environment configuration for the FastAPI backend.
No hardcoded paths, keys, or ports — everything comes from the environment
(with sane local-dev defaults) and is loaded once per process.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class Settings(BaseModel):
    app_name: str = "Agentic RAG API"
    version: str = "1.0.0"
    environment: str = Field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))

    # LLM — must be "gemini" or "groq" (OpenAI/Ollama are not supported by the API; see llm_providers.py)
    llm_provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "gemini"))

    # Web search fallback tool: "serper" | "firecrawl" | "none"
    web_search_provider: str = Field(default_factory=lambda: os.getenv("WEB_SEARCH_PROVIDER", "serper"))

    # Vector store
    qdrant_persist_dir: Optional[str] = Field(
        default_factory=lambda: os.getenv("QDRANT_PERSIST_DIR", "./qdrant_storage") or None
    )
    retrieval_top_k: int = Field(default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "5")))
    retrieval_fetch_k: int = Field(default_factory=lambda: int(os.getenv("RETRIEVAL_FETCH_K", "20")))
    use_reranker: bool = Field(default_factory=lambda: os.getenv("USE_RERANKER", "true").lower() == "true")
    reranker_model: str = Field(default_factory=lambda: os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base"))

    # Uploads
    max_upload_mb: int = Field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_MB", "25")))
    supported_extensions: List[str] = Field(default_factory=lambda: ["pdf", "docx", "txt", "csv"])

    # CORS — comma-separated origins, e.g. "http://localhost:5173,https://myapp.vercel.app"
    cors_origins: List[str] = Field(
        default_factory=lambda: [
            o.strip()
            for o in os.getenv(
                "CORS_ORIGINS",
                "http://localhost:5173,http://localhost:3000,http://localhost:8080",
            ).split(",")
            if o.strip()
        ]
    )

    # Research mode default (per-request override still allowed via API)
    research_mode_default: bool = Field(
        default_factory=lambda: os.getenv("RESEARCH_MODE_DEFAULT", "false").lower() == "true"
    )

    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    @field_validator("retrieval_top_k")
    @classmethod
    def _positive_top_k(cls, v: int) -> int:
        if v < 1:
            raise ValueError("RETRIEVAL_TOP_K must be >= 1")
        return v

    @field_validator("max_upload_mb")
    @classmethod
    def _positive_upload_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError("MAX_UPLOAD_MB must be >= 1")
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
