"""
llm_providers.py
=================
LLM provider selection for the FastAPI backend (the deployable surface).

Per the platform requirements: OpenAI and Ollama are not supported here.
Only Gemini and Groq are valid providers. LLM_PROVIDER must be explicitly
set to one of them, with the matching API key present -- there is no silent
fallback to an unconfigured default.

Note: the original Streamlit entry point `app.py` is explicitly an
"OpenAI backend" demo (its own docstring says so) and is left as-is per
"preserve existing functionality" -- it does not import this module and is
not part of the deployed API. `app_deep_seek.py` / `app_llama3.2.py` are
local-LLM demos, also untouched, also not part of the deployed API. This
module only governs what agentic_rag.api serves.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

SUPPORTED_PROVIDERS = ("gemini", "groq")


class LLMConfigError(RuntimeError):
    """Raised when the configured provider is missing, unsupported, or missing credentials."""


def _build_llm(provider: str):
    from crewai import LLM  # imported lazily so importing this module never requires crewai at collection time

    provider = (provider or "").lower().strip()

    if provider in ("openai", "ollama", "", "default"):
        raise LLMConfigError(
            f"LLM_PROVIDER='{provider or '(unset)'}' is not supported by the API. "
            f"Set LLM_PROVIDER to one of: {', '.join(SUPPORTED_PROVIDERS)}."
        )

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise LLMConfigError("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set.")
        model = os.getenv("GEMINI_MODEL", "gemini/gemini-2.0-flash")
        return LLM(model=model, api_key=api_key)

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise LLMConfigError("LLM_PROVIDER=groq but GROQ_API_KEY is not set.")
        model = os.getenv("GROQ_MODEL", "groq/llama-3.3-70b-versatile")
        return LLM(model=model, api_key=api_key)

    raise LLMConfigError(
        f"Unknown LLM_PROVIDER '{provider}'. Supported: {', '.join(SUPPORTED_PROVIDERS)}."
    )


@lru_cache(maxsize=8)
def _cached_llm(provider: str):
    return _build_llm(provider)


def get_llm(provider: Optional[str] = None):
    """
    Return a configured LLM for the given provider name (or the
    LLM_PROVIDER env var if not specified). Cached per-provider so we
    don't reconstruct a client on every request. Raises LLMConfigError
    for anything other than 'gemini' or 'groq'.
    """
    return _cached_llm((provider or os.getenv("LLM_PROVIDER", "")).lower().strip())


def active_model_name(provider: Optional[str] = None) -> str:
    """Human-readable model identifier for analytics/health endpoints."""
    provider = (provider or os.getenv("LLM_PROVIDER", "")).lower().strip()
    if provider == "gemini":
        return os.getenv("GEMINI_MODEL", "gemini/gemini-2.0-flash")
    if provider == "groq":
        return os.getenv("GROQ_MODEL", "groq/llama-3.3-70b-versatile")
    return provider or "(unconfigured)"
