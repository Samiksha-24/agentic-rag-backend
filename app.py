"""
app.py — Agentic RAG (OpenAI backend)
======================================
Uses OpenAI (via crewai's default LLM resolution from OPENAI_API_KEY / MODEL
env vars, same as the original project) and Serper for web search.

Run:  streamlit run app.py
"""

import os
import sys

from crewai_tools import SerperDevTool

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from agentic_rag.app_runtime import run_app

run_app(
    web_search_tool_factory=SerperDevTool,
    llm_factory=None,  # crewai resolves the LLM from OPENAI_API_KEY / MODEL env vars, as before
    badge_text="⚡ OpenAI backend",
)
