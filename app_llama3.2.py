"""
app_llama3.2.py — Agentic RAG (Llama 3.2, local via Ollama)
=============================================================
Requires Ollama running locally with the llama3.2 model pulled
(`ollama pull llama3.2`). Uses FireCrawl for web search.

Run:  streamlit run app_llama3.2.py
"""

import os
import sys

import streamlit as st
from crewai import LLM

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from agentic_rag.tools.custom_tool import FireCrawlWebSearchTool
from agentic_rag.app_runtime import run_app


@st.cache_resource
def _load_llm():
    return LLM(model="ollama/llama3.2", base_url="http://localhost:11434")


run_app(
    web_search_tool_factory=FireCrawlWebSearchTool,
    llm_factory=_load_llm,
    badge_text="🦙 Llama 3.2 · local (Ollama)",
)
