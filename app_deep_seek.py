"""
app_deep_seek.py — Agentic RAG (DeepSeek-R1, local via Ollama)
================================================================
Requires Ollama running locally with the deepseek-r1:7b model pulled
(`ollama pull deepseek-r1:7b`). Uses FireCrawl for web search.

Run:  streamlit run app_deep_seek.py
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
    return LLM(model="ollama/deepseek-r1:7b", base_url="http://localhost:11434")


run_app(
    web_search_tool_factory=FireCrawlWebSearchTool,
    llm_factory=_load_llm,
    badge_text="🦙 DeepSeek-R1 · local (Ollama)",
)
