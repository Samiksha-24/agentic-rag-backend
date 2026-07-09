"""
app_runtime.py
===============
Shared Streamlit application logic used by app.py, app_deep_seek.py and
app_llama3.2.py. Each entry point only supplies its LLM + web-search-tool
configuration and calls `run_app(...)`; everything else (UI, crew wiring,
state, live agent trace, citations) lives here once instead of being
duplicated three times.
"""

import os
import tempfile
import gc

import streamlit as st

from agentic_rag.tools.custom_tool import MultiDocumentSearchTool
from agentic_rag.crew_builder import build_crew as _build_crew
from agentic_rag import ui_components as ui

SUPPORTED_TYPES = ["pdf", "docx", "txt", "csv"]


def _init_state():
    defaults = {"messages": [], "pdf_tool": None, "trace_steps": [], "research_mode": False}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if st.session_state.pdf_tool is None:
        st.session_state.pdf_tool = MultiDocumentSearchTool()


def _reset_chat():
    st.session_state.messages = []
    st.session_state.trace_steps = []
    gc.collect()


def _make_step_callback(trace_placeholder):
    def _callback(step):
        icon, title, detail = "🔹", "Agent step", ""
        try:
            tool = getattr(step, "tool", None)
            tool_input = getattr(step, "tool_input", None)
            thought = getattr(step, "thought", None) or getattr(step, "log", "")
            if tool:
                icon = "📄" if "Document" in str(tool) else "🌐"
                title = f"Using tool: {tool}"
                detail = f"Query: {tool_input}" if tool_input else str(thought)[:180]
            elif thought:
                title = "Reasoning"
                detail = str(thought)[:180]
        except Exception:
            detail = str(step)[:180]

        st.session_state.trace_steps.append(ui.new_trace_step(icon, title, detail))
        trace_placeholder.markdown(ui.render_trace_steps(st.session_state.trace_steps), unsafe_allow_html=True)
    return _callback


def run_app(web_search_tool_factory, llm_factory=None, badge_text: str = "⚡ OpenAI backend"):
    """
    web_search_tool_factory: zero-arg callable returning a fresh web search tool instance
    llm_factory: optional zero-arg callable returning a configured LLM (None => provider default, e.g. OpenAI env vars)
    badge_text: shown in the header to indicate which backend this entry point runs
    """
    st.set_page_config(page_title="Agentic RAG", page_icon="🧠", layout="centered")

    _init_state()
    ui.inject_design_system()
    ui.render_header(
        title="Agentic RAG",
        subtitle="Multi-document knowledge base with live agentic reasoning",
        badge_text=badge_text,
    )

    with st.sidebar:
        st.markdown('<p class="arag-section-title">Knowledge base</p>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload documents", type=SUPPORTED_TYPES, accept_multiple_files=True, label_visibility="collapsed"
        )

        if uploaded_files:
            existing = {d["name"] for d in st.session_state.pdf_tool.list_documents()}
            new_files = [f for f in uploaded_files if f.name not in existing]
            if new_files:
                with st.spinner(f"Indexing {len(new_files)} document(s)…"):
                    for f in new_files:
                        with tempfile.TemporaryDirectory() as temp_dir:
                            temp_path = os.path.join(temp_dir, f.name)
                            with open(temp_path, "wb") as out:
                                out.write(f.getvalue())
                            st.session_state.pdf_tool.add_document(temp_path)
                st.success(f"Indexed {len(new_files)} document(s).")

        st.markdown('<p class="arag-section-title">Indexed documents</p>', unsafe_allow_html=True)
        removed = ui.render_document_list(st.session_state.pdf_tool.list_documents())
        if removed:
            st.session_state.pdf_tool.remove_document(removed)
            st.rerun()

        st.markdown('<p class="arag-section-title">Settings</p>', unsafe_allow_html=True)
        st.session_state.research_mode = st.toggle(
            "Research Mode",
            value=st.session_state.research_mode,
            help="Adds a verification agent that cross-checks and flags low-confidence claims before the final answer.",
        )

        col1, col2 = st.columns(2)
        with col1:
            st.button("Clear chat", on_click=_reset_chat, use_container_width=True)
        with col2:
            transcript = "\n\n".join(f"**{m['role'].title()}:** {m['content']}" for m in st.session_state.messages)
            st.download_button(
                "Export", data=transcript or "No messages yet.", file_name="agentic_rag_chat.md",
                mime="text/markdown", use_container_width=True,
            )

    if not st.session_state.messages:
        ui.render_empty_state(
            title="Ask something about your documents",
            description="Upload a PDF, DOCX, TXT or CSV on the left, then ask a question below. "
                         "I'll search your documents first and fall back to the web if needed.",
            icon="💬",
        )
    else:
        for message in st.session_state.messages:
            ui.render_message(message["role"], message["content"], message.get("sources"))

    prompt = st.chat_input("Ask a question about your documents...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        ui.render_message("user", prompt)

        trace_expander = st.expander("🧠 Agent reasoning (live)", expanded=True)
        with trace_expander:
            trace_placeholder = st.empty()
        st.session_state.trace_steps = []

        llm = llm_factory() if llm_factory else None
        crew = _build_crew(
            st.session_state.pdf_tool,
            web_search_tool_factory(),
            llm,
            st.session_state.research_mode,
            _make_step_callback(trace_placeholder),
        )

        with st.spinner("Thinking..."):
            history_text = "\n".join(
                f"{m['role'].title()}: {m['content']}" for m in st.session_state.messages[-7:-1]
            )
            result = crew.kickoff(inputs={"query": prompt, "history": history_text}).raw

        sources = list(st.session_state.pdf_tool.last_sources) or [{"type": "none", "label": "No source found"}]
        st.session_state.messages.append({"role": "assistant", "content": result, "sources": sources})
        ui.render_message("assistant", result, sources)
