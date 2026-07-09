"""
ui_components.py
=================
Reusable, presentation-only Streamlit helpers shared by app.py, app_deep_seek.py
and app_llama3.2.py so all three entry points render a single, consistent
design system instead of duplicating markup/CSS in every file.

Nothing in this module touches business logic (CrewAI, Qdrant, tools) — it
only renders HTML/CSS driven by data passed in, so it is safe to import from
any of the three Streamlit apps without changing their behavior.
"""

import base64
import os
from datetime import datetime
from pathlib import Path

import streamlit as st

_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
def inject_design_system():
    """Load the shared style.css once per session."""
    css_path = _ASSETS_DIR / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def _b64(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
def render_header(title: str, subtitle: str, badge_text: str, icon: str = "🧠"):
    st.markdown(
        f"""
        <div class="arag-header">
            <div class="arag-header__icon">{icon}</div>
            <div>
                <p class="arag-header__title">{title}</p>
                <p class="arag-header__subtitle">{subtitle}</p>
            </div>
            <div class="arag-header__badge">{badge_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Document manager (Feature #1 — multi-document intelligence)
# --------------------------------------------------------------------------- #
def render_document_list(documents: list[dict], on_remove_key_prefix: str = "rm_doc"):
    """
    documents: list of {"name": str, "chunks": int, "indexed_at": str}
    Renders a card per document with a remove button. Returns the name of the
    document the user clicked "remove" on this run, or None.
    """
    removed = None
    if not documents:
        st.markdown(
            """
            <div class="arag-card arag-card--dashed">
                No documents indexed yet. Upload one or more PDFs above to get started.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return removed

    for doc in documents:
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(
                f"""
                <div class="arag-doc-item">
                    <div class="arag-doc-item__icon">📄</div>
                    <div>
                        <div class="arag-doc-item__name">{doc['name']}</div>
                        <div class="arag-doc-item__meta">{doc['chunks']} chunks · indexed {doc['indexed_at']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            if st.button("✕", key=f"{on_remove_key_prefix}_{doc['name']}", help=f"Remove {doc['name']}"):
                removed = doc["name"]
    return removed


# --------------------------------------------------------------------------- #
# Chat rendering
# --------------------------------------------------------------------------- #
def render_message(role: str, content: str, sources: list[dict] | None = None):
    """
    role: "user" | "assistant"
    sources: optional list of {"type": "doc"|"web"|"none", "label": str}
             rendered as citation chips under assistant messages.
    """
    is_user = role == "user"
    avatar_cls = "arag-avatar--user" if is_user else "arag-avatar--assistant"
    bubble_cls = "arag-bubble--user" if is_user else "arag-bubble--assistant"
    row_cls = "arag-msg--user" if is_user else ""
    avatar_char = "🙂" if is_user else "✨"

    sources_html = ""
    if sources:
        chips = []
        for s in sources:
            chip_cls = {"doc": "arag-chip--doc", "web": "arag-chip--web"}.get(s["type"], "arag-chip--none")
            icon = {"doc": "📄", "web": "🌐"}.get(s["type"], "❓")
            chips.append(f'<span class="arag-chip {chip_cls}">{icon} {s["label"]}</span>')
        sources_html = f'<div class="arag-sources">{"".join(chips)}</div>'

    st.markdown(
        f"""
        <div class="arag-msg {row_cls}">
            <div class="arag-avatar {avatar_cls}">{avatar_char}</div>
            <div class="arag-bubble {bubble_cls}">{content}{sources_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(title: str, description: str, icon: str = "💬"):
    st.markdown(
        f"""
        <div class="arag-empty-state">
            <div class="arag-empty-state__icon">{icon}</div>
            <div class="arag-empty-state__title">{title}</div>
            <div>{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Agent trace (Feature #2 — transparent agentic orchestration)
# --------------------------------------------------------------------------- #
def render_trace_steps(steps: list[dict]) -> str:
    """
    steps: list of {"icon": str, "title": str, "detail": str}
    Returns the HTML string (caller places it inside a placeholder so it can
    be re-rendered live, step by step, while the crew is running).
    """
    if not steps:
        return '<div class="arag-trace-step__body">Waiting for the agents to start…</div>'

    rows = []
    for s in steps:
        rows.append(
            f"""
            <div class="arag-trace-step">
                <div class="arag-trace-step__dot">{s['icon']}</div>
                <div class="arag-trace-step__body"><strong>{s['title']}</strong><br/>{s['detail']}</div>
            </div>
            """
        )
    return "".join(rows)


def new_trace_step(icon: str, title: str, detail: str) -> dict:
    return {"icon": icon, "title": title, "detail": detail, "ts": datetime.now().strftime("%H:%M:%S")}
