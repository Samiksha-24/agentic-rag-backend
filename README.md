# Agentic RAG using CrewAI 

A Streamlit chat app where you upload documents and a small **CrewAI** multi-agent
pipeline answers your questions — searching your documents first, and falling
back to the web when the answer isn't in them.

This version keeps the original project's business logic and all three run
commands working exactly as before, and adds a consistent design system, a
proper multi-document knowledge base with citations, and a live view into
what the agents are actually doing.

---

## 1. What this project does

- You upload one or more documents (PDF / DOCX / TXT / CSV).
- Text is extracted, split into semantic chunks (Chonkie), embedded, and
  stored in a Qdrant vector index.
- You ask a question in chat.
- A **retriever agent** searches your documents first, and only falls back to
  a live web search if it can't find the answer there.
- A **response synthesizer agent** turns whatever was retrieved into a clear
  answer.
- Optional **Research Mode** inserts a third agent that cross-checks the
  retrieved information before the final answer is written.

Three interchangeable backends are provided, unchanged in intent from the
original project:

| File | LLM | Web search |
|---|---|---|
| `app.py` | OpenAI (via `OPENAI_API_KEY`/`MODEL`) | Serper |
| `app_deep_seek.py` | DeepSeek-R1 7B, local via Ollama | FireCrawl |
| `app_llama3.2.py` | Llama 3.2, local via Ollama | FireCrawl |

---

## 2. Architecture / structure

```
app.py                          # OpenAI entry point (thin wrapper)
app_deep_seek.py                # DeepSeek-R1 / Ollama entry point (thin wrapper)
app_llama3.2.py                 # Llama 3.2 / Ollama entry point (thin wrapper)

assets/
  style.css                     # single design-system stylesheet (colors, type, components)

.streamlit/
  config.toml                   # native Streamlit theme matching style.css

src/agentic_rag/
  app_runtime.py                 # shared UI + crew-orchestration logic used by all 3 entry points
  ui_components.py               # presentation-only render_* helpers (header, chat, docs, trace)
  tools/
    custom_tool.py                # DocumentSearchTool (unchanged) + MultiDocumentSearchTool + FireCrawlWebSearchTool
  crew.py, main.py, config/*.yaml # original CrewAI-project CLI scaffolding — untouched
```

Previously, `app.py`, `app_deep_seek.py` and `app_llama3.2.py` each contained
a full copy of the UI and crew-building code (~190 lines each, ~95%
duplicated). That's now factored into `app_runtime.py`, so each entry point
is just its LLM + web-search configuration (~20 lines) calling `run_app(...)`.

---

## 3. UI/UX redesign

**Design rationale.** The original UI was Streamlit's unstyled defaults: no
color system, no spacing scale, plain `st.chat_message`, a single-column
sidebar. The redesign introduces one stylesheet (`assets/style.css`) that
every screen in every entry point shares, so nothing drifts out of sync.

- **Palette:** dark surface (`#0F1117`/`#171A23`) with an indigo→violet
  gradient accent (`#6366F1`→`#8B5CF6`) — legible, calm, and reads as an AI
  product rather than a form.
- **Typography:** Inter for UI text, a consistent 4/8/12/16/24/32px spacing
  scale, and a small set of border-radius tokens so cards/buttons/chips feel
  like one family.
- **Components:** a reusable card, document-list item, chat bubble (with
  distinct user/assistant styling), source-citation chip, and a live
  agent-trace step — all defined once in CSS and rendered through
  `ui_components.py` functions instead of ad-hoc `st.markdown` calls scattered
  through each app.
- **Empty/loading states:** a proper empty state before the first message,
  spinners during indexing/thinking, and disabled/interactive states on the
  document list.
- **Responsiveness:** the layout is centered with a max width for desktop
  readability and collapses gracefully on narrow (mobile) viewports via the
  `@media` rules in `style.css`.
- **Accessibility:** sufficient color contrast on text/backgrounds, icon +
  text pairing (not color alone) for source types, and native Streamlit
  form controls (file uploader, toggle, buttons) are kept — so keyboard
  navigation and screen-reader behavior isn't broken by custom HTML.

---

## 4. The two most valuable features — and their extensions

Reviewing the original app, most of it (chat loop, sidebar upload, "Clear
chat") is UI plumbing. Two things actually determine whether the product is
useful:

### Feature #1 — Document retrieval (the "R" in RAG)
This is the app's entire value proposition: whether it can find the right
passage in your document. The original tool indexed exactly **one** PDF, held
it only **in memory** for the session, and returned raw joined text with
**no indication of where an answer came from** — so answers were
unverifiable.

**Extended into `MultiDocumentSearchTool`:**
- Multiple documents in one knowledge base (PDF, DOCX, TXT, CSV — via the
  same MarkItDown extraction, plus `pypdf` for page-accurate text where
  available), each addable/removable independently without rebuilding the
  whole index.
- **Page-aware chunking** — every chunk is tagged with its source file and
  page number.
- **Citations** — every `_run()` call records `last_sources`
  (`document · p.N`, similarity score), which the UI renders as chips under
  each answer, so you can see exactly which document/page an answer is
  grounded in (or that no document matched, and the web fallback fired).
- Optional on-disk persistence (`persist_dir`) so the index can survive
  across restarts, instead of being wiped every session.
- The original `DocumentSearchTool` class is untouched and still exported —
  `crew.py` and the notebooks keep working unmodified.

### Feature #2 — Agentic orchestration (the "Agentic" in Agentic RAG)
The multi-agent routing between "search my docs" and "search the web" is
what differentiates this from a plain RAG chatbot — but in the original app
it was a black box: you saw a spinner, then an answer, with no way to tell
which path was taken or why.

**Extended with a live agent-reasoning panel and optional Research Mode:**
- A `step_callback` is attached to every agent. Each tool call / reasoning
  step is rendered **live**, in place, in a "🧠 Agent reasoning" panel above
  the answer — which tool fired, what query it ran, and a preview of the
  agent's thought — so orchestration is observable, not opaque.
- **Research Mode** (sidebar toggle) inserts a third **verification agent**
  between retrieval and synthesis, whose job is to flag unsupported or
  low-confidence claims and note whether each came from a document or the
  web, before the final answer is written — a natural, incremental extension
  of the existing sequential pipeline rather than a rewrite of it.

Both extensions plug into the exact same `Process.sequential` Crew pattern
the project already used — nothing about the underlying orchestration model
changed, it's just observable and, optionally, one step longer now.

### A note on the pasted "Enterprise Platform" spec
Partway through this session a much larger spec was pasted in (multi-format
knowledge library with folders/tags/versioning, a 5-agent Research Workspace
with PDF/DOCX report export, knowledge graphs, shared workspaces, dashboards,
etc.). That's a multi-week product, not something that can be honestly
delivered as working code in one pass — so rather than generating unused
scaffolding or fake UI for features that don't run, I kept everything in
this delivery real and testable, and folded the realistic parts of that
list (multi-format documents, citations with page numbers, a verification
agent, confidence via retrieval score) into the two extensions above. The
rest is listed under **Roadmap** below.

---

## 5. Backward compatibility

- `DocumentSearchTool` (constructor, `_run` signature and return type) is
  **byte-for-byte unchanged** — anything importing it (`crew.py`, the
  notebooks) keeps working.
- All three original run commands still work exactly as before:
  `streamlit run app.py`, `streamlit run app_deep_seek.py`,
  `streamlit run app_llama3.2.py`.
- Same env vars as before (`OPENAI_API_KEY`, `MODEL`, `SERPER_API_KEY`,
  `FIRECRAWL_API_KEY`), same Ollama model names.
- Fixed one pre-existing bug: `app_deep_seek.py`/`app_llama3.2.py` imported a
  `FireCrawlWebSearchTool` from `custom_tool.py` that was never actually
  defined there, so the local-LLM apps' web-search fallback could never
  import. It's now implemented (`firecrawl-py` + `FIRECRAWL_API_KEY`).

---

## 6. Setup and run instructions

**Requirements:** Python 3.11–3.13.

```bash
pip install crewai crewai-tools "chonkie[semantic]" markitdown qdrant-client fastembed pypdf firecrawl-py
```

**Environment variables** (`.env`, see `.env.example`):

```bash
# Only needed for app.py (OpenAI backend)
MODEL=your_model_name
OPENAI_API_KEY=your_openai_api_key
SERPER_API_KEY=your_serper_api_key

# Only needed for app_deep_seek.py / app_llama3.2.py (local Ollama backends)
FIRECRAWL_API_KEY=your_firecrawl_api_key
```

**Run:**

```bash
streamlit run app.py              # OpenAI + Serper
streamlit run app_deep_seek.py    # DeepSeek-R1 via Ollama + FireCrawl (needs `ollama pull deepseek-r1:7b`)
streamlit run app_llama3.2.py     # Llama 3.2 via Ollama + FireCrawl (needs `ollama pull llama3.2`)
```

Then, in the app: upload one or more PDF/DOCX/TXT/CSV files in the sidebar,
optionally toggle **Research Mode**, and ask a question in the chat box.

---

## 7. Roadmap (not implemented in this pass)

Kept honest and out of the code until they can be built and tested properly:

- Folder/tag/category organization and document versioning in the knowledge base
- Full 5-agent Research Workspace (planner → retrieval → web → verification → report writer) with a task/timeline visualization
- PDF/DOCX export of generated reports
- Knowledge graph visualization, multi-document comparison, follow-up question generation
- Persistent chat/session history and shared workspaces across users
- Auth, multi-tenancy, and a real deployment/observability story (logging, rate limits, error tracking)

---

## 8. Contribution

Contributions are welcome! Please fork the repository and submit a pull
request with your improvements.
