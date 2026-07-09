# Agentic RAG — FastAPI backend

Wraps the CrewAI crew behind a REST API. Streamlit apps (`app.py`,
`app_deep_seek.py`, `app_llama3.2.py`) are untouched and still work
standalone — both surfaces share `crew_builder.build_crew()` so behavior
can't drift between them, though only the API path uses the new hybrid
retrieval tool and Gemini/Groq providers described below.

## Pipeline

```
User Query + conversation history
    -> Query Rewriter Agent      resolves follow-ups ("when was it released?")
    -> Retriever Agent           HybridDocumentSearchTool:
                                    dense (Qdrant) + BM25, fused via
                                    reciprocal rank fusion, reranked with a
                                    BGE cross-encoder
    -> [Verifier Agent]          research_mode only
    -> Response Synthesizer Agent
```

Every `/chat` response includes a **confidence score** (0-100) blending
retrieval similarity, reranker score, and dense/BM25 agreement — see
`tools/hybrid_search.py::_run` for the exact formula.

## Run it

```bash
pip install -e .            # picks up fastapi/uvicorn/rank-bm25/sentence-transformers
cp .env.example .env        # set LLM_PROVIDER=gemini (+GEMINI_API_KEY) or groq (+GROQ_API_KEY)
uvicorn agentic_rag.api.main:app --reload --port 8000
```

Interactive docs at `http://localhost:8000/docs`.

**LLM providers: Gemini and Groq only.** `LLM_PROVIDER=openai` or `ollama`
(or unset) raises a clear `LLMConfigError` at request time and a warning
from `/health` — there's no silent fallback. The legacy Streamlit apps are
separate demos and aren't affected by this restriction.

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Process status, vector store readiness, active model, config warnings |
| POST | `/chat` | Ask a question. Body: `{message, session_id?, research_mode?}`. Returns answer + citations + confidence + agent trace + latency |
| GET | `/chat/{session_id}/history` | Full turn history for a session (includes per-turn confidence) |
| POST | `/upload` | Multipart upload, one or more files (pdf/docx/txt/csv) |
| GET | `/documents` | List indexed documents + chunk counts |
| DELETE | `/documents/{name}` | Remove one document from the index (also drops it from BM25) |
| GET | `/sources?session_id=` | Citations for the last answer (doc, page, preview, score) |
| GET | `/analytics` | Query counts, latency averages, confidence average, real (process-local) daily volume + hourly latency series, active model |

## Config

Everything is env-var driven (`.env.example`) — no hardcoded paths, keys,
or ports:

- `LLM_PROVIDER` — `gemini` / `groq` (required, no OpenAI/Ollama)
- `WEB_SEARCH_PROVIDER` — `serper` / `firecrawl` / `none`
- `QDRANT_PERSIST_DIR` — on-disk vector index path (blank = in-memory)
- `RETRIEVAL_TOP_K` / `RETRIEVAL_FETCH_K` — final results vs. fusion candidate pool size
- `USE_RERANKER` / `RERANKER_MODEL` — BGE cross-encoder toggle + model name
- `CORS_ORIGINS` — comma-separated frontend origins
- `MAX_UPLOAD_MB`, `RESEARCH_MODE_DEFAULT`

## Known limitations (by design, for now)

- **State is in-process, in-memory** (`api/state.py`): sessions, analytics,
  and the BM25 index reset on restart and aren't shared across multiple
  worker processes. Fine for a single-instance deploy; first thing to swap
  for Redis/Postgres + a proper document store if you scale out.
- **Conversation memory** is implemented via the query rewriter reading
  recent turns (`state.recent_history_text`), not a separate memory module
  — this resolves pronoun/follow-up references without a second LLM call
  per turn. It resets when a session is forgotten (process restart).
- **Retrieval/generation latency split** is a heuristic derived from agent
  step timestamps — good enough for the analytics dashboard, not for SLAs.
- **BM25 index rebuilds fully on every add/remove document** — fine for
  tens to low hundreds of documents; swap for an incremental index if the
  corpus grows much larger.
- **Reranker degrades gracefully**: if `sentence-transformers` or the BGE
  model can't load (e.g. not installed, no internet in a locked-down
  deploy), retrieval still works using the fusion score directly, and
  `/health` won't flag it — check server logs for the one-time warning.
- **No authentication yet** — endpoints are open.

## Files

```
src/agentic_rag/
  crew_builder.py            # Query Rewriter -> Retriever -> [Verifier] -> Response, shared by Streamlit + API
  llm_providers.py           # Gemini/Groq-only LLM factory
  tools/
    hybrid_search.py         # HybridDocumentSearchTool: BM25 + dense + RRF + BGE rerank + confidence
    custom_tool.py            # original tools, untouched, still used by Streamlit apps
  api/
    main.py
    config.py
    schemas.py
    state.py                  # sessions + timestamped analytics + BM25/document tool singleton
    dependencies.py
    routers/{chat,documents,analytics,sources,health}.py
```

## Not touched

`app.py`, `app_deep_seek.py`, `app_llama3.2.py`, `tools/custom_tool.py`,
`config/agents.yaml`, `config/tasks.yaml` — original functionality
preserved as-is. `crew.py` had one hardcoded absolute path fixed (now
env-var driven) but is otherwise unchanged; it's legacy CLI boilerplate not
used by either the Streamlit apps or the API.
