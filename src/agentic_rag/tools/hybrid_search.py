"""
hybrid_search.py
=================
HybridDocumentSearchTool — the retrieval upgrade requested for the API path:

    User Query -> Hybrid Retriever (dense Qdrant + BM25, fused with RRF)
              -> BGE cross-encoder reranker
              -> confidence score
              -> top_k passages returned to the CrewAI retriever agent

This class *extends* MultiDocumentSearchTool (tools/custom_tool.py) rather
than replacing it. Document extraction and chunking
(`_extract_pages` / `_create_chunks`, both inherited unchanged) are reused
exactly as-is — only indexing bookkeeping and the retrieval algorithm
change, so the original tool stays untouched and the Streamlit apps that
construct it directly are unaffected.

Optional dependencies, both degrade gracefully if unavailable:
  - rank_bm25 (BM25Okapi)              -> falls back to dense-only search
  - sentence-transformers CrossEncoder  -> falls back to the RRF fusion
    score in place of a reranker score (confidence calculation notes this)
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict

from agentic_rag.tools.custom_tool import MultiDocumentSearchTool, MultiDocumentSearchToolInput

logger = logging.getLogger("agentic_rag.hybrid_search")

RRF_K = 60  # standard reciprocal-rank-fusion constant


class HybridDocumentSearchTool(MultiDocumentSearchTool):
    name: str = "DocumentSearchTool"
    description: str = (
        "Hybrid (dense vector + BM25 keyword) search across all indexed documents, "
        "fused with reciprocal rank fusion and reranked with a cross-encoder. "
        "Returns the most relevant passages with source document, page, and a confidence score."
    )
    args_schema: type[BaseModel] = MultiDocumentSearchToolInput
    model_config = ConfigDict(extra="allow")

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        top_k: int = 5,
        fetch_k: int = 20,
        use_reranker: bool = True,
        reranker_model: Optional[str] = None,
    ):
        super().__init__(persist_dir=persist_dir, top_k=top_k)
        self.fetch_k = max(fetch_k, top_k)
        self.use_reranker = use_reranker
        self.reranker_model_name = reranker_model or os.getenv(
            "RERANKER_MODEL", "BAAI/bge-reranker-base"
        )

        self._chunk_store: Dict[int, dict] = {}  # id -> {text, source, page}
        self._bm25 = None
        self._bm25_ids: List[int] = []
        self._reranker = None
        self._reranker_unavailable = False

        self.last_confidence: Optional[float] = None
        self.last_retrieval_debug: dict = {}

    # ---- indexing: reuses inherited _extract_pages / _create_chunks -- #
    def add_document(self, file_path: str) -> int:
        name = os.path.basename(file_path)
        if name in self._documents:
            self.remove_document(name)

        pages = self._extract_pages(file_path)
        docs, metadata, ids = [], [], []

        for page_num, page_text in enumerate(pages, start=1):
            if not page_text or not page_text.strip():
                continue
            for chunk in self._create_chunks(page_text):
                docs.append(chunk.text)
                metadata.append({"source": name, "page": page_num})
                ids.append(self._next_id)
                self._chunk_store[self._next_id] = {
                    "text": chunk.text,
                    "source": name,
                    "page": page_num,
                }
                self._next_id += 1

        if docs:
            self.client.add(
                collection_name=self.COLLECTION,
                documents=docs,
                metadata=metadata,
                ids=ids,
            )

        self._documents[name] = {"chunks": len(docs), "indexed_at": time.strftime("%H:%M:%S")}
        self._rebuild_bm25()
        return len(docs)

    def remove_document(self, name: str) -> None:
        super().remove_document(name)  # removes from Qdrant + self._documents
        stale_ids = [i for i, c in self._chunk_store.items() if c["source"] == name]
        for i in stale_ids:
            self._chunk_store.pop(i, None)
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        """Rebuild the BM25 index from the current chunk store. O(n) rebuild
        on every add/remove — fine for the document counts this tool is
        aimed at (tens to low hundreds of docs); swap for an incremental
        index if the corpus grows much larger."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            if self._bm25 is None:
                logger.warning("rank_bm25 not installed — hybrid search will run dense-only.")
            self._bm25 = None
            self._bm25_ids = []
            return

        ids = list(self._chunk_store.keys())
        if not ids:
            self._bm25 = None
            self._bm25_ids = []
            return
        tokenized = [self._tokenize(self._chunk_store[i]["text"]) for i in ids]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_ids = ids

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.lower().split()

    # ---- reranker (lazy-loaded, optional) ------------------------------ #
    def _get_reranker(self):
        if self._reranker is not None or self._reranker_unavailable:
            return self._reranker
        try:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self.reranker_model_name)
        except Exception as exc:
            logger.warning(
                "Reranker '%s' unavailable (%s) — falling back to fusion score.",
                self.reranker_model_name, exc,
            )
            self._reranker_unavailable = True
            self._reranker = None
        return self._reranker

    # ---- retrieval: dense + BM25 -> RRF fusion -> rerank -> confidence - #
    def _dense_candidates(self, query: str) -> List[Tuple[int, float, dict]]:
        try:
            hits = self.client.query(collection_name=self.COLLECTION, query_text=query, limit=self.fetch_k)
        except Exception as exc:
            logger.warning("Dense retrieval failed: %s", exc)
            return []

        out = []
        for h in hits:
            hid = getattr(h, "id", None)
            meta = getattr(h, "metadata", {}) or {}
            score = float(getattr(h, "score", 0.0))
            if hid is None:
                # defensive fallback if this qdrant-client version doesn't
                # surface point ids on query results — match by text instead
                doc_text = getattr(h, "document", "")
                hid = next((i for i, c in self._chunk_store.items() if c["text"] == doc_text), None)
            if hid is not None:
                out.append((hid, score, meta))
        return out

    def _bm25_candidates(self, query: str) -> List[Tuple[int, float]]:
        if self._bm25 is None or not self._bm25_ids:
            return []
        scores = self._bm25.get_scores(self._tokenize(query))
        ranked = sorted(zip(self._bm25_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[: self.fetch_k]

    def _fuse(
        self, dense: List[Tuple[int, float, dict]], bm25: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """Reciprocal rank fusion over the two candidate lists."""
        rrf_scores: Dict[int, float] = {}
        for rank, (cid, _score, _meta) in enumerate(dense):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, (cid, _score) in enumerate(bm25):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[: self.fetch_k]

    def _run(self, query: str) -> str:
        if not self._documents:
            self.last_sources = []
            self.last_confidence = None
            return "No documents are indexed yet."

        dense = self._dense_candidates(query)
        bm25 = self._bm25_candidates(query)
        fused = self._fuse(dense, bm25)

        if not fused:
            self.last_sources = []
            self.last_confidence = None
            return "No relevant passages were found for this query."

        dense_ids = {cid for cid, _s, _m in dense}
        bm25_ids = {cid for cid, _s in bm25}
        dense_scores = {cid: s for cid, s, _m in dense}

        candidates = [
            (cid, self._chunk_store[cid])
            for cid, _rrf in fused
            if cid in self._chunk_store
        ]

        reranker = self._get_reranker() if self.use_reranker else None
        used_reranker = reranker is not None

        if used_reranker:
            pairs = [[query, c["text"]] for _cid, c in candidates]
            rerank_scores = reranker.predict(pairs)
            scored = [
                (cid, float(score), c) for (cid, c), score in zip(candidates, rerank_scores)
            ]
        else:
            fused_lookup = dict(fused)
            scored = [(cid, fused_lookup[cid], c) for cid, c in candidates]

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[: self.top_k]

        # ---- confidence: blend retrieval similarity, rerank score, and
        # cross-method agreement (how many of the final top_k were found
        # by *both* dense and BM25, a proxy for retrieval consensus) ---- #
        top_ids = [cid for cid, _s, _c in top]
        agreement = (
            sum(1 for cid in top_ids if cid in dense_ids and cid in bm25_ids) / len(top_ids)
            if top_ids else 0.0
        )
        avg_dense_sim = (
            sum(max(min(dense_scores.get(cid, 0.0), 1.0), 0.0) for cid in top_ids) / len(top_ids)
            if top_ids else 0.0
        )
        if used_reranker:
            # CrossEncoder logits aren't bounded [0,1] — squash with a sigmoid.
            avg_rerank = sum(1 / (1 + math.exp(-s)) for _cid, s, _c in top) / len(top)
            confidence = 100 * (0.35 * avg_dense_sim + 0.4 * avg_rerank + 0.25 * agreement)
        else:
            # No reranker available: weight dense similarity + agreement only.
            confidence = 100 * (0.6 * avg_dense_sim + 0.4 * agreement)

        self.last_confidence = round(max(0.0, min(confidence, 100.0)), 1)
        self.last_retrieval_debug = {
            "dense_candidates": len(dense),
            "bm25_candidates": len(bm25),
            "fused_candidates": len(fused),
            "used_reranker": used_reranker,
            "agreement": round(agreement, 2),
        }

        def _display_score(cid: int, rerank_or_fused_score: float) -> float:
            if used_reranker:
                return round(1 / (1 + math.exp(-rerank_or_fused_score)), 3)
            return round(max(min(float(rerank_or_fused_score), 1.0), 0.0), 3)

        self.last_sources = [
            {
                "type": "doc",
                "label": f"{c['source']} · p.{c['page']}",
                "doc": c["source"],
                "page": c["page"],
                "score": _display_score(cid, s),
                "preview": c["text"][:220].strip(),
            }
            for cid, s, c in top
        ]

        blocks = [f"[Source: {c['source']}, page {c['page']}]\n{c['text']}" for _cid, _s, c in top]
        return "\n___\n".join(blocks)
