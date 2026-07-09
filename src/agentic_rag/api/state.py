"""
state.py
========
Process-local application state.

This is deliberately an in-memory store (thread-safe via a single lock),
matching the persistence model of the underlying retrieval tool (a
local/on-disk Qdrant collection + in-memory BM25 index, not a separate
database). It is fine for a single-instance deployment (Render/Railway/one
container). Swapping this for Redis/Postgres-backed sessions later is a
drop-in change -- every route only talks to this module, never to globals
directly.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from agentic_rag.tools.hybrid_search import HybridDocumentSearchTool


@dataclass
class ChatTurn:
    role: str
    content: str
    sources: Optional[List[dict]] = None
    confidence: Optional[float] = None


@dataclass
class Session:
    id: str
    history: List[ChatTurn] = field(default_factory=list)
    last_sources: List[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass
class QueryRecord:
    timestamp: float
    total_ms: float
    retrieval_ms: Optional[float]
    generation_ms: Optional[float]
    confidence: Optional[float]


class AppState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.document_tool: Optional[HybridDocumentSearchTool] = None
        self.sessions: Dict[str, Session] = {}

        # analytics: every recorded query, timestamped, so the analytics
        # endpoint can bucket real (if process-local) time series instead
        # of returning fabricated numbers.
        self.total_queries = 0
        self._query_records: List[QueryRecord] = []

    # ---- lifecycle -------------------------------------------------- #
    def init_document_tool(
        self,
        persist_dir: Optional[str],
        top_k: int,
        fetch_k: int = 20,
        use_reranker: bool = True,
        reranker_model: Optional[str] = None,
    ) -> None:
        with self._lock:
            if self.document_tool is None:
                self.document_tool = HybridDocumentSearchTool(
                    persist_dir=persist_dir,
                    top_k=top_k,
                    fetch_k=fetch_k,
                    use_reranker=use_reranker,
                    reranker_model=reranker_model,
                )

    # ---- sessions ----------------------------------------------------- #
    def get_or_create_session(self, session_id: Optional[str]) -> Session:
        with self._lock:
            if session_id and session_id in self.sessions:
                return self.sessions[session_id]
            new_id = session_id or str(uuid.uuid4())
            session = Session(id=new_id)
            self.sessions[new_id] = session
            return session

    def get_session(self, session_id: str) -> Optional[Session]:
        with self._lock:
            return self.sessions.get(session_id)

    def recent_history_text(self, session_id: str, max_turns: int = 6) -> str:
        """Formats the last `max_turns` turns as plain text for the query
        rewriter agent's {history} placeholder. Empty string for new sessions."""
        with self._lock:
            session = self.sessions.get(session_id)
            if not session or not session.history:
                return ""
            recent = session.history[-max_turns:]
            return "\n".join(f"{t.role.title()}: {t.content}" for t in recent)

    def append_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[List[dict]] = None,
        confidence: Optional[float] = None,
    ) -> None:
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                session = Session(id=session_id)
                self.sessions[session_id] = session
            session.history.append(ChatTurn(role=role, content=content, sources=sources, confidence=confidence))
            if sources is not None:
                session.last_sources = sources

    # ---- analytics ------------------------------------------------- #
    def record_query(
        self,
        total_ms: float,
        retrieval_ms: Optional[float] = None,
        generation_ms: Optional[float] = None,
        confidence: Optional[float] = None,
    ) -> None:
        with self._lock:
            self.total_queries += 1
            self._query_records.append(
                QueryRecord(
                    timestamp=time.time(),
                    total_ms=total_ms,
                    retrieval_ms=retrieval_ms,
                    generation_ms=generation_ms,
                    confidence=confidence,
                )
            )

    def analytics_snapshot(self) -> dict:
        with self._lock:
            def avg(values: List[float]) -> Optional[float]:
                return round(sum(values) / len(values), 2) if values else None

            response_times = [r.total_ms for r in self._query_records]
            retrieval_times = [r.retrieval_ms for r in self._query_records if r.retrieval_ms is not None]
            generation_times = [r.generation_ms for r in self._query_records if r.generation_ms is not None]
            confidences = [r.confidence for r in self._query_records if r.confidence is not None]

            # Real (process-local) daily query volume, last 14 days.
            daily = defaultdict(int)
            for r in self._query_records:
                day = datetime.fromtimestamp(r.timestamp).strftime("%Y-%m-%d")
                daily[day] += 1
            query_volume = [{"date": d, "queries": n} for d, n in sorted(daily.items())][-14:]

            # Real (process-local) latency averaged by hour-of-day.
            hourly_retrieval = defaultdict(list)
            hourly_generation = defaultdict(list)
            for r in self._query_records:
                hour = datetime.fromtimestamp(r.timestamp).hour
                if r.retrieval_ms is not None:
                    hourly_retrieval[hour].append(r.retrieval_ms)
                if r.generation_ms is not None:
                    hourly_generation[hour].append(r.generation_ms)
            hours = sorted(set(hourly_retrieval) | set(hourly_generation))
            latency_by_hour = [
                {
                    "hour": f"{h:02d}:00",
                    "retrieval_ms": round(sum(hourly_retrieval[h]) / len(hourly_retrieval[h]), 1) if hourly_retrieval[h] else None,
                    "generation_ms": round(sum(hourly_generation[h]) / len(hourly_generation[h]), 1) if hourly_generation[h] else None,
                }
                for h in hours
            ]

            # Real new-session counts per day (last 7 days) -- an honest
            # substitute for "active users", which we can't measure without auth.
            sessions_daily = defaultdict(int)
            for s in self.sessions.values():
                day = datetime.fromtimestamp(s.created_at).strftime("%a")
                sessions_daily[day] += 1
            weekday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            new_sessions_by_day = [{"day": d, "sessions": sessions_daily.get(d, 0)} for d in weekday_order]

            return {
                "total_queries": self.total_queries,
                "active_sessions": len(self.sessions),
                "avg_response_time_ms": avg(response_times),
                "avg_retrieval_latency_ms": avg(retrieval_times),
                "avg_generation_latency_ms": avg(generation_times),
                "avg_confidence": avg(confidences),
                "uptime_seconds": round(time.time() - self.started_at, 1),
                "query_volume_daily": query_volume,
                "latency_by_hour": latency_by_hour,
                "new_sessions_by_day": new_sessions_by_day,
            }


# module-level singleton -- one per process, shared by all requests
app_state = AppState()
