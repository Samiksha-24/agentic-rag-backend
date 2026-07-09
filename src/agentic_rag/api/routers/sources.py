from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, Query

from agentic_rag.api.dependencies import get_document_tool
from agentic_rag.api.schemas import SourceItem
from agentic_rag.api.state import app_state
from agentic_rag.tools.hybrid_search import HybridDocumentSearchTool

router = APIRouter(tags=["sources"])


@router.get("/sources", response_model=List[SourceItem])
def get_sources(
    session_id: Optional[str] = Query(default=None, description="If provided, returns that session's last sources."),
    tool: HybridDocumentSearchTool = Depends(get_document_tool),
) -> List[SourceItem]:
    """
    Returns the source citations for the most recent answer: a specific
    session's last answer if session_id is given, otherwise the most
    recent query handled by this process (any session).
    """
    if session_id:
        session = app_state.get_session(session_id)
        raw = session.last_sources if session else []
    else:
        raw = tool.last_sources or []

    return [SourceItem(**s) for s in raw]
