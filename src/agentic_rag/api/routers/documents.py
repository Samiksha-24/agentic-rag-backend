from __future__ import annotations

import logging
import os
import tempfile
from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

from agentic_rag.api.config import Settings, get_settings
from agentic_rag.api.dependencies import get_document_tool
from agentic_rag.api.schemas import (
    DeleteDocumentResponse,
    DocumentInfo,
    DocumentListResponse,
    UploadResponse,
    UploadResult,
)
from agentic_rag.tools.hybrid_search import HybridDocumentSearchTool

logger = logging.getLogger("agentic_rag.api.documents")

router = APIRouter(tags=["documents"])


def _to_document_info(docs: List[dict]) -> List[DocumentInfo]:
    return [DocumentInfo(name=d["name"], chunks=d["chunks"], indexed_at=d["indexed_at"]) for d in docs]


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(tool: HybridDocumentSearchTool = Depends(get_document_tool)) -> DocumentListResponse:
    docs = tool.list_documents()
    return DocumentListResponse(
        documents=_to_document_info(docs),
        total_documents=len(docs),
        total_chunks=sum(d["chunks"] for d in docs),
    )


@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_documents(
    files: List[UploadFile] = File(...),
    settings: Settings = Depends(get_settings),
    tool: HybridDocumentSearchTool = Depends(get_document_tool),
) -> UploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    uploaded: List[UploadResult] = []
    skipped: List[str] = []

    for f in files:
        ext = (f.filename or "").rsplit(".", 1)[-1].lower() if "." in (f.filename or "") else ""
        if ext not in settings.supported_extensions:
            skipped.append(f"{f.filename} (unsupported type '.{ext}')")
            continue

        contents = await f.read()
        if len(contents) > max_bytes:
            skipped.append(f"{f.filename} (exceeds {settings.max_upload_mb}MB limit)")
            continue
        if len(contents) == 0:
            skipped.append(f"{f.filename} (empty file)")
            continue

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = os.path.join(temp_dir, f.filename)
            with open(temp_path, "wb") as out:
                out.write(contents)
            try:
                chunk_count = tool.add_document(temp_path)
            except Exception as exc:
                logger.exception("Failed to index %s", f.filename)
                skipped.append(f"{f.filename} (indexing failed: {exc})")
                continue

        uploaded.append(UploadResult(name=f.filename, chunks_indexed=chunk_count, status="indexed"))

    return UploadResponse(
        uploaded=uploaded,
        skipped=skipped,
        documents=_to_document_info(tool.list_documents()),
    )


@router.delete("/documents/{name}", response_model=DeleteDocumentResponse)
def delete_document(name: str, tool: HybridDocumentSearchTool = Depends(get_document_tool)) -> DeleteDocumentResponse:
    existing = {d["name"] for d in tool.list_documents()}
    if name not in existing:
        raise HTTPException(status_code=404, detail=f"Document '{name}' is not indexed.")
    tool.remove_document(name)
    return DeleteDocumentResponse(name=name, deleted=True)
