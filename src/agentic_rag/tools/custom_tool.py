import os
import time
from typing import Type, Optional

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
from markitdown import MarkItDown
from chonkie import SemanticChunker
from qdrant_client import QdrantClient
from typing import ClassVar


# ============================================================================
# ORIGINAL TOOL — kept fully backward compatible.
# Anything that imports `DocumentSearchTool(file_path=...)` and calls `_run`
# continues to work exactly as before (crew.py, notebooks, etc.).
# ============================================================================
class DocumentSearchToolInput(BaseModel):
    """Input schema for DocumentSearchTool."""
    query: str = Field(..., description="Query to search the document.")


class DocumentSearchTool(BaseTool):
    name: str = "DocumentSearchTool"
    description: str = "Search the document for the given query."
    args_schema: Type[BaseModel] = DocumentSearchToolInput

    model_config = ConfigDict(extra="allow")

    def __init__(self, file_path: str):
        """Initialize the searcher with a PDF file path and set up the Qdrant collection."""
        super().__init__()
        self.file_path = file_path
        self.client = QdrantClient(":memory:")  # For small experiments
        self._process_document()

    def _extract_text(self) -> str:
        """Extract raw text from PDF using MarkItDown."""
        md = MarkItDown()
        result = md.convert(self.file_path)
        return result.text_content

    def _create_chunks(self, raw_text: str) -> list:
        """Create semantic chunks from raw text."""
        chunker = SemanticChunker(
            embedding_model="minishlab/potion-base-8M",
            threshold=0.5,
            chunk_size=512,
            min_sentences=1
        )
        return chunker.chunk(raw_text)

    def _process_document(self):
        """Process the document and add chunks to Qdrant collection."""
        raw_text = self._extract_text()
        chunks = self._create_chunks(raw_text)

        docs = [chunk.text for chunk in chunks]
        metadata = [{"source": os.path.basename(self.file_path)} for _ in range(len(chunks))]
        ids = list(range(len(chunks)))

        self.client.add(
            collection_name="demo_collection",
            documents=docs,
            metadata=metadata,
            ids=ids
        )

    def _run(self, query: str) -> str:
        """Search the document with a query string."""
        relevant_chunks = self.client.query(
            collection_name="demo_collection",
            query_text=query
        )
        docs = [chunk.document for chunk in relevant_chunks]
        separator = "\n___\n"
        return separator.join(docs)


# ============================================================================
# NEW TOOL — MultiDocumentSearchTool
# Extends the exact same retrieval concept (MarkItDown -> Chonkie -> Qdrant)
# with the improvements needed for the redesigned app:
#   - multiple documents in one knowledge base (not just one PDF)
#   - persistent, on-disk Qdrant index (survives across reruns of the app,
#     not just in-memory for one session)
#   - page-aware chunking, so every chunk carries a real page number
#   - per-document add / remove without rebuilding the whole index
#   - `last_sources` is populated on every `_run` call so the UI layer can
#     render citations ("PDF report.pdf, p.4") without parsing agent logs
# It does NOT replace DocumentSearchTool — both are exported so existing
# code (crew.py) keeps working unchanged.
# ============================================================================
class MultiDocumentSearchToolInput(BaseModel):
    """Input schema for MultiDocumentSearchTool."""
    query: str = Field(..., description="Query to search across all indexed documents.")


class MultiDocumentSearchTool(BaseTool):
    name: str = "DocumentSearchTool"
    description: str = (
        "Search across all currently indexed PDF documents for the given query. "
        "Returns the most relevant passages along with their source document and page number."
    )
    args_schema: Type[BaseModel] = MultiDocumentSearchToolInput

    model_config = ConfigDict(extra="allow")

    COLLECTION: ClassVar[str] = "agentic_rag_documents"

    def __init__(self, persist_dir: Optional[str] = None, top_k: int = 5):
        """
        persist_dir: if provided, the vector index is stored on disk at this
                     path and survives across app restarts. If None, an
                     in-memory (session-only) index is used instead — the
                     same default behavior as the original tool.
        """
        super().__init__()
        self.persist_dir = persist_dir
        self.top_k = top_k
        self.client = QdrantClient(path=persist_dir) if persist_dir else QdrantClient(":memory:")
        self._documents: dict[str, dict] = {}   # name -> {chunks, indexed_at}
        self._next_id = 0
        self.last_sources: list[dict] = []       # populated after every _run()

    # ---- indexing ---------------------------------------------------- #
    def _extract_pages(self, file_path: str) -> list[str]:
        """
        Extract text page-by-page when possible (pypdf), so citations can
        reference a real page number. Falls back to a single "page" using
        MarkItDown (whole-document extraction) for non-PDF or if pypdf is
        unavailable, which matches the original tool's behavior.
        """
        if file_path.lower().endswith(".pdf"):
            try:
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                pages = [page.extract_text() or "" for page in reader.pages]
                if any(p.strip() for p in pages):
                    return pages
            except Exception:
                pass
        md = MarkItDown()
        return [md.convert(file_path).text_content]

    def _create_chunks(self, raw_text: str) -> list:
        chunker = SemanticChunker(
            embedding_model="minishlab/potion-base-8M",
            threshold=0.5,
            chunk_size=512,
            min_sentences=1,
        )
        return chunker.chunk(raw_text)

    def add_document(self, file_path: str) -> int:
        """Index a new document (or re-index if the same name already exists). Returns chunk count."""
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
                self._next_id += 1

        if docs:
            self.client.add(
                collection_name=self.COLLECTION,
                documents=docs,
                metadata=metadata,
                ids=ids,
            )

        self._documents[name] = {
            "chunks": len(docs),
            "indexed_at": time.strftime("%H:%M:%S"),
        }
        return len(docs)

    def remove_document(self, name: str):
        """Remove a single document's chunks from the index without touching the rest."""
        try:
            self.client.delete(
                collection_name=self.COLLECTION,
                points_selector={"filter": {"must": [{"key": "source", "match": {"value": name}}]}},
            )
        except Exception:
            pass
        self._documents.pop(name, None)

    def list_documents(self) -> list[dict]:
        return [{"name": n, **meta} for n, meta in self._documents.items()]

    def has_documents(self) -> bool:
        return len(self._documents) > 0

    # ---- retrieval ----------------------------------------------------- #
    def _run(self, query: str) -> str:
        if not self._documents:
            self.last_sources = []
            return "No documents are indexed yet."

        try:
            relevant_chunks = self.client.query(
                collection_name=self.COLLECTION,
                query_text=query,
                limit=self.top_k,
            )
        except Exception as e:
            self.last_sources = []
            return f"Document search failed: {e}"

        self.last_sources = [
            {
                "type": "doc",
                "label": f"{c.metadata.get('source', 'document')} · p.{c.metadata.get('page', '?')}",
                "score": round(getattr(c, "score", 0.0), 3),
            }
            for c in relevant_chunks
        ]

        blocks = []
        for c in relevant_chunks:
            src = c.metadata.get("source", "document")
            page = c.metadata.get("page", "?")
            blocks.append(f"[Source: {src}, page {page}]\n{c.document}")
        return "\n___\n".join(blocks)


# ============================================================================
# FireCrawlWebSearchTool
# NOTE: app_deep_seek.py / app_llama3.2.py already imported a
# `FireCrawlWebSearchTool` from this module, but the class was never actually
# defined here — a pre-existing bug in the original project that meant the
# local-LLM apps' web-search fallback could never import successfully. It is
# implemented here now (using the `firecrawl-py` SDK + FIRECRAWL_API_KEY, as
# documented in the README) so those two entry points work as originally
# intended, with no change to their public interface.
# ============================================================================
class FireCrawlWebSearchToolInput(BaseModel):
    """Input schema for FireCrawlWebSearchTool."""
    query: str = Field(..., description="Query to search the web for.")


class FireCrawlWebSearchTool(BaseTool):
    name: str = "FireCrawlWebSearchTool"
    description: str = "Search the web for the given query using FireCrawl."
    args_schema: Type[BaseModel] = FireCrawlWebSearchToolInput

    model_config = ConfigDict(extra="allow")

    def __init__(self, limit: int = 5):
        super().__init__()
        self.limit = limit

    def _run(self, query: str) -> str:
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            return "Web search is unavailable: FIRECRAWL_API_KEY is not set in the environment."
        try:
            from firecrawl import FirecrawlApp
        except ImportError:
            return "Web search is unavailable: the 'firecrawl-py' package is not installed."

        try:
            app = FirecrawlApp(api_key=api_key)
            results = app.search(query, limit=self.limit)
            entries = getattr(results, "data", results) or []
            blocks = []
            for r in entries:
                title = r.get("title", "") if isinstance(r, dict) else getattr(r, "title", "")
                url = r.get("url", "") if isinstance(r, dict) else getattr(r, "url", "")
                snippet = r.get("description", "") if isinstance(r, dict) else getattr(r, "description", "")
                blocks.append(f"[{title}]({url})\n{snippet}")
            return "\n___\n".join(blocks) if blocks else "No web results found."
        except Exception as e:
            return f"Web search failed: {e}"


# Test the implementation
def test_document_searcher():
    # Test file path — env-var driven with a relative default, was hardcoded
    # to a developer's local machine path.
    import os
    default_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "knowledge", "dspy.pdf")
    )
    pdf_path = os.getenv("PDF_TOOL_PATH", default_path)

    # Create instance
    searcher = DocumentSearchTool(file_path=pdf_path)

    # Test search
    result = searcher._run("What is the purpose of DSpy?")
    print("Search Results:", result)


if __name__ == "__main__":
    test_document_searcher()
