"""End-to-end RAG orchestrator for document ingestion, retrieval, and context formatting."""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from app.config import settings
from app.rag.document_loader import load_all_documents
from app.rag.chunking import chunk_all_documents
from app.rag.vector_store import ChromaVectorStore, get_vector_store


class RAGPipeline:
    """Orchestrates document loading, chunking, embedding, vector storage, and retrieval."""

    def __init__(self, vector_store: Optional[ChromaVectorStore] = None) -> None:
        self.vector_store = vector_store or get_vector_store()

    def ingest_knowledge_base(
        self,
        docs_dir: Optional[Union[str, Path]] = None,
        force_reload: bool = False,
    ) -> Dict[str, Any]:
        """Loads all documents from the medical knowledge base directory into ChromaDB.

        Skips re-indexing if documents are already stored unless force_reload=True.
        """
        target_dir = Path(docs_dir) if docs_dir else settings.BASE_DIR / "data" / "sample_docs"
        current_count = self.vector_store.count()

        if current_count > 0 and not force_reload:
            return {
                "status": "already_indexed",
                "chunks_count": current_count,
                "message": f"Knowledge base already contains {current_count} chunks. Use force_reload to overwrite.",
            }

        if force_reload:
            self.vector_store.reset()

        # Step 1 & 2: Load and Clean
        documents = load_all_documents(target_dir)
        if not documents:
            return {
                "status": "empty",
                "chunks_count": 0,
                "message": f"No valid documents found in {target_dir}",
            }

        # Step 3: Chunk
        chunks = chunk_all_documents(documents, chunk_size=500, chunk_overlap=100)

        # Step 4 & 5: Embed and Store in ChromaDB
        indexed_count = self.vector_store.add_chunks(chunks)

        return {
            "status": "success",
            "documents_loaded": len(documents),
            "chunks_indexed": indexed_count,
            "message": f"Successfully ingested {len(documents)} documents ({indexed_count} chunks) into ChromaDB.",
        }

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Performs semantic similarity search to retrieve relevant medical chunks.

        Returns attributed chunks with source, score, and excerpt text.
        """
        k = top_k or settings.RAG_TOP_K
        threshold = min_score if min_score is not None else settings.RAG_SIMILARITY_THRESHOLD

        return self.vector_store.similarity_search(
            query=query,
            top_k=k,
            min_score=threshold,
        )

    def format_rag_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Formats retrieved chunks into structured context for the LLM prompt.

        Enforces full chunk text, [n] source IDs, title, URL, and similarity scores.
        """
        if not retrieved_chunks:
            return (
                "No specific clinical knowledge base documents matched this query with sufficient confidence.\n"
                "If information is missing, explicitly state: 'not found in knowledge base'."
            )

        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            source = chunk.get("source", "Medical Guide")
            title = chunk.get("title", "")
            url = chunk.get("url", "")
            score = chunk.get("score", 0.0)
            text = chunk.get("text", "").strip()

            header = f"[{i}] SOURCE: {source} | SOURCE ID: [{i}]"
            if title:
                header += f" | TITLE: {title}"
            if url:
                header += f" | URL: {url}"
            header += f" | CONFIDENCE: {int(score * 100)}% (SCORE: {score:.2f})"

            context_blocks.append(f"{header}\nCHUNK TEXT:\n{text}\n")

        return "\n".join(context_blocks)


_global_rag_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Singleton getter for the RAGPipeline."""
    global _global_rag_pipeline
    if _global_rag_pipeline is None:
        _global_rag_pipeline = RAGPipeline()
    return _global_rag_pipeline
