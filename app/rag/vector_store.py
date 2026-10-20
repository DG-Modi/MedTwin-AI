"""ChromaDB vector store client for medical document indexing and semantic retrieval."""

from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.rag.embeddings import EmbeddingModel, get_embedding_model


class ChromaVectorStore:
    """Manages ChromaDB vector database interactions for medical documents."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "medical_knowledge",
        embedding_model: Optional[EmbeddingModel] = None,
        is_ephemeral: bool = False,
    ) -> None:
        self.persist_dir = persist_directory or settings.CHROMA_PERSIST_DIR
        self.collection_name = collection_name
        self.embedding_model = embedding_model or get_embedding_model()
        self.is_ephemeral = is_ephemeral

        # Ensure directory exists if persistent
        if not self.is_ephemeral:
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_dir)
        else:
            self.client = chromadb.EphemeralClient()

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Embeds and upserts document chunks into ChromaDB.

        Returns the number of chunks successfully stored.
        """
        if not chunks:
            return 0

        ids = [chunk["id"] for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [
            {
                "source": chunk.get("source", "unknown"),
                "title": chunk.get("title", ""),
                "chunk_index": chunk.get("chunk_index", 0),
                "char_count": chunk.get("char_count", len(chunk["text"])),
            }
            for chunk in chunks
        ]

        # Generate dense embeddings
        embeddings = self.embedding_model.embed_documents(texts)

        # Batch upsert into ChromaDB
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        return len(chunks)

    def similarity_search(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Retrieves top_k most semantically relevant chunks for a user query.

        Calculates cosine similarity score from distance and returns attributed results.
        """
        if not query.strip() or self.count() == 0:
            return []

        # Embed user query
        query_embedding = self.embedding_model.embed_query(query)

        # Query ChromaDB collection
        n_results = min(top_k, self.count())
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: List[Dict[str, Any]] = []

        ids_list = results.get("ids", [[]])[0]
        docs_list = results.get("documents", [[]])[0]
        metas_list = results.get("metadatas", [[]])[0]
        dists_list = results.get("distances", [[]])[0]

        for i, chunk_id in enumerate(ids_list):
            distance = dists_list[i] if i < len(dists_list) else 1.0
            # Convert cosine distance to similarity score in [0.0, 1.0]
            similarity = max(0.0, min(1.0, 1.0 - distance))

            if similarity >= min_score:
                meta = metas_list[i] if i < len(metas_list) else {}
                retrieved.append({
                    "id": chunk_id,
                    "text": docs_list[i] if i < len(docs_list) else "",
                    "source": meta.get("source", "unknown"),
                    "title": meta.get("title", "Clinical Document"),
                    "chunk_index": meta.get("chunk_index", 0),
                    "score": round(similarity, 4),
                })

        return retrieved

    def count(self) -> int:
        """Returns the total number of chunks indexed in the collection."""
        return self.collection.count()

    def reset(self) -> None:
        """Deletes all items in collection and reinitializes."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )


_global_vector_store: Optional[ChromaVectorStore] = None


def get_vector_store() -> ChromaVectorStore:
    """Returns the application singleton ChromaVectorStore."""
    global _global_vector_store
    if _global_vector_store is None:
        _global_vector_store = ChromaVectorStore()
    return _global_vector_store
