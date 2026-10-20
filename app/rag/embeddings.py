"""Embedding model wrapper using SentenceTransformers for semantic dense vectors."""

from typing import List, Optional
from sentence_transformers import SentenceTransformer
from app.config import settings


class EmbeddingModel:
    """Singleton wrapper for sentence embedding generation."""

    _instance: Optional["EmbeddingModel"] = None
    _model: Optional[SentenceTransformer] = None

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self._load_model()

    def _load_model(self) -> None:
        """Loads the SentenceTransformer model if not already loaded."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

    @classmethod
    def get_instance(cls, model_name: Optional[str] = None) -> "EmbeddingModel":
        """Returns the singleton instance of the EmbeddingModel."""
        if cls._instance is None:
            cls._instance = cls(model_name=model_name)
        return cls._instance

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates dense vector embeddings for a list of document strings."""
        if not texts:
            return []
        embeddings = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Generates a dense vector embedding for a single search query."""
        if not text.strip():
            # Return zero vector of appropriate dimension (384 for all-MiniLM-L6-v2)
            dim = self._model.get_sentence_embedding_dimension() or 384
            return [0.0] * dim
        embedding = self._model.encode(text, show_progress_bar=False, normalize_embeddings=True)
        return embedding.tolist()


def get_embedding_model() -> EmbeddingModel:
    """Dependency / helper to acquire the singleton EmbeddingModel instance."""
    return EmbeddingModel.get_instance()
