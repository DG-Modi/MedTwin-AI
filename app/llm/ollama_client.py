"""Ollama client and compatibility module for MedTwin AI."""

from app.llm.ollama_provider import OllamaProvider, OllamaClient


def get_llm_client():
    """Deferred factory call to break circular import with app.llm.factory."""
    from app.llm.factory import get_llm_provider
    return get_llm_provider()


__all__ = ["OllamaClient", "OllamaProvider", "get_llm_client"]
