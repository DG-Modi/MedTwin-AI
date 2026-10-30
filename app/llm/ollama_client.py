"""Ollama client and compatibility module for MedTwin AI."""

from app.llm.ollama_provider import OllamaProvider, OllamaClient
from app.llm.factory import get_llm_provider

# Backward compatibility alias
get_llm_client = get_llm_provider

__all__ = ["OllamaClient", "OllamaProvider", "get_llm_client", "get_llm_provider"]
