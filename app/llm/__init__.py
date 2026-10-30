"""LLM Providers and prompt engineering package for MedTwin AI."""

from app.llm.base import LLMProvider
from app.llm.ollama_provider import OllamaProvider, OllamaClient
from app.llm.hf_provider import HuggingFaceProvider
from app.llm.factory import AutoLLMProvider, get_llm_provider, reset_llm_provider
from app.llm.prompts import MEDICAL_SYSTEM_PROMPT, build_medical_prompt

# Backward compatibility alias
get_llm_client = get_llm_provider

__all__ = [
    "LLMProvider",
    "OllamaProvider",
    "OllamaClient",
    "HuggingFaceProvider",
    "AutoLLMProvider",
    "get_llm_provider",
    "get_llm_client",
    "reset_llm_provider",
    "MEDICAL_SYSTEM_PROMPT",
    "build_medical_prompt",
]
