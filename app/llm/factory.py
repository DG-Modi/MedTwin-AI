"""LLM Provider factory and Auto fallback provider for MedTwin AI."""

import logging
from typing import Optional, List
from app.config import settings
from app.llm.base import LLMProvider
from app.llm.ollama_provider import OllamaProvider
from app.llm.hf_provider import HuggingFaceProvider

logger = logging.getLogger(__name__)


class AutoLLMProvider(LLMProvider):
    """Orchestrates multi-provider fallback.

    In 'auto' mode:
    1. Attempts inference using Ollama (local open-source).
    2. If Ollama is unavailable or fails, falls back to Hugging Face Inference API.
    3. If neither provider succeeds, raises an honest ConnectionError. Never returns a fake template.
    """

    name: str = "auto"

    def __init__(
        self,
        ollama_provider: Optional[OllamaProvider] = None,
        hf_provider: Optional[HuggingFaceProvider] = None,
    ) -> None:
        self.ollama = ollama_provider or OllamaProvider()
        self.hf = hf_provider or HuggingFaceProvider()
        self._active_provider: Optional[LLMProvider] = None

    @property
    def model(self) -> str:
        if self._active_provider:
            return self._active_provider.model
        return self.ollama.model

    def get_display_name(self) -> str:
        if self._active_provider:
            return self._active_provider.get_display_name()
        # Default display name when idle
        short_ollama = self.ollama.model
        return f"{short_ollama} (Auto)"

    async def is_available(self) -> bool:
        """Checks if either Ollama or Hugging Face is reachable."""
        if await self.ollama.is_available():
            self._active_provider = self.ollama
            return True
        if await self.hf.is_available():
            self._active_provider = self.hf
            return True
        return False

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Executes generation with strict fallback order."""
        ollama_err = None
        # 1. Attempt Ollama first
        try:
            if await self.ollama.is_available():
                ans = await self.ollama.generate(prompt=prompt, system_prompt=system_prompt)
                self._active_provider = self.ollama
                return ans
            else:
                ollama_err = "Ollama daemon is not reachable on configured port."
        except Exception as exc:
            ollama_err = str(exc)
            logger.warning(f"Ollama generation failed in auto mode: {exc}. Attempting Hugging Face fallback...")

        # 2. Attempt Hugging Face second
        hf_err = None
        try:
            if self.hf.token:
                ans = await self.hf.generate(prompt=prompt, system_prompt=system_prompt)
                self._active_provider = self.hf
                return ans
            else:
                hf_err = "HF_API_TOKEN is not configured."
        except Exception as exc:
            hf_err = str(exc)
            logger.warning(f"Hugging Face fallback generation failed: {exc}.")

        # 3. Honest failure across all configured providers
        self._active_provider = None
        raise ConnectionError(
            f"No LLM provider available. "
            f"Ollama failure: {ollama_err or 'unreachable'}; "
            f"Hugging Face failure: {hf_err or 'unreachable'}. "
            "Please ensure Ollama is running or provide a valid HF_API_TOKEN in .env."
        )


_global_provider: Optional[LLMProvider] = None


def get_llm_provider(provider_type: Optional[str] = None) -> LLMProvider:
    """Factory function returning the configured LLM provider singleton or requested provider."""
    global _global_provider
    ptype = (provider_type or settings.LLM_PROVIDER).lower()

    if provider_type is not None:
        # Create non-cached instance if specific provider requested
        if ptype == "ollama":
            return OllamaProvider()
        elif ptype in ("huggingface", "hf"):
            return HuggingFaceProvider()
        elif ptype == "auto":
            return AutoLLMProvider()
        raise ValueError(f"Unknown LLM provider '{provider_type}'. Must be 'ollama', 'huggingface', or 'auto'.")

    if _global_provider is None:
        if ptype == "ollama":
            _global_provider = OllamaProvider()
        elif ptype in ("huggingface", "hf"):
            _global_provider = HuggingFaceProvider()
        else:
            _global_provider = AutoLLMProvider()

    return _global_provider


def reset_llm_provider():
    """Resets the singleton for clean testing."""
    global _global_provider
    _global_provider = None

