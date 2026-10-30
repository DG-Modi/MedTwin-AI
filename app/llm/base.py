"""Base interface for LLM providers in MedTwin AI."""

from abc import ABC, abstractmethod
from typing import Optional


class LLMProvider(ABC):
    """Abstract base class for all clinical LLM providers."""

    name: str = "base"
    model: str = ""

    @abstractmethod
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generates a text completion given a prompt and optional system instructions.

        Must raise ConnectionError or RuntimeError on failure with an honest error message.
        Must never return a fake placeholder response.
        """
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Checks if the provider service is reachable and functional."""
        pass

    def get_display_name(self) -> str:
        """Returns a human-readable display string (e.g. 'Llama 3.2 via Ollama' or 'HF: model')."""
        return f"{self.name}: {self.model}"

