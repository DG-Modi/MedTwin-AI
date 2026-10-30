"""Ollama LLM client with streaming, timeout resilience, and honest error reporting."""

import json
import logging
from typing import Optional, Dict, Any, List
import httpx

from app.config import settings
from app.llm.prompts import MEDICAL_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class OllamaClient:
    """Communicates with a local or remote Ollama instance via its HTTP API."""

    model: str = settings.OLLAMA_MODEL

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
    ) -> None:
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.OLLAMA_MODEL
        self.timeout = timeout_seconds or settings.OLLAMA_TIMEOUT_SECONDS
        self.enable_fallback = settings.ENABLE_MOCK_LLM_FALLBACK

    async def is_available(self) -> bool:
        """Checks if the Ollama daemon is reachable and responding."""
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Sends an inference request to Ollama /api/generate.

        If Ollama fails or is unreachable, logs the failure with full context and returns
        or raises an honest error ('The local model is unavailable'), never a fake generic template.
        """
        system = system_prompt or MEDICAL_SYSTEM_PROMPT
        endpoint = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
        }

        logger.info(f"Dispatching inference request to Ollama endpoint '{endpoint}' (model='{self.model}')")

        try:
            async with httpx.AsyncClient(timeout=float(self.timeout)) as client:
                response = await client.post(endpoint, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    ans = data.get("response", "").strip()
                    if ans:
                        logger.info(f"Ollama inference successful for model '{self.model}' ({len(ans)} characters)")
                        return ans
                    logger.warning(f"Ollama returned HTTP 200 at {endpoint} but generated empty text.")
                else:
                    logger.error(
                        f"Ollama returned error HTTP {response.status_code} at {endpoint}: {response.text}"
                    )
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError) as exc:
            logger.error(
                f"Ollama connection/timeout failure at '{self.base_url}': {type(exc).__name__} - {exc}"
            )
        except Exception as exc:
            logger.error(f"Unexpected error communicating with Ollama at '{self.base_url}': {exc}")

        # When the LLM fails, return an honest error, NOT a fake generic template.
        raise ConnectionError(
            f"The local model is unavailable at '{self.base_url}'. Please ensure Ollama is running and model '{self.model}' is installed."
        )


_global_llm_client: Optional[OllamaClient] = None


def get_llm_client() -> OllamaClient:
    """Singleton getter for the Ollama client."""
    global _global_llm_client
    if _global_llm_client is None:
        _global_llm_client = OllamaClient()
    return _global_llm_client
