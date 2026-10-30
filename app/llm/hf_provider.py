"""Hugging Face Inference API provider for cloud/serverless model execution."""

import json
import logging
from typing import Optional, Dict, Any
import httpx

from app.config import settings
from app.llm.base import LLMProvider
from app.llm.prompts import MEDICAL_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class HuggingFaceProvider(LLMProvider):
    """Communicates with the Hugging Face Inference API via HTTP/httpx."""

    name: str = "huggingface"

    def __init__(
        self,
        token: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
    ) -> None:
        self.token = token if token is not None else settings.HF_API_TOKEN
        self.model = model or settings.HF_MODEL
        self.timeout = timeout_seconds or getattr(settings, "HF_TIMEOUT_SECONDS", 30)

    def get_display_name(self) -> str:
        # e.g. "HF: meta-llama/Llama-3.2-3B-Instruct" or "HF: Mistral-7B-Instruct"
        short_model = self.model.split("/")[-1] if "/" in self.model else self.model
        return f"HF: {short_model}"

    async def is_available(self) -> bool:
        """Checks if Hugging Face Inference API is reachable and token is valid."""
        if not self.token:
            return False
        endpoint = f"https://api-inference.huggingface.co/models/{self.model}"
        headers = {"Authorization": f"Bearer {self.token}"}
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(endpoint, headers=headers)
                # 200: model ready, 302: redirect, 503: model loading but endpoint alive
                return res.status_code in (200, 302, 503)
        except Exception:
            return False

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Sends an inference request to Hugging Face Inference API.

        Raises honest ConnectionError if token missing, service unavailable, or model loading.
        Never returns a fake template.
        """
        if not self.token:
            raise ConnectionError(
                "Hugging Face API token is missing. Please set HF_API_TOKEN in your environment or .env file."
            )

        system = system_prompt or MEDICAL_SYSTEM_PROMPT
        endpoint = f"https://api-inference.huggingface.co/models/{self.model}"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        # Format chat template compatible with instruction models
        formatted_input = f"<|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>\n"
        payload = {
            "inputs": formatted_input,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": 0.2,
                "return_full_text": False,
            },
        }

        logger.info(f"Dispatching inference request to Hugging Face endpoint '{endpoint}' (model='{self.model}')")

        try:
            async with httpx.AsyncClient(timeout=float(self.timeout)) as client:
                response = await client.post(endpoint, json=payload, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list) and len(data) > 0:
                        ans = data[0].get("generated_text", "").strip()
                        if ans:
                            return ans
                    elif isinstance(data, dict):
                        if "generated_text" in data:
                            return data["generated_text"].strip()
                        if "error" in data:
                            raise ConnectionError(f"Hugging Face API returned error: {data['error']}")
                    logger.warning(f"Hugging Face returned 200 but generated empty text.")
                elif response.status_code == 503:
                    err_data = response.json() if "application/json" in response.headers.get("content-type", "") else {}
                    err_msg = err_data.get("error", "Model currently loading")
                    raise ConnectionError(f"Hugging Face model '{self.model}' is loading: {err_msg}")
                elif response.status_code in (401, 403):
                    raise ConnectionError(f"Hugging Face authentication failed (HTTP {response.status_code}). Verify HF_API_TOKEN.")
                else:
                    raise ConnectionError(f"Hugging Face API returned HTTP {response.status_code}: {response.text}")
        except ConnectionError:
            raise
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError) as exc:
            logger.error(f"Hugging Face connection failure at '{endpoint}': {type(exc).__name__} - {exc}")
            raise ConnectionError(f"Hugging Face Inference API is unavailable at '{endpoint}': {exc}")
        except Exception as exc:
            logger.error(f"Unexpected error communicating with Hugging Face: {exc}")
            raise ConnectionError(f"Unexpected error calling Hugging Face model '{self.model}': {exc}")

        raise ConnectionError(f"Failed to generate answer from Hugging Face model '{self.model}'.")

