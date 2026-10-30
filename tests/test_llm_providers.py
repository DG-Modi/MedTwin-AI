"""Tests for LLM providers: OllamaProvider, HuggingFaceProvider, and AutoLLMProvider."""

import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

from app.llm.base import LLMProvider
from app.llm.ollama_provider import OllamaProvider
from app.llm.hf_provider import HuggingFaceProvider
from app.llm.factory import AutoLLMProvider, get_llm_provider, reset_llm_provider


def test_provider_selection():
    """Verify provider factory correctly instantiates providers based on input parameter."""
    ollama = get_llm_provider("ollama")
    assert isinstance(ollama, OllamaProvider)
    assert ollama.name == "ollama"
    assert "Ollama" in ollama.get_display_name()

    hf = get_llm_provider("huggingface")
    assert isinstance(hf, HuggingFaceProvider)
    assert hf.name == "huggingface"
    assert "HF:" in hf.get_display_name()

    auto = get_llm_provider("auto")
    assert isinstance(auto, AutoLLMProvider)
    assert auto.name == "auto"

    with pytest.raises(ValueError):
        get_llm_provider("unsupported_provider")


@pytest.mark.asyncio
async def test_auto_fallback_order_ollama_first():
    """Verify AutoLLMProvider tries and succeeds with Ollama first when available."""
    mock_ollama = MagicMock(spec=OllamaProvider)
    mock_ollama.name = "ollama"
    mock_ollama.model = "llama3.2"
    mock_ollama.is_available = AsyncMock(return_value=True)
    mock_ollama.generate = AsyncMock(return_value="Answer from Ollama")

    mock_hf = MagicMock(spec=HuggingFaceProvider)
    mock_hf.name = "huggingface"
    mock_hf.token = "fake_token"
    mock_hf.is_available = AsyncMock(return_value=True)
    mock_hf.generate = AsyncMock(return_value="Answer from HF")

    auto_provider = AutoLLMProvider(ollama_provider=mock_ollama, hf_provider=mock_hf)
    result = await auto_provider.generate(prompt="What are normal BP ranges?")

    assert result == "Answer from Ollama"
    mock_ollama.generate.assert_called_once()
    mock_hf.generate.assert_not_called()


@pytest.mark.asyncio
async def test_auto_fallback_order_hf_second():
    """Verify AutoLLMProvider falls back to Hugging Face when Ollama is unavailable."""
    mock_ollama = MagicMock(spec=OllamaProvider)
    mock_ollama.name = "ollama"
    mock_ollama.model = "llama3.2"
    mock_ollama.is_available = AsyncMock(return_value=False)
    mock_ollama.generate = AsyncMock(side_effect=ConnectionError("Ollama offline"))

    mock_hf = MagicMock(spec=HuggingFaceProvider)
    mock_hf.name = "huggingface"
    mock_hf.model = "meta-llama/Llama-3.2-3B-Instruct"
    mock_hf.token = "hf_token_valid"
    mock_hf.is_available = AsyncMock(return_value=True)
    mock_hf.generate = AsyncMock(return_value="Answer from Hugging Face")

    auto_provider = AutoLLMProvider(ollama_provider=mock_ollama, hf_provider=mock_hf)
    result = await auto_provider.generate(prompt="Explain diabetes target ranges.")

    assert result == "Answer from Hugging Face"
    mock_hf.generate.assert_called_once()


@pytest.mark.asyncio
async def test_auto_fallback_both_fail_honest_error():
    """Verify AutoLLMProvider raises honest ConnectionError when both Ollama and HF fail."""
    mock_ollama = MagicMock(spec=OllamaProvider)
    mock_ollama.name = "ollama"
    mock_ollama.model = "llama3.2"
    mock_ollama.is_available = AsyncMock(return_value=False)
    mock_ollama.generate = AsyncMock(side_effect=ConnectionError("Ollama offline"))

    mock_hf = MagicMock(spec=HuggingFaceProvider)
    mock_hf.name = "huggingface"
    mock_hf.model = "meta-llama/Llama-3.2-3B-Instruct"
    mock_hf.token = None  # No token
    mock_hf.is_available = AsyncMock(return_value=False)
    mock_hf.generate = AsyncMock(side_effect=ConnectionError("HF token missing"))

    auto_provider = AutoLLMProvider(ollama_provider=mock_ollama, hf_provider=mock_hf)

    with pytest.raises(ConnectionError) as exc_info:
        await auto_provider.generate(prompt="Can I take aspirin?")

    error_msg = str(exc_info.value).lower()
    assert "no llm provider available" in error_msg
    assert "ollama" in error_msg


@pytest.mark.asyncio
async def test_hf_provider_mock_httpx_generate():
    """Verify HuggingFaceProvider parses generated_text from successful HTTP 200 response."""
    hf_provider = HuggingFaceProvider(token="hf_mock_token_123", model="meta-llama/Llama-3.2-3B-Instruct")

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = [{"generated_text": "Clinical guidance for hypertension."}]

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        result = await hf_provider.generate(prompt="What is hypertension?")
        assert result == "Clinical guidance for hypertension."


@pytest.mark.asyncio
async def test_hf_provider_missing_token_error():
    """Verify HuggingFaceProvider raises honest ConnectionError when token is missing."""
    hf_provider = HuggingFaceProvider(token=None)
    with pytest.raises(ConnectionError) as exc_info:
        await hf_provider.generate(prompt="Test prompt")
    assert "token is missing" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_hf_provider_http_503_loading_error():
    """Verify HuggingFaceProvider reports model loading on HTTP 503."""
    hf_provider = HuggingFaceProvider(token="hf_valid_token")

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 503
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {"error": "Model currently loading", "estimated_time": 20.0}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        with pytest.raises(ConnectionError) as exc_info:
            await hf_provider.generate(prompt="Query")
        assert "loading" in str(exc_info.value).lower()

