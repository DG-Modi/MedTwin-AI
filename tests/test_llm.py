"""Unit tests for the Ollama client and prompt building."""

import pytest
from app.llm.prompts import build_medical_prompt, MEDICAL_SYSTEM_PROMPT
from app.llm.ollama_client import OllamaClient


def test_build_medical_prompt():
    """Verify prompt assembler incorporates query, digital twin, and RAG context."""
    query = "What is the DASH diet?"
    rag_context = "### RETRIEVED MEDICAL CONTEXT:\n[1] SOURCE: hypertension.md\nDASH diet lowers sodium."
    twin_context = "### PATIENT DIGITAL TWIN PROFILE:\n- Age: 50 | Conditions: Hypertension"

    prompt = build_medical_prompt(
        query=query,
        rag_context=rag_context,
        twin_context=twin_context,
    )

    assert "hypertension.md" in prompt
    assert "DASH diet lowers sodium" in prompt
    assert "Conditions: Hypertension" in prompt
    assert "USER MEDICAL QUESTION:" in prompt
    assert "What is the DASH diet?" in prompt


@pytest.mark.asyncio
async def test_ollama_client_fallback_generation():
    """Verify OllamaClient produces grounded, safe educational response when Ollama is offline."""
    client = OllamaClient(base_url="http://localhost:99999", timeout_seconds=1)

    prompt = (
        "### PATIENT DIGITAL TWIN PROFILE:\n- Chronic Health Conditions: Type 2 Diabetes\n"
        "### RETRIEVED MEDICAL KNOWLEDGE BASE CONTEXT:\n"
        "[1] SOURCE: type_2_diabetes.md | CONFIDENCE: 85%\n"
        "Target HbA1c is below 7.0% for most adults.\n"
        "USER MEDICAL QUESTION:\nWhat is the target HbA1c?"
    )

    response = await client.generate(prompt)

    assert isinstance(response, str)
    assert len(response) > 50
    assert "type_2_diabetes.md" in response
    assert "educational" in response.lower()
    assert "Type 2 Diabetes" in response


@pytest.mark.asyncio
async def test_ollama_client_is_available():
    """Verify is_available correctly reports False when port is closed."""
    client = OllamaClient(base_url="http://localhost:99999", timeout_seconds=1)
    available = await client.is_available()
    assert available is False
