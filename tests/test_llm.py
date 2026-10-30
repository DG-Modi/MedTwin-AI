"""Unit tests for the Ollama client and prompt building."""

import pytest
from app.llm.prompts import build_medical_prompt, MEDICAL_SYSTEM_PROMPT
from app.llm.ollama_client import OllamaClient


def test_build_medical_prompt():
    """Verify prompt assembler incorporates query, digital twin, and RAG context."""
    query = "What is the DASH diet?"
    rag_context = "[1] SOURCE ID: [1] | FILE: hypertension.md\nDASH diet lowers sodium."
    twin_context = "- Patient/User Name: Alex Morgan\n- Age: 45\n- Chronic Health Conditions: Hypertension"

    prompt = build_medical_prompt(
        query=query,
        rag_context=rag_context,
        twin_context=twin_context,
    )

    assert "hypertension.md" in prompt
    assert "DASH diet lowers sodium" in prompt
    assert "Chronic Health Conditions: Hypertension" in prompt
    assert "USER MEDICAL QUESTION:" in prompt
    assert "What is the DASH diet?" in prompt
    assert "not found in knowledge base" in prompt


@pytest.mark.asyncio
async def test_ollama_client_failure_returns_error_instead_of_template():
    """Verify OllamaClient raises ConnectionError with honest error message when Ollama is offline."""
    client = OllamaClient(base_url="http://localhost:99999", timeout_seconds=1)

    prompt = (
        "### PATIENT DIGITAL TWIN CONTEXT:\n- Chronic Health Conditions: Type 2 Diabetes\n"
        "### RETRIEVED KNOWLEDGE BASE CONTEXT:\n"
        "[1] SOURCE ID: [1] | FILE: type_2_diabetes.md\n"
        "Target HbA1c is below 7.0% for most adults.\n"
        "USER MEDICAL QUESTION:\nWhat is the target HbA1c?"
    )

    with pytest.raises(ConnectionError) as exc_info:
        await client.generate(prompt)

    err_msg = str(exc_info.value)
    assert "The local model is unavailable" in err_msg
    assert "documented for this topic" not in err_msg
    assert "inhaler" not in err_msg.lower()


@pytest.mark.asyncio
async def test_ollama_client_is_available():
    """Verify is_available correctly reports False when port is closed."""
    client = OllamaClient(base_url="http://localhost:99999", timeout_seconds=1)
    available = await client.is_available()
    assert available is False
