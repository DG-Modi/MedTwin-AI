"""Tests for configuration settings and defaults."""

from app.config import settings


def test_settings_default_values():
    """Verify default settings values are correctly loaded."""
    assert settings.APP_NAME == "MedTwin AI"
    assert settings.DATABASE_URL.startswith("sqlite")
    assert settings.OLLAMA_BASE_URL == "http://localhost:11434"
    assert settings.OLLAMA_MODEL == "llama3"
    assert settings.ENABLE_MOCK_LLM_FALLBACK is False
    assert settings.RAG_TOP_K == 4
    assert len(settings.EMERGENCY_KEYWORDS) > 5
    assert "chest pain" in settings.EMERGENCY_KEYWORDS


def test_medical_disclaimer_present():
    """Verify medical disclaimer is present and non-empty."""
    assert "educational medical-information assistant" in settings.MEDICAL_DISCLAIMER.lower()
    assert "emergency" in settings.EMERGENCY_NOTICE.lower()
