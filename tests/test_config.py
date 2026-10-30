"""Tests for configuration settings and defaults."""

from app.config import Settings, settings


def test_settings_default_values():
    """Verify default settings values are correctly loaded without environment pollution."""
    fresh_settings = Settings(_env_file=None)
    assert fresh_settings.APP_NAME == "MedTwin AI"
    assert fresh_settings.DATABASE_URL.startswith("sqlite")
    assert fresh_settings.OLLAMA_BASE_URL == "http://localhost:11434"
    assert fresh_settings.OLLAMA_MODEL == "llama3.2"
    assert fresh_settings.LLM_PROVIDER == "auto"
    assert fresh_settings.ENABLE_MOCK_LLM_FALLBACK is False
    assert fresh_settings.RAG_TOP_K == 4
    assert len(fresh_settings.EMERGENCY_KEYWORDS) > 5
    assert "chest pain" in fresh_settings.EMERGENCY_KEYWORDS


def test_medical_disclaimer_present():
    """Verify medical disclaimer is present and non-empty."""
    assert "educational medical-information assistant" in settings.MEDICAL_DISCLAIMER.lower()
    assert "emergency" in settings.EMERGENCY_NOTICE.lower()
