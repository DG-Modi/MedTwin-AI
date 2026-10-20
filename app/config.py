"""Configuration management for MedTwin AI.

Uses pydantic-settings to load environment variables from .env with fallback defaults.
"""

from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and configuration parameters."""

    # Project root directory
    BASE_DIR: Path = Path(__file__).resolve().parent.parent

    # Application details
    APP_NAME: str = "MedTwin AI"
    APP_VERSION: str = "0.1.0"
    APP_ENV: str = "development"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database
    DATABASE_URL: str = "sqlite:///./medtwin.db"

    # Ollama LLM Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"
    OLLAMA_TIMEOUT_SECONDS: int = 60
    ENABLE_MOCK_LLM_FALLBACK: bool = True

    # RAG & Embeddings
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHROMA_PERSIST_DIR: str = "./data/chromadb"
    RAG_TOP_K: int = 3
    RAG_SIMILARITY_THRESHOLD: float = 0.3

    # Caching
    CACHE_TTL_SECONDS: int = 3600

    # Medical Disclaimer & Emergency Safety Settings
    MEDICAL_DISCLAIMER: str = (
        "Disclaimer: MedTwin AI is an educational medical-information assistant and NOT "
        "a diagnostic tool or healthcare provider. This tool does not provide medical diagnoses, "
        "treatment plans, or prescriptions. Always consult a qualified physician or healthcare "
        "provider for any medical concerns or emergency situations."
    )
    EMERGENCY_NOTICE: str = (
        "EMERGENCY DETECTED: If you or someone around you is experiencing life-threatening symptoms "
        "(such as severe chest pain, shortness of breath, sudden numbness, uncontrolled bleeding, "
        "or loss of consciousness), please immediately call emergency services (e.g., 911, 112, or your local emergency number) "
        "or go to the nearest emergency room immediately."
    )

    # Keywords triggering immediate emergency safety escalation
    EMERGENCY_KEYWORDS: List[str] = [
        "chest pain",
        "heart attack",
        "cannot breathe",
        "difficulty breathing",
        "severe shortness of breath",
        "stroke",
        "slurred speech",
        "unconscious",
        "suicide",
        "kill myself",
        "overdose",
        "severe bleeding",
        "anaphylaxis",
        "coughing blood",
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Global singleton settings instance
settings = Settings()
