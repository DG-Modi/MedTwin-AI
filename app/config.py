"""Configuration management for MedTwin AI.

Uses pydantic-settings to load environment variables from .env with fallback defaults.
"""

from pathlib import Path
from typing import List, Optional
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

    # LLM Provider Configuration (ollama | huggingface | auto)
    LLM_PROVIDER: str = "auto"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"
    OLLAMA_TIMEOUT_SECONDS: int = 60
    HF_API_TOKEN: Optional[str] = None
    HF_MODEL: str = "meta-llama/Llama-3.2-3B-Instruct"
    HF_TIMEOUT_SECONDS: int = 30
    ENABLE_MOCK_LLM_FALLBACK: bool = False
    ENABLE_GROUNDED_RAG_SYNTHESIZER: bool = False

    # Emergency Services Configuration
    EMERGENCY_NUMBERS: str = "112, 108, 911"

    # RAG & Embeddings
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHROMA_PERSIST_DIR: str = "./data/chromadb"
    RAG_TOP_K: int = 4
    RAG_SIMILARITY_THRESHOLD: float = 0.25

    # Caching
    CACHE_TTL_SECONDS: int = 3600

    # Voice Input & Audio Transcription (Whisper)
    WHISPER_MODEL_NAME: str = "tiny"
    ENABLE_VOICE_INPUT: bool = True


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
