"""Shared test fixtures for MedTwin AI automated testing suite."""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import io
from typing import Dict, Any, List, Generator
import numpy as np
import pytest
from scipy.io import wavfile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from fastapi.testclient import TestClient

from app.api.main import app
from app.db.session import Base, get_db
from app.core.cache import QueryCache


@pytest.fixture
def sample_user_profile() -> Dict[str, Any]:
    """Provides a sample user health profile for digital twin testing."""
    return {
        "user_id": "test_patient_001",
        "name": "Alex Morgan",
        "age": 45,
        "gender": "male",
        "allergies": ["penicillin", "peanuts"],
        "chronic_conditions": ["Type 2 Diabetes", "Hypertension"],
        "current_medications": ["Metformin 500mg BID", "Lisinopril 10mg QD"],
        "smoking_status": "never",
        "activity_level": "moderate",
    }


@pytest.fixture
def sample_patient_profiles() -> Dict[str, Dict[str, Any]]:
    """Provides a set of varied clinical profiles for multi-patient testing."""
    return {
        "diabetic": {
            "user_id": "patient_diab_01",
            "name": "Alex Morgan",
            "age": 45,
            "gender": "Male",
            "allergies": ["penicillin", "peanuts"],
            "chronic_conditions": ["Type 2 Diabetes", "Hypertension"],
            "current_medications": ["Metformin 500mg BID", "Lisinopril 10mg QD"],
            "lifestyle_notes": "Moderate daily walking, non-smoker, DASH-style diet",
        },
        "asthmatic": {
            "user_id": "patient_asthma_02",
            "name": "Elena Rostova",
            "age": 32,
            "gender": "Female",
            "allergies": ["aspirin", "NSAIDs", "cat dander"],
            "chronic_conditions": ["Persistent Asthma", "Allergic Rhinitis"],
            "current_medications": ["Albuterol HFA PRN", "Fluticasone BID"],
            "lifestyle_notes": "Indoor yoga, air purifier, avoids NSAIDs",
        },
        "migraineur": {
            "user_id": "patient_migraine_03",
            "name": "Marcus Vance",
            "age": 58,
            "gender": "Male",
            "allergies": ["sulfa drugs"],
            "chronic_conditions": ["Chronic Migraines", "GERD"],
            "current_medications": ["Sumatriptan 50mg PRN", "Omeprazole 20mg QD"],
            "lifestyle_notes": "High stress desk work, avoids red wine and aged cheese",
        },
    }


@pytest.fixture
def mock_retrieved_documents() -> List[Dict[str, Any]]:
    """Provides sample retrieved medical chunks with source attribution."""
    return [
        {
            "id": "doc1_chunk1",
            "text": "Target HbA1c is generally below 7.0% for most non-pregnant adults with Type 2 Diabetes.",
            "source": "type_2_diabetes.md",
            "score": 0.88,
        },
        {
            "id": "doc2_chunk1",
            "text": "DASH diet emphasizes fruits, vegetables, whole grains, and sodium under 2,300 mg daily for hypertension.",
            "source": "hypertension.md",
            "score": 0.81,
        },
    ]


from sqlalchemy.pool import StaticPool

@pytest.fixture
def test_db_session() -> Generator[Session, None, None]:
    """Provides an isolated in-memory SQLite database session with clean schema creation and teardown."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(test_db_session: Session) -> Generator[TestClient, None, None]:
    """Provides a FastAPI TestClient with database dependency overridden to an isolated in-memory DB."""
    def override_get_db():
        try:
            yield test_db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def sample_synthetic_wav() -> bytes:
    """Generates an in-memory 0.5s 16kHz sine wave audio WAV file for voice testing."""
    sr = 16000
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    data = (0.5 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, data)
    return buf.getvalue()


@pytest.fixture
def fresh_cache() -> QueryCache:
    """Provides an empty, isolated QueryCache instance."""
    return QueryCache(max_size=50, ttl_seconds=300)
