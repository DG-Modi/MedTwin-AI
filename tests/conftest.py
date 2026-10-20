"""Shared test fixtures for MedTwin AI."""

import pytest
from typing import Dict, Any


@pytest.fixture
def sample_user_profile() -> Dict[str, Any]:
    """Provides a sample user health profile for digital twin testing."""
    return {
        "user_id": "test_patient_001",
        "name": "Alex",
        "age": 45,
        "gender": "male",
        "allergies": ["penicillin", "peanuts"],
        "chronic_conditions": ["Type 2 Diabetes", "Hypertension"],
        "current_medications": ["Metformin 500mg BID", "Lisinopril 10mg QD"],
        "smoking_status": "never",
        "activity_level": "moderate",
    }


@pytest.fixture
def mock_retrieved_documents() -> list[dict]:
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
