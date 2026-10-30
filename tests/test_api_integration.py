"""End-to-end integration tests for FastAPI REST API endpoints."""

import io
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import UserProfile


def test_health_check_endpoint(client: TestClient):
    """Verify GET /health returns 200 with service status, app version, and vector store count."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app_name"] == settings.APP_NAME
    assert "vector_store_chunks" in data
    assert isinstance(data["vector_store_chunks"], int)


def test_twin_crud_lifecycle(client: TestClient):
    """Verify complete CRUD lifecycle of a Digital Twin profile via /api/twin."""
    # 1. Create Profile
    create_payload = {
        "user_id": "patient_api_test_01",
        "name": "Sarah Connor",
        "age": 39,
        "gender": "Female",
        "allergies": ["amoxicillin"],
        "chronic_conditions": ["Mild Asthma"],
        "current_medications": ["Albuterol inhaler"],
        "lifestyle_notes": "Crossfit training, non-smoker",
    }
    create_res = client.post("/api/twin", json=create_payload)
    assert create_res.status_code == 200
    created = create_res.json()
    assert created["user_id"] == "patient_api_test_01"
    assert created["name"] == "Sarah Connor"
    assert "amoxicillin" in created["allergies"]
    assert "Mild Asthma" in created["chronic_conditions"]

    # 2. Retrieve Profile by ID
    get_res = client.get("/api/twin/patient_api_test_01")
    assert get_res.status_code == 200
    fetched = get_res.json()
    assert fetched["name"] == "Sarah Connor"
    assert fetched["age"] == 39

    # 3. Retrieve Formatted Prompt Context
    ctx_res = client.get("/api/twin/patient_api_test_01/context")
    assert ctx_res.status_code == 200
    ctx_data = ctx_res.json()
    assert ctx_data["user_id"] == "patient_api_test_01"
    assert "Sarah Connor" in ctx_data["formatted_context"]
    assert "Mild Asthma" in ctx_data["formatted_context"]
    assert "amoxicillin" in ctx_data["formatted_context"]

    # 4. Update Profile
    update_payload = dict(create_payload)
    update_payload["age"] = 40
    update_payload["allergies"] = ["amoxicillin", "codeine"]
    update_res = client.post("/api/twin", json=update_payload)
    assert update_res.status_code == 200
    updated = update_res.json()
    assert updated["age"] == 40
    assert len(updated["allergies"]) == 2


def test_twin_not_found(client: TestClient):
    """Verify GET /api/twin/{user_id} returns 404 for non-existent patient."""
    response = client.get("/api/twin/non_existent_patient_99999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_twin_context_not_found(client: TestClient):
    """Verify GET /api/twin/{user_id}/context returns 404 for non-existent patient."""
    response = client.get("/api/twin/non_existent_patient_99999/context")
    assert response.status_code == 404


def test_list_twins_endpoint(client: TestClient):
    """Verify GET /api/twin lists registered profiles with limit."""
    # Seed two patients
    client.post("/api/twin", json={"user_id": "p_list_01", "name": "Patient One"})
    client.post("/api/twin", json={"user_id": "p_list_02", "name": "Patient Two"})

    response = client.get("/api/twin?limit=10")
    assert response.status_code == 200
    profiles = response.json()
    assert isinstance(profiles, list)
    user_ids = [p["user_id"] for p in profiles]
    assert "p_list_01" in user_ids
    assert "p_list_02" in user_ids


def test_semantic_search_endpoint(client: TestClient):
    """Verify GET /api/search performs vector similarity retrieval with query validation."""
    response = client.get("/api/search", params={"query": "diabetes symptoms", "top_k": 2})
    assert response.status_code == 200
    results = response.json()
    assert isinstance(results, list)
    if results:
        first = results[0]
        assert "source" in first
        assert "score" in first
        assert "text" in first
        assert 0.0 <= first["score"] <= 1.0


def test_semantic_search_short_query_validation(client: TestClient):
    """Verify GET /api/search validates query length >= 2 chars."""
    response = client.get("/api/search", params={"query": "a"})
    assert response.status_code == 422


def test_chat_endpoint_full_pipeline(client: TestClient):
    """Verify POST /api/chat returns structured clinical answer, sources, and disclaimer."""
    # Seed patient first
    client.post("/api/twin", json={
        "user_id": "chat_test_p01",
        "name": "David Miller",
        "age": 52,
        "chronic_conditions": ["Hypertension"],
        "current_medications": ["Lisinopril 10mg QD"],
    })

    payload = {
        "user_id": "chat_test_p01",
        "session_id": "session_integ_01",
        "message": "What is normal blood pressure according to clinical guidelines?",
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "answer" in data
    assert len(data["answer"]) > 10
    assert "sources" in data
    assert "disclaimer" in data
    assert "educational" in data["disclaimer"].lower() or "disclaimer" in data["disclaimer"].lower()
    assert data["session_id"] == "session_integ_01"
    assert data["is_emergency"] is False


def test_chat_endpoint_emergency_detection(client: TestClient):
    """Verify POST /api/chat escalates emergency symptoms immediately with is_emergency=True."""
    payload = {
        "user_id": "emergency_patient",
        "session_id": "emerg_sess_01",
        "message": "I am having severe crushing chest pain radiating to my left jaw",
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["is_emergency"] is True
    assert "emergency" in data["answer"].lower() or "911" in data["answer"] or "112" in data["answer"]


def test_cache_stats_endpoint(client: TestClient):
    """Verify GET /api/cache/stats returns telemetry."""
    response = client.get("/api/cache/stats")
    assert response.status_code == 200
    stats = response.json()
    assert "hits" in stats
    assert "misses" in stats
    assert "hit_rate" in stats
    assert "total_queries" in stats


def test_transcribe_audio_endpoint_with_synthetic_wav(client: TestClient, sample_synthetic_wav: bytes):
    """Verify POST /api/transcribe successfully decodes and transcribes audio WAV payload."""
    files = {"audio_file": ("patient_audio.wav", sample_synthetic_wav, "audio/wav")}
    response = client.post("/api/transcribe", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "duration_seconds" in data
    assert data["duration_seconds"] > 0
    assert data["status"] in ("success", "fallback")
