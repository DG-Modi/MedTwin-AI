"""Comprehensive tests for Digital Twin Cache Invalidation and Deterministic Allergy Safety.

Validates the 6 critical safety requirements:
1. same query + same twin = cache hit
2. same query + changed allergies = cache miss and the answer contains "ALLERGY CONFLICT"
3. Save Twin invalidates the cache
4. the allergy check matches drug classes (penicillin -> amoxicillin)
5. an unrelated allergy produces no conflict
6. the cache key differs between two different patients
Bonus: Never cache emergency responses, never serve cached answer when twin hash differs.
"""

import pytest
from fastapi.testclient import TestClient
from app.core.cache import generate_cache_key, generate_twin_hash, get_cache
from app.core.guardrails import check_deterministic_allergy_conflict


def test_cache_key_differs_between_two_different_patients():
    """Requirement 6: Verify cache key differs between two different patients for the same query."""
    query = "Can I take Ibuprofen with my current medications?"
    profile_a = {
        "user_id": "patient_alpha",
        "age": 40,
        "gender": "Male",
        "allergies": ["peanuts"],
        "chronic_conditions": ["Hypertension"],
        "current_medications": ["Lisinopril"],
        "twin_version": 1,
    }
    profile_b = {
        "user_id": "patient_beta",
        "age": 40,
        "gender": "Male",
        "allergies": ["peanuts"],
        "chronic_conditions": ["Hypertension"],
        "current_medications": ["Lisinopril"],
        "twin_version": 1,
    }

    key_a = generate_cache_key(user_id="patient_alpha", query=query, user_profile=profile_a)
    key_b = generate_cache_key(user_id="patient_beta", query=query, user_profile=profile_b)

    assert key_a != key_b, "Cache keys must differ between two distinct patients to ensure clinical isolation"
    assert len(key_a) == 64
    assert len(key_b) == 64


def test_allergy_check_drug_classes_deterministic():
    """Requirement 4: Verify allergy check matches drug classes (e.g., penicillin -> amoxicillin)."""
    # Penicillin allergy should match amoxicillin query
    res = check_deterministic_allergy_conflict(
        query="Can I take amoxicillin for my throat infection?",
        allergies=["penicillin"],
    )
    assert res is not None
    assert res["conflict"] is True
    assert "ALLERGY CONFLICT" in res["alert"]
    assert "penicillin" in res["alert"]

    # NSAID allergy should match ibuprofen, advil, naproxen, aspirin queries
    res_nsaid = check_deterministic_allergy_conflict(
        query="Can I take Advil or Motrin for a headache?",
        allergies=["NSAIDs"],
    )
    assert res_nsaid is not None
    assert res_nsaid["conflict"] is True
    assert "NSAIDs" in res_nsaid["alert"]

    # Ibuprofen allergy should match ibuprofen query
    res_ibu = check_deterministic_allergy_conflict(
        query="Can I take Ibuprofen with my current medications?",
        allergies=["ibuprofen"],
    )
    assert res_ibu is not None
    assert res_ibu["conflict"] is True
    assert "ibuprofen" in res_ibu["alert"]
    assert "Do not take it, and speak to your doctor or pharmacist." in res_ibu["alert"]


def test_unrelated_allergy_produces_no_conflict():
    """Requirement 5: Verify an unrelated allergy produces no conflict."""
    # Patient has penicillin allergy, asking about ibuprofen
    res = check_deterministic_allergy_conflict(
        query="Can I take Ibuprofen with my current medications?",
        allergies=["penicillin", "peanuts"],
    )
    assert res is None, "Penicillin and peanuts allergies must not trigger conflict for ibuprofen query"

    # Patient has sulfa allergy, asking about metformin
    res_sulfa = check_deterministic_allergy_conflict(
        query="What is the recommended dose of metformin?",
        allergies=["sulfa drugs"],
    )
    assert res_sulfa is None


def test_same_query_same_twin_is_cache_hit(client: TestClient):
    """Requirement 1: same query + same twin = cache hit."""
    patient_id = "patient_same_twin_test"
    query = "What is the recommended sodium intake in the DASH diet?"

    # Initial profile setup
    save_res = client.post("/api/twin", json={
        "user_id": patient_id,
        "name": "Same Twin User",
        "age": 45,
        "gender": "Male",
        "allergies": ["dust"],
        "chronic_conditions": ["Hypertension"],
        "current_medications": ["Lisinopril 10mg"],
        "lifestyle_notes": "DASH diet adherent",
    })
    assert save_res.status_code == 200

    # Query 1: Fresh live generation
    res1 = client.post("/api/chat", json={
        "user_id": patient_id,
        "session_id": "session_same_twin_1",
        "message": query,
    })
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["cached"] is False

    # Query 2: Identical twin + identical query -> Cache Hit
    res2 = client.post("/api/chat", json={
        "user_id": patient_id,
        "session_id": "session_same_twin_1",
        "message": query,
    })
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["cached"] is True
    assert data2["answer"] == data1["answer"]


def test_same_query_changed_allergies_cache_miss_and_allergy_conflict(client: TestClient):
    """Requirement 2: same query + changed allergies = cache miss and the answer contains ALLERGY CONFLICT."""
    patient_id = "patient_allergy_bug_test"
    query = "Can I take Ibuprofen with my current medications?"

    # 1. Start with no ibuprofen allergy
    save1 = client.post("/api/twin", json={
        "user_id": patient_id,
        "name": "Jane Doe",
        "age": 38,
        "gender": "Female",
        "allergies": ["penicillin"],
        "chronic_conditions": ["Hypertension"],
        "current_medications": ["Lisinopril 10mg QD"],
        "lifestyle_notes": "Non-smoker",
    })
    assert save1.status_code == 200

    # First query before changing allergies
    res1 = client.post("/api/chat", json={
        "user_id": patient_id,
        "session_id": "session_allergy_1",
        "message": query,
    })
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["cached"] is False
    assert "ALLERGY CONFLICT" not in data1["answer"]

    # 2. Change Digital Twin allergies to "ibuprofen" and click Save Twin
    save2 = client.post("/api/twin", json={
        "user_id": patient_id,
        "name": "Jane Doe",
        "age": 38,
        "gender": "Female",
        "allergies": ["ibuprofen"],
        "chronic_conditions": ["Hypertension"],
        "current_medications": ["Lisinopril 10mg QD"],
        "lifestyle_notes": "Non-smoker",
    })
    assert save2.status_code == 200
    updated_profile = save2.json()
    assert updated_profile["twin_version"] > 1

    # 3. Ask same query: "Can I take Ibuprofen with my current medications?"
    res2 = client.post("/api/chat", json={
        "user_id": patient_id,
        "session_id": "session_allergy_1",
        "message": query,
    })
    assert res2.status_code == 200
    data2 = res2.json()

    # Must be a CACHE MISS (fresh live evaluation)
    assert data2["cached"] is False, "A changed twin MUST produce a cache miss"

    # Must contain "ALLERGY CONFLICT" at the TOP of the answer
    assert "ALLERGY CONFLICT" in data2["answer"]
    assert "Your profile lists an allergy to ibuprofen. Do not take it, and speak to your doctor or pharmacist." in data2["answer"]
    # Verify alert is at top of answer
    assert data2["answer"].strip().startswith("🚨 **ALLERGY CONFLICT**")

    # Verify Reasoning step contains "Allergy check: CONFLICT"
    reasoning_step_names = [s["name"] for s in data2.get("reasoning_steps", [])]
    assert any("Allergy check: CONFLICT" in name for name in reasoning_step_names)


def test_save_twin_invalidates_cache(client: TestClient):
    """Requirement 3: Save Twin clears or version-bumps cache entries so subsequent queries produce cache miss."""
    patient_id = "patient_invalidation_test"
    query = "What are the common symptoms of high blood pressure?"

    client.post("/api/twin", json={
        "user_id": patient_id,
        "name": "Invalidation Tester",
        "age": 50,
        "gender": "Male",
        "allergies": [],
        "chronic_conditions": ["Hypertension"],
        "current_medications": [],
    })

    # Query 1: Uncached
    r1 = client.post("/api/chat", json={
        "user_id": patient_id,
        "session_id": "sess_inv",
        "message": query,
    })
    assert r1.status_code == 200
    assert r1.json()["cached"] is False

    # Query 2: Cached
    r2 = client.post("/api/chat", json={
        "user_id": patient_id,
        "session_id": "sess_inv",
        "message": query,
    })
    assert r2.status_code == 200
    assert r2.json()["cached"] is True

    # Save Twin -> triggers invalidate_patient() and version bump
    save_res = client.post("/api/twin", json={
        "user_id": patient_id,
        "name": "Invalidation Tester",
        "age": 51,  # updated age
        "gender": "Male",
        "allergies": [],
        "chronic_conditions": ["Hypertension"],
        "current_medications": [],
    })
    assert save_res.status_code == 200

    # Query 3: Must now be cache miss because cache was invalidated
    r3 = client.post("/api/chat", json={
        "user_id": patient_id,
        "session_id": "sess_inv",
        "message": query,
    })
    assert r3.status_code == 200
    assert r3.json()["cached"] is False, "Save Twin must invalidate the cache for that patient"


def test_never_cache_emergency_responses(client: TestClient):
    """Requirement 4: Never cache emergency responses."""
    patient_id = "patient_emergency_test"
    emergency_query = "I am having severe crushing chest pain radiating down my left arm"

    r1 = client.post("/api/chat", json={
        "user_id": patient_id,
        "session_id": "sess_emerg",
        "message": emergency_query,
    })
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["is_emergency"] is True
    assert d1["cached"] is False

    # Query again with same emergency prompt
    r2 = client.post("/api/chat", json={
        "user_id": patient_id,
        "session_id": "sess_emerg",
        "message": emergency_query,
    })
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["is_emergency"] is True
    assert d2["cached"] is False, "Emergency responses must NEVER be cached"
