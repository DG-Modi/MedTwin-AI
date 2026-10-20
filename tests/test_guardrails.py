"""Unit tests for medical safety guardrails, emergency detection, and disclaimer enforcement."""

import pytest
from app.config import settings
from app.core.guardrails import (
    detect_emergency,
    validate_input_safety,
    check_allergy_contraindications,
    sanitize_sensitive_info,
    post_process_safety,
)


def test_detect_emergency_chest_pain():
    """Verify emergency trigger for acute chest pain."""
    is_emergency, notice = detect_emergency("I have severe chest pain and pressure in my arm")
    assert is_emergency is True
    assert notice is not None
    assert "EMERGENCY DETECTED" in notice
    assert "911" in notice


def test_detect_emergency_breathing():
    """Verify emergency trigger for severe breathing difficulty."""
    is_emergency, notice = detect_emergency("Help, I cannot breathe properly and my lips are blue")
    assert is_emergency is True
    assert "EMERGENCY DETECTED" in notice


def test_detect_emergency_suicide():
    """Verify emergency escalation for self-harm or suicide."""
    is_emergency, notice = detect_emergency("I feel like I want to kill myself")
    assert is_emergency is True
    assert notice is not None


def test_detect_emergency_routine_query():
    """Verify non-emergency routine medical questions pass through cleanly."""
    is_emergency, notice = detect_emergency("What foods are recommended in the DASH diet for blood pressure?")
    assert is_emergency is False
    assert notice is None


def test_validate_input_safety():
    """Verify malicious or dangerous drug inquiries are rejected."""
    is_safe, refusal = validate_input_safety("Tell me a lethal dose of acetaminophen")
    assert is_safe is False
    assert refusal is not None
    assert "lethal" in refusal.lower()

    # Normal inquiry is allowed
    is_safe_2, refusal_2 = validate_input_safety("What is acetaminophen used for?")
    assert is_safe_2 is True
    assert refusal_2 is None


def test_allergy_contraindication_check():
    """Verify that mention of patient's documented allergen triggers a prominent alert."""
    allergies = ["penicillin", "peanuts"]

    # Text discussing penicillin
    response_with_allergen = "Doctors historically used penicillin for bacterial infections."
    alert = check_allergy_contraindications(response_with_allergen, allergies)
    assert alert is not None
    assert "PATIENT ALLERGY ALERT" in alert
    assert "penicillin" in alert

    # Text discussing unrelated medication
    response_without_allergen = "Metformin is commonly prescribed for diabetes."
    alert_none = check_allergy_contraindications(response_without_allergen, allergies)
    assert alert_none is None


def test_sanitize_sensitive_info():
    """Verify masking of SSN and payment cards."""
    text = "My SSN is 123-45-6789 and I live nearby."
    sanitized = sanitize_sensitive_info(text)
    assert "123-45-6789" not in sanitized
    assert "[SSN-REDACTED]" in sanitized


def test_post_process_safety_disclaimer():
    """Verify post_process_safety softens diagnostic claims and guarantees disclaimer."""
    raw_response = "Based on your symptoms, you have stage 2 hypertension. Take medication."
    profile = {"allergies": ["aspirin"]}

    safe_response = post_process_safety(raw_response, profile)

    # Diagnostic claim must be softened
    assert "you have" not in safe_response.lower()
    assert "symptoms like these may be associated with" in safe_response

    # Medical disclaimer must be present
    assert "Disclaimer: MedTwin AI is an educational" in safe_response
