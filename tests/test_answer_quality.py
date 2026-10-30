"""Tests for Step 3: Answer Quality, Formatting, Clinical Rules, and Guardrails."""

import re
import pytest
from unittest.mock import MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.db.session import Base
from app.db.models import UserProfile
from app.core.guardrails import (
    fix_unbalanced_markdown,
    post_process_safety,
    check_drug_interactions,
    check_deterministic_allergy_conflict,
)
from app.rag.synthesizer import synthesize_grounded_response
from app.core.cot_pipeline import ChainOfThoughtPipeline
from app.llm.ollama_client import OllamaClient


def test_fix_unbalanced_markdown():
    """Verify orphaned closing bold tags are fixed and outputs have balanced ** tags."""
    # Orphaned closing bold from stripped prefix
    broken_acei = "- Hypertension & ACE Inhibitors (Lisinopril)**: Ibuprofen directly antagonizes Lisinopril."
    fixed_acei = fix_unbalanced_markdown(broken_acei)
    assert fixed_acei.count("**") % 2 == 0
    assert "- **Hypertension & ACE Inhibitors (Lisinopril)**:" in fixed_acei

    broken_metformin = "- Diabetes & Metformin**: Renal impairment induced by Ibuprofen leads to accumulation."
    fixed_metformin = fix_unbalanced_markdown(broken_metformin)
    assert fixed_metformin.count("**") % 2 == 0
    assert "- **Diabetes & Metformin**:" in fixed_metformin

    # Already balanced string should remain balanced
    balanced = "- **Target Blood Pressure**: **< 130/80 mmHg**."
    assert fix_unbalanced_markdown(balanced) == balanced


@pytest.mark.asyncio
async def test_ibuprofen_query_with_alex_twin_triggers_acei_and_metformin_rules():
    """Verify ibuprofen query with Alex twin triggers ACE inhibitor and Metformin rules,

    leads with a one-line verdict, softens phrasing, and includes allergy check.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    alex = UserProfile(
        user_id="alex_morgan_01",
        name="Alex Morgan",
        age=45,
        allergies=["penicillin", "peanuts"],
        chronic_conditions=["Type 2 Diabetes", "Hypertension"],
        current_medications=["Metformin 500mg BID", "Lisinopril 10mg QD"],
        lifestyle_notes="Moderate daily walking, DASH diet",
    )
    db.add(alex)
    db.commit()

    offline_llm = OllamaClient(base_url="http://127.0.0.1:99999", timeout_seconds=1)
    pipeline = ChainOfThoughtPipeline(llm_client=offline_llm, enable_grounded_fallback=True)

    res = await pipeline.execute(
        query="Can I take Ibuprofen for a headache?",
        user_id="alex_morgan_01",
        session_id="alex_session_01",
        db=db,
    )

    answer = res["answer"]

    # 1. Balanced markdown
    assert answer.count("**") % 2 == 0

    # 2. Both rules triggered
    assert "Lisinopril" in answer
    assert "Metformin" in answer
    assert "kidney" in answer.lower() or "renal" in answer.lower()

    # 3. Phrasing: uses "can increase the risk" and avoids "dramatically" or "life-threatening"
    assert "can increase" in answer.lower() or "can reduce" in answer.lower()
    assert "dramatically" not in answer.lower()
    assert "life-threatening" not in answer.lower()

    # 4. Phrased alternatives as "ask your doctor or pharmacist whether ... is a safer option for you"
    assert "ask your doctor or pharmacist whether" in answer.lower()
    assert "safer option for you" in answer.lower()
    assert answer.lower().count("ask your doctor or pharmacist whether acetaminophen is a safer option for you") == 1
    assert "**What you can do**: Ask your doctor or pharmacist whether acetaminophen is a safer option for you." in answer
    assert "Safer Alternatives" not in answer

    # 5. One-line verdict at the top
    assert answer.startswith("**Verdict**:")

    # 6. Allergy check line present
    assert "Allergy Check" in answer
    assert "penicillin" in answer.lower()
    assert "peanuts" in answer.lower()

    # 7. Single medical disclaimer at end
    assert answer.count(settings.MEDICAL_DISCLAIMER) == 1
    assert not answer.startswith("Disclaimer: Disclaimer:")

    db.close()


@pytest.mark.asyncio
async def test_twin_without_meds_does_not_trigger_interaction_rules():
    """Verify twin without Lisinopril or Metformin does NOT trigger ACEI or Metformin rules."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    healthy_twin = UserProfile(
        user_id="healthy_user_01",
        name="Jordan Lee",
        age=28,
        allergies=[],
        chronic_conditions=[],
        current_medications=[],
        lifestyle_notes="Running 3 times a week",
    )
    db.add(healthy_twin)
    db.commit()

    offline_llm = OllamaClient(base_url="http://127.0.0.1:99999", timeout_seconds=1)
    pipeline = ChainOfThoughtPipeline(llm_client=offline_llm, enable_grounded_fallback=True)

    res = await pipeline.execute(
        query="Can I take Ibuprofen for a headache?",
        user_id="healthy_user_01",
        session_id="healthy_session_01",
        db=db,
    )

    answer = res["answer"]
    assert "Lisinopril" not in answer
    assert "Metformin" not in answer
    assert answer.count("**") % 2 == 0

    db.close()


@pytest.mark.asyncio
async def test_ibuprofen_allergy_produces_allergy_conflict():
    """Verify twin with ibuprofen allergy produces ALLERGY CONFLICT at the top of the answer."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    allergic_twin = UserProfile(
        user_id="allergic_user_01",
        name="Casey allergic",
        age=30,
        allergies=["ibuprofen"],
        chronic_conditions=[],
        current_medications=[],
    )
    db.add(allergic_twin)
    db.commit()

    offline_llm = OllamaClient(base_url="http://127.0.0.1:99999", timeout_seconds=1)
    pipeline = ChainOfThoughtPipeline(llm_client=offline_llm, enable_grounded_fallback=True)

    res = await pipeline.execute(
        query="Can I take Ibuprofen for a headache?",
        user_id="allergic_user_01",
        session_id="allergic_session_01",
        db=db,
    )

    answer = res["answer"]
    assert "ALLERGY CONFLICT" in answer
    assert answer.startswith("🚨 **ALLERGY CONFLICT**")
    assert "ibuprofen" in answer.lower()
    assert answer.count(settings.MEDICAL_DISCLAIMER) == 1

    db.close()


def test_disclaimer_appears_exactly_once():
    """Verify medical disclaimer appears exactly once on any response."""
    raw = "Here is some clinical guidance.\n\n---\n*Disclaimer: MedTwin AI is an educational medical-information assistant and NOT a diagnostic tool or healthcare provider. This tool does not provide medical diagnoses, treatment plans, or prescriptions. Always consult a qualified physician or healthcare provider for any medical concerns or emergency situations.*"
    processed = post_process_safety(raw)
    assert processed.count(settings.MEDICAL_DISCLAIMER) == 1

