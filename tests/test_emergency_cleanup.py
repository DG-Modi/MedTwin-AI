"""Tests for Step 2: Emergency message cleanup, hotline numbers, twin conditions, and bypass logic."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.session import Base
from app.db.models import UserProfile
from app.core.guardrails import check_emergency_keywords, format_emergency_response
from app.core.cot_pipeline import ChainOfThoughtPipeline
from app.llm.base import LLMProvider
from app.rag.pipeline import RAGPipeline


def test_chest_pain_triggers_112_108_and_no_988():
    """Verify chest pain contains 112 and 108, and does NOT contain 988 suicide lifeline."""
    triggered, keyword = check_emergency_keywords("I have severe crushing chest pain")
    assert triggered is True
    assert keyword is not None

    output = format_emergency_response(matched_keyword=keyword, query="I have severe crushing chest pain")
    assert "112" in output
    assert "108" in output
    assert "988" not in output
    assert "ask the emergency dispatcher before taking any aspirin" in output


def test_suicide_query_shows_988():
    """Verify 988 lifeline is shown exclusively when query indicates suicide/self-harm."""
    triggered, keyword = check_emergency_keywords("I feel like I want to kill myself")
    assert triggered is True

    output = format_emergency_response(matched_keyword=keyword, query="I feel like I want to kill myself")
    assert "988" in output
    assert "Suicide & Crisis Lifeline" in output


def test_twin_conditions_appear_in_emergency_output():
    """Verify patient cardiac risk conditions and medications appear in emergency message for responders."""
    twin_profile = {
        "name": "Alex Morgan",
        "chronic_conditions": ["Type 2 Diabetes", "Hypertension"],
        "current_medications": ["Metformin 500mg BID", "Lisinopril 10mg QD"],
    }
    output = format_emergency_response(
        matched_keyword="chest pain",
        twin_profile=twin_profile,
        query="crushing chest pain",
    )
    assert "Tell the responders about your conditions and current medications" in output
    assert "Hypertension" in output
    assert "Type 2 Diabetes" in output
    assert "Lisinopril 10mg QD" in output


@pytest.mark.parametrize(
    "query_text",
    [
        "can't breathe",
        "cannot breathe",
        "face drooping and slurred speech",
        "chest feels tight and I'm sweating",
    ],
)
def test_emergency_variants_trigger(query_text: str):
    """Verify emergency variants ('can't breathe', 'face drooping...', 'chest feels tight...') trigger escalation."""
    triggered, keyword = check_emergency_keywords(query_text)
    assert triggered is True
    assert keyword is not None


def test_blood_pressure_targets_does_not_trigger():
    """Verify non-emergency routine query 'blood pressure targets' does NOT trigger emergency escalation."""
    triggered, keyword = check_emergency_keywords("What are my blood pressure targets?")
    assert triggered is False
    assert keyword is None


@pytest.mark.asyncio
async def test_emergency_bypasses_llm_and_retrieval():
    """Verify emergency queries completely skip RAG retrieval and LLM generation."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    mock_llm = MagicMock(spec=LLMProvider)
    mock_llm.generate = AsyncMock()

    mock_rag = MagicMock(spec=RAGPipeline)
    mock_rag.retrieve = MagicMock()

    pipeline = ChainOfThoughtPipeline(rag_pipeline=mock_rag, llm_client=mock_llm)

    res = await pipeline.execute(
        query="I have severe crushing chest pain radiating to left arm",
        user_id="patient_001",
        session_id="emergency_test_session",
        db=db,
    )

    assert res["is_emergency"] is True
    mock_llm.generate.assert_not_called()
    mock_rag.retrieve.assert_not_called()

    # Verify reasoning step text
    step_summaries = [s.summary for s in res["reasoning_steps"]]
    assert any("Red-flag check = TRIGGERED" in s for s in step_summaries)
    assert any("RAG skipped, LLM skipped" in s for s in step_summaries)

    db.close()

