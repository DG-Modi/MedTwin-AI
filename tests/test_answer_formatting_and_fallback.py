"""Tests for Answer Formatting: Single Acetaminophen Recommendation and Retrieval-Only LLM Fallback."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.db.session import Base
from app.db.models import UserProfile
from app.db.schemas import ChatRequest
from app.core.guardrails import (
    check_drug_interactions,
    post_process_safety,
)
from app.core.cot_pipeline import ChainOfThoughtPipeline
from app.services.chat_service import ChatService


@pytest.mark.asyncio
async def test_mock_failing_llm_response_contains_alerts_sources_retrieval_only_and_single_acetaminophen():
    """Verify that when LLM fails:

    1. The honest error message is retained.
    2. Allergy check line is displayed.
    3. Interaction alerts are shown.
    4. Top retrieved guideline excerpts are displayed under 'Model unavailable: showing guideline excerpts only' with source file, URL, and score.
    5. The response is marked as 'retrieval-only' in the Reasoning steps panel.
    6. The acetaminophen sentence appears EXACTLY once in a single 'What you can do' line.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    alex = UserProfile(
        user_id="alex_fallback_01",
        name="Alex Morgan",
        age=45,
        allergies=["penicillin", "peanuts"],
        chronic_conditions=["Type 2 Diabetes", "Hypertension"],
        current_medications=["Metformin 500mg BID", "Lisinopril 10mg QD"],
        lifestyle_notes="Moderate exercise",
    )
    db.add(alex)
    db.commit()

    # Mock failing LLM
    mock_llm = MagicMock()
    mock_llm.model = "llama3.2"
    mock_llm.generate = AsyncMock(side_effect=RuntimeError("Local Ollama service timed out or offline."))

    # Mock RAG pipeline returning cited chunks
    mock_rag = MagicMock()
    mock_rag.retrieve.return_value = [
        {
            "id": "nsaid_0",
            "source": "nsaid_ibuprofen_safety.md",
            "title": "Clinical Safety: NSAID (Ibuprofen) Precautions & Drug Interactions",
            "score": 0.88,
            "url": "https://www.ncbi.nlm.nih.gov/books/NBK537074/",
            "text": "Prostaglandins maintain renal blood flow. NSAIDs induce renal ischemia and elevate systemic blood pressure.",
        },
        {
            "id": "htn_0",
            "source": "hypertension.md",
            "title": "Hypertension Clinical Guidelines",
            "score": 0.81,
            "url": "https://www.ahajournals.org/doi/10.1161/HYP.0000000000000065",
            "text": "Blood pressure targets for non-pregnant adults with hypertension should be < 130/80 mmHg.",
        },
    ]
    mock_rag.format_rag_context.return_value = "Formatted RAG Context Excerpt"

    pipeline = ChainOfThoughtPipeline(
        rag_pipeline=mock_rag,
        llm_client=mock_llm,
        enable_grounded_fallback=False,  # Force offline fallback path
    )

    result = await pipeline.execute(
        query="Can I take Ibuprofen for my headache?",
        user_id="alex_fallback_01",
        session_id="sess_fallback_01",
        db=db,
    )

    answer = result["answer"]
    steps = result["reasoning_steps"]

    # 1. Honest error message is preserved
    assert "The local model is unavailable" in answer

    # 2. Allergy check line is displayed
    assert "**Allergy Check**: No conflict found with penicillin, peanuts." in answer

    # 3. Interaction alerts are shown
    assert "### ⚠️ Interaction Alerts:" in answer
    assert "Lisinopril" in answer
    assert "Metformin" in answer

    # 4. Heading 'Model unavailable: showing guideline excerpts only'
    assert "### Model unavailable: showing guideline excerpts only" in answer

    # 5. Top retrieved guideline excerpts contain source file, URL, and score
    assert "nsaid_ibuprofen_safety.md" in answer
    assert "https://www.ncbi.nlm.nih.gov/books/NBK537074/" in answer
    assert "Score: 0.88" in answer
    assert len(result["sources"]) >= 1

    # 6. Response marked as 'retrieval-only' in Reasoning steps panel
    step_6 = next((s for s in steps if s.stage == "llm_generation"), None)
    assert step_6 is not None
    assert step_6.status == "retrieval-only"
    assert "Retrieval-Only" in step_6.name
    assert "retrieval-only" in step_6.summary.lower()
    assert step_6.details.get("retrieval_only") is True

    # 7. Acetaminophen suggestion appears EXACTLY once, in single 'What you can do' line
    aceta_sentence = "ask your doctor or pharmacist whether acetaminophen is a safer option for you"
    assert answer.lower().count(aceta_sentence) == 1
    assert "**What you can do**: Ask your doctor or pharmacist whether acetaminophen is a safer option for you." in answer
    assert "Safer Alternatives" not in answer

    db.close()


def test_hypertension_warning_has_no_duplicate_acetaminophen():
    """Verify HYPERTENSION WARNING rule text does not contain acetaminophen,

    and post_process_safety shows the acetaminophen suggestion once in 'What you can do'.
    """
    warnings = check_drug_interactions(
        query_text="Can I take Ibuprofen?",
        medications=[],
        conditions=["Hypertension"],
        allergies=[],
    )

    assert len(warnings) == 1
    htn_warn = warnings[0]
    assert htn_warn["rule_id"] == "CONTRAINDICATION_NSAID_HYPERTENSION"
    # Ensure rule text does NOT contain acetaminophen
    assert "acetaminophen" not in htn_warn["clinical_warning"].lower()
    assert "tylenol" not in htn_warn["clinical_warning"].lower()

    # Pass through post_process_safety
    processed = post_process_safety(
        response_text="General headache advice.",
        user_profile={"chronic_conditions": ["Hypertension"], "current_medications": []},
        query_text="Can I take Ibuprofen?",
    )

    # Must contain the single 'What you can do' line
    assert "**What you can do**: Ask your doctor or pharmacist whether acetaminophen is a safer option for you." in processed
    # Acetaminophen sentence appears exactly once
    aceta_sentence = "ask your doctor or pharmacist whether acetaminophen is a safer option for you"
    assert processed.lower().count(aceta_sentence) == 1
    assert "Safer Alternatives" not in processed


@pytest.mark.asyncio
async def test_chat_service_failing_llm_produces_retrieval_only_and_single_acetaminophen(test_db_session):
    """Verify ChatService with a mocked failing LLM returns retrieval-only steps and single acetaminophen line."""
    alex = UserProfile(
        user_id="user_cs_fail",
        name="Alex Morgan",
        age=45,
        allergies=["penicillin"],
        chronic_conditions=["Hypertension"],
        current_medications=["Lisinopril 10mg"],
    )
    test_db_session.add(alex)
    test_db_session.commit()

    mock_llm = MagicMock()
    mock_llm.model = "llama3.2"
    mock_llm.generate = AsyncMock(side_effect=RuntimeError("Ollama daemon down"))

    mock_rag = MagicMock()
    mock_rag.retrieve.return_value = [
        {
            "id": "nsaid_1",
            "source": "nsaid_ibuprofen_safety.md",
            "title": "NSAID Safety",
            "score": 0.85,
            "url": "https://www.ncbi.nlm.nih.gov/books/NBK537074/",
            "text": "NSAIDs can elevate blood pressure and reduce ACE inhibitor efficacy.",
        }
    ]
    mock_rag.format_rag_context.return_value = "NSAID RAG Context"

    service = ChatService(llm_client=mock_llm, rag_pipeline=mock_rag)
    service.cot.enable_grounded_fallback = False

    req = ChatRequest(
        user_id="user_cs_fail",
        session_id="sess_cs_fail",
        message="Can I take Ibuprofen?",
    )

    res = await service.process_chat(test_db_session, req)

    assert res.cached is False
    assert "The local model is unavailable" in res.answer
    assert "**Allergy Check**: No conflict found with penicillin." in res.answer
    assert "### ⚠️ Interaction Alerts:" in res.answer
    assert "Lisinopril" in res.answer
    assert "### Model unavailable: showing guideline excerpts only" in res.answer
    assert "nsaid_ibuprofen_safety.md" in res.answer
    assert "https://www.ncbi.nlm.nih.gov/books/NBK537074/" in res.answer

    # Step marked retrieval-only
    step_retrieval = next((s for s in res.reasoning_steps if s.stage == "llm_generation"), None)
    assert step_retrieval is not None
    assert step_retrieval.status == "retrieval-only"

    # Acetaminophen sentence appears exactly once
    aceta_sentence = "ask your doctor or pharmacist whether acetaminophen is a safer option for you"
    assert res.answer.lower().count(aceta_sentence) == 1
    assert "**What you can do**: Ask your doctor or pharmacist whether acetaminophen is a safer option for you." in res.answer
