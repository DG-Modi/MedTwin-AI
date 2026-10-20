"""Comprehensive unit and integration tests for chat orchestration, caching, safety, and API pipeline."""

import time
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.config import settings
from app.db.session import Base
from app.db.models import UserProfile, Conversation, ChatMessage
from app.db.schemas import ChatRequest, UserProfileCreate
from app.core.cache import QueryCache, generate_cache_key
from app.core.digital_twin import create_or_update_profile
from app.rag.pipeline import RAGPipeline
from app.rag.vector_store import ChromaVectorStore
from app.llm.ollama_client import OllamaClient
from app.services.chat_service import ChatService
from app.api.main import app


@pytest.fixture
def test_db():
    """Provides a clean in-memory SQLite database session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def mock_rag():
    """Mock RAG pipeline returning attributed clinical chunks."""
    rag = MagicMock(spec=RAGPipeline)
    rag.retrieve.return_value = [
        {
            "id": "diabetes_chunk_1",
            "source": "type_2_diabetes.md",
            "title": "Clinical Guide: Type 2 Diabetes",
            "score": 0.89,
            "text": "Target HbA1c is below 7.0% for most non-pregnant adults.",
        }
    ]
    rag.format_rag_context.return_value = (
        "### RETRIEVED MEDICAL CONTEXT:\n"
        "[1] SOURCE: type_2_diabetes.md | CONFIDENCE: 89%\n"
        "Target HbA1c is below 7.0% for most non-pregnant adults."
    )
    return rag


@pytest.fixture
def mock_llm():
    """Mock Ollama client generating clinical text."""
    llm = MagicMock(spec=OllamaClient)
    llm.generate = AsyncMock(
        return_value=(
            "According to [type_2_diabetes.md], the target HbA1c is generally below 7.0%. "
            "Regular aerobic exercise and a balanced diet are recommended."
        )
    )
    return llm


@pytest.fixture
def test_service(mock_rag, mock_llm):
    """Instantiates ChatService with clean cache and mocked external clients."""
    fresh_cache = QueryCache(default_ttl=3600)
    return ChatService(cache=fresh_cache, rag_pipeline=mock_rag, llm_client=mock_llm)


# 1. Normal Chat Request
@pytest.mark.asyncio
async def test_normal_chat_request(test_service, test_db):
    req = ChatRequest(user_id="user_1", session_id="sess_1", message="What is target HbA1c?")
    res = await test_service.process_chat(test_db, req)
    assert res.session_id == "sess_1"
    assert "type_2_diabetes.md" in res.answer
    assert len(res.sources) == 1
    assert res.sources[0].source == "type_2_diabetes.md"
    assert res.cached is False
    assert res.is_emergency is False
    assert settings.MEDICAL_DISCLAIMER in res.disclaimer


# 2. Empty Message Validation
@pytest.mark.asyncio
async def test_empty_message(test_service, test_db):
    with pytest.raises(HTTPException) as exc_info:
        req = ChatRequest(user_id="user_1", session_id="sess_1", message="   ")
        await test_service.process_chat(test_db, req)
    assert exc_info.value.status_code == 400


# 3. Invalid User Validation
@pytest.mark.asyncio
async def test_invalid_user(test_service, test_db):
    with pytest.raises(HTTPException) as exc_info:
        req = ChatRequest(user_id="  ", session_id="sess_1", message="Hello")
        await test_service.process_chat(test_db, req)
    assert exc_info.value.status_code == 400


# 4. Emergency Question
@pytest.mark.asyncio
async def test_emergency_question(test_service, test_db):
    req = ChatRequest(user_id="user_1", session_id="sess_emerg", message="I have severe chest pain and cannot breathe")
    res = await test_service.process_chat(test_db, req)
    assert res.is_emergency is True
    assert "EMERGENCY DETECTED" in res.answer
    assert "911" in res.answer
    assert res.sources == []
    assert res.cached is False
    # Verify not calling LLM or RAG
    test_service.rag.retrieve.assert_not_called()
    test_service.llm.generate.assert_not_called()


# 5 & 6. Cache Miss & Cache Hit
@pytest.mark.asyncio
async def test_cache_miss_then_hit(test_service, test_db):
    req = ChatRequest(user_id="user_cache", session_id="sess_cache", message="Explain HbA1c guidelines")

    # First request: Cache Miss
    res1 = await test_service.process_chat(test_db, req)
    assert res1.cached is False
    assert test_service.cache.misses >= 1

    # Second identical request: Cache Hit
    res2 = await test_service.process_chat(test_db, req)
    assert res2.cached is True
    assert res2.answer == res1.answer
    assert test_service.cache.hits >= 1


# 7. Cache Isolation Between Users
@pytest.mark.asyncio
async def test_cache_isolation_between_users(test_service, test_db):
    # User A with Diabetes
    pA = UserProfileCreate(user_id="user_A", chronic_conditions=["Type 2 Diabetes"])
    create_or_update_profile(test_db, pA)

    # User B with Asthma
    pB = UserProfileCreate(user_id="user_B", chronic_conditions=["Asthma"])
    create_or_update_profile(test_db, pB)

    reqA = ChatRequest(user_id="user_A", session_id="sess_A", message="What exercises are safe?")
    reqB = ChatRequest(user_id="user_B", session_id="sess_B", message="What exercises are safe?")

    resA = await test_service.process_chat(test_db, reqA)
    assert resA.cached is False

    # User B should MISS cache because of differing user health profile
    resB = await test_service.process_chat(test_db, reqB)
    assert resB.cached is False


# 8. Digital Twin Retrieval
@pytest.mark.asyncio
async def test_digital_twin_retrieval(test_service, test_db):
    p = UserProfileCreate(
        user_id="patient_twin",
        name="Sarah",
        age=52,
        allergies=["sulfa"],
        chronic_conditions=["Hypertension"],
    )
    create_or_update_profile(test_db, p)

    req = ChatRequest(user_id="patient_twin", session_id="sess_twin", message="Diet for high blood pressure")
    await test_service.process_chat(test_db, req)

    # Verify LLM was called with prompt containing twin details
    call_args = test_service.llm.generate.call_args[1]
    prompt_used = call_args["prompt"]
    assert "Sarah" in prompt_used
    assert "Hypertension" in prompt_used


# 9 & 10. RAG Retrieval & Source Attribution
@pytest.mark.asyncio
async def test_rag_retrieval_and_source_attribution(test_service, test_db):
    req = ChatRequest(user_id="user_rag", session_id="sess_rag", message="What is normal blood pressure?")
    res = await test_service.process_chat(test_db, req)

    test_service.rag.retrieve.assert_called_once_with(query="What is normal blood pressure?", top_k=settings.RAG_TOP_K)
    assert len(res.sources) > 0
    assert res.sources[0].score == 0.89
    assert res.sources[0].source == "type_2_diabetes.md"


# 11. Ollama Success Using Mock
@pytest.mark.asyncio
async def test_ollama_success_using_mock(test_service, test_db):
    test_service.llm.generate.return_value = "Educated answer citing [type_2_diabetes.md]."
    req = ChatRequest(user_id="user_mock", session_id="sess_mock", message="Diabetes query")
    res = await test_service.process_chat(test_db, req)
    assert "Educated answer" in res.answer


# 12. Ollama Failure Graceful Handling
@pytest.mark.asyncio
async def test_ollama_failure_graceful_handling(test_service, test_db):
    test_service.llm.generate.side_effect = Exception("Ollama connection timed out")
    req = ChatRequest(user_id="user_fail", session_id="sess_fail", message="Any query")
    res = await test_service.process_chat(test_db, req)
    assert res.cached is False
    assert "technical difficulties" in res.answer


# 13. Post-generation Safety Check (non-diagnostic)
@pytest.mark.asyncio
async def test_post_generation_safety_non_diagnostic(test_service, test_db):
    test_service.llm.generate.return_value = "You have diabetes and need immediate drugs."
    req = ChatRequest(user_id="user_diag", session_id="sess_diag", message="Do I have diabetes?")
    res = await test_service.process_chat(test_db, req)
    # The guardrail rewrites "you have"
    assert "you have" not in res.answer.lower()
    assert "symptoms like these may be associated with" in res.answer


# 14. Allergy Conflict Detection
@pytest.mark.asyncio
async def test_allergy_conflict_detection(test_service, test_db):
    p = UserProfileCreate(user_id="patient_allergy", allergies=["penicillin"])
    create_or_update_profile(test_db, p)

    test_service.llm.generate.return_value = "Doctors may discuss penicillin for bacterial infections."
    req = ChatRequest(user_id="patient_allergy", session_id="sess_allergy", message="Infection questions")
    res = await test_service.process_chat(test_db, req)

    assert "PATIENT ALLERGY ALERT" in res.answer
    assert "penicillin" in res.answer


# 15. Conversation Persistence
@pytest.mark.asyncio
async def test_conversation_persistence(test_service, test_db):
    req = ChatRequest(user_id="user_persist", session_id="sess_persist", message="Test persistence")
    await test_service.process_chat(test_db, req)

    conv = test_db.query(Conversation).filter(Conversation.session_id == "sess_persist").first()
    assert conv is not None
    assert conv.user_id == "user_persist"

    messages = test_db.query(ChatMessage).filter(ChatMessage.conversation_id == conv.id).all()
    assert len(messages) == 2  # 1 user + 1 assistant
    assert messages[0].sender == "user"
    assert messages[1].sender == "assistant"


# 16. TTL Expiration
def test_ttl_expiration():
    short_cache = QueryCache(default_ttl=1)
    short_cache.set("temp_key", {"data": "test"}, ttl_seconds=1)
    assert short_cache.get("temp_key") is not None

    time.sleep(1.05)
    # Expired
    assert short_cache.get("temp_key") is None


# 17. End-to-End Orchestration via FastAPI TestClient
def test_end_to_end_api_route():
    client = TestClient(app)
    payload = {
        "user_id": "test_api_patient",
        "session_id": "api_session_999",
        "message": "What is normal blood pressure according to guidelines?",
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "api_session_999"
    assert "answer" in data
    assert "disclaimer" in data
    assert data["cached"] is False

    # Second call should be cached!
    response2 = client.post("/api/chat", json=payload)
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["cached"] is True
