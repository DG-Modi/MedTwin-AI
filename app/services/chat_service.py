"""Chat orchestration service connecting guardrails, digital twin, caching, RAG, and Ollama."""

import logging
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from app.config import settings
from app.db.models import Conversation, ChatMessage
from app.db.schemas import ChatRequest, ChatResponse, RetrievedSource
from app.core.cache import QueryCache, get_cache, generate_cache_key
from app.core.guardrails import (
    detect_emergency,
    validate_input_safety,
    post_process_safety,
)
from app.core.digital_twin import get_profile, format_twin_context
from app.rag.pipeline import RAGPipeline, get_rag_pipeline
from app.llm.ollama_client import OllamaClient, get_llm_client
from app.llm.prompts import build_medical_prompt

logger = logging.getLogger(__name__)


class ChatService:
    """Orchestrates end-to-end medical chat processing with safety boundaries and query caching."""

    def __init__(
        self,
        cache: Optional[QueryCache] = None,
        rag_pipeline: Optional[RAGPipeline] = None,
        llm_client: Optional[OllamaClient] = None,
    ) -> None:
        self.cache = cache or get_cache()
        self.rag = rag_pipeline or get_rag_pipeline()
        self.llm = llm_client or get_llm_client()

    async def process_chat(self, db: Session, request: ChatRequest) -> ChatResponse:
        """Processes a chat request through the full safety-bounded RAG pipeline."""
        # 1. Input Validation
        user_id = str(request.user_id).strip()
        session_id = request.session_id.strip()
        raw_message = request.message.strip()

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User ID cannot be empty or invalid.",
            )
        if not session_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Session ID cannot be empty.",
            )
        if not raw_message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message cannot be empty or whitespace.",
            )
        if len(raw_message) > 2000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message exceeds maximum allowed length of 2000 characters.",
            )

        # 2. Pre-execution Safety: Emergency Check
        is_emergency, emergency_notice = detect_emergency(raw_message)
        if is_emergency:
            # Record emergency interaction in database
            self._save_interaction(
                db=db,
                session_id=session_id,
                user_id=user_id,
                user_message=raw_message,
                assistant_response=emergency_notice or settings.EMERGENCY_NOTICE,
                sources=[],
                is_emergency=True,
                cached=False,
            )
            return ChatResponse(
                session_id=session_id,
                answer=emergency_notice or settings.EMERGENCY_NOTICE,
                sources=[],
                disclaimer=settings.MEDICAL_DISCLAIMER,
                cached=False,
                is_emergency=True,
            )

        # 3. Pre-execution Safety: Harmful / Illicit Request Check
        is_safe, refusal_msg = validate_input_safety(raw_message)
        if not is_safe:
            return ChatResponse(
                session_id=session_id,
                answer=refusal_msg or "I cannot assist with this request.",
                sources=[],
                disclaimer=settings.MEDICAL_DISCLAIMER,
                cached=False,
                is_emergency=False,
            )

        # 4. Load Digital Twin Profile from SQLite
        profile = get_profile(db, user_id)
        profile_dict = profile.to_dict() if profile else None

        # 5. Build Cache Key
        cache_key = generate_cache_key(
            query=raw_message,
            user_id=user_id,
            user_profile=profile_dict,
        )

        # 6. Cache Lookup
        cached_entry = self.cache.get(cache_key)
        if cached_entry is not None:
            logger.info(f"Cache HIT for query: '{raw_message[:40]}...' (key={cache_key[:24]})")
            return ChatResponse(
                session_id=session_id,
                answer=cached_entry["answer"],
                sources=cached_entry.get("sources", []),
                disclaimer=cached_entry.get("disclaimer", settings.MEDICAL_DISCLAIMER),
                cached=True,
                is_emergency=False,
            )

        # 7. CACHE MISS -> RAG Semantic Retrieval
        logger.info(f"Cache MISS for query: '{raw_message[:40]}...'. Running RAG retrieval.")
        retrieved_chunks = self.rag.retrieve(query=raw_message, top_k=settings.RAG_TOP_K)

        sources = [
            RetrievedSource(
                source=c.get("source", "knowledge_base"),
                score=c.get("score", 0.0),
                text=c.get("text", ""),
            )
            for c in retrieved_chunks
        ]

        rag_context_text = self.rag.format_rag_context(retrieved_chunks)
        twin_context_text = format_twin_context(profile)

        # 8. Load Conversation History for Context
        chat_history = self._get_conversation_history(db, session_id=session_id, limit=4)

        # 9. Prompt Synthesis
        prompt = build_medical_prompt(
            query=raw_message,
            rag_context=rag_context_text,
            twin_context=twin_context_text,
            chat_history=chat_history,
        )

        # 10. Ollama Generation
        try:
            raw_answer = await self.llm.generate(prompt=prompt)
        except Exception as exc:
            logger.error(f"Error during LLM generation: {exc}")
            raw_answer = (
                "I am currently experiencing technical difficulties retrieving an educational response. "
                "Please consult a qualified healthcare provider directly."
            )

        # 11. Post-generation Safety Validation (allergy alerts, non-diagnostic checks, disclaimer)
        safe_answer = post_process_safety(raw_answer, user_profile=profile_dict)

        # 12. Persist Interaction to SQLite
        self._save_interaction(
            db=db,
            session_id=session_id,
            user_id=user_id,
            user_message=raw_message,
            assistant_response=safe_answer,
            sources=[s.model_dump() for s in sources],
            is_emergency=False,
            cached=False,
        )

        # 13. Store Safe Response in Cache
        self.cache.set(
            key=cache_key,
            value={
                "answer": safe_answer,
                "sources": sources,
                "disclaimer": settings.MEDICAL_DISCLAIMER,
            },
        )

        return ChatResponse(
            session_id=session_id,
            answer=safe_answer,
            sources=sources,
            disclaimer=settings.MEDICAL_DISCLAIMER,
            cached=False,
            is_emergency=False,
        )

    def _get_conversation_history(
        self,
        db: Session,
        session_id: str,
        limit: int = 4,
    ) -> List[Dict[str, str]]:
        """Retrieves the recent chat turns for a session."""
        conv = db.query(Conversation).filter(Conversation.session_id == session_id).first()
        if not conv:
            return []

        messages = (
            db.query(ChatMessage)
            .filter(ChatMessage.conversation_id == conv.id)
            .order_by(ChatMessage.id.desc())
            .limit(limit)
            .all()
        )

        history = []
        for msg in reversed(messages):
            history.append({
                "role": "user" if msg.sender == "user" else "assistant",
                "content": msg.content,
            })
        return history

    def _save_interaction(
        self,
        db: Session,
        session_id: str,
        user_id: str,
        user_message: str,
        assistant_response: str,
        sources: List[Dict[str, Any]],
        is_emergency: bool = False,
        cached: bool = False,
    ) -> None:
        """Persists the user query and assistant response in SQLite."""
        try:
            conv = db.query(Conversation).filter(Conversation.session_id == session_id).first()
            if not conv:
                conv = Conversation(
                    session_id=session_id,
                    user_id=user_id,
                    title=user_message[:60] if user_message else "Medical Consultation",
                )
                db.add(conv)
                db.flush()

            # Record user message
            msg_user = ChatMessage(
                conversation_id=conv.id,
                sender="user",
                content=user_message,
                sources=[],
                is_emergency=is_emergency,
                cached=cached,
            )
            db.add(msg_user)

            # Record assistant response
            msg_assistant = ChatMessage(
                conversation_id=conv.id,
                sender="assistant",
                content=assistant_response,
                sources=sources,
                is_emergency=is_emergency,
                cached=cached,
            )
            db.add(msg_assistant)
            db.commit()
        except Exception as exc:
            db.rollback()
            logger.error(f"Failed to persist chat message: {exc}")


_global_chat_service: Optional[ChatService] = None


def get_chat_service() -> ChatService:
    """Singleton getter for the ChatService."""
    global _global_chat_service
    if _global_chat_service is None:
        _global_chat_service = ChatService()
    return _global_chat_service
