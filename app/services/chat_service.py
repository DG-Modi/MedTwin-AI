"""Chat orchestration service connecting guardrails, digital twin, two-layer caching, and the CoT pipeline."""

import time
import logging
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from app.config import settings
from app.db.models import Conversation, ChatMessage, QueryLog, Document, utc_now
from app.db.schemas import ChatRequest, ChatResponse, RetrievedSource, ReasoningStep
from app.core.cache import QueryCache, get_cache, generate_cache_key, generate_twin_hash
from app.core.cot_pipeline import ChainOfThoughtPipeline, get_cot_pipeline
from app.core.digital_twin import get_profile
from app.rag.embeddings import get_embedding_model
from app.core.guardrails import detect_emergency, check_emergency_keywords, format_emergency_response

logger = logging.getLogger(__name__)


class ChatService:
    """Orchestrates end-to-end medical chat processing with safety boundaries, CoT reasoning, and query caching."""

    def __init__(
        self,
        cache: Optional[QueryCache] = None,
        cot_pipeline: Optional[ChainOfThoughtPipeline] = None,
        rag_pipeline: Optional[Any] = None,
        llm_client: Optional[Any] = None,
    ) -> None:
        self.cache = cache or get_cache()
        self.cot = cot_pipeline or (
            ChainOfThoughtPipeline(rag_pipeline=rag_pipeline, llm_client=llm_client)
            if (rag_pipeline or llm_client)
            else get_cot_pipeline()
        )
        self.rag = self.cot.rag
        self.llm = self.cot.llm

    @property
    def model_name(self) -> str:
        m = getattr(self.llm, "model", None)
        if isinstance(m, str) and m.strip():
            return m.strip()
        return settings.OLLAMA_MODEL

    async def process_chat(self, db: Session, request: ChatRequest) -> ChatResponse:
        """Processes a chat request through the full two-layer cache and 7-stage CoT pipeline."""
        t_start = time.time()

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
        if len(raw_message) > 3000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message exceeds maximum allowed length of 3000 characters.",
            )

        # 2. Emergency Pre-Execution Check
        is_emergency, keyword = check_emergency_keywords(raw_message)
        if is_emergency:
            profile = get_profile(db, user_id)
            profile_dict = profile.to_dict() if profile else None
            emergency_notice = format_emergency_response(
                matched_keyword=keyword,
                twin_profile=profile_dict,
                query=raw_message,
            )
            reasoning_steps = [
                ReasoningStep(
                    stage="emergency_detection",
                    name="2. Red-Flag & Emergency Escalation Check",
                    status="flagged",
                    summary=f"Red-flag check = TRIGGERED with matched keyword '{keyword}'. Emergency protocol activated: RAG skipped, LLM skipped.",
                    details={"triggered": True, "matched_keyword": keyword, "rag_skipped": True, "llm_skipped": True},
                )
            ]
            self._save_interaction(
                db=db,
                session_id=session_id,
                user_id=user_id,
                user_message=raw_message,
                assistant_response=emergency_notice,
                sources=[],
                reasoning_steps=[s.model_dump() for s in reasoning_steps],
                is_emergency=True,
                cached=False,
                latency_ms=0.0,
            )
            return ChatResponse(
                session_id=session_id,
                answer=emergency_notice,
                sources=[],
                disclaimer=settings.MEDICAL_DISCLAIMER,
                cached=False,
                is_emergency=True,
                reasoning_steps=reasoning_steps,
                retrieval_ms=0.0,
                llm_ms=0.0,
                total_latency_ms=0.0,
                model_name=self.model_name,
            )

        # 3. Digital Twin Context retrieval
        profile = get_profile(db, user_id)
        profile_dict = profile.to_dict() if profile else None
        twin_hash = generate_twin_hash(profile_dict=profile_dict, user_id=user_id)

        # 4. Two-Layer Cache Lookup
        cache_key = generate_cache_key(
            user_id=user_id,
            query=raw_message,
            user_profile=profile_dict,
            model=self.model_name,
        )

        # Layer 1: Exact-match
        cached_entry = self.cache.get(cache_key, current_twin_hash=twin_hash)
        cache_hit = False
        cache_type = None

        if cached_entry is not None:
            cache_hit = True
            cache_type = "exact"
            total_lat = round((time.time() - t_start) * 1000, 2)
            logger.info(f"Layer 1 EXACT Cache HIT for query: '{raw_message[:40]}...' (key={cache_key[:16]})")

            self._log_query(
                db=db,
                session_id=session_id,
                patient_id=user_id,
                query_text=raw_message,
                cache_hit=True,
                cache_type="exact",
                total_latency_ms=total_lat,
                retrieval_ms=0.0,
                llm_ms=0.0,
                is_emergency=cached_entry.get("is_emergency", False),
                model_name=self.model_name,
            )

            # Convert cached sources to Pydantic objects if dicts
            sources = [
                RetrievedSource(**s) if isinstance(s, dict) else s
                for s in cached_entry.get("sources", [])
            ]
            reasoning_steps = [
                ReasoningStep(**s) if isinstance(s, dict) else s
                for s in cached_entry.get("reasoning_steps", [])
            ]

            # Append synthetic cached reasoning step for transparency
            cached_steps = [
                ReasoningStep(
                    stage="cache_lookup",
                    name="⚡ Two-Layer Query Cache Engine",
                    status="completed",
                    summary="Layer 1 Exact Match HIT (0ms): Returned verified clinical answer from patient-isolated cache.",
                    details={"cache_type": "exact", "cache_key": cache_key[:16]},
                )
            ] + reasoning_steps

            return ChatResponse(
                session_id=session_id,
                answer=cached_entry["answer"],
                sources=sources,
                disclaimer=cached_entry.get("disclaimer", settings.MEDICAL_DISCLAIMER),
                cached=True,
                cache_type="exact",
                is_emergency=cached_entry.get("is_emergency", False),
                reasoning_steps=cached_steps,
                retrieval_ms=0.0,
                llm_ms=0.0,
                total_latency_ms=total_lat,
                model_name=self.model_name,
            )

        # Layer 2: Semantic vector match
        query_embedding = None
        try:
            embed_model = get_embedding_model()
            query_embedding = embed_model.embed_query(raw_message)
            sem_result = self.cache.get_semantic(
                query=raw_message,
                patient_id=user_id,
                query_embedding=query_embedding,
                current_twin_hash=twin_hash,
            )
            if sem_result is not None:
                sem_key, sem_entry, sim_score = sem_result
                total_lat = round((time.time() - t_start) * 1000, 2)
                logger.info(f"Layer 2 SEMANTIC Cache HIT (sim={sim_score:.3f}) for: '{raw_message[:40]}...'")

                self._log_query(
                    db=db,
                    session_id=session_id,
                    patient_id=user_id,
                    query_text=raw_message,
                    cache_hit=True,
                    cache_type="semantic",
                    total_latency_ms=total_lat,
                    retrieval_ms=0.0,
                    llm_ms=0.0,
                    is_emergency=sem_entry.get("is_emergency", False),
                    model_name=self.model_name,
                )

                sources = [
                    RetrievedSource(**s) if isinstance(s, dict) else s
                    for s in sem_entry.get("sources", [])
                ]
                reasoning_steps = [
                    ReasoningStep(
                        stage="cache_lookup",
                        name="⚡ Two-Layer Query Cache Engine",
                        status="completed",
                        summary=f"Layer 2 Semantic Match HIT ({sim_score:.1%} cosine similarity >= 92% threshold).",
                        details={"cache_type": "semantic", "similarity": sim_score},
                    )
                ] + [
                    ReasoningStep(**s) if isinstance(s, dict) else s
                    for s in sem_entry.get("reasoning_steps", [])
                ]

                return ChatResponse(
                    session_id=session_id,
                    answer=sem_entry["answer"],
                    sources=sources,
                    disclaimer=sem_entry.get("disclaimer", settings.MEDICAL_DISCLAIMER),
                    cached=True,
                    cache_type="semantic",
                    is_emergency=sem_entry.get("is_emergency", False),
                    reasoning_steps=reasoning_steps,
                    retrieval_ms=0.0,
                    llm_ms=0.0,
                    total_latency_ms=total_lat,
                    model_name=self.model_name,
                )
        except Exception as ex:
            logger.debug(f"Semantic cache lookup skipped or unavailable: {ex}")

        # 4. CACHE MISS -> Execute full 7-stage CoT Pipeline
        logger.info(f"Cache MISS for query: '{raw_message[:40]}...'. Executing 7-Stage Chain of Thought Pipeline.")
        chat_history = self._get_conversation_history(db, session_id=session_id, limit=4)

        cot_result = await self.cot.execute(
            query=raw_message,
            user_id=user_id,
            session_id=session_id,
            db=db,
            chat_history=chat_history,
        )

        answer = cot_result["answer"]
        sources = cot_result["sources"]
        is_emergency = cot_result["is_emergency"]
        reasoning_steps = cot_result["reasoning_steps"]
        timings = cot_result.get("timings", {})
        total_latency_ms = round((time.time() - t_start) * 1000, 2)

        # 5. Persist interaction to SQLite
        self._save_interaction(
            db=db,
            session_id=session_id,
            user_id=user_id,
            user_message=raw_message,
            assistant_response=answer,
            sources=[s.model_dump() if hasattr(s, "model_dump") else s for s in sources],
            reasoning_steps=[s.model_dump() if hasattr(s, "model_dump") else s for s in reasoning_steps],
            is_emergency=is_emergency,
            cached=False,
            latency_ms=total_latency_ms,
        )

        # 6. Store non-emergency answers in Two-Layer Cache
        if not is_emergency:
            self.cache.set(
                key=cache_key,
                value={
                    "answer": answer,
                    "sources": [s.model_dump() if hasattr(s, "model_dump") else s for s in sources],
                    "disclaimer": settings.MEDICAL_DISCLAIMER,
                    "is_emergency": False,
                    "reasoning_steps": [s.model_dump() if hasattr(s, "model_dump") else s for s in reasoning_steps],
                },
                query_text=raw_message,
                embedding=query_embedding,
                patient_id=user_id,
                cache_type="exact",
                twin_hash=twin_hash,
            )

        # 7. Telemetry Log in SQLite
        self._log_query(
            db=db,
            session_id=session_id,
            patient_id=user_id,
            query_text=raw_message,
            cache_hit=False,
            cache_type=None,
            total_latency_ms=total_latency_ms,
            retrieval_ms=timings.get("retrieval_ms", 0.0),
            llm_ms=timings.get("llm_ms", 0.0),
            is_emergency=is_emergency,
            model_name=self.model_name,
        )

        return ChatResponse(
            session_id=session_id,
            answer=answer,
            sources=sources,
            disclaimer=settings.MEDICAL_DISCLAIMER,
            cached=False,
            cache_type=None,
            is_emergency=is_emergency,
            reasoning_steps=reasoning_steps,
            retrieval_ms=timings.get("retrieval_ms", 0.0),
            llm_ms=timings.get("llm_ms", 0.0),
            total_latency_ms=total_latency_ms,
            model_name=self.model_name,
        )

    def _get_conversation_history(
        self,
        db: Session,
        session_id: str,
        limit: int = 4,
    ) -> List[Dict[str, str]]:
        """Retrieves recent chat turns for a session."""
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
        reasoning_steps: Optional[List[Dict[str, Any]]] = None,
        is_emergency: bool = False,
        cached: bool = False,
        latency_ms: Optional[float] = None,
    ) -> None:
        """Persists user query and assistant response in SQLite."""
        import json

        def _clean_json(obj):
            try:
                if hasattr(obj, "model_dump_json"):
                    return json.loads(obj.model_dump_json())
                return json.loads(json.dumps(obj, default=str))
            except Exception:
                return str(obj)

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
                reasoning_steps=[],
                is_emergency=is_emergency,
                cached=cached,
                latency_ms=0.0,
            )
            db.add(msg_user)

            # Record assistant response
            msg_assistant = ChatMessage(
                conversation_id=conv.id,
                sender="assistant",
                content=assistant_response,
                sources=_clean_json(sources),
                reasoning_steps=_clean_json(reasoning_steps or []),
                is_emergency=is_emergency,
                cached=cached,
                latency_ms=latency_ms,
            )
            db.add(msg_assistant)
            db.commit()
        except Exception as exc:
            db.rollback()
            logger.error(f"Failed to persist chat message: {exc}")

    def _log_query(
        self,
        db: Session,
        session_id: str,
        patient_id: str,
        query_text: str,
        cache_hit: bool,
        cache_type: Optional[str],
        total_latency_ms: float,
        retrieval_ms: float,
        llm_ms: float,
        is_emergency: bool,
        model_name: str,
    ) -> None:
        """Logs query execution telemetry in SQLite `query_logs` table."""
        try:
            log_entry = QueryLog(
                session_id=session_id,
                patient_id=patient_id,
                query_text=query_text,
                cache_hit=cache_hit,
                cache_type=cache_type,
                total_latency_ms=total_latency_ms,
                retrieval_ms=retrieval_ms,
                llm_ms=llm_ms,
                is_emergency=is_emergency,
                model_name=model_name,
                created_at=utc_now(),
            )
            db.add(log_entry)
            db.commit()
        except Exception as ex:
            db.rollback()
            logger.error(f"Failed to record QueryLog: {ex}")


_global_chat_service: Optional[ChatService] = None


def get_chat_service() -> ChatService:
    """Singleton getter for the ChatService."""
    global _global_chat_service
    if _global_chat_service is None:
        _global_chat_service = ChatService()
    return _global_chat_service
