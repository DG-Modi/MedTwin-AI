"""FastAPI routing for MedTwin AI endpoints."""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status, UploadFile, File
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.config import settings
from app.db.session import get_db
from app.db.models import Document, QueryLog
from app.db.schemas import (
    UserProfileCreate,
    UserProfileUpdate,
    UserProfileResponse,
    RetrievedSource,
    ChatRequest,
    ChatResponse,
    DocumentResponse,
)
from app.core.digital_twin import (
    get_profile,
    create_or_update_profile,
    list_profiles,
    format_twin_context,
)
from app.core.cache import get_cache
from app.rag.pipeline import get_rag_pipeline
from app.llm.factory import get_llm_provider
from app.services.chat_service import ChatService, get_chat_service
from app.services.voice_service import VoiceService, get_voice_service
from pipeline.run import run_ingestion_pipeline

router = APIRouter()


@router.get("/health", tags=["System"])
def health_check(db: Session = Depends(get_db)):
    """System health check, active model, and vector database status."""
    rag = get_rag_pipeline()
    chunks_count = rag.vector_store.count()
    doc_count = db.query(Document).count()
    llm = get_llm_provider()
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "active_model": llm.get_display_name(),
        "llm_provider": llm.name,
        "vector_store_chunks": chunks_count,
        "documents_in_db": doc_count,
        "environment": settings.APP_ENV,
    }


@router.post("/api/ingest", tags=["RAG Knowledge Base"])
def ingest_documents(
    force_reload: bool = Query(True, description="Force re-indexing"),
    db: Session = Depends(get_db),
):
    """Triggers end-to-end ingestion of clinical documents into ChromaDB and SQLite."""
    result = run_ingestion_pipeline(force_reload=force_reload, db=db)
    return result


@router.get("/api/documents", response_model=List[DocumentResponse], tags=["RAG Knowledge Base"])
def list_ingested_documents(db: Session = Depends(get_db)):
    """Retrieves all ingested medical guidelines from the SQLite documents table."""
    docs = db.query(Document).order_by(Document.id.asc()).all()
    return [
        DocumentResponse(
            id=d.id,
            source=d.source,
            title=d.title,
            url=d.url,
            category=d.category,
            chunk_count=d.chunk_count,
            ingested_at=d.ingested_at.isoformat() if d.ingested_at else None,
        )
        for d in docs
    ]


@router.get("/api/search", response_model=List[RetrievedSource], tags=["RAG Knowledge Base"])
def semantic_search(
    query: str = Query(..., min_length=2, description="Medical query or symptom"),
    top_k: int = Query(3, ge=1, le=10),
    min_score: float = Query(0.1, ge=0.0, le=1.0),
):
    """Performs semantic similarity search against the ChromaDB medical knowledge base."""
    rag = get_rag_pipeline()
    results = rag.retrieve(query=query, top_k=top_k, min_score=min_score)
    return [
        RetrievedSource(
            source=r["source"],
            score=r["score"],
            text=r["text"],
            title=r.get("title", "Clinical Guide"),
            url=r.get("url", None),
        )
        for r in results
    ]


@router.post("/api/twin", response_model=UserProfileResponse, tags=["Digital Twin"])
def save_user_profile(
    profile_in: UserProfileCreate,
    db: Session = Depends(get_db),
):
    """Creates or updates a patient Digital Twin profile."""
    profile = create_or_update_profile(db, profile_in)
    return profile


@router.get("/api/twin/{user_id}", response_model=UserProfileResponse, tags=["Digital Twin"])
def get_user_profile(
    user_id: str,
    db: Session = Depends(get_db),
):
    """Retrieves a patient Digital Twin profile by user ID."""
    profile = get_profile(db, user_id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile with user_id '{user_id}' not found.",
        )
    return profile


@router.get("/api/twin/{user_id}/context", tags=["Digital Twin"])
def get_user_twin_context(
    user_id: str,
    db: Session = Depends(get_db),
):
    """Returns the formatted educational prompt context for the given user's Digital Twin."""
    profile = get_profile(db, user_id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile with user_id '{user_id}' not found.",
        )
    context_text = format_twin_context(profile)
    return {
        "user_id": user_id,
        "formatted_context": context_text,
    }


@router.get("/api/twin", response_model=List[UserProfileResponse], tags=["Digital Twin"])
def list_all_profiles(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Lists registered patient Digital Twin profiles."""
    return list_profiles(db, limit=limit)


@router.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(
    request: ChatRequest,
    db: Session = Depends(get_db),
    service: ChatService = Depends(get_chat_service),
):
    """Orchestrates an end-to-end medical chat interaction with safety guardrails, Digital Twin, and caching."""
    return await service.process_chat(db=db, request=request)


@router.get("/api/cache/stats", tags=["System"])
def get_cache_telemetry(db: Session = Depends(get_db)):
    """Returns two-layer query cache performance metrics and recent query logs."""
    stats = get_cache().stats()

    # Query log metrics from SQLite
    try:
        avg_cached = (
            db.query(func.avg(QueryLog.total_latency_ms))
            .filter(QueryLog.cache_hit == True)
            .scalar()
        )
        avg_uncached = (
            db.query(func.avg(QueryLog.total_latency_ms))
            .filter(QueryLog.cache_hit == False)
            .scalar()
        )
        recent_logs = (
            db.query(QueryLog)
            .order_by(QueryLog.id.desc())
            .limit(10)
            .all()
        )
        recent_data = [l.to_dict() for l in recent_logs]
    except Exception:
        avg_cached = 0.0
        avg_uncached = 0.0
        recent_data = []

    rag = get_rag_pipeline()
    chunks_count = rag.vector_store.count()

    return {
        **stats,
        "avg_latency_cached_ms": round(avg_cached or 0.0, 2),
        "avg_latency_uncached_ms": round(avg_uncached or 0.0, 2),
        "active_model": settings.OLLAMA_MODEL,
        "vector_store_chunks": chunks_count,
        "recent_queries": recent_data,
    }


@router.post("/api/transcribe", tags=["Voice Input"])
async def transcribe_audio_endpoint(
    audio_file: UploadFile = File(..., description="Audio file (WAV or raw PCM)"),
    language: Optional[str] = Query("en", description="Spoken language code (e.g. en)"),
    service: VoiceService = Depends(get_voice_service),
):
    """Transcribes an uploaded audio recording into text using Whisper."""
    try:
        content = await audio_file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty audio payload provided.",
            )
        result = service.transcribe(content, language=language)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Speech transcription failed: {str(e)}",
        )
