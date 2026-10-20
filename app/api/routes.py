"""FastAPI routing for MedTwin AI endpoints."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.config import settings
from app.db.session import get_db
from app.db.schemas import (
    UserProfileCreate,
    UserProfileUpdate,
    UserProfileResponse,
    RetrievedSource,
)
from app.core.digital_twin import (
    get_profile,
    create_or_update_profile,
    list_profiles,
    format_twin_context,
)
from app.rag.pipeline import get_rag_pipeline

router = APIRouter()


@router.get("/health", tags=["System"])
def health_check(db: Session = Depends(get_db)):
    """System health check and vector database status."""
    rag = get_rag_pipeline()
    chunks_count = rag.vector_store.count()
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "vector_store_chunks": chunks_count,
        "environment": settings.APP_ENV,
    }


@router.post("/api/ingest", tags=["RAG Knowledge Base"])
def ingest_documents(force_reload: bool = Query(False, description="Force re-indexing")):
    """Triggers ingestion of clinical documents into ChromaDB."""
    rag = get_rag_pipeline()
    result = rag.ingest_knowledge_base(force_reload=force_reload)
    return result


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
