"""SQLAlchemy ORM models for MedTwin AI.

Defines schemas for Digital Twin health profiles, conversations, chat messages,
document ingestion tracking, two-layer cache entries, and query telemetry logs.
"""

from datetime import datetime, timezone
from typing import List, Optional, Any, Dict
from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    Text,
    DateTime,
    Boolean,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import relationship
from app.db.session import Base


def utc_now() -> datetime:
    """Returns the current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class UserProfile(Base):
    """Digital Twin health profile representing patient background context.

    Used strictly for educational context personalization, NOT diagnosis.
    """

    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(64), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=True)
    age = Column(Integer, nullable=True)
    gender = Column(String(32), nullable=True)

    # Structured medical background stored as JSON lists
    allergies = Column(JSON, default=list, nullable=False)
    chronic_conditions = Column(JSON, default=list, nullable=False)
    current_medications = Column(JSON, default=list, nullable=False)

    lifestyle_notes = Column(Text, nullable=True)
    clinical_tags = Column(JSON, default=list, nullable=True)
    twin_version = Column(Integer, default=1, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        """Converts model instance to a standard Python dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "allergies": self.allergies or [],
            "chronic_conditions": self.chronic_conditions or [],
            "current_medications": self.current_medications or [],
            "lifestyle_notes": self.lifestyle_notes or "",
            "clinical_tags": self.clinical_tags or [],
            "twin_version": getattr(self, "twin_version", 1) or 1,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# Aliases for assignment nomenclature
Patient = UserProfile


class Conversation(Base):
    """Represents a chat session with the assistant."""

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), unique=True, index=True, nullable=False)
    user_id = Column(String(64), ForeignKey("user_profiles.user_id"), nullable=True)
    title = Column(String(200), default="New Medical Consultation", nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    user = relationship("UserProfile", back_populates="conversations")
    messages = relationship(
        "ChatMessage",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ChatMessage.id",
    )


SessionModel = Conversation


class ChatMessage(Base):
    """Represents an individual message in a conversation."""

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    sender = Column(String(20), nullable=False)  # "user" | "assistant" | "system"
    content = Column(Text, nullable=False)

    # Retrieval, reasoning, and safety metadata
    sources = Column(JSON, default=list, nullable=False)
    reasoning_steps = Column(JSON, default=list, nullable=True)
    is_emergency = Column(Boolean, default=False, nullable=False)
    cached = Column(Boolean, default=False, nullable=False)
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    conversation = relationship("Conversation", back_populates="messages")


Message = ChatMessage


class Document(Base):
    """Ingested knowledge base article metadata logged in SQLite."""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String(255), unique=True, index=True, nullable=False)
    url = Column(String(500), nullable=True)
    title = Column(String(255), nullable=False)
    category = Column(String(100), nullable=True)
    chunk_count = Column(Integer, default=0, nullable=False)
    ingested_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "url": self.url,
            "title": self.title,
            "category": self.category,
            "chunk_count": self.chunk_count,
            "ingested_at": self.ingested_at.isoformat() if self.ingested_at else None,
        }


class CacheEntryModel(Base):
    """Persistent storage for exact-match and semantic cache entries."""

    __tablename__ = "cache_entries"

    id = Column(Integer, primary_key=True, index=True)
    cache_key = Column(String(128), unique=True, index=True, nullable=False)
    cache_type = Column(String(32), default="exact", nullable=False)  # "exact" | "semantic"
    query_text = Column(Text, nullable=False)
    patient_id = Column(String(64), index=True, nullable=True)
    answer = Column(Text, nullable=False)
    sources = Column(JSON, default=list, nullable=False)
    reasoning_steps = Column(JSON, default=list, nullable=True)
    expires_at = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)


class QueryLog(Base):
    """Telemetry log recording query latency, stage timings, and cache metrics."""

    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), index=True, nullable=True)
    patient_id = Column(String(64), index=True, nullable=True)
    query_text = Column(Text, nullable=False)
    cache_hit = Column(Boolean, default=False, nullable=False)
    cache_type = Column(String(32), nullable=True)  # "exact", "semantic", or None
    total_latency_ms = Column(Float, default=0.0, nullable=False)
    retrieval_ms = Column(Float, default=0.0, nullable=False)
    llm_ms = Column(Float, default=0.0, nullable=False)
    is_emergency = Column(Boolean, default=False, nullable=False)
    model_name = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "patient_id": self.patient_id,
            "query_text": self.query_text,
            "cache_hit": self.cache_hit,
            "cache_type": self.cache_type,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "retrieval_ms": round(self.retrieval_ms, 2),
            "llm_ms": round(self.llm_ms, 2),
            "is_emergency": self.is_emergency,
            "model_name": self.model_name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
