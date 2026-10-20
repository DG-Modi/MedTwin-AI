"""SQLAlchemy ORM models for MedTwin AI.

Defines schemas for Digital Twin health profiles, conversations, and chat messages.
"""

from datetime import datetime, timezone
from typing import List, Optional, Any
from sqlalchemy import (
    Column,
    Integer,
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
    """Digital Twin health profile representing user background context.

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
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")

    def to_dict(self) -> dict[str, Any]:
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
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Conversation(Base):
    """Represents a chat session with the assistant."""

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), unique=True, index=True, nullable=False)
    user_id = Column(String(64), ForeignKey("user_profiles.user_id"), nullable=True)
    title = Column(String(200), default="New Medical Consultation", nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    user = relationship("UserProfile", back_populates="conversations")
    messages = relationship("ChatMessage", back_populates="conversation", cascade="all, delete-orphan", order_by="ChatMessage.id")


class ChatMessage(Base):
    """Represents an individual message in a conversation."""

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    sender = Column(String(20), nullable=False)  # "user" | "assistant" | "system"
    content = Column(Text, nullable=False)

    # Retrieval and safety metadata
    sources = Column(JSON, default=list, nullable=False)  # [{source: "...", score: 0.85, text: "..."}]
    is_emergency = Column(Boolean, default=False, nullable=False)
    cached = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    conversation = relationship("Conversation", back_populates="messages")
