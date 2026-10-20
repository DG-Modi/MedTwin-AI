"""Pydantic schemas for data validation and API serialization."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class RetrievedSource(BaseModel):
    """Metadata describing a retrieved document chunk from the knowledge base."""

    source: str = Field(..., description="Document source filename or identifier")
    score: float = Field(..., description="Semantic similarity score (0 to 1)")
    text: str = Field(..., description="Excerpt from the source document")


class UserProfileBase(BaseModel):
    """Base schema for Digital Twin health profile."""

    user_id: str = Field(..., min_length=1, max_length=64, description="Unique user or patient identifier")
    name: Optional[str] = Field(None, max_length=100, description="User's preferred name")
    age: Optional[int] = Field(None, ge=0, le=130, description="Age in years")
    gender: Optional[str] = Field(None, max_length=32, description="Biological sex or gender identity")
    allergies: List[str] = Field(default_factory=list, description="List of confirmed allergies (e.g. penicillin)")
    chronic_conditions: List[str] = Field(default_factory=list, description="Known chronic health conditions")
    current_medications: List[str] = Field(default_factory=list, description="Current prescription or OTC medications")
    lifestyle_notes: Optional[str] = Field(None, description="Relevant lifestyle details (diet, smoking, exercise)")


class UserProfileCreate(UserProfileBase):
    """Schema for creating a new Digital Twin profile."""
    pass


class UserProfileUpdate(BaseModel):
    """Schema for updating an existing Digital Twin profile."""

    name: Optional[str] = Field(None, max_length=100)
    age: Optional[int] = Field(None, ge=0, le=130)
    gender: Optional[str] = Field(None, max_length=32)
    allergies: Optional[List[str]] = None
    chronic_conditions: Optional[List[str]] = None
    current_medications: Optional[List[str]] = None
    lifestyle_notes: Optional[str] = None


class UserProfileResponse(UserProfileBase):
    """Schema for returning Digital Twin profile data."""

    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ChatMessageCreate(BaseModel):
    """Schema for adding a message to a conversation."""

    sender: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1)
    sources: List[RetrievedSource] = Field(default_factory=list)
    is_emergency: bool = False
    cached: bool = False


class ChatMessageResponse(BaseModel):
    """Schema for returning a chat message."""

    id: int
    conversation_id: int
    sender: str
    content: str
    sources: List[RetrievedSource] = Field(default_factory=list)
    is_emergency: bool
    cached: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ConversationResponse(BaseModel):
    """Schema for returning a conversation thread with messages."""

    id: int
    session_id: str
    user_id: Optional[str] = None
    title: str
    created_at: datetime
    messages: List[ChatMessageResponse] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)
