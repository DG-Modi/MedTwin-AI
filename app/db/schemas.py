"""Pydantic schemas for data validation and API serialization."""

from datetime import datetime
from typing import List, Optional, Any, Dict, Union
from pydantic import BaseModel, Field, ConfigDict


class RetrievedSource(BaseModel):
    """Metadata describing a retrieved document chunk from the knowledge base."""

    source: str = Field(..., description="Document source filename or identifier")
    score: float = Field(..., description="Semantic similarity score (0 to 1)")
    text: str = Field(..., description="Excerpt from the source document")
    title: Optional[str] = Field("Clinical Guide", description="Article or section title")
    url: Optional[str] = Field(None, description="Source web URL (e.g. CDC, NIH, WHO)")


class ReasoningStep(BaseModel):
    """Explicit stage result in Chain of Thought (CoT) pipeline."""

    stage: str = Field(..., description="Stage identifier (e.g., intent, emergency, safety_rules)")
    name: str = Field(..., description="Human-readable stage title")
    status: str = Field("completed", description="Status (completed, passed, flagged, bypassed)")
    summary: str = Field(..., description="Concise structured summary of stage outcome")
    details: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata or flags")


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
    clinical_tags: Optional[List[str]] = Field(default_factory=list, description="Clinical risk flags / tags")
    twin_version: int = Field(1, description="Monotonically increasing profile version counter for cache invalidation")


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
    clinical_tags: Optional[List[str]] = None


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
    reasoning_steps: List[ReasoningStep] = Field(default_factory=list)
    is_emergency: bool = False
    cached: bool = False


class ChatMessageResponse(BaseModel):
    """Schema for returning a chat message."""

    id: int
    conversation_id: int
    sender: str
    content: str
    sources: List[RetrievedSource] = Field(default_factory=list)
    reasoning_steps: List[ReasoningStep] = Field(default_factory=list)
    is_emergency: bool
    cached: bool
    latency_ms: Optional[float] = None
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


class ChatRequest(BaseModel):
    """Schema for incoming chat messages to POST /api/chat."""

    user_id: Any = Field(..., description="Unique user or patient identifier (string or integer)")
    session_id: str = Field(..., min_length=1, max_length=128, description="Chat session identifier")
    message: str = Field(..., min_length=1, max_length=2000, description="User's medical query")

    model_config = ConfigDict(extra="ignore")


class ChatResponse(BaseModel):
    """Schema for returning structured chat response from POST /api/chat."""

    session_id: str
    answer: str
    sources: List[RetrievedSource] = Field(default_factory=list)
    disclaimer: str
    cached: bool = False
    cache_type: Optional[str] = None
    is_emergency: bool = False
    reasoning_steps: List[ReasoningStep] = Field(default_factory=list)
    retrieval_ms: Optional[float] = 0.0
    llm_ms: Optional[float] = 0.0
    total_latency_ms: Optional[float] = 0.0
    model_name: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class DocumentResponse(BaseModel):
    """Schema for knowledge base document library entries."""

    id: int
    source: str
    title: str
    url: Optional[str] = None
    category: Optional[str] = None
    chunk_count: int
    ingested_at: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class IngestionStatsResponse(BaseModel):
    """Schema for ingestion run statistics."""

    status: str
    documents_loaded: int
    chunks_indexed: int
    last_run: Optional[str] = None
    message: str


class CacheStatsResponse(BaseModel):
    """Schema for two-layer cache statistics."""

    hits: int
    misses: int
    total_lookups: int
    hit_ratio: float
    active_entries: int
    exact_cache_size: int
    semantic_cache_size: int
    recent_queries: List[Dict[str, Any]] = Field(default_factory=list)
