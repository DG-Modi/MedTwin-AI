"""Unit tests for the Digital Twin context formatter, database models, and schemas."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pydantic import ValidationError

from app.db.session import Base
from app.db.models import UserProfile
from app.db.schemas import UserProfileCreate, UserProfileUpdate
from app.core.digital_twin import (
    format_twin_context,
    create_or_update_profile,
    get_profile,
    list_profiles,
)


@pytest.fixture
def in_memory_db():
    """Provides a fresh, isolated in-memory SQLite database session for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()


def test_format_twin_context_none():
    """Verify format_twin_context handles empty/None gracefully."""
    context = format_twin_context(None)
    assert "None provided" in context
    assert "general medical educational knowledge" in context


def test_format_twin_context_full(sample_user_profile):
    """Verify that full profile details and safety disclaimers are formatted into context."""
    context = format_twin_context(sample_user_profile)
    assert "Alex" in context
    assert "Age: 45" in context
    assert "penicillin, peanuts" in context
    assert "Type 2 Diabetes, Hypertension" in context
    assert "Metformin 500mg BID" in context
    # Check clinical safety boundaries
    assert "NOT A DIAGNOSIS" in context
    assert "Never prescribe medications or modify existing dosages" in context


def test_create_and_get_profile(in_memory_db, sample_user_profile):
    """Test creating and retrieving a user profile in the database."""
    profile_in = UserProfileCreate(**sample_user_profile)
    created = create_or_update_profile(in_memory_db, profile_in)

    assert created.id is not None
    assert created.user_id == "test_patient_001"
    assert created.allergies == ["penicillin", "peanuts"]
    assert "Hypertension" in created.chronic_conditions

    # Retrieve from DB
    retrieved = get_profile(in_memory_db, "test_patient_001")
    assert retrieved is not None
    assert retrieved.name == "Alex"
    assert retrieved.age == 45


def test_update_existing_profile(in_memory_db, sample_user_profile):
    """Test updating fields on an existing profile."""
    profile_in = UserProfileCreate(**sample_user_profile)
    create_or_update_profile(in_memory_db, profile_in)

    # Update age and add an allergy
    update_data = UserProfileUpdate(
        age=46,
        allergies=["penicillin", "peanuts", "sulfa"],
    )
    updated = create_or_update_profile(in_memory_db, update_data, user_id="test_patient_001")

    assert updated.age == 46
    assert "sulfa" in updated.allergies
    # Other fields remain preserved
    assert updated.name == "Alex"


def test_profile_schema_validation():
    """Verify Pydantic validates bounds like age and required user_id."""
    with pytest.raises(ValidationError):
        UserProfileCreate(user_id="user_123", age=200)  # Age > 130 invalid

    with pytest.raises(ValidationError):
        UserProfileCreate(user_id="", age=30)  # Empty user_id invalid


def test_list_profiles(in_memory_db):
    """Verify listing multiple profiles."""
    p1 = UserProfileCreate(user_id="user_a", name="Alice", age=30)
    p2 = UserProfileCreate(user_id="user_b", name="Bob", age=40)
    create_or_update_profile(in_memory_db, p1)
    create_or_update_profile(in_memory_db, p2)

    profiles = list_profiles(in_memory_db)
    assert len(profiles) == 2
    user_ids = [p.user_id for p in profiles]
    assert "user_a" in user_ids and "user_b" in user_ids
