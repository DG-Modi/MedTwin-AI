"""Digital Twin context module for MedTwin AI.

Manages user health profiles (age, allergies, chronic conditions, current medications)
and formats safe educational context for the medical assistant without diagnosing.
"""

from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from app.db.models import UserProfile
from app.db.schemas import UserProfileCreate, UserProfileUpdate


def get_profile(db: Session, user_id: str) -> Optional[UserProfile]:
    """Retrieves a user profile by user_id."""
    return db.query(UserProfile).filter(UserProfile.user_id == user_id).first()


def list_profiles(db: Session, limit: int = 50) -> List[UserProfile]:
    """Lists available user profiles."""
    return db.query(UserProfile).limit(limit).all()


def create_or_update_profile(
    db: Session,
    profile_data: UserProfileCreate | UserProfileUpdate | Dict[str, Any],
    user_id: Optional[str] = None,
) -> UserProfile:
    """Creates a new user profile or updates an existing one."""
    if isinstance(profile_data, dict):
        uid = user_id or profile_data.get("user_id")
        data_dict = profile_data
    else:
        uid = user_id or getattr(profile_data, "user_id", None)
        data_dict = profile_data.model_dump(exclude_unset=True)

    if not uid:
        raise ValueError("user_id must be provided to create or update a profile.")

    existing = get_profile(db, uid)

    if existing:
        for key, value in data_dict.items():
            if value is not None and hasattr(existing, key):
                setattr(existing, key, value)
        db.commit()
        db.refresh(existing)
        return existing
    else:
        new_profile = UserProfile(
            user_id=uid,
            name=data_dict.get("name"),
            age=data_dict.get("age"),
            gender=data_dict.get("gender"),
            allergies=data_dict.get("allergies") or [],
            chronic_conditions=data_dict.get("chronic_conditions") or [],
            current_medications=data_dict.get("current_medications") or [],
            lifestyle_notes=data_dict.get("lifestyle_notes"),
        )
        db.add(new_profile)
        db.commit()
        db.refresh(new_profile)
        return new_profile


def format_twin_context(profile: Optional[UserProfile | Dict[str, Any]]) -> str:
    """Formats a user's health profile into a safe context block for the LLM.

    Strict medical safety boundaries are enforced:
    - Never diagnose.
    - Never prescribe medication or adjust dosage.
    - Use allergies to explicitly flag contraindications in educational explanations.
    """
    if not profile:
        return "Digital Twin Context: None provided. Responding with general medical educational knowledge."

    if isinstance(profile, UserProfile):
        p_dict = profile.to_dict()
    else:
        p_dict = profile

    user_name = p_dict.get("name") or "User"
    age = p_dict.get("age", "Not specified")
    gender = p_dict.get("gender", "Not specified")

    allergies = p_dict.get("allergies") or []
    allergies_str = ", ".join(allergies) if allergies else "None reported"

    conditions = p_dict.get("chronic_conditions") or []
    conditions_str = ", ".join(conditions) if conditions else "None reported"

    meds = p_dict.get("current_medications") or []
    meds_str = ", ".join(meds) if meds else "None reported"

    lifestyle = p_dict.get("lifestyle_notes") or "None noted"

    context_lines = [
        "### PATIENT DIGITAL TWIN PROFILE (FOR EDUCATIONAL PERSONALIZATION ONLY - NOT A DIAGNOSIS):",
        f"- Patient/User Name: {user_name}",
        f"- Age: {age} | Gender: {gender}",
        f"- Documented Allergies: {allergies_str}",
        f"- Chronic Health Conditions: {conditions_str}",
        f"- Current Regimen/Medications: {meds_str}",
        f"- Lifestyle & Notes: {lifestyle}",
        "",
        "CRITICAL CLINICAL BOUNDARIES FOR DIGITAL TWIN CONTEXT:",
        "1. This profile is for educational personalization only; do NOT attempt to diagnose new conditions.",
        "2. Never prescribe medications or modify existing dosages.",
        "3. If the user's query references an allergen or a potential contraindication with their documented profile, highlight this educational concern clearly.",
    ]

    return "\n".join(context_lines)
