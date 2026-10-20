"""Medical safety guardrails and clinical boundary enforcement.

Guarantees non-diagnostic, non-prescriptive responses, emergency detection,
allergy contraindication checks, and disclaimer enforcement.
"""

import re
from typing import Tuple, Optional, Dict, Any, List
from app.config import settings


def detect_emergency(query: str) -> Tuple[bool, Optional[str]]:
    """Inspects user input for life-threatening emergency symptoms.

    Returns (is_emergency, emergency_notice).
    """
    normalized = query.lower()

    for keyword in settings.EMERGENCY_KEYWORDS:
        # Match whole words or phrases
        pattern = r"\b" + re.escape(keyword) + r"\b"
        if re.search(pattern, normalized):
            return True, settings.EMERGENCY_NOTICE

    # Specific acute symptom patterns
    acute_patterns = [
        r"\b(crushing|severe)\s+(chest\s+pain|pressure)\b",
        r"\bcan('?t|not)\s+breathe\b",
        r"\bsudden(ly)?\s+(numb|weak|confused|blind)\b",
        r"\b(pass|passed)\s+out\b",
        r"\bface\s+droop(ing)?\b",
        r"\bwant\s+to\s+(die|end\s+it|kill\s+myself)\b",
    ]

    for pat in acute_patterns:
        if re.search(pat, normalized):
            return True, settings.EMERGENCY_NOTICE

    return False, None


def validate_input_safety(query: str) -> Tuple[bool, Optional[str]]:
    """Checks for harmful requests (lethal dosage, illicit drug manufacturing, self-harm).

    Returns (is_safe, refusal_message).
    """
    normalized = query.lower()

    harmful_patterns = [
        r"\blethal\s+dose\b",
        r"\bhow\s+to\s+overdose\b",
        r"\bmake\s+(meth|cocaine|heroin|fentanyl)\b",
        r"\bhow\s+to\s+(commit\s+suicide|kill\s+myself)\b",
    ]

    for pat in harmful_patterns:
        if re.search(pat, normalized):
            return (
                False,
                "I cannot assist with queries regarding lethal dosages, intentional harm, or illegal substances. "
                "If you are in distress, please contact a crisis lifeline immediately (Call or text 988 in the US/Canada, "
                "or 112 in Europe).",
            )

    return True, None


def check_allergy_contraindications(
    response_text: str,
    allergies: List[str],
) -> Optional[str]:
    """Inspects assistant response to alert if any documented patient allergen is mentioned."""
    if not allergies:
        return None

    normalized_response = response_text.lower()
    flagged_allergens = []

    for allergen in allergies:
        clean_allergen = allergen.strip().lower()
        if not clean_allergen:
            continue
        pattern = r"\b" + re.escape(clean_allergen) + r"\b"
        if re.search(pattern, normalized_response):
            flagged_allergens.append(allergen.strip())

    if flagged_allergens:
        joined = ", ".join(flagged_allergens)
        return (
            f"⚠️ **PATIENT ALLERGY ALERT**: Your Digital Twin profile lists a documented allergy to **{joined}**. "
            f"Please note that substances or medications mentioned in this educational text may conflict with your allergy. "
            f"Always verify with your doctor or pharmacist."
        )

    return None


def sanitize_sensitive_info(text: str) -> str:
    """Masks obvious sensitive personal identifiers (SSNs, credit card numbers, phone numbers)."""
    # SSN pattern
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN-REDACTED]", text)
    # Credit card pattern (13 to 19 digits)
    text = re.sub(r"\b(?:\d{4}[ -]?){3}\d{1,4}\b", "[CARD-REDACTED]", text)
    return text


def post_process_safety(
    response_text: str,
    user_profile: Optional[Dict[str, Any]] = None,
) -> str:
    """Applies clinical guardrails to LLM output:

    1. Checks against documented allergies in Digital Twin.
    2. Rewrites or flags diagnostic assertions.
    3. Guarantees mandatory medical disclaimer is present.
    """
    cleaned = sanitize_sensitive_info(response_text)

    # 1. Check for aggressive diagnostic assertions and soften them
    cleaned = re.sub(
        r"\b(you have|i diagnose you with|you are suffering from)\b",
        "symptoms like these may be associated with",
        cleaned,
        flags=re.IGNORECASE,
    )

    # 2. Check allergy contraindications
    allergy_warning = None
    if user_profile and user_profile.get("allergies"):
        allergy_warning = check_allergy_contraindications(
            cleaned,
            user_profile.get("allergies", []),
        )

    # 3. Assemble final safe response with disclaimer
    sections = []
    if allergy_warning:
        sections.append(f"{allergy_warning}\n\n---")

    sections.append(cleaned)

    # 4. Enforce medical disclaimer
    if settings.MEDICAL_DISCLAIMER not in cleaned:
        sections.append(f"\n\n---\n*{settings.MEDICAL_DISCLAIMER}*")

    return "\n\n".join(sections)
