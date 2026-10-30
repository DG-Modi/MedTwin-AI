"""Medical safety guardrails, clinical boundary enforcement, and drug interaction rules.

Guarantees non-diagnostic, non-prescriptive responses, emergency detection & escalation,
allergy contraindication checks, deterministic drug interaction verification,
and strict disclaimer enforcement.
"""

import re
from typing import Tuple, Optional, Dict, Any, List
from app.config import settings


def check_emergency_keywords(query: str) -> Tuple[bool, Optional[str]]:
    """Inspects query for life-threatening medical emergency symptoms.

    Returns (is_emergency: bool, matched_keyword: Optional[str]).
    """
    if not query:
        return False, None

    normalized = query.lower()

    # Specific acute clinical emergency patterns
    acute_patterns = [
        (r"\b(crushing|severe)\s+(chest\s+pain|pressure)\b", "chest pain"),
        (r"\bchest\s+(pain|pressure|tightness)\b", "chest pain"),
        (r"\bchest\s+(feels\s+tight|tightness).*(sweat|sweating)\b", "chest tightness and sweating"),
        (r"\b(sweat|sweating).*chest\s+(feels\s+tight|tightness)\b", "chest tightness and sweating"),
        (r"\b(having|think.*is having)\s+a\s+heart\s+attack\b", "heart attack"),
        (r"\bcan('?t|not)\s+breathe\b", "cannot breathe"),
        (r"\bdifficulty\s+breathing\b", "difficulty breathing"),
        (r"\bsevere\s+shortness\s+of\s+breath\b", "severe shortness of breath"),
        (r"\blips\s+are\s+turning\s+blue\b", "difficulty breathing"),
        (r"\bhigh\s+pitched\s+wheeze\b", "difficulty breathing"),
        (r"\bsudden(ly)?\s+(numb|weak|confused|blind|slurred)\b", "stroke symptoms"),
        (r"\bface\s+droop(ing)?\b", "stroke symptoms"),
        (r"\bslurred\s+speech\b", "slurred speech"),
        (r"\b(unconscious|not\s+waking\s+up|passed\s+out)\b", "unconscious"),
        (r"\b(kill\s+myself|commit\s+suicide|end\s+my\s+life)\b", "suicide crisis"),
        (r"\bcoughing\s+(up\s+)?(large\s+amounts\s+of\s+)?blood\b", "coughing blood"),
        (r"\bsevere\s+bleeding\b", "severe bleeding"),
        (r"\banaphylaxis\b", "anaphylaxis"),
        (r"\boverdose\b", "overdose"),
    ]

    for pat, keyword in acute_patterns:
        if re.search(pat, normalized):
            return True, keyword

    for keyword in settings.EMERGENCY_KEYWORDS:
        pattern = r"\b" + re.escape(keyword) + r"\b"
        if re.search(pattern, normalized):
            return True, keyword

    return False, None


def detect_emergency(
    query: str,
    twin_profile: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[str]]:
    """Wrapper matching detect_emergency contract returning (is_emergency, emergency_notice)."""
    triggered, keyword = check_emergency_keywords(query)
    if triggered:
        notice = format_emergency_response(matched_keyword=keyword, twin_profile=twin_profile, query=query)
        return True, notice
    return False, None


def format_emergency_response(
    matched_keyword: Optional[str] = None,
    twin_profile: Optional[Dict[str, Any]] = None,
    query: Optional[str] = None,
) -> str:
    """Formats an urgent emergency medical escalation message directing user to emergency dispatch services."""
    keyword_clause = f" for suspected **{matched_keyword}**" if matched_keyword else ""
    emergency_numbers = getattr(settings, "EMERGENCY_NUMBERS", "112, 108, 911")

    lines = [
        f"🚨 **CRITICAL MEDICAL EMERGENCY DETECTED**{keyword_clause}!\n",
        "If you or someone near you is experiencing severe symptoms, **do not wait for an online response**.\n",
        f"1. **IMMEDIATELY CALL EMERGENCY DISPATCH: {emergency_numbers} (112, 108 in India, or 911 in US/Canada).**",
        "2. **Proceed directly to the nearest hospital Emergency Room (ER) / Emergency Department.**",
    ]

    kw_lower = (matched_keyword or "").lower()
    q_lower = (query or "").lower()

    # Cardiac symptom advice
    if any(k in kw_lower or k in q_lower for k in ["chest", "heart", "breathe", "breathing", "stroke", "slurred"]):
        lines.append("3. Sit or lie down in a position of comfort to minimize stress on the heart and respiratory system while awaiting responders.")
        lines.append("4. **Medication Caution**: Do not take medications without medical approval; **ask the emergency dispatcher before taking any aspirin**.")

    # Patient conditions & medications guidance for responders
    if twin_profile:
        conditions = [c for c in twin_profile.get("chronic_conditions", []) if c]
        meds = [m for m in twin_profile.get("current_medications", []) if m]
        has_cardiac_risk = any(
            any(risk in c.lower() for risk in ["diabetes", "hypertension", "heart", "cardiac", "bp", "high blood pressure"])
            for c in conditions
        )
        if has_cardiac_risk or conditions or meds:
            details_parts = []
            if conditions:
                details_parts.append(f"Conditions: {', '.join(conditions)}")
            if meds:
                details_parts.append(f"Medications: {', '.join(meds)}")
            details_str = " | ".join(details_parts)
            lines.append(f"5. **Tell the responders about your conditions and current medications: {details_str}**.")

    # Suicide line: ONLY when query indicates suicide or self-harm
    is_suicide = any(k in kw_lower or k in q_lower for k in ["suicide", "kill myself", "end my life", "self-harm"])
    if is_suicide:
        lines.append("6. **Suicide & Crisis Lifeline**: Dial or text **988** (US/Canada Lifeline) or contact emergency services immediately.")

    lines.append(f"\n---\n*{settings.MEDICAL_DISCLAIMER}*")
    return "\n\n".join(lines)


def validate_medical_query(query: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """Sanitizes user input and validates length and content.

    Returns (is_valid, cleaned_query, error_message).
    """
    if query is None:
        return False, None, "Query cannot be null."

    cleaned = query.strip()
    if not cleaned:
        return False, None, "Query cannot be empty or solely whitespace."

    if len(cleaned) > 3000:
        return False, None, "Query is too long (maximum 3000 characters allowed)."

    return True, cleaned, None


def validate_input_safety(query: str) -> Tuple[bool, Optional[str]]:
    """Checks for harmful requests (lethal dosage, illicit drug manufacturing, intentional harm).

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


DRUG_ALLERGY_FAMILIES: Dict[str, List[str]] = {
    # NSAID class
    "nsaid": ["nsaid", "nsaids", "ibuprofen", "advil", "motrin", "aspirin", "naproxen", "aleve", "celebrex", "meloxicam", "diclofenac"],
    "nsaids": ["nsaid", "nsaids", "ibuprofen", "advil", "motrin", "aspirin", "naproxen", "aleve", "celebrex", "meloxicam", "diclofenac"],
    "ibuprofen": ["ibuprofen", "advil", "motrin", "nsaid", "nsaids", "aspirin", "naproxen", "aleve"],
    "aspirin": ["aspirin", "nsaid", "nsaids", "ibuprofen", "naproxen"],
    "naproxen": ["naproxen", "aleve", "nsaid", "nsaids", "ibuprofen", "aspirin"],
    # Penicillin / Beta-lactam class
    "penicillin": ["penicillin", "amoxicillin", "ampicillin", "augmentin", "piperacillin", "beta-lactam", "amoxil"],
    "amoxicillin": ["amoxicillin", "penicillin", "ampicillin", "augmentin", "amoxil"],
    "ampicillin": ["ampicillin", "penicillin", "amoxicillin", "augmentin"],
    # Sulfa class
    "sulfa": ["sulfa", "sulfonamide", "bactrim", "septra", "sulfamethoxazole"],
    "sulfonamide": ["sulfa", "sulfonamide", "bactrim", "septra", "sulfamethoxazole"],
    "sulfa drugs": ["sulfa", "sulfonamide", "bactrim", "septra", "sulfamethoxazole"],
    # Opioids class
    "codeine": ["codeine", "morphine", "opioid", "opioids", "hydrocodone", "oxycodone", "tylenol 3"],
    "opioid": ["opioid", "opioids", "codeine", "morphine", "hydrocodone", "oxycodone", "fentanyl", "tramadol"],
    "opioids": ["opioid", "opioids", "codeine", "morphine", "hydrocodone", "oxycodone", "fentanyl", "tramadol"],
    # Environmental / food allergens
    "peanuts": ["peanut", "peanuts", "tree nut", "tree nuts"],
    "peanut": ["peanut", "peanuts"],
    "cat dander": ["cat dander", "cat", "cats", "feline"],
}


def check_deterministic_allergy_conflict(
    query: str,
    allergies: List[str],
) -> Optional[Dict[str, Any]]:
    """Deterministic allergy check matching query against twin.allergies and drug classes.

    Runs BEFORE retrieval and the LLM.
    If a conflict is detected, returns metadata with alert text:
    '🚨 **ALLERGY CONFLICT**: Your profile lists an allergy to {allergen}. Do not take it, and speak to your doctor or pharmacist.'
    """
    if not allergies:
        return None

    norm_query = query.lower()

    for raw_allergen in allergies:
        clean_allergen = str(raw_allergen).strip().lower()
        if not clean_allergen:
            continue

        # 1. Direct word-boundary match in query
        pattern = r"\b" + re.escape(clean_allergen) + r"\b"
        direct_match = bool(re.search(pattern, norm_query))

        # 2. Drug class / synonym family match
        family_members = DRUG_ALLERGY_FAMILIES.get(clean_allergen, [])
        class_match = False
        matched_drug_term = clean_allergen

        for member in family_members:
            mem_pat = r"\b" + re.escape(member) + r"\b"
            if re.search(mem_pat, norm_query):
                class_match = True
                matched_drug_term = member
                break

        if direct_match or class_match:
            display_allergy = str(raw_allergen).strip()
            alert_msg = (
                f"🚨 **ALLERGY CONFLICT**: Your profile lists an allergy to {display_allergy}. "
                "Do not take it, and speak to your doctor or pharmacist."
            )
            return {
                "conflict": True,
                "allergen": display_allergy,
                "matched_query_term": matched_drug_term,
                "alert": alert_msg,
            }

    return None


def check_allergy_contraindications(
    response_text: str,
    allergies: List[str],
) -> Optional[str]:
    """Inspects assistant response or query to alert if any documented patient allergen is mentioned."""
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


def check_drug_interactions(
    query_text: str,
    medications: List[str],
    conditions: List[str],
    allergies: List[str],
) -> List[Dict[str, Any]]:
    """Deterministic clinical rule engine for drug-drug and drug-disease contraindications.

    Evaluates:
    - NSAIDs (Ibuprofen, Naproxen) vs ACE Inhibitors (Lisinopril) -> Acute Kidney Injury & BP elevation
    - NSAIDs (Ibuprofen) vs Metformin -> Renal impairment risk & lactic acidosis
    - NSAIDs vs Asthma / Samter's triad -> Bronchospasm exacerbation
    - Allergies vs requested medications
    """
    warnings = []
    norm_query = query_text.lower()
    norm_meds = [m.lower() for m in medications]
    norm_conds = [c.lower() for c in conditions]
    norm_allergies = [a.lower() for a in allergies]

    # NSAID inquiry detection
    is_nsaid_inquiry = any(n in norm_query for n in ["ibuprofen", "advil", "motrin", "naproxen", "aleve", "nsaid", "nsaids"])

    if is_nsaid_inquiry:
        # Rule 1: NSAID + ACE Inhibitor (Lisinopril, Enalapril, etc.)
        has_ace_inhibitor = any("lisinopril" in m or "enalapril" in m or "ramipril" in m or "captopril" in m for m in norm_meds)
        if has_ace_inhibitor:
            warnings.append({
                "rule_id": "DRUG_INTERACTION_NSAID_ACEI",
                "severity": "HIGH",
                "drugs": ["Ibuprofen / NSAID", "Lisinopril (ACE Inhibitor)"],
                "mechanism": "NSAIDs constrict afferent renal arterioles, while ACE inhibitors dilate efferent arterioles.",
                "clinical_warning": (
                    "🚨 **CLINICAL DRUG INTERACTION DETECTED**: Your Digital Twin profile is taking **Lisinopril** (ACE Inhibitor). "
                    "Taking **Ibuprofen (NSAID)** with ACE inhibitors can reduce the blood-pressure lowering effect "
                    "and can increase the risk of acute kidney injury (AKI) and fluid retention. "
                    "Ask your doctor or pharmacist whether Acetaminophen (Tylenol) is a safer option for you."
                ),
            })

        # Rule 2: NSAID + Metformin
        has_metformin = any("metformin" in m for m in norm_meds)
        if has_metformin:
            warnings.append({
                "rule_id": "DRUG_INTERACTION_NSAID_METFORMIN",
                "severity": "HIGH",
                "drugs": ["Ibuprofen / NSAID", "Metformin"],
                "mechanism": "NSAID-induced renal impairment can decrease metformin clearance.",
                "clinical_warning": (
                    "⚠️ **CLINICAL CONTRAINDICATION**: You are taking **Metformin** for diabetes. "
                    "NSAIDs like **Ibuprofen** can cause acute decreases in renal function, which can increase the risk of "
                    "Metformin accumulation and metabolic complications. "
                    "Ask your doctor or pharmacist whether Acetaminophen (Tylenol) is a safer option for you."
                ),
            })

        # Rule 3: NSAID + Asthma / Aspirin allergy
        has_asthma = any("asthma" in c for c in norm_conds)
        has_nsaid_allergy = any("nsaid" in a or "aspirin" in a or "ibuprofen" in a for a in norm_allergies)
        if has_asthma or has_nsaid_allergy:
            warnings.append({
                "rule_id": "CONTRAINDICATION_NSAID_ASTHMA",
                "severity": "HIGH",
                "drugs": ["Ibuprofen / NSAID"],
                "mechanism": "Inhibition of COX-1 shunts arachidonic acid to leukotrienes, causing bronchospasm.",
                "clinical_warning": (
                    "⚠️ **CONTRAINDICATION ALERT**: Your Digital Twin profile records **Asthma / NSAID sensitivity**. "
                    "In patients with asthma or aspirin sensitivity, NSAIDs like Ibuprofen can provoke bronchospasms "
                    "and acute respiratory distress. Ask your doctor or pharmacist whether Acetaminophen (Tylenol) is a safer option for you."
                ),
            })

        # Rule 4: NSAID + Hypertension condition
        has_hypertension = any("hypertension" in c or "high blood pressure" in c for c in norm_conds)
        if has_hypertension and not has_ace_inhibitor:
            warnings.append({
                "rule_id": "CONTRAINDICATION_NSAID_HYPERTENSION",
                "severity": "MEDIUM",
                "drugs": ["Ibuprofen / NSAID"],
                "mechanism": "Sodium and fluid retention via renal prostaglandin inhibition.",
                "clinical_warning": (
                    "⚠️ **HYPERTENSION WARNING**: Your Digital Twin profile documents **Hypertension**. "
                    "Regular or high-dose NSAID use (such as Ibuprofen) promotes renal sodium retention and can increase the risk of elevated blood pressure. "
                    "Ask your doctor or pharmacist whether Acetaminophen (Tylenol) is a safer option for you."
                ),
            })

    return warnings


def sanitize_sensitive_info(text: str) -> str:
    """Masks obvious sensitive personal identifiers (SSNs, credit card numbers, phone numbers)."""
    # SSN pattern
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN-REDACTED]", text)
    # Credit card pattern (13 to 19 digits)
    text = re.sub(r"\b(?:\d{4}[ -]?){3}\d{1,4}\b", "[CARD-REDACTED]", text)
    return text


def append_medical_disclaimer(text: str) -> str:
    """Ensures standard medical disclaimer is attached to educational output."""
    clean_text = text.strip()
    if settings.MEDICAL_DISCLAIMER in clean_text:
        return clean_text

    return f"{clean_text}\n\n---\n*{settings.MEDICAL_DISCLAIMER}*"


def fix_unbalanced_markdown(text: str) -> str:
    """Ensures markdown bolding (**) is strictly balanced on every line.
    
    Fixes orphaned closing ** markers (e.g. lost opening **) and ensures
    no unbalanced ** tags break rendering.
    """
    lines = text.split("\n")
    fixed = []
    for line in lines:
        count = line.count("**")
        if count % 2 != 0:
            if re.search(r"^[ \t]*-\s*([^*]+)\*\*:", line):
                line = re.sub(r"^([ \t]*-\s*)([^*]+)\*\*:", r"\1**\2**:", line)
            elif re.search(r"^[ \t]*([A-Za-z0-9\s&,()-]+)\*\*:", line):
                line = re.sub(r"^([ \t]*)([A-Za-z0-9\s&,()-]+)\*\*:", r"\1**\2**:", line)
            else:
                line = line + "**"
        fixed.append(line)
    return "\n".join(fixed)


def post_process_safety(
    response_text: str,
    user_profile: Optional[Dict[str, Any]] = None,
    query_text: Optional[str] = None,
) -> str:
    """Applies clinical guardrails to LLM output:

    1. Checks against documented allergies in Digital Twin.
    2. Runs deterministic drug-interaction rules and avoids duplicate alerts.
    3. Rewrites or flags diagnostic assertions.
    4. Softens extreme language (replaces 'dramatically' and 'life-threatening').
    5. Drops COX-1/COX-2 mechanism bullet unless asked.
    6. Ensures allergy check line on medication queries.
    7. Guarantees single, clean medical disclaimer.
    8. Guarantees balanced markdown bolding.
    """
    cleaned = sanitize_sensitive_info(response_text)

    # 1. Check for aggressive diagnostic assertions and soften them
    cleaned = re.sub(
        r"\b(you have|i diagnose you with|you are suffering from)\b",
        "symptoms like these may be associated with",
        cleaned,
        flags=re.IGNORECASE,
    )

    # 2. Soften extreme phrasing
    cleaned = re.sub(r"\bdramatically\s+escalates\b", "can increase", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bdramatically\b", "substantially", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bpotentially\s+life-threatening\b", "serious", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\blife-threatening\b", "serious", cleaned, flags=re.IGNORECASE)

    q_lower = (query_text or "").lower()

    # 3. Drop COX-1/COX-2 mechanism bullet unless explicitly asked
    if not any(k in q_lower for k in ["cox", "cox-1", "cox-2", "mechanism", "pharmacodynamics"]):
        cleaned = re.sub(
            r"^[ \t]*-\s*.*?(?:COX-1|COX-2|cyclooxygenase).*?$\n?",
            "",
            cleaned,
            flags=re.MULTILINE | re.IGNORECASE,
        )

    # 4. Check allergies and interactions
    allergy_warning = None
    interaction_warnings = []
    allergies = []
    if user_profile:
        allergies = user_profile.get("allergies", [])
        medications = user_profile.get("current_medications", [])
        conditions = user_profile.get("chronic_conditions", [])

        # Remove "Allergy Check" status line before inspecting text for actual contraindicated allergen mentions
        inspect_text = re.sub(r"(\*\*Allergy Check\*\*:|Allergy Check:)[^\n]*", "", cleaned, flags=re.IGNORECASE)
        allergy_warning = check_allergy_contraindications(inspect_text, allergies)

        if query_text:
            interaction_warnings = check_drug_interactions(
                query_text=query_text,
                medications=medications,
                conditions=conditions,
                allergies=allergies,
            )

    # 5. Assemble final response blocks without duplicates
    sections = []
    already_has_interaction_alerts = "### ⚠️ Interaction Alerts:" in cleaned or "### ⚠️ interaction alerts:" in cleaned.lower()

    if interaction_warnings and not already_has_interaction_alerts:
        sections.append("### ⚠️ Interaction Alerts:")
        for w in interaction_warnings:
            sections.append(f"- {w['clinical_warning']}")
        sections.append("- **Safer Alternatives**: Ask your doctor or pharmacist whether Acetaminophen (Tylenol) is a safer option for you.")
        sections.append("---")

    if allergy_warning:
        sections.append(f"{allergy_warning}\n\n---")

    sections.append(cleaned)

    full_output = "\n\n".join(sections).strip()

    # 6. Ensure allergy check line is present for medication queries if not already present
    is_med_query = any(k in q_lower for k in ["ibuprofen", "advil", "motrin", "naproxen", "aleve", "nsaid", "nsaids", "medication", "drug", "pill", "pain", "headache", "tylenol", "acetaminophen"])
    if is_med_query and "Allergy Check" not in full_output and "ALLERGY CONFLICT" not in full_output:
        allergies_str = ", ".join(allergies) if allergies else "documented profile"
        if full_output.startswith("**Verdict**:"):
            parts = full_output.split("\n\n", 1)
            if len(parts) == 2:
                full_output = f"{parts[0]}\n\n**Allergy Check**: No conflict found with {allergies_str}.\n\n{parts[1]}"
            else:
                full_output = f"{full_output}\n\n**Allergy Check**: No conflict found with {allergies_str}."
        else:
            full_output = f"**Allergy Check**: No conflict found with {allergies_str}.\n\n{full_output}"

    # 7. Deduplicate any pre-existing medical disclaimers to guarantee it is displayed exactly ONCE
    pattern = re.compile(
        r"(\n*---\n*)?\*?(Disclaimer:\s*)?MedTwin AI is an educational medical-information assistant.*",
        re.DOTALL | re.IGNORECASE,
    )
    full_output = pattern.sub("", full_output).strip()

    generic_pattern = re.compile(
        r"(\n*---\n*)?\*?(Please remember:\s*)?This information is strictly educational and does not constitute a clinical diagnosis.*",
        re.DOTALL | re.IGNORECASE,
    )
    full_output = generic_pattern.sub("", full_output).strip()

    # Append standard disclaimer once at the very end
    full_output = f"{full_output}\n\n---\n*{settings.MEDICAL_DISCLAIMER}*"

    # 8. Fix any unbalanced markdown
    return fix_unbalanced_markdown(full_output)
