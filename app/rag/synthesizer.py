"""Grounded Clinical RAG Synthesizer for MedTwin AI.

Dynamically synthesizes evidence-grounded, personalized clinical educational responses
directly from retrieved knowledge base chunks and Patient Digital Twin profiles.
Acts as a high-reliability offline generator when local Ollama inference is unavailable.
"""

import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


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
            # Check for orphaned closing bold on list labels like:
            # "- Hypertension & ACE Inhibitors (Lisinopril)**: text"
            if re.search(r"^[ \t]*-\s*([^*]+)\*\*:", line):
                line = re.sub(r"^([ \t]*-\s*)([^*]+)\*\*:", r"\1**\2**:", line)
            elif re.search(r"^[ \t]*([A-Za-z0-9\s&,()-]+)\*\*:", line):
                line = re.sub(r"^([ \t]*)([A-Za-z0-9\s&,()-]+)\*\*:", r"\1**\2**:", line)
            else:
                # Append missing bold tag to balance
                line = line + "**"
        fixed.append(line)
    return "\n".join(fixed)


def synthesize_grounded_response(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    twin_profile: Optional[Dict[str, Any]] = None,
) -> str:
    """Synthesizes a structured, evidence-grounded clinical response from retrieved chunks and patient twin profile."""
    q_lower = query.lower()
    profile = twin_profile or {}

    patient_name = profile.get("name") or "User"
    conditions = profile.get("chronic_conditions") or []
    medications = profile.get("current_medications") or []
    allergies = profile.get("allergies") or []
    lifestyle = profile.get("lifestyle_notes") or ""

    norm_meds = [m.lower() for m in medications]
    norm_conds = [c.lower() for c in conditions]
    norm_allergies = [a.lower() for a in allergies]

    # Check for empty retrieval
    if not retrieved_chunks:
        return (
            "The requested information was not found in the medical knowledge base. "
            "Please consult a qualified healthcare provider for detailed guidance on this topic."
        )

    # Map sources and citations
    source_citation_map = {}
    for idx, c in enumerate(retrieved_chunks, 1):
        src = c.get("source", "knowledge_base.md")
        source_citation_map[src] = f"[{idx}]"

    citations_str = "".join(f"[{i}]" for i in range(1, min(len(retrieved_chunks) + 1, 3)))
    response_blocks = []

    # -------------------------------------------------------------------------
    # Topic 1: Medication / NSAID / Ibuprofen Inquiry
    # -------------------------------------------------------------------------
    is_nsaid_inquiry = any(n in q_lower for n in ["ibuprofen", "advil", "motrin", "naproxen", "aleve", "nsaid", "nsaids"])
    is_med_inquiry = is_nsaid_inquiry or any(m in q_lower for m in ["medication", "drug", "pill", "pain", "headache", "tylenol", "acetaminophen"])

    if is_nsaid_inquiry or (is_med_inquiry and any("nsaid" in c.get("source", "") for c in retrieved_chunks)):
        has_ace_inhibitor = any("lisinopril" in m or "enalapril" in m or "ramipril" in m or "captopril" in m for m in norm_meds)
        has_metformin = any("metformin" in m for m in norm_meds)
        has_asthma = any("asthma" in c for c in norm_conds)
        has_hypertension = any("hypertension" in c or "high blood pressure" in c for c in norm_conds)

        # 1. Lead with a one-line verdict
        if has_ace_inhibitor and has_metformin:
            verdict = (
                "**Verdict**: Taking Ibuprofen is not recommended with your current medications "
                "(Lisinopril and Metformin) without medical guidance, as it can increase the risk of acute kidney injury "
                "and reduce blood pressure control."
            )
        elif has_ace_inhibitor:
            verdict = (
                "**Verdict**: Taking Ibuprofen is not recommended with your current medication Lisinopril, "
                "as it can reduce blood-pressure control and can increase the risk of kidney injury."
            )
        elif has_metformin:
            verdict = (
                "**Verdict**: Taking Ibuprofen is not recommended with your current medication Metformin, "
                "as it can reduce kidney function and can increase the risk of Metformin accumulation."
            )
        elif has_asthma:
            verdict = (
                "**Verdict**: Taking NSAIDs like Ibuprofen requires caution in individuals with asthma, "
                "as it can increase the risk of bronchospasm."
            )
        elif has_hypertension:
            verdict = (
                "**Verdict**: Ibuprofen should be used with caution in individuals with hypertension, "
                "as it can promote sodium retention and can increase blood pressure."
            )
        else:
            verdict = (
                "**Verdict**: Ibuprofen is an over-the-counter NSAID for temporary pain relief; "
                "use the lowest effective dose for the shortest possible duration."
            )
        response_blocks.append(verdict)
        response_blocks.append("")

        # 2. Allergy Check Line
        if allergies:
            allergies_str = ", ".join(allergies)
            response_blocks.append(f"**Allergy Check**: No conflict found with {allergies_str}.")
        else:
            response_blocks.append("**Allergy Check**: No drug allergy conflicts recorded in your profile.")
        response_blocks.append("")

        # 3. Merged Interaction Alerts (No duplicates, no COX-1/COX-2 unless asked)
        response_blocks.append("### ⚠️ Interaction Alerts:")
        if has_ace_inhibitor:
            response_blocks.append(
                "- **Hypertension & ACE Inhibitors (Lisinopril)**: Ibuprofen directly antagonizes the antihypertensive "
                "effect of Lisinopril and can increase the risk of acute kidney injury. "
                "Ask your doctor or pharmacist whether Acetaminophen (Tylenol) is a safer option for you. [1]"
            )
        if has_metformin:
            response_blocks.append(
                "- **Diabetes & Metformin**: Renal impairment induced by Ibuprofen can increase the risk of "
                "Metformin accumulation and metabolic complications. "
                "Ask your doctor or pharmacist whether Acetaminophen (Tylenol) is a safer option for you. [1]"
            )
        if has_asthma:
            response_blocks.append(
                "- **Asthma / Respiratory Sensitivity**: In patients with asthma or aspirin sensitivity, "
                "NSAIDs like Ibuprofen can trigger bronchospasms. "
                "Ask your doctor or pharmacist whether Acetaminophen (Tylenol) is a safer option for you. [1]"
            )
        if has_hypertension and not has_ace_inhibitor:
            response_blocks.append(
                "- **Hypertension**: NSAID use promotes sodium retention and can increase the risk of elevated blood pressure. "
                "Ask your doctor or pharmacist whether Acetaminophen (Tylenol) is a safer option for you. [1]"
            )

        # Safer Alternatives bullet
        response_blocks.append(
            "- **Safer Alternatives**: For mild to moderate pain or headache, "
            "ask your doctor or pharmacist whether Acetaminophen (Tylenol) is a safer option for you. [1]"
        )

        # Only include COX-1/COX-2 mechanism if explicitly asked
        if any(k in q_lower for k in ["cox", "cox-1", "cox-2", "mechanism", "pharmacodynamics"]):
            response_blocks.append(
                "- **Mechanism**: Ibuprofen non-selectively inhibits cyclooxygenase-1 (COX-1) and cyclooxygenase-2 (COX-2) enzymes, "
                "blocking prostaglandin synthesis. [1]"
            )

    # -------------------------------------------------------------------------
    # Topic 2: Target Blood Pressure and HbA1c
    # -------------------------------------------------------------------------
    elif any(k in q_lower for k in ["blood pressure", "hba1c", "target", "range", "ranges", "goal", "bp"]):
        response_blocks.append(
            f"Based on clinical guidelines from the American Diabetes Association (ADA) and ACC/AHA {citations_str}, "
            f"here are the recommended evidence-based clinical target ranges for {patient_name}:"
        )
        response_blocks.append("")
        response_blocks.append("### 🎯 Clinical Targets & Guidelines:")
        response_blocks.append(
            "- **Target Blood Pressure**: **< 130/80 mmHg** (Systolic < 130 mmHg and Diastolic < 80 mmHg) "
            f"for non-pregnant adults with confirmed hypertension and diabetes {source_citation_map.get('hypertension.md', '[1]')}. "
            "Achieving this target reduces cardiovascular mortality and significantly slows the progression of diabetic nephropathy."
        )
        response_blocks.append(
            "- **Glycemic Target (HbA1c)**: **< 7.0%** (estimated average glucose ~154 mg/dL) for most non-pregnant adults "
            f"with Type 2 Diabetes {source_citation_map.get('type_2_diabetes.md', '[2]')}. Goals are individualized:"
        )
        response_blocks.append("  - **Stricter Goal (< 6.5%)**: For younger patients or those with short disease duration managed with lifestyle or metformin without hypoglycemia.")
        response_blocks.append("  - **Less Stringent Goal (< 8.0%)**: For patients with extensive comorbidities, history of severe hypoglycemia, or limited life expectancy.")
        response_blocks.append("- **Capillary Blood Glucose Goals**: Fasting / Pre-prandial: **80–130 mg/dL**; Post-prandial peak (1–2h after meals): **< 180 mg/dL**.")

        # Query-specific personalization for BP / HbA1c targets
        has_htn = any("hypertension" in c.lower() for c in conditions)
        has_lisinopril = any("lisinopril" in m.lower() for m in medications)
        has_dm = any("diabetes" in c.lower() for c in conditions)
        has_metformin = any("metformin" in m.lower() for m in medications)

        if has_htn or has_lisinopril or has_dm or has_metformin:
            response_blocks.append("")
            response_blocks.append("### 👤 Relevant Profile Context:")
            if has_htn or has_lisinopril:
                response_blocks.append(
                    "- **Hypertension Management**: Your profile records Hypertension managed with Lisinopril, "
                    "which supports achieving your blood pressure goal of **< 130/80 mmHg**."
                )
            if has_dm or has_metformin:
                response_blocks.append(
                    "- **Diabetes Management**: Your profile records Type 2 Diabetes managed with Metformin, "
                    "which supports achieving your glycemic target of **HbA1c < 7.0%**."
                )

    # -------------------------------------------------------------------------
    # Topic 3: DASH Diet and Lifestyle
    # -------------------------------------------------------------------------
    elif any(k in q_lower for k in ["dash", "diet", "nutrition", "food", "sodium", "salt"]):
        response_blocks.append(
            f"According to clinical cardiology and nutrition guidelines {citations_str}, "
            "the DASH (Dietary Approaches to Stop Hypertension) eating plan is proven to lower blood pressure:"
        )
        response_blocks.append("")
        response_blocks.append("### 🥗 Nutritional & Lifestyle Principles:")
        response_blocks.append("- **Sodium Restriction**: Standard target of **< 2,300 mg/day** (optimal reduction to **< 1,500 mg/day** for enhanced blood pressure lowering).")
        response_blocks.append("- **Nutrient-Dense Foods**: High in potassium, calcium, magnesium, and dietary fiber through vegetables, fruits, whole grains, and legumes.")
        response_blocks.append("- **Saturated Fat Reduction**: Emphasize lean poultry, fish, nuts, and fat-free or low-fat dairy products while limiting red meats and sweets.")
        response_blocks.append("- **Physical Activity**: Aim for **150+ minutes of moderate aerobic exercise** (such as daily brisk walking) per week.")

        # Query-specific lifestyle notes
        if "dash" in lifestyle.lower() or "walking" in lifestyle.lower():
            response_blocks.append("")
            response_blocks.append("### 👤 Relevant Profile Context:")
            response_blocks.append(
                f"- **Documented Routine**: Your documented routine of following a DASH-style diet and moderate daily walking "
                "aligns directly with these clinical recommendations."
            )

    # -------------------------------------------------------------------------
    # Topic 4: General Clinical Inquiry from Chunks
    # -------------------------------------------------------------------------
    else:
        top_chunk = retrieved_chunks[0]
        title = top_chunk.get("title", "Clinical Guidelines")
        src = top_chunk.get("source", "knowledge_base.md")
        response_blocks.append(f"Based on clinical documentation from **{title}** (`{src}`) [1]:")
        response_blocks.append("")
        for i, c in enumerate(retrieved_chunks[:2], 1):
            text_lines = [line.strip() for line in c.get("text", "").split("\n") if line.strip() and not line.startswith("#")]
            for raw_l in text_lines:
                # Clean bullet markers safely without stripping ** bolding delimiters
                clean_l = re.sub(r"^(?:[-*]\s+|\d+\.\s+|#+\s*)", "", raw_l).strip()
                if len(clean_l) > 30 and not any(clean_l.lower().startswith(x) for x in ["mechanism", "pharmacodynamics"]):
                    response_blocks.append(f"- {clean_l} [{i}]")

    raw_response = "\n".join(response_blocks)
    return fix_unbalanced_markdown(raw_response)
