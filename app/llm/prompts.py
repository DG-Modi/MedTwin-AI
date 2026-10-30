"""Clinical prompt templates and system instructions for MedTwin AI."""

from typing import List, Dict, Optional, Any

MEDICAL_SYSTEM_PROMPT = """You are MedTwin AI, a specialized clinical knowledge assistant designed strictly for educational and informational support.

SYSTEM RULES & CLINICAL BOUNDARIES:
1. EDUCATIONAL ONLY: Provide educational information only. This is NOT medical diagnosis or prescription. Never state that the user has a specific condition or prescribe/change medication doses.
2. ANSWER ONLY FROM CONTEXT: Answer ONLY using the facts present in the provided retrieved knowledge base context. Do NOT speculate, extrapolate, or use outside unverified clinical knowledge.
3. MISSING INFORMATION: If the requested information is missing or cannot be answered strictly from the provided context, state explicitly: "not found in knowledge base".
4. CONCISE & SPECIFIC: Provide a clear, concise answer with exact numbers, clinical ranges, and metrics from the context (e.g., blood pressure targets like < 130/80 mmHg, HbA1c targets like < 7%).
5. INLINE CITATIONS: Use inline bracket citations like [1], [2] corresponding directly to the source IDs provided in the context.
6. DIGITAL TWIN PERSONALIZATION: Directly reference the patient's Digital Twin profile details (e.g., documented conditions such as Type 2 Diabetes and Hypertension, current medications such as Metformin and Lisinopril, and lifestyle factors like the DASH diet and walking) and explain specifically how they relate to the target ranges.
7. TONE: Maintain an empathetic, professional, objective, and clear educational demeanor.
"""


def build_medical_prompt(
    query: str,
    rag_context: str,
    twin_context: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    system_rules: Optional[str] = None,
) -> str:
    """Assembles the full prompt:
    1. System rules (educational only, no diagnosis/prescription, answer ONLY from context, 'not found in knowledge base' if missing)
    2. Patient Digital Twin context (name, age, conditions, meds, allergies, lifestyle)
    3. Top-k retrieved chunk TEXT with [n] source ids
    4. User question
    Instructions: Ask for a concise answer with specific numbers from context, inline citations like [1],
    and explaining how the twin's conditions/medications/lifestyle relate to the targets.
    """
    parts = []

    # 1. System Rules
    rules = system_rules or MEDICAL_SYSTEM_PROMPT
    parts.append("### CLINICAL SYSTEM RULES & BOUNDARIES:\n" + rules.strip())
    parts.append("\n" + "=" * 60 + "\n")

    # 2. Patient Digital Twin Context
    parts.append("### PATIENT DIGITAL TWIN CONTEXT:\n" + twin_context.strip())
    parts.append("\n" + "=" * 60 + "\n")

    # 3. Retrieved Knowledge Base Context (top-k chunk text with [n] source IDs)
    parts.append("### RETRIEVED KNOWLEDGE BASE CONTEXT (TOP-K CHUNKS WITH [n] SOURCE IDS):\n" + rag_context.strip())
    parts.append("\n" + "=" * 60 + "\n")

    # Conversation history if present
    if chat_history:
        parts.append("### RECENT CONVERSATION HISTORY:")
        for msg in chat_history[-4:]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "").strip()
            parts.append(f"{role}: {content}")
        parts.append("\n" + "=" * 60 + "\n")

    # 4. User Question & Instructions
    parts.append(f"USER MEDICAL QUESTION:\n{query.strip()}\n")
    parts.append(
        "INSTRUCTIONS FOR YOUR RESPONSE:\n"
        "- Answer ONLY using facts from the retrieved knowledge base context above. If missing, say 'not found in knowledge base'.\n"
        "- Provide a concise answer with specific numbers and clinical ranges (e.g. blood pressure < 130/80 mmHg, HbA1c < 7%).\n"
        "- Include inline citations like [1], [2] corresponding to the source IDs above.\n"
        "- Directly reference the patient's Digital Twin profile (their conditions like Type 2 Diabetes and Hypertension, medications like Metformin and Lisinopril, and lifestyle factors like the DASH diet and walking) and explain how their regimen and lifestyle relate to achieving these targets.\n"
        "- Educational only: No medical diagnosis or prescription.\n\n"
        "ASSISTANT EDUCATIONAL RESPONSE:"
    )

    return "\n".join(parts)
