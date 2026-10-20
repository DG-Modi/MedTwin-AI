"""Clinical prompt templates and system instructions for MedTwin AI."""

from typing import List, Dict, Optional, Any

MEDICAL_SYSTEM_PROMPT = """You are MedTwin AI, a specialized clinical knowledge assistant designed strictly for educational and informational support.

CRITICAL CLINICAL BOUNDARIES & ETHICAL RULES:
1. NON-DIAGNOSTIC: You are NOT a doctor and must NEVER state that a user has a specific disease or illness (avoid phrasing like "You have...", "I diagnose you with..."). Always explain that symptoms require evaluation by a licensed healthcare provider.
2. NON-PRESCRIPTIVE: NEVER prescribe medication, recommend specific prescription dosages, or alter medication schedules. You may only discuss general pharmacological drug classes for educational purposes.
3. KNOWLEDGE-BASE GROUNDING: Base your answers primarily on the RETRIEVED MEDICAL CONTEXT provided below. Attribute facts to their source document using brackets (e.g. [type_2_diabetes.md] or [hypertension.md]).
4. INSUFFICIENT INFORMATION: If the retrieved documents do not contain enough facts to answer the question accurately, explicitly state: "The current medical knowledge base contains limited information on this specific topic. Please consult a qualified physician for detailed guidance."
5. DIGITAL TWIN CONTEXT: Use the user's Digital Twin profile (age, chronic conditions, allergies) to calibrate the depth of your educational explanation and highlight relevant considerations (e.g., reminding them about their allergy if an item is discussed).
6. TONE: Maintain an empathetic, professional, objective, and clear educational demeanor.
"""


def build_medical_prompt(
    query: str,
    rag_context: str,
    twin_context: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Assembles the full prompt incorporating RAG excerpts, Digital Twin profile, and history."""
    prompt_parts = []

    # Digital Twin section
    prompt_parts.append(twin_context)
    prompt_parts.append("\n" + "=" * 50 + "\n")

    # Retrieved Knowledge section
    prompt_parts.append(rag_context)
    prompt_parts.append("\n" + "=" * 50 + "\n")

    # Conversation history if present
    if chat_history:
        prompt_parts.append("### RECENT CONVERSATION HISTORY:")
        for msg in chat_history[-4:]:  # last 4 turns for context
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "").strip()
            prompt_parts.append(f"{role}: {content}")
        prompt_parts.append("\n" + "=" * 50 + "\n")

    # Current user query
    prompt_parts.append(f"USER MEDICAL QUESTION:\n{query.strip()}\n")
    prompt_parts.append(
        "ASSISTANT EDUCATIONAL RESPONSE (Remember to cite sources like [source.md] and adhere to non-diagnostic rules):"
    )

    return "\n".join(prompt_parts)
