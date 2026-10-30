"""Chain of Thought (CoT) Clinical Orchestration Pipeline for MedTwin AI.

Implements explicit, traceable stages as code:
  Stage 1: Intent Classification
  Stage 2: Red-Flag / Emergency Detection
  Stage 3: Digital Twin Context Injection
  Stage 4: Knowledge Base Retrieval & Citation Scoring
  Stage 5: Deterministic Safety & Drug Interaction Checks
  Stage 6: LLM Reasoning & Educational Answer Generation
  Stage 7: Disclaimer & Clinical Post-Processing

Produces structured, concise reasoning summaries for the user interface without leaking
raw internal LLM thinking.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session

from app.config import settings
from app.db.schemas import ReasoningStep, RetrievedSource
from app.core.guardrails import (
    check_emergency_keywords,
    format_emergency_response,
    validate_input_safety,
    check_drug_interactions,
    check_allergy_contraindications,
    check_deterministic_allergy_conflict,
    post_process_safety,
)
from app.core.digital_twin import get_profile, format_twin_context
from app.rag.pipeline import RAGPipeline, get_rag_pipeline
from app.rag.synthesizer import synthesize_grounded_response
from app.llm.ollama_client import OllamaClient, get_llm_client
from app.llm.prompts import build_medical_prompt

logger = logging.getLogger(__name__)


class ChainOfThoughtPipeline:
    """Executes the 7-stage clinical reasoning pipeline with telemetry and safety guarantees."""

    def __init__(
        self,
        rag_pipeline: Optional[RAGPipeline] = None,
        llm_client: Optional[OllamaClient] = None,
        enable_grounded_fallback: Optional[bool] = None,
    ) -> None:
        self.rag = rag_pipeline or get_rag_pipeline()
        self.llm = llm_client or get_llm_client()
        self.enable_grounded_fallback = (
            enable_grounded_fallback
            if enable_grounded_fallback is not None
            else settings.ENABLE_GROUNDED_RAG_SYNTHESIZER
        )

    async def execute(
        self,
        query: str,
        user_id: str,
        session_id: str,
        db: Session,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Executes all 7 reasoning stages in sequence."""
        t_start = time.time()
        steps: List[ReasoningStep] = []
        timings = {"retrieval_ms": 0.0, "llm_ms": 0.0, "total_ms": 0.0}

        # ------------------------------------------------------------------
        # STAGE 1: Intent Classification
        # ------------------------------------------------------------------
        t0 = time.time()
        intent = self._classify_intent(query)
        stage_intent = ReasoningStep(
            stage="intent_classification",
            name="1. Clinical Intent Classification",
            status="completed",
            summary=f"Detected Intent: **{intent['primary']}** (Confidence: {intent['confidence']:.0%}). Topic: {intent['topic']}.",
            details=intent,
        )
        steps.append(stage_intent)

        # ------------------------------------------------------------------
        # STAGE 3 PRE-LOAD: Digital Twin Context retrieval
        # ------------------------------------------------------------------
        profile = get_profile(db, user_id)
        profile_dict = profile.to_dict() if profile else {}

        # ------------------------------------------------------------------
        # STAGE 2: Red-Flag & Emergency Detection
        # ------------------------------------------------------------------
        t0 = time.time()
        is_emergency, emergency_keyword = check_emergency_keywords(query)
        if is_emergency:
            emergency_text = format_emergency_response(
                matched_keyword=emergency_keyword,
                twin_profile=profile_dict,
                query=query,
            )
            stage_emergency = ReasoningStep(
                stage="emergency_detection",
                name="2. Red-Flag & Emergency Escalation Check",
                status="flagged",
                summary=f"Red-flag check = TRIGGERED with matched keyword '{emergency_keyword}'. Emergency protocol activated: RAG skipped, LLM skipped.",
                details={"triggered": True, "matched_keyword": emergency_keyword, "rag_skipped": True, "llm_skipped": True},
            )
            steps.append(stage_emergency)
            timings["total_ms"] = round((time.time() - t_start) * 1000, 2)
            return {
                "answer": emergency_text,
                "sources": [],
                "is_emergency": True,
                "reasoning_steps": steps,
                "timings": timings,
            }

        # Check for illicit/harmful requests
        is_safe, refusal_msg = validate_input_safety(query)
        if not is_safe:
            stage_safety_refusal = ReasoningStep(
                stage="emergency_detection",
                name="2. Safety Policy Enforcement",
                status="refused",
                summary="Query violated safety policy regarding harmful or restricted substances.",
                details={"refusal": True},
            )
            steps.append(stage_safety_refusal)
            timings["total_ms"] = round((time.time() - t_start) * 1000, 2)
            return {
                "answer": refusal_msg or "I cannot assist with this request.",
                "sources": [],
                "is_emergency": False,
                "reasoning_steps": steps,
                "timings": timings,
            }

        steps.append(ReasoningStep(
            stage="emergency_detection",
            name="2. Red-Flag & Emergency Escalation",
            status="passed",
            summary="Zero acute emergency red flags identified. Safe to proceed with educational reasoning.",
            details={"is_emergency": False},
        ))

        # ------------------------------------------------------------------
        # STAGE 3: Digital Twin Context Injection
        # ------------------------------------------------------------------
        twin_context_text = format_twin_context(profile)

        conditions_str = ", ".join(profile_dict.get("chronic_conditions", [])) or "None reported"
        meds_str = ", ".join(profile_dict.get("current_medications", [])) or "None reported"
        allergies_str = ", ".join(profile_dict.get("allergies", [])) or "None reported"

        steps.append(ReasoningStep(
            stage="twin_injection",
            name="3. Patient Digital Twin Context Injection",
            status="completed",
            summary=f"Profile loaded for Patient ID '{user_id}'. Active Conditions: [{conditions_str}], Meds: [{meds_str}], Allergies: [{allergies_str}].",
            details={"profile": profile_dict},
        ))

        # ------------------------------------------------------------------
        # ALLERGY RULE: Deterministic check BEFORE Retrieval & LLM
        # ------------------------------------------------------------------
        allergy_conflict = check_deterministic_allergy_conflict(
            query=query,
            allergies=profile_dict.get("allergies", []),
        )
        if allergy_conflict:
            allergy_step = ReasoningStep(
                stage="allergy_check",
                name="Allergy check: CONFLICT",
                status="flagged",
                summary=(
                    f"ALLERGY CONFLICT DETECTED: Profile lists allergy to '{allergy_conflict['allergen']}' "
                    f"matching query term '{allergy_conflict['matched_query_term']}'. Immediate clinical warning enforced at top of response."
                ),
                details=allergy_conflict,
            )
        else:
            allergy_step = ReasoningStep(
                stage="allergy_check",
                name="Allergy check: no conflict",
                status="passed",
                summary="No allergy conflict detected between query and patient profile allergies.",
                details={"conflict": False},
            )
        steps.append(allergy_step)

        # ------------------------------------------------------------------
        # STAGE 4: Knowledge Base Retrieval & Citation Scoring
        # ------------------------------------------------------------------
        t_retrieval_0 = time.time()
        retrieved_chunks = self.rag.retrieve(query=query, top_k=settings.RAG_TOP_K)
        timings["retrieval_ms"] = round((time.time() - t_retrieval_0) * 1000, 2)

        sources: List[RetrievedSource] = []
        for c in retrieved_chunks:
            sources.append(RetrievedSource(
                source=c.get("source", "knowledge_base"),
                score=c.get("score", 0.0),
                text=c.get("text", ""),
                title=c.get("title", "Clinical Guide"),
                url=c.get("url", None),
            ))

        retrieval_summary = (
            f"Retrieved {len(sources)} relevant chunk(s) from ChromaDB vector store "
            f"(Top score: {sources[0].score:.0%})." if sources else "No high-confidence chunks retrieved (weak match fallback active)."
        )
        steps.append(ReasoningStep(
            stage="retrieval",
            name="4. Knowledge Base Retrieval & Citation Scoring",
            status="completed",
            summary=retrieval_summary,
            details={"chunks_retrieved": len(sources), "top_score": sources[0].score if sources else 0.0},
        ))

        rag_context_text = self.rag.format_rag_context(retrieved_chunks)

        # ------------------------------------------------------------------
        # STAGE 5: Deterministic Clinical Safety & Drug Interaction Checks
        # ------------------------------------------------------------------
        interaction_warnings = check_drug_interactions(
            query_text=query,
            medications=profile_dict.get("current_medications", []),
            conditions=profile_dict.get("chronic_conditions", []),
            allergies=profile_dict.get("allergies", []),
        )

        allergy_alerts = []
        if profile_dict.get("allergies"):
            alert = check_allergy_contraindications(query, profile_dict["allergies"])
            if alert:
                allergy_alerts.append(alert)

        if interaction_warnings or allergy_alerts:
            flag_count = len(interaction_warnings) + len(allergy_alerts)
            details_list = [w["rule_id"] for w in interaction_warnings]
            steps.append(ReasoningStep(
                stage="deterministic_safety",
                name="5. Deterministic Safety & Contraindication Engine",
                status="flagged",
                summary=f"FLAGGED {flag_count} clinical safety issue(s): {', '.join(details_list) if details_list else 'Allergy match'}. Mandatory warning injected.",
                details={"interactions": interaction_warnings, "allergies": allergy_alerts},
            ))
        else:
            steps.append(ReasoningStep(
                stage="deterministic_safety",
                name="5. Deterministic Safety & Contraindication Engine",
                status="passed",
                summary="Deterministic rule check verified no active contraindications between query and patient medications/allergies.",
                details={"interactions_found": 0},
            ))

        # ------------------------------------------------------------------
        # STAGE 6: LLM Reasoning & Educational Answer Generation
        # ------------------------------------------------------------------
        prompt = build_medical_prompt(
            query=query,
            rag_context=rag_context_text,
            twin_context=twin_context_text,
            chat_history=chat_history,
        )

        t_llm_0 = time.time()
        llm_failed = False
        used_grounded_engine = False
        try:
            raw_answer = await self.llm.generate(prompt=prompt)
        except Exception as exc:
            logger.error(f"Error during LLM generation: {exc}")
            if self.enable_grounded_fallback:
                logger.info("Local Ollama offline. Generating grounded clinical response from retrieved RAG guidelines and digital twin...")
                raw_answer = synthesize_grounded_response(
                    query=query,
                    retrieved_chunks=retrieved_chunks,
                    twin_profile=profile_dict,
                )
                used_grounded_engine = True
            else:
                raw_answer = (
                    "The local model is unavailable. Unable to generate an AI response at this time. "
                    "Please ensure the local Ollama LLM service is running and accessible."
                )
                llm_failed = True
        timings["llm_ms"] = round((time.time() - t_llm_0) * 1000, 2)

        model_str = getattr(self.llm, "model", settings.OLLAMA_MODEL) or settings.OLLAMA_MODEL
        if llm_failed:
            steps.append(ReasoningStep(
                stage="llm_generation",
                name="6. Structured Clinical Answer Synthesis",
                status="failed",
                summary=f"The local model is unavailable. Ollama service call failed ({timings['llm_ms']:.0f}ms).",
                details={"model": model_str, "status": "unavailable", "llm_ms": timings["llm_ms"]},
            ))
        elif used_grounded_engine:
            steps.append(ReasoningStep(
                stage="llm_generation",
                name="6. Structured Clinical Answer Synthesis",
                status="completed",
                summary=f"Synthesized evidence-grounded answer using Grounded Clinical RAG Engine (Ollama offline) ({timings['llm_ms']:.0f}ms).",
                details={"engine": "grounded_clinical_rag", "llm_ms": timings["llm_ms"]},
            ))
        else:
            steps.append(ReasoningStep(
                stage="llm_generation",
                name="6. Structured Clinical Answer Synthesis",
                status="completed",
                summary=f"Synthesized evidence-grounded educational answer using active model `{model_str}` ({timings['llm_ms']:.0f}ms).",
                details={"model": model_str, "llm_ms": timings["llm_ms"]},
            ))

        # ------------------------------------------------------------------
        # STAGE 7: Disclaimer & Clinical Post-Processing
        # ------------------------------------------------------------------
        safe_answer = post_process_safety(
            response_text=raw_answer,
            user_profile=profile_dict,
            query_text=query,
        )

        # Place deterministic allergy conflict alert at the TOP of the answer
        if allergy_conflict:
            safe_answer = f"{allergy_conflict['alert']}\n\n---\n\n{safe_answer}"

        steps.append(ReasoningStep(
            stage="disclaimer_postprocessing",
            name="7. Disclaimer & Clinical Post-Processing",
            status="completed",
            summary="Verified non-diagnostic boundaries, softened speculative language, and appended mandatory medical disclaimer.",
            details={"disclaimer_enforced": True},
        ))

        timings["total_ms"] = round((time.time() - t_start) * 1000, 2)

        return {
            "answer": safe_answer,
            "sources": sources,
            "is_emergency": False,
            "reasoning_steps": steps,
            "timings": timings,
        }

    def _classify_intent(self, query: str) -> Dict[str, Any]:
        """Classifies clinical intent deterministically based on key clinical semantics."""
        q = query.lower()
        if any(w in q for w in ["chest pain", "heart attack", "stroke", "cannot breathe", "suicide", "bleeding"]):
            return {"primary": "Emergency / Acute Symptom", "confidence": 0.98, "topic": "Acute Critical Care"}
        if any(w in q for w in ["ibuprofen", "advil", "aspirin", "lisinopril", "metformin", "medication", "pill", "dose", "drug"]):
            return {"primary": "Medication Safety & Pharmacology", "confidence": 0.95, "topic": "Drug Interactions & Dosing"}
        if any(w in q for w in ["diet", "food", "dash", "sodium", "salt", "nutrition", "exercise", "lifestyle"]):
            return {"primary": "Lifestyle & Nutritional Guidance", "confidence": 0.92, "topic": "Preventive Cardiology & Diet"}
        if any(w in q for w in ["hba1c", "blood pressure", "target", "range", "level", "goal", "reading"]):
            return {"primary": "Clinical Target & Biomarker Evaluation", "confidence": 0.94, "topic": "Disease Management Targets"}
        if any(w in q for w in ["symptom", "pain", "headache", "cough", "fever", "migraine", "asthma"]):
            return {"primary": "Symptom Inquiry & Differential Education", "confidence": 0.90, "topic": "General Medicine"}

        return {"primary": "General Medical Educational Inquiry", "confidence": 0.85, "topic": "Health Education"}


_global_cot_pipeline: Optional[ChainOfThoughtPipeline] = None


def get_cot_pipeline() -> ChainOfThoughtPipeline:
    """Singleton getter for the Chain of Thought pipeline."""
    global _global_cot_pipeline
    if _global_cot_pipeline is None:
        _global_cot_pipeline = ChainOfThoughtPipeline()
    return _global_cot_pipeline
