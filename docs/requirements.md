# Requirements Document: MedTwin AI

## 1. Project Overview & Problem Statement
MedTwin AI is an open-source, edge-first clinical conversational assistant designed for educational medical information support (Problem Statement 2). It combines **Retrieval-Augmented Generation (RAG)**, **Patient Digital Twins**, **Voice Input (Whisper)**, **Two-Layer Caching**, a **Pluggable Multi-Provider LLM Layer (Ollama & Hugging Face)**, and an **Explicit 7-Stage Chain of Thought (CoT)** pipeline to deliver safe, evidence-grounded medical guidance.

---

## 2. Explicit Assumptions

1. **Educational & Non-Diagnostic Scope**: MedTwin AI is strictly an **educational and informational health assistant** and does NOT function as a licensed Software as a Medical Device (SaMD), clinical diagnostic engine, or prescription prescriber.
2. **Open-Source Weight Philosophy**: The system prioritizes open-weights models (`llama3.2` 3B / `Mistral-7B`, `whisper-tiny` 39M, `all-MiniLM-L6-v2` 80MB) to preserve clinical transparency, reproducibility, and data privacy.
3. **Pluggable Deployment Environment**: The application runs either completely offline on local hardware via Ollama or in cloud-hosted environments (Streamlit Cloud) using serverless inference via the Hugging Face Inference API.
4. **Standard CPU Baseline**: Core retrieval, vector matching, SQLite persistence, and speech-to-text inference are designed to execute smoothly on standard multi-core consumer CPUs without requiring a dedicated discrete GPU.
5. **Patient Responsibility**: The user is responsible for accurately entering their digital twin parameters (allergies, chronic conditions, and medications); the system operates deterministically on the provided record.

---

## 3. Scope Definition

### ✅ In-Scope (Delivered Features)
* **Authoritative Clinical Data Ingestion**: Automated ingestion and recursive chunking of guidelines from CDC, NIH MedlinePlus, WHO, and ADA into ChromaDB with cosine vector indexing.
* **Patient Digital Twin Engine**: Persistent CRUD storage of demographics, allergies, chronic conditions, medications, and lifestyle notes with dynamic prompt context binding.
* **Pluggable LLM Provider Layer**: Unified `LLMProvider` interface with `OllamaProvider`, `HuggingFaceProvider`, and auto-fallback chain, guaranteeing honest retrieval-only excerpts if all LLMs are offline.
* **Deterministic Clinical Guardrails**: Hardcoded drug-drug and drug-disease interaction engine (NSAIDs vs Lisinopril, Metformin, Asthma, and Hypertension) with softened language ("can increase the risk") and allergy check lines.
* **Acute Emergency Protocol**: Immediate escalation for chest pain, dyspnea, stroke, and hemorrhage with configurable hotlines (112, 108, 911), 988 lifeline isolation, responder briefing with twin conditions, and LLM bypass.
* **Two-Layer Cache Engine**: Layer 1 Exact SHA-256 (0ms) + Layer 2 Semantic Match (cosine similarity >= 0.92) with strict patient isolation and automatic invalidation on twin updates.
* **Voice Speech-to-Text**: Built-in Whisper-tiny microphone recorder running on CPU with auto-submit to chat and full-text Read Aloud TTS.
* **Interactive UI & Observability**: Streamlit frontend with 5 shortened tabs ("Chat & Voice", "Knowledge Search", "Library", "Analytics", "Docs"), startup self-check diagnostics, reasoning steps expanders, and telemetry analytics.
* **Comprehensive Automated Tests**: 121+ unit and integration tests with pytest-cov code coverage.

### ❌ Out-of-Scope (Explicit Non-Goals)
* **Direct Clinical Prescriptions**: Modifying prescription dosages, calculating custom milligram regimens, or ordering prescription refills.
* **Hospital EHR / EMR Live Integration**: Direct bidirectional HL7/FHIR EHR interfaces with hospital electronic medical records.
* **Telehealth Video Consultations**: WebRTC live video doctor-patient calls.
* **Medical Device Telemetry Streaming**: Real-time Bluetooth / IoT streaming from ECG monitors or continuous glucose monitors (CGMs).

---

## 4. Deal-Breakers (Zero-Tolerance Safety Constraints)

1. **Emergency Escalation Failure**: Failure to immediately intercept acute life-threatening symptoms (e.g. crushing chest pain, slurred speech, dyspnea, suicide crisis) and direct the user to emergency services (112, 108, 911) is an **absolute deal-breaker**. Emergency responses must **never** be cached and must **bypass LLM generation**.
2. **Hallucinated or Fabricated Medical Advice**: Inventing clinical facts, recommending unverified home remedies, or citing fake medical guidelines. The system must fail honestly with retrieved excerpts rather than return a fake template.
3. **Cross-Patient Data Leakage**: Serving one patient's cached query, medications, or health background to another patient is a **critical violation of patient isolation**. Cache keys must incorporate the patient ID and twin version.
4. **Allergy & Contraindication Neglect**: Advising a user to take a medication that directly conflicts with a documented allergy (e.g., ibuprofen when allergic to ibuprofen) or active prescription (e.g., ibuprofen with Lisinopril) without high-priority warnings and safer alternatives.
5. **Prescriptive Assertions & Unsoftened Language**: Asserting definitive clinical diagnoses (e.g. "I diagnose you with...") or using extreme hyperbolic phrasing ("dramatically escalates", "life-threatening") instead of evidence-grounded risk communication ("can increase the risk").
6. **Missing Medical Disclaimer**: Delivering educational advice without the mandatory disclaimer identifying MedTwin AI as an educational assistant and directing the user to consult a physician.

---

## 5. Functional Requirements (FR) Summary

| Requirement ID | Component | Description |
| :--- | :--- | :--- |
| **FR-1** | Clinical Data Pipeline | Scraping, recursive chunking, dense embeddings, and ChromaDB persistence for 8 official CDC/NIH/WHO guidelines. |
| **FR-2** | Digital Twin Manager | Persistent SQLite CRUD for patient health profiles with version tracking and context synthesis. |
| **FR-3** | Multi-Provider LLM | Pluggable `LLMProvider` interface supporting Ollama (`llama3.2`), Hugging Face Inference API, and offline Grounded RAG. |
| **FR-4** | Deterministic Safety | Rule-based engine detecting drug-drug and drug-disease contraindications with alternatives phrased for physician verification. |
| **FR-5** | Emergency Protocol | Regex-driven acute red-flag interceptor directing users to 112, 108, 911, and nearest emergency room. |
| **FR-6** | Two-Layer Cache | Dual exact SHA-256 and semantic vector cache with patient isolation and auto-invalidation on twin save. |
| **FR-7** | Voice STT & TTS | Local Whisper-tiny speech-to-text audio ingestion with auto-submission and HTML5 speech synthesis. |
| **FR-8** | Traceable CoT | 7-stage chain of thought pipeline rendering collapsible reasoning steps with confidence scores and source citations. |

---

## 6. Non-Functional Requirements (NFR)

* **NFR-1 (Clinical Safety)**: Strict non-diagnostic, non-prescriptive phrasing enforced via post-processing guardrails.
* **NFR-2 (Privacy & Isolation)**: SHA-256 patient isolation keys prevent cross-patient data sharing.
* **NFR-3 (Performance & Latency)**: Cached queries resolve in `< 5ms`; local LLM inference under 5s; Whisper-tiny CPU transcription under 1.5s.
* **NFR-4 (Portability & Cloud Readiness)**: Dual execution paths: 100% offline edge execution via Ollama or cloud deployment via Streamlit Cloud and Hugging Face.
* **NFR-5 (Test Coverage & Hygiene)**: Automated test suite with 80%+ coverage across core clinical services and guardrails.
