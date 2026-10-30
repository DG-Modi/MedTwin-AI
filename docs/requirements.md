# Requirements Document: MedTwin AI

## 1. Project Overview & Problem Statement
MedTwin AI is an open-source, edge-ready clinical conversational assistant designed for educational medical information support (Problem Statement 2). It combines **Retrieval-Augmented Generation (RAG)**, **Patient Digital Twins**, **Voice Input (Whisper)**, **Two-Layer Caching**, and an **Explicit 7-Stage Chain of Thought (CoT)** pipeline to deliver safe, evidence-grounded medical guidance.

---

## 2. Functional Requirements

### FR-1: Authoritative Clinical Data Pipeline
- Automated scraping and ingestion of 8 clinical guidelines from official authorities:
  - Centers for Disease Control and Prevention (CDC)
  - National Institutes of Health (NIH / MedlinePlus)
  - World Health Organization (WHO)
  - American Diabetes Association (ADA)
- Coverage: Type 2 Diabetes, Hypertension, NSAID & Ibuprofen Safety, Acute Chest Pain Emergency, DASH Diet, HbA1c & BP targets, Asthma Action Plan, Migraine Differential.
- Text cleaning, recursive chunking with 100-character overlap, dense embedding with `all-MiniLM-L6-v2`, ChromaDB upsert, and SQLite `documents` logging.
- Standalone execution via `python -m pipeline.run` and interactive UI execution via "Run Ingestion" button.

### FR-2: Patient Digital Twin Profile Manager
- Persistent storage of patient health parameters in SQLite:
  - Unique Patient ID, Name, Age, Gender
  - Confirmed Allergies (e.g., Penicillin, Peanuts, Sulfa)
  - Chronic Conditions (e.g., Type 2 Diabetes, Hypertension, Asthma)
  - Current Prescription & OTC Medications (e.g., Metformin 500mg BID, Lisinopril 10mg QD)
  - Lifestyle & Dietary Notes (e.g., DASH diet adherence, smoking status)
- Interactive presets for rapid evaluation (Alex Morgan, Elena Rostova, Marcus Vance).
- Context injection preview displaying active prompt modifications.

### FR-3: Two-Layer Query Cache & Telemetry
- **Layer 1 (Exact Match)**: Deterministic 64-character SHA-256 hash derived from normalized query + patient digital twin context.
- **Layer 2 (Semantic Match)**: Vector cosine similarity threshold (~0.92) comparing query embeddings within the same patient context.
- Configurable TTL (3600 seconds) and LRU eviction.
- Performance telemetry recorded in SQLite `query_logs`:
  - Hit/miss counts, hit ratio, active cache entries.
  - Average latency cached vs uncached.
  - Per-stage execution timings (retrieval ms, LLM ms, total latency).
- Second identical query demonstrably returns in 0ms tagged as `⚡ Cached`.

### FR-4: Explicit 7-Stage Chain of Thought (CoT) Pipeline
Execution pipeline strictly structured into discrete stages as code:
1. **Intent Classification**: Identifies clinical category (Emergency, Medication Safety, Nutrition, Biomarker, General).
2. **Red-Flag & Emergency Escalation**: Regex-driven acute symptom detector (crushing chest pain, slurred speech, dyspnea, suicide crisis).
3. **Digital Twin Context Injection**: Merges patient profile and highlights clinical risk factors.
4. **Knowledge Base Retrieval & Scoring**: Queries ChromaDB, calculates cosine similarity, formats citations.
5. **Deterministic Safety & Contraindications Engine**: Evaluates rule-based interactions (NSAID vs Lisinopril, NSAID vs Metformin, NSAID vs Asthma/Allergies).
6. **Structured LLM Generation**: Grounded reasoning prompt with citation requirements and insufficient-information handling.
7. **Disclaimer & Clinical Post-Processing**: Softens diagnostic phrasing and appends clean mandatory medical disclaimer.
- Renders collapsible **Reasoning steps** panel under each answer displaying concise stage summaries.

### FR-5: Voice Input & Speech-to-Text (STT)
- OpenAI Whisper (`whisper-tiny`) CPU inference.
- Direct microphone audio capture via Streamlit `st.audio_input`.
- In-memory resampler converting arbitrary sample rates and channels to 16kHz mono float32.
- 1-click test simulated voice query buttons.
- Auto-send to chat toggle upon speech detection.
- Audio Read-Aloud (browser SpeechSynthesis) for accessibility.

### FR-6: Knowledge Base Search & Library
- Direct vector similarity search with confidence score threshold and top-k slider.
- Complete clinical reference library displaying source documents, URLs, and chunk counts.

---

## 3. Non-Functional Requirements (NFRs)
- **NFR-1: Clinical Safety**: Zero prescriptive dosing, zero definitive disease diagnosis.
- **NFR-2: Privacy & Isolation**: SHA-256 cache keys incorporate patient profile hashes; Patient A's query cache is strictly isolated from Patient B.
- **NFR-3: Latency**: Cached queries resolve in `< 5ms`; local LLM responses within 2–5 seconds.
- **NFR-4: Portability & Offline Resilience**: Operates 100% offline using Ollama or embedded grounded fallback with zero required cloud API keys.
- **NFR-5: Test Coverage**: 80%+ test coverage across unit and integration suites.

---

## 4. Assumptions & Scope Decisions
- **Assumption 1**: MedTwin AI is strictly an **educational and informational assistant**, NOT a licensed software as a medical device (SaMD).
- **Assumption 2**: Edge deployment on modern laptop CPU/GPU is supported via lightweight models (`whisper-tiny` 39M, `all-MiniLM-L6-v2` 80MB, `llama3` 8B).
- **Scope Inclusion**: Focused clinical domains: Type 2 Diabetes, Hypertension, NSAID safety, Heart Attack emergencies, DASH nutrition.
- **Scope Exclusion**: Hospital EHR HL7/FHIR live integration and direct prescription submission are out of scope for this 20-hour MVP.

---

## 5. Deal-Breaker Features
1. **Emergency Escalation Failure**: Failure to immediately flag crushing chest pain and direct to 911/112 is unacceptable.
2. **Hallucinated Medical Advice**: Fabricating clinical facts or citing non-existent medical sources is prohibited.
3. **Cross-Patient Data Leakage**: Returning cached medical context of one patient to another with different clinical parameters is an absolute deal-breaker.
4. **Prescription Dosing**: Suggesting or adjusting specific medication dosages is prohibited.
