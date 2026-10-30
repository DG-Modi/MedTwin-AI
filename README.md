# 🩺 MedTwin AI: Clinical Digital Twin & RAG-Powered Conversational Health Assistant

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Persistent-orange.svg)](https://www.trychroma.com/)
[![Tests](https://img.shields.io/badge/Pytest-87%20Passed%20(100%25)-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

> **Problem Statement 2: Medical-Domain RAG Chatbot MVP with Voice, Patient Digital Twins, Two-Layer Caching, and 7-Stage Chain of Thought Pipeline.**

MedTwin AI is an edge-first, open-source clinical knowledge assistant designed strictly for medical information support. It features patient-specific Digital Twin personalization, deterministic drug interaction verification, life-threatening emergency escalation, dense semantic vector retrieval, and speech-to-text dictation.

---

## 🌟 Key Architectural Features

1. **Authoritative Clinical Data Pipeline (`pipeline/`)**:
   - Automated ingestion of 8 authoritative clinical guidelines from the **CDC**, **NIH / MedlinePlus**, **WHO**, and **ADA**.
   - Covers: Type 2 Diabetes, Hypertension, NSAID & Ibuprofen Safety, Acute Chest Pain Emergency Signs, DASH Diet, HbA1c & Blood Pressure Targets, Asthma Management, and Migraine Evaluation.
   - Text cleaning, recursive chunking (500 chars, 100 overlap), dense embeddings with `all-MiniLM-L6-v2`, ChromaDB upsert, and SQLite `documents` logging.
   - Runnable via `python -m pipeline.run` or via the interactive UI "Run Ingestion" button.

2. **Patient Digital Twin Engine**:
   - Persistent SQLite storage for demographics, confirmed allergies, chronic conditions, active prescriptions, and lifestyle notes.
   - Includes 3 clinical case study presets (**Alex Morgan**, **Elena Rostova**, **Marcus Vance**).
   - Dynamic prompt injection calibrating educational explanations and contraindication checks.

3. **Explicit 7-Stage Chain of Thought (CoT) Pipeline**:
   Implemented strictly as traceable code across 7 sequential stages:
   - **Stage 1: Intent Classification**: Categorizes query (Emergency, Pharmacology, Lifestyle, Biomarker, Triage).
   - **Stage 2: Red-Flag & Emergency Escalation**: Intercepts life-threatening symptoms and directs immediately to 911/112 & ER.
   - **Stage 3: Digital Twin Context Injection**: Binds patient clinical background and risk profile.
   - **Stage 4: Knowledge Base Retrieval & Citation Scoring**: Searches ChromaDB with score reranking and source metadata.
   - **Stage 5: Deterministic Safety & Drug Interaction Engine**: Evaluates rule-based contraindications:
     - *Ibuprofen (NSAID) vs Lisinopril (ACE Inhibitor)*: Constriction of afferent + dilation of efferent arterioles -> Acute Kidney Injury (AKI) & BP elevation.
     - *Ibuprofen vs Metformin*: Renal clearance impairment -> Lactic acidosis risk.
     - *Ibuprofen vs Asthma*: Leukotriene shunting -> Severe bronchospasm (AERD).
   - **Stage 6: Structured LLM Reasoning**: Grounded prompt synthesis enforcing strict citations `[source.md]` with weak-retrieval handling.
   - **Stage 7: Disclaimer & Post-Processing**: Softens diagnostic phrasing and appends clean mandatory medical disclaimer.
   - Renders a collapsible **Reasoning steps** panel under each answer.

4. **Two-Layer Query Cache & Telemetry**:
   - **Layer 1 (Exact Match)**: Deterministic 64-character SHA-256 hash derived from normalized query + patient profile hash.
   - **Layer 2 (Semantic Match)**: Vector cosine similarity on query embeddings (threshold ~0.92) within the same patient context.
   - TTL expiration (3600s) and LRU eviction.
   - Comprehensive telemetry in SQLite `query_logs`: hit/miss counts, hit rate, average cached vs uncached latency, and per-stage timings.
   - Demonstrably returns second identical query in 0ms with `⚡ Cached` tag.

5. **Voice Input (Whisper STT) & Audio Playback (TTS)**:
   - Microphone audio recording with **OpenAI Whisper-tiny** CPU inference.
   - In-memory waveform resampler converting arbitrary audio payloads to 16kHz mono float32.
   - Optional auto-submit toggle ("Auto-send to chat when speech detected").
   - 1-click test simulated voice query buttons.
   - Audio Read-Aloud button (browser SpeechSynthesis) for accessibility.

6. **5-Tab Polished UI**:
   - `💬 Chat & Voice`: Consultation, CoT reasoning panel, citation cards, TTS audio, quick prompts.
   - `🔎 Knowledge Search`: Direct ChromaDB vector search with confidence scores and top-k slider.
   - `📚 Library`: Catalog of 8 ingested guidelines, source URLs, and "Run Ingestion Pipeline" button.
   - `📈 Analytics`: Real-time cache performance dashboard, latency comparisons, and recent query logs.
   - `📐 Docs`: Interactive Mermaid Architecture Diagram, Data Flow Diagram (DFD), Database ER Diagram, and Library Tradeoff Evaluation.

---

## 🚀 Quickstart Installation Guide

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- [Ollama](https://ollama.com/) (Optional: if Ollama is offline, MedTwin AI automatically uses its grounded fallback generator with 100% testability).

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/your-username/MedTwin-AI.git
cd MedTwin-AI

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Pull Ollama Model (Optional)
```bash
ollama pull llama3.2
# or
ollama pull llama3
```

### 3. Run Data Ingestion Pipeline
Scrapes guidelines from CDC, NIH, WHO, and ADA, builds ChromaDB vector index, and initializes SQLite tables:
```bash
python -m pipeline.run
```
*Expected output*: `Successfully ingested 8 documents (29 chunks) in ~11s`.

### 4. Run Automated Test Suite
Run the 87 automated unit and integration tests:
```bash
python run_tests.py
# or
pytest -v
```
*Expected output*: `87 passed in ~19s (100% passing)`.

### 5. Launch Application Services
Open two terminal windows:

**Terminal 1 — FastAPI Backend**:
```bash
uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```
- API Docs (Swagger): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health Status: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

**Terminal 2 — Streamlit Frontend**:
```bash
streamlit run streamlit_app.py
```
- Interactive Frontend: [http://localhost:8501](http://localhost:8501)

---

## ⏱️ 2-Minute Interview Demo Script

Follow this sequential walkthrough to demonstrate all required capabilities:

| Step | Action | What to Observe / Verify |
| :--- | :--- | :--- |
| **1. Digital Twin Context** | Select preset **Alex Morgan** from sidebar. Notice conditions (`Type 2 Diabetes`, `Hypertension`) and meds (`Metformin`, `Lisinopril`). | Expand **🔍 View Injected LLM Prompt Context** in sidebar to show personalized context formatting. |
| **2. Drug Interaction Safety** | Click suggested button: `💊 Can I take Ibuprofen with my meds?` | **Stage 5 Deterministic Safety Engine** flags `DRUG_INTERACTION_NSAID_ACEI` and `DRUG_INTERACTION_NSAID_METFORMIN`. The response explicitly warns against Ibuprofen due to Lisinopril/Metformin renal risk and suggests Acetaminophen. |
| **3. CoT Reasoning Panel** | Expand **🧠 Clinical Reasoning Steps (Chain of Thought)** under the response. | Displays explicit summaries for each stage: Intent, Emergency check passed, Twin injected, 3 chunks retrieved, Safety flagged, LLM synthesized, Disclaimer verified. |
| **4. Voice Dictation & STT** | Open **🎙️ Whisper Voice Input** expander. Click **🎙️ Load Simulated Voice Prompt** or record voice with mic. Click **🚀 Send Voice Query**. | Audio waveform is decoded in-memory and transcribed via Whisper-tiny CPU inference, automatically populating chat. Click **🔊 Read Aloud** to test browser audio synthesis. |
| **5. Life-Threatening Emergency** | Click button: `🚨 Crushing chest pain emergency`. | Normal LLM generation is **immediately bypassed**. A red emergency alert appears instructing user to call **911 / 112** and proceed to the nearest ER. Sources list is 0. |
| **6. Demonstrable Two-Layer Cache** | Submit the identical query again: `What are the DASH diet recommendations for blood pressure?` | Second query returns in **0ms** displaying badge `⚡ Cached Query (EXACT) (0ms)`. Open **📈 Analytics** tab to show live cache hit count increment and hit rate percentage. |
| **7. Pipeline & Ingestion** | Navigate to **📚 Library** tab. Click **📥 Run Ingestion Pipeline**. | Live spinner scrapes guidelines, chunks, embeds with `all-MiniLM-L6-v2`, upserts 29 chunks into ChromaDB, and logs ingestion timestamp in SQLite `documents` table. |

---

## 🗂️ Project Directory Structure

```
MedTwin-AI/
├── app/
│   ├── api/
│   │   ├── main.py              # FastAPI application entrypoint & middleware
│   │   └── routes.py            # Endpoints: /api/chat, /api/twin, /api/search, /api/documents, /api/cache
│   ├── core/
│   │   ├── cache.py             # Two-layer cache (Exact SHA-256 + Semantic ~0.92) & LRU eviction
│   │   ├── cot_pipeline.py      # Explicit 7-stage Chain of Thought reasoning pipeline
│   │   ├── digital_twin.py      # Patient profile manager & prompt context builder
│   │   └── guardrails.py        # Emergency keyword detector, drug interactions & disclaimer enforcement
│   ├── db/
│   │   ├── models.py            # SQLAlchemy models: UserProfile, Conversation, ChatMessage, Document, QueryLog
│   │   ├── schemas.py           # Pydantic validation schemas & request/response models
│   │   └── session.py           # Database engine & session generator
│   ├── frontend/
│   │   └── app.py               # Streamlit 5-tab responsive UI with dark theme
│   ├── llm/
│   │   ├── ollama_client.py     # Ollama HTTP client & grounded educational fallback
│   │   └── prompts.py           # System prompt templates & clinical instruction sets
│   ├── rag/
│   │   ├── chunking.py          # Recursive text chunking with semantic boundary preservation
│   │   ├── embeddings.py        # Sentence-transformers all-MiniLM-L6-v2 wrapper
│   │   ├── pipeline.py          # RAG retrieval & citation formatting
│   │   └── vector_store.py      # Persistent ChromaDB client with cosine similarity
│   └── services/
│       ├── chat_service.py      # Chat orchestration coordinating cache, CoT, and SQLite logging
│       └── voice_service.py     # Whisper-tiny STT & audio waveform resampler
├── data/
│   ├── chromadb/                # Persistent ChromaDB vector index
│   └── sample_docs/             # 8 Authoritative clinical guidelines (CDC, NIH, WHO, ADA)
├── docs/
│   ├── requirements.md          # Functional & non-functional requirements, scope decisions
│   ├── HLD.md                   # High-Level Architecture & component design
│   ├── LLD.md                   # Low-Level Design & method contracts
│   ├── data_flow_diagram.md     # Level-0 and Level-1 Mermaid Data Flow Diagrams (DFD)
│   ├── db_design.md             # Entity Relationship (ER) Diagram & schema specifications
│   ├── architecture_diagram.md  # System Architecture Mermaid diagram
│   ├── libraries.md             # Technology comparison matrix & trade-offs
│   └── plan_of_action.md        # 20-Hour implementation timeline & deliverables
├── pipeline/
│   ├── scraper.py               # Clinical guideline scraper (BeautifulSoup + Requests)
│   └── run.py                   # Data pipeline runner: scrape -> clean -> chunk -> embed -> SQLite
├── tests/                       # 87 Passing Automated Tests
│   ├── conftest.py              # Shared fixtures (in-memory DB, TestClient, mock LLM/RAG, audio WAV)
│   ├── test_api_integration.py  # End-to-end FastAPI endpoint integration tests
│   ├── test_cache_isolation.py  # Cache isolation, TTL expiration, and telemetry tests
│   ├── test_chat_pipeline.py    # Chat orchestration, CoT stages, and fallback tests
│   ├── test_clinical_safety.py  # Emergency escalation, disclaimer, and prompt injection tests
│   ├── test_config.py           # Configuration defaults and environment overrides
│   ├── test_digital_twin.py     # Digital Twin CRUD and context formatting tests
│   ├── test_guardrails.py       # Clinical guardrails and contraindication checks
│   ├── test_llm.py              # Ollama client and prompt synthesis tests
│   ├── test_rag.py              # Loader, chunker, vector store, and retrieval tests
│   └── test_voice_service.py    # Whisper-tiny STT and waveform resampling tests
├── docker-compose.yml           # Multi-container orchestration (FastAPI + Streamlit)
├── Dockerfile.backend           # Container image for FastAPI backend
├── Dockerfile.frontend          # Container image for Streamlit frontend
├── pytest.ini                   # Pytest configuration
├── run_tests.py                 # Convenience CLI test runner
└── requirements.txt             # Project Python dependencies
```

---

## ⚖️ Ethical & Clinical Boundaries
- **Educational Scope Only**: MedTwin AI is strictly an educational tool and **NOT** a diagnostic device.
- **Non-Prescriptive**: Does not prescribe medications or alter dosages.
- **Mandatory Disclaimer**: Every response enforces the verified clinical disclaimer.
- **Emergency Escalation**: Immediate guidance to dial 911/112 and proceed to nearest ER.
#   M e d T w i n - A I  
 