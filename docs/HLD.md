# High-Level Design (HLD): MedTwin AI

## 1. System Overview
MedTwin AI is an edge-first, open-source clinical RAG conversational assistant. It provides a modular, multi-tier architecture designed for high maintainability, strict clinical boundary enforcement, and deterministic performance.

```mermaid
graph TD
    subgraph Frontend["Presentation Tier (Streamlit)"]
        UI_Chat["Chat & Voice Consultation"]
        UI_KB["Knowledge Base Search"]
        UI_Lib["Clinical Document Library"]
        UI_Analytics["Cache & Telemetry Analytics"]
        UI_Docs["Architecture & System Documentation"]
        Sidebar["Patient Digital Twin Manager"]
    end

    subgraph Backend["Application Server Tier (FastAPI)"]
        API_Routes["FastAPI Routers (/api/chat, /api/twin, /api/search, /api/cache)"]
        Guardrails["Pre-Execution Safety & Emergency Engine"]
        CacheMgr["Two-Layer Cache Manager (Exact + Semantic)"]
        CoTEngine["7-Stage Chain of Thought Orchestrator"]
        VoiceEngine["Whisper Audio Transcription Service"]
    end

    subgraph DataTier["Data & Model Storage Tier"]
        ChromaStore[("ChromaDB Vector Store (all-MiniLM-L6-v2)")]
        SQLiteDB[("SQLite Relational Database (SQLAlchemy 2.0)")]
        OllamaLLM["Ollama Daemon (Llama 3.x / Llama 3.2)"]
    end

    Frontend -->|HTTP / JSON| Backend
    Backend --> DataTier
```

---

## 2. Component Deconstruction

### 2.1 Presentation Tier (Streamlit)
- **Framework**: Streamlit with custom CSS design tokens and dark mode styling.
- **Twin Manager Sidebar**: Allows interactive customization or 1-click loading of validated patient clinical profiles (Alex Morgan, Elena Rostova, Marcus Vance).
- **5 Core Workspaces**:
  1. **Chat & Voice**: Real-time consultation, collapsible 7-stage CoT reasoning panel, citation cards, Whisper mic recording, audio playback.
  2. **Knowledge Search**: Direct semantic search against ChromaDB with confidence scores.
  3. **Library**: Catalog of 8 official clinical guidelines with "Run Ingestion" button.
  4. **Analytics**: Real-time cache telemetry, hit/miss metrics, latency comparisons, and recent query logs.
  5. **Docs**: Embedded Mermaid diagrams, data flow models, and technology evaluations.

### 2.2 Application Tier (FastAPI)
- **ASGI Server**: Uvicorn with asynchronous endpoints.
- **Pydantic Validation**: Strict schema enforcement for queries, twin profiles, and responses.
- **Dependency Injection**: Modular provision of database sessions, cache instances, voice services, and RAG pipelines.
- **Two-Layer Cache**:
  - *Layer 1*: Exact SHA-256 hash match on `normalized_query + patient_profile_hash`.
  - *Layer 2*: Semantic vector similarity on embeddings with cosine threshold ~0.92 within the same patient context.
- **7-Stage CoT Pipeline**:
  - Stage 1: Intent Classification
  - Stage 2: Emergency / Red-Flag Detection
  - Stage 3: Digital Twin Profile Injection
  - Stage 4: ChromaDB Vector Retrieval & Scoring
  - Stage 5: Deterministic Safety Rules (NSAID vs Lisinopril/Metformin/Asthma)
  - Stage 6: Grounded LLM Answer Synthesis
  - Stage 7: Disclaimer & Clinical Post-Processing

### 2.3 Data & Model Storage Tier
- **ChromaDB**: Embedded persistent vector store utilizing `hnsw:space: cosine` indexing.
- **Sentence-Transformers**: `all-MiniLM-L6-v2` generating 384-dimensional dense embeddings in <15ms.
- **SQLite Database**: ACID persistence across 6 normalized tables (`user_profiles`, `conversations`, `chat_messages`, `documents`, `cache_entries`, `query_logs`).
- **Ollama Engine**: Local HTTP inference on port 11434 serving Llama 3 open-source weights.
