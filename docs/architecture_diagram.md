# Architecture Diagram: MedTwin AI

## 1. System Architecture Diagram

```mermaid
graph TD
    subgraph Client["Presentation Layer (Port 8501)"]
        StreamlitApp["Streamlit Frontend (app/frontend/app.py)"]
        Microphone["Microphone Input (st.audio_input - Visible by Default)"]
        TTS["Audio Read-Aloud (HTML5 SpeechSynthesis)"]
    end

    subgraph API["Application Layer (Port 8000)"]
        FastAPI["FastAPI REST Server (app/api/main.py)"]
        Router["API Routers (app/api/routes.py)"]
        ChatSrv["ChatService (app/services/chat_service.py)"]
        VoiceSrv["VoiceService (app/services/voice_service.py)"]
    end

    subgraph Core["Clinical Intelligence & Guardrails"]
        CoT["7-Stage Chain of Thought Pipeline (app/core/cot_pipeline.py)"]
        Guardrails["Clinical Safety & Guardrails (app/core/guardrails.py)"]
        TwinMgr["Digital Twin Engine (app/core/digital_twin.py)"]
        Cache["Two-Layer Query Cache with Invalidation (app/core/cache.py)"]
    end

    subgraph Data["Persistence & Vector Intelligence"]
        Chroma[("ChromaDB Persistent Store (data/chromadb)")]
        SQLite[("SQLite / PostgreSQL Database (medtwin.db)")]
        Embeddings["all-MiniLM-L6-v2 (sentence-transformers)"]
    end

    subgraph Inference["Pluggable LLM & Voice Inference Layer"]
        Whisper["OpenAI Whisper-tiny (CPU Speech-to-Text)"]
        LLMFactory["LLM Provider Layer (app/llm/factory.py)"]
        Ollama["OllamaProvider (llama3.2 - Local Port 11434)"]
        HF["HuggingFaceProvider (HF Inference API via httpx)"]
        GroundedRAG["Grounded Clinical RAG Synthesizer (app/rag/synthesizer.py)"]
    end

    subgraph Ingestion["Automated Data Pipeline"]
        PipelineRun["Pipeline Runner (pipeline/run.py)"]
        Scraper["Clinical Web Scraper (pipeline/scraper.py)"]
        CDC_NIH["Authoritative Sources (CDC, NIH, WHO, ADA)"]
    end

    StreamlitApp <-->|HTTP / REST JSON| FastAPI
    Microphone --> StreamlitApp
    TTS <-- StreamlitApp

    FastAPI --> Router
    Router --> ChatSrv
    Router --> VoiceSrv

    ChatSrv --> Cache
    ChatSrv --> CoT
    CoT --> Guardrails
    CoT --> TwinMgr
    CoT --> Embeddings
    CoT --> Chroma
    CoT --> LLMFactory
    LLMFactory --> Ollama
    LLMFactory --> HF
    LLMFactory -.->|On LLM Unavailable / Offline| GroundedRAG

    VoiceSrv --> Whisper

    TwinMgr --> SQLite
    ChatSrv --> SQLite

    Scraper --> CDC_NIH
    PipelineRun --> Scraper
    PipelineRun --> Chroma
    PipelineRun --> SQLite
```

---

## 2. Design Principles
1. **Edge-First & Pluggable Cloud Hybrid**: Operates 100% offline via local Ollama (`llama3.2`) or seamlessly deploys to cloud platforms (Streamlit Cloud) via the Hugging Face Inference API.
2. **Deterministic Safety Over Speculation**: Red-flag symptoms and drug contraindications are intercepted by deterministic rule engines with configurable numbers (112, 108, 911), eliminating LLM jailbreak vulnerabilities.
3. **Traceable Reasoning**: Each step of the 7-stage CoT pipeline is recorded and exposed to the user in a collapsible panel for clinical auditability.
4. **Context Isolation**: Multi-patient cache isolation keys bind the patient ID and twin version, preventing PHI leakage and immediately invalidating stale cache on profile saves.
