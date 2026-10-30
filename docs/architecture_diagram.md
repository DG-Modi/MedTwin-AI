# Architecture Diagram: MedTwin AI

## 1. System Architecture Diagram

```mermaid
graph TD
    subgraph Client["Presentation Layer (Port 8501)"]
        StreamlitApp["Streamlit Frontend (app/frontend/app.py)"]
        Microphone["Microphone Input (st.audio_input)"]
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
        Cache["Two-Layer Query Cache (app/core/cache.py)"]
    end

    subgraph Data["Persistence & Vector Intelligence"]
        Chroma[("ChromaDB Persistent Store (data/chromadb)")]
        SQLite[("SQLite Database (medtwin.db)")]
        Embeddings["all-MiniLM-L6-v2 (sentence-transformers)"]
    end

    subgraph Inference["Local LLM Inference Layer"]
        Whisper["OpenAI Whisper-tiny (CPU Speech-to-Text)"]
        Ollama["Ollama Daemon (Llama 3 / Llama 3.2 - Port 11434)"]
        GroundedMock["Grounded Clinical Fallback Generator"]
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
    CoT --> Ollama
    Ollama -.->|On Offline/Timeout| GroundedMock

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
1. **Edge-First & Fully Local**: No cloud API keys or SaaS dependencies required. Operates on standard workstation hardware.
2. **Deterministic Safety Over Speculation**: Red-flag symptoms and drug contraindications are intercepted by deterministic rule engines, eliminating LLM prompt-jailbreak vulnerability.
3. **Traceable Reasoning**: Each step of the 7-stage CoT pipeline is recorded and exposed to the user in a collapsible panel for clinical auditability.
4. **Context Isolation**: Multi-patient cache isolation prevents PHI leakage across consultation threads.
