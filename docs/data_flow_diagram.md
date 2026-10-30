# Data Flow Diagram (DFD): MedTwin AI

## 1. Level-0 Context DFD
The Level-0 Context Diagram illustrates the interaction between external users and the MedTwin AI system boundaries.

```mermaid
graph LR
    User([Patient / Clinician]) -->|Voice WAV / Text Query| MedTwinAI[MedTwin AI Medical Assistant]
    MedTwinAI -->|Personalized Educational Response + Citations| User
    CDC_NIH([CDC / NIH / WHO Web Sources]) -->|Clinical Guidelines HTML/MD| MedTwinAI
    MedTwinAI -->|Audit Logs & Telemetry| SQLite[(Local SQLite / PostgreSQL Database)]
```

---

## 2. Level-1 Detailed Data Flow Diagram
The Level-1 DFD captures internal data transformations through each processing stage:

```mermaid
sequenceDiagram
    autonumber
    actor User as Patient User
    participant Frontend as Streamlit UI
    participant Backend as FastAPI Server
    participant Cache as Two-Layer Cache
    participant Safety as Guardrails & Drug Rules
    participant Chroma as ChromaDB Vector Store
    participant LLM as LLM Provider (Ollama / HuggingFace / Fallback)
    participant DB as SQLite DB

    User->>Frontend: Enters query or speaks into microphone
    opt Voice Recording
        Frontend->>Frontend: Whisper-tiny CPU transcription (Auto-submits to chat)
    end
    Frontend->>Backend: POST /api/chat (user_id, session_id, message)
    Backend->>Safety: check_emergency_keywords(message)
    alt Emergency Red Flag Detected (e.g. crushing chest pain)
        Safety-->>Backend: Immediate emergency protocol (112, 108, 911) + Responder brief with twin conditions
        Backend->>DB: Log emergency interaction in chat_messages & query_logs (Never cached)
        Backend-->>Frontend: Immediate emergency response (Bypasses LLM & RAG)
    else Routine Clinical Inquiry
        Backend->>DB: Fetch Patient Digital Twin context (allergies, meds, conditions)
        Backend->>Cache: Layer 1 Exact Match (SHA-256 patient isolation key)
        alt Cache HIT (Layer 1)
            Cache-->>Backend: Return cached answer (0ms)
            Backend->>DB: Record Cache HIT in query_logs
            Backend-->>Frontend: Return response ("⚡ Cached EXACT")
        else Cache MISS (Layer 1)
            Backend->>Cache: Layer 2 Semantic Match (Cosine Sim >= 0.92)
            alt Semantic HIT (Layer 2)
                Cache-->>Backend: Return semantically matching answer
                Backend->>DB: Record Semantic HIT in query_logs
                Backend-->>Frontend: Return response ("⚡ Cached SEMANTIC")
            else Full MISS
                Backend->>Safety: Deterministic Allergy Conflict Check
                alt Direct Allergy Conflict (e.g. Ibuprofen allergy)
                    Safety-->>Backend: Enforce ALLERGY CONFLICT alert at top of response
                end
                Backend->>Chroma: Similarity search with all-MiniLM-L6-v2 (top_k=3)
                Chroma-->>Backend: Top-k clinical chunks + similarity scores
                Backend->>Safety: Check Drug Interactions (NSAID vs Lisinopril/Metformin/Asthma)
                Backend->>LLM: generate(prompt) [Ollama -> Hugging Face -> Grounded Synthesizer]
                LLM-->>Backend: Grounded clinical response text
                Backend->>Safety: post_process_safety (soften claims, balanced markdown, single disclaimer)
                Backend->>Cache: Store non-emergency response in Layer 1 & index in Layer 2
                Backend->>DB: Persist chat turns & query telemetry log
                Backend-->>Frontend: Return structured response (answer, citations, CoT steps)
            end
        end
    end
    Frontend-->>User: Display answer, collapsible reasoning steps, citation links, and TTS Read Aloud
```
