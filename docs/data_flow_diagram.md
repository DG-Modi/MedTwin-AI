# Data Flow Diagram (DFD): MedTwin AI

## 1. Level-0 Context DFD
The Level-0 Context Diagram illustrates the interaction between external users and the MedTwin AI system boundaries.

```mermaid
graph LR
    User([Patient / Clinician]) -->|Voice WAV / Text Query| MedTwinAI[MedTwin AI Medical Assistant]
    MedTwinAI -->|Personalized Educational Response + Citations| User
    CDC_NIH([CDC / NIH / WHO Web Sources]) -->|Clinical Guidelines HTML/MD| MedTwinAI
    MedTwinAI -->|Audit Logs & Telemetry| SQLite[(Local SQLite Database)]
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
    participant Ollama as Ollama Llama 3
    participant DB as SQLite DB

    User->>Frontend: Enters query or records voice via mic
    opt Voice Recording
        Frontend->>Frontend: Whisper-tiny audio transcription (CPU)
    end
    Frontend->>Backend: POST /api/chat (user_id, session_id, message)
    Backend->>Safety: detect_emergency(message)
    alt Emergency Red Flag Detected (e.g. crushing chest pain)
        Safety-->>Backend: Urgent 911 / 112 notification + ER guidance
        Backend->>DB: Log emergency interaction in chat_messages & query_logs
        Backend-->>Frontend: Immediate emergency response (Bypasses LLM inference)
    else Non-Emergency Inquiry
        Backend->>DB: Fetch Patient Digital Twin context (allergies, meds, conditions)
        Backend->>Cache: Layer 1 Exact Match (SHA-256)
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
                Backend->>Chroma: Similarity search with all-MiniLM-L6-v2 (top_k=3)
                Chroma-->>Backend: Top-k clinical chunks + similarity scores
                Backend->>Safety: Check Drug Interactions (NSAID vs Lisinopril/Metformin/Asthma)
                Backend->>Ollama: POST /api/generate (RAG excerpts + Twin Context + Query)
                Ollama-->>Backend: Grounded clinical response text
                Backend->>Safety: post_process_safety (Softens diagnosis + appends disclaimer)
                Backend->>Cache: Store response in Layer 1 & index in Layer 2
                Backend->>DB: Persist chat turns & query telemetry log
                Backend-->>Frontend: Return structured response (answer, citations, CoT steps)
            end
        end
    end
    Frontend-->>User: Display answer, collapsible reasoning steps, and citation links
```
