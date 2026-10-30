# Database Design (ER): MedTwin AI

## 1. Relational Database Overview
MedTwin AI utilizes **SQLite** orchestrated through **SQLAlchemy 2.0 ORM**. SQLite provides embedded, zero-configuration ACID persistence without external server dependencies, guaranteeing 100% local operation and clinical privacy.

```mermaid
erDiagram
    user_profiles ||--o{ conversations : "owns"
    conversations ||--o{ chat_messages : "contains"
    user_profiles ||--o{ query_logs : "telemetry"

    user_profiles {
        int id PK "Auto-incrementing ID"
        string user_id UK "Unique Patient Identifier (e.g. patient_001)"
        string name "Patient full name"
        int age "Age in years (1 to 120)"
        string gender "Biological sex or gender identity"
        json allergies "Confirmed allergens list"
        json chronic_conditions "Diagnosed chronic conditions list"
        json current_medications "Active prescriptions & dosages list"
        text lifestyle_notes "Dietary adherence, exercise, smoking"
        json clinical_tags "Risk categorization tags"
        datetime created_at "Record creation UTC"
        datetime updated_at "Last modification UTC"
    }

    conversations {
        int id PK "Session record ID"
        string session_id UK "Session token (e.g. session_a1b2c3d4)"
        string user_id FK "References user_profiles.user_id"
        string title "Consultation summary heading"
        datetime created_at "Conversation initialization UTC"
    }

    chat_messages {
        int id PK "Message ID"
        int conversation_id FK "References conversations.id"
        string sender "user | assistant | system"
        text content "Message body text"
        json sources "Attributed RAG citations and scores"
        json reasoning_steps "7-Stage CoT execution summaries"
        boolean is_emergency "Emergency flag (True/False)"
        boolean cached "Cached flag (True/False)"
        float latency_ms "End-to-end response time in ms"
        datetime created_at "Message timestamp UTC"
    }

    documents {
        int id PK "Document ID"
        string source UK "Filename identifier (e.g. type_2_diabetes.md)"
        string title "Official clinical guide title"
        string url "CDC/NIH/WHO Web URL"
        string category "Clinical specialty (Endocrinology, Cardiology)"
        int chunk_count "Total chunks indexed in ChromaDB"
        datetime ingested_at "Ingestion timestamp UTC"
    }

    cache_entries {
        int id PK "Cache record ID"
        string cache_key UK "Deterministic SHA-256 digest"
        string cache_type "exact | semantic"
        text query_text "Normalized medical question"
        string patient_id "Patient context isolation key"
        text answer "Clinical response text"
        json sources "Retrieved sources metadata"
        json reasoning_steps "CoT stage summaries"
        float expires_at "Epoch expiration timestamp"
        datetime created_at "Cache creation UTC"
    }

    query_logs {
        int id PK "Log ID"
        string session_id "Session token"
        string patient_id "Patient ID"
        text query_text "Query string"
        boolean cache_hit "True if served from cache"
        string cache_type "exact | semantic | null"
        float total_latency_ms "Total response time ms"
        float retrieval_ms "Chroma retrieval time ms"
        float llm_ms "Ollama generation time ms"
        boolean is_emergency "True if red flag triggered"
        string model_name "Active LLM model name"
        datetime created_at "Log creation UTC"
    }
```

---

## 2. Table Schemas & Indexing Strategy

### 2.1 Table: `user_profiles`
- **Primary Key**: `id` (INTEGER, AUTOINCREMENT)
- **Unique Index**: `user_id` (VARCHAR(64), UNIQUE)
- **Clinical Privacy**: Background medical history stored in structured JSON columns (`allergies`, `chronic_conditions`, `current_medications`).

### 2.2 Table: `conversations`
- **Primary Key**: `id` (INTEGER, AUTOINCREMENT)
- **Unique Index**: `session_id` (VARCHAR(64), UNIQUE)
- **Foreign Key**: `user_id` -> `user_profiles.user_id` with `ON DELETE CASCADE`.

### 2.3 Table: `chat_messages`
- **Primary Key**: `id` (INTEGER, AUTOINCREMENT)
- **Foreign Key**: `conversation_id` -> `conversations.id` with `ON DELETE CASCADE`.
- **JSON Metadata**: `sources` and `reasoning_steps` stored as structured JSON arrays.

### 2.4 Table: `documents`
- **Primary Key**: `id` (INTEGER, AUTOINCREMENT)
- **Unique Index**: `source` (VARCHAR(255), UNIQUE)
- **Audit Columns**: `url`, `category`, `chunk_count`, `ingested_at`.

### 2.5 Table: `query_logs`
- **Primary Key**: `id` (INTEGER, AUTOINCREMENT)
- **Indexes**: `session_id`, `patient_id` for fast telemetry filtering.
- **Metrics**: `total_latency_ms`, `retrieval_ms`, `llm_ms`, `cache_hit`, `is_emergency`.
