# 🧪 MedTwin AI – Automated Testing & Verification Suite

This document outlines the automated testing framework, test suite architecture, and verification instructions for MedTwin AI.

---

## 📁 Test Suite Architecture

All automated test suites are located in the [`tests/`](tests/) directory and are configured via [`pytest.ini`](pytest.ini).

| Test Module | Category | Description |
| :--- | :--- | :--- |
| [`test_api_integration.py`](tests/test_api_integration.py) | **E2E API Integration** | Tests all 10 FastAPI endpoints (`/health`, `/api/ingest`, `/api/search`, `/api/twin`, `/api/twin/{id}`, `/api/twin/{id}/context`, `/api/chat`, `/api/cache/stats`, `/api/transcribe`). |
| [`test_llm_providers.py`](tests/test_llm_providers.py) | **LLM Provider Layer** | Tests provider interface, Ollama/HuggingFace selection, auto-fallback order, and honest error on offline services. |
| [`test_emergency_cleanup.py`](tests/test_emergency_cleanup.py) | **Emergency Protocol** | Tests acute emergency triggers (112, 108, 911), 988 isolation to suicide/self-harm, responder profile injection, and LLM bypass. |
| [`test_answer_quality.py`](tests/test_answer_quality.py) | **Answer Quality & Guardrails** | Tests markdown bolding balance, rule-based interaction merging, softened phrasing, allergy checks, and single disclaimer. |
| [`test_twin_cache_allergy_safety.py`](tests/test_twin_cache_allergy_safety.py) | **Twin Cache & Allergy Safety** | Tests patient cache isolation, twin update invalidation, deterministic allergy conflict alerts, and emergency cache bypass. |
| [`test_clinical_safety.py`](tests/test_clinical_safety.py) | **Clinical Safety & Guardrails** | Tests life-threatening emergency escalation (911/112), contraindications, disclaimer enforcement, and prompt injection resistance. |
| [`test_cache_isolation.py`](tests/test_cache_isolation.py) | **Cache & Privacy** | Tests SHA-256 patient isolation keys (preventing cross-patient data leaks), TTL expiration, LRU eviction, and hit/miss telemetry. |
| [`test_voice_service.py`](tests/test_voice_service.py) | **Voice & Whisper STT** | Tests audio waveform parsing, in-memory 16kHz mono resampling, OpenAI Whisper model transcription, silent audio rejection, and fallback simulation. |
| [`test_chat_pipeline.py`](tests/test_chat_pipeline.py) | **Chat Orchestration** | Tests multi-turn conversation threads, session persistence, RAG clinical attribution, and end-to-end response generation. |
| [`test_rag.py`](tests/test_rag.py) | **RAG & Vector DB** | Tests ChromaDB persistent store, sentence-transformers dense embeddings (`all-MiniLM-L6-v2`), document chunking, and similarity threshold filtering. |
| [`test_digital_twin.py`](tests/test_digital_twin.py) | **Digital Twin Engine** | Tests patient health profile schemas, SQLite CRUD operations, and personalized prompt context synthesis. |
| [`test_guardrails.py`](tests/test_guardrails.py) | **Safety Detection** | Unit tests for emergency keyword matching, query sanitization, and emergency advisory templates. |
| [`test_llm.py`](tests/test_llm.py) | **LLM & Inference** | Tests Ollama client communication, timeout handling, and deterministic clinical fallback generation. |
| [`test_config.py`](tests/test_config.py) | **Configuration** | Validates environment variable loading, default values, and pydantic settings. |
| [`conftest.py`](tests/conftest.py) | **Fixtures** | Shared fixtures: `client` (`TestClient(app)`), `test_db_session` (in-memory SQLite), `sample_synthetic_wav`, `sample_patient_profiles`, `fresh_cache`. |

---

## 🚀 How to Run the Tests

You can run the tests using either `pytest` directly or the included `run_tests.py` runner script:

### 1. Run the Entire Test Suite
```powershell
.\.venv\Scripts\python.exe -m pytest
```
*Or using the runner script:*
```powershell
.\.venv\Scripts\python.exe run_tests.py
```

### 2. Run Verbose Mode with Detailed Test Names
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

### 3. Run Specific Sub-Suites

#### API Integration Tests:
```powershell
.\.venv\Scripts\python.exe run_tests.py --api
# Or:
.\.venv\Scripts\python.exe -m pytest tests/test_api_integration.py -v
```

#### Clinical Safety & Emergency Guardrails Tests:
```powershell
.\.venv\Scripts\python.exe run_tests.py --safety
# Or:
.\.venv\Scripts\python.exe -m pytest tests/test_clinical_safety.py -v
```

#### Whisper Voice Input & Audio Tests:
```powershell
.\.venv\Scripts\python.exe run_tests.py --voice
# Or:
.\.venv\Scripts\python.exe -m pytest tests/test_voice_service.py -v
```

#### Chat Orchestration & RAG Pipeline Tests:
```powershell
.\.venv\Scripts\python.exe run_tests.py --chat
# Or:
.\.venv\Scripts\python.exe -m pytest tests/test_chat_pipeline.py -v
```

#### Cache & Patient Privacy Isolation Tests:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_cache_isolation.py -v
```

---

## 🛡️ Critical Clinical Safety Assertions Verified

1. **Patient Data Isolation**: `generate_cache_key(user_id="patient_A", query="...")` produces a completely different key than `patient_B`, preventing medical information leakage across users.
2. **Emergency Bypass**: Life-threatening symptoms (`severe chest pain`, `slurred speech`, `cannot breathe`, `suicide`) trigger immediate `is_emergency=True` responses directing the user to dial 911 / 112, short-circuiting downstream LLM generation.
3. **Mandatory Educational Disclaimer**: Every assistant response is verified to contain educational disclaimers stating that MedTwin AI is not a diagnostic medical doctor.
4. **Prompt Injection Resistance**: Prompt injections attempting to force prescription dosages or strip disclaimers are sanitized.
