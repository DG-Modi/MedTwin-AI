# 🩺 MedTwin AI: Clinical Digital Twin & Conversational Health Assistant

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg)](https://streamlit.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17%20Local-336791.svg)](https://www.postgresql.org/)
[![SQLite](https://img.shields.io/badge/SQLite-Integrated-003B57.svg)](https://www.sqlite.org/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange.svg)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **"What if your AI medical assistant actually remembered your health background and actively protected you from dangerous drug interactions?"**

**MedTwin AI** is an intelligent, edge-first clinical knowledge assistant. It combines **authoritative medical guidelines** (from the CDC, WHO, and NIH) with a personalized **Patient Digital Twin** to deliver safe, educational answers to your health and medication questions—via text or voice.

---

## 📖 Table of Contents
1. [🌟 What is MedTwin AI? (In Plain English)](#-what-is-medtwin-ai-in-plain-english)
2. [💡 The Problem It Solves](#-the-problem-it-solves)
3. [🧠 How Does It Work? The 3 Core Pillars](#-how-does-it-work-the-3-core-pillars)
4. [🔄 The 7-Stage Chain of Thought (CoT) Engine](#-the-7-stage-chain-of-thought-cot-engine)
5. [🗄️ Database Architecture (SQLite & PostgreSQL)](#️-database-architecture-sqlite--postgresql)
6. [🎙️ Voice Dictation (Whisper Speech-to-Text)](#️-voice-dictation-whisper-speech-to-text)
7. [⚡ Two-Layer Fast Caching (0ms Repetitive Queries)](#-two-layer-fast-caching-0ms-repetitive-queries)
8. [🚀 Step-by-Step Installation Guide (For Anyone)](#-step-by-step-installation-guide-for-anyone)
9. [💻 How to Run the Project](#-how-to-run-the-project)
10. [🧪 Automated Testing](#-automated-testing)
11. [⚠️ Safety & Ethical Boundaries](#️-safety--ethical-boundaries)

---

## 🌟 What is MedTwin AI? (In Plain English)

Imagine you visit a doctor or pharmacist. Before they advise you on taking a medicine, they always ask:
- *"Are you allergic to anything?"*
- *"Do you have conditions like high blood pressure or asthma?"*
- *"What other pills are you currently taking?"*

Normal AI chatbots (like standard ChatGPT) **do not remember this**. If you ask them *"Can I take Ibuprofen for a headache?"*, they will usually say *"Yes, Ibuprofen relieves headaches."* 

**However, if you take blood pressure medicine (like Lisinopril), Ibuprofen can trigger acute kidney failure and dangerous blood pressure spikes!**

**MedTwin AI fixes this:**
1. You create your **Digital Twin** (your virtual health profile: allergies, conditions, and active prescriptions).
2. It stores your profile securely in your **own laptop's database** (SQLite or PostgreSQL).
3. Whenever you ask a question, it automatically cross-checks your unique health profile and official medical guidelines to ensure the advice is **safe for you**.

---

## 💡 The Problem It Solves

| Ordinary AI Chatbots | 🩺 MedTwin AI |
| :--- | :--- |
| ❌ Has no memory of your medical history | ✅ Remembers your allergies, medications & conditions |
| ❌ Can hallucinate or invent medical treatments | ✅ Grounded strictly in CDC, NIH, ADA, and WHO clinical guides |
| ❌ Might miss life-threatening drug conflicts | ✅ Hardcoded Drug Interaction Safety Engine blocks dangerous combos |
| ❌ Gives slow answers every time | ✅ Two-layer smart cache answers identical questions in **0ms** |
| ❌ Might give diagnostic false hope during heart attacks | ✅ Instantly detects emergencies and directs you to 911 / 112 & the ER |

---

## 🧠 How Does It Work? The 3 Core Pillars

### 1. The Patient Digital Twin (Your Health Memory)
A "Digital Twin" is simply a digital version of your medical background. It securely tracks:
* **Demographics**: Name, Age, Gender
* **Allergies**: Penicillin, Peanuts, Sulfa, Aspirin, etc.
* **Chronic Conditions**: Type 2 Diabetes, Hypertension, Asthma, Migraines
* **Current Medications**: Lisinopril, Metformin, Albuterol inhaler, etc.
* **Lifestyle Notes**: Diet, exercise habits, smoking status

### 2. RAG Knowledge Base (Retrieval-Augmented Generation)
Instead of relying on guesswork, MedTwin AI searches a local vector database (**ChromaDB**) holding authoritative clinical guidelines from:
* 🏛️ **CDC** (Centers for Disease Control and Prevention)
* 🏛️ **NIH / MedlinePlus** (National Institutes of Health)
* 🏛️ **WHO** (World Health Organization)
* 🏛️ **ADA** (American Diabetes Association)

### 3. Deterministic Clinical Safety Engine
AI models can sometimes be unpredictable. MedTwin AI uses **hardcoded clinical rules** that run *before* the AI is even allowed to respond:
* **Ibuprofen + Lisinopril**: Detected! Warns of Acute Kidney Injury (AKI) and blood pressure spikes.
* **Ibuprofen + Metformin**: Detected! Warns of impaired renal drug clearance and lactic acidosis.
* **Ibuprofen + Asthma**: Detected! Warns of severe bronchospasms (Aspirin-Exacerbated Respiratory Disease).
* **Chest Pain / Stroke Keywords**: Detected! Instantly triggers emergency escalation protocol.

---

## 🔄 The 7-Stage Chain of Thought (CoT) Engine

Every single time a user asks a question, it passes through 7 traceable stages:

```mermaid
flowchart TD
    A["👤 User asks query (Voice or Text)"] --> B["Stage 1: Intent Classification"]
    B --> C{"Stage 2: Emergency Red-Flag Check"}
    C -- "🚨 Life-Threatening (Chest Pain, Stroke)" --> D["Direct Emergency Alert (Call 911 / ER)"]
    C -- "Routine Health Query" --> E["Stage 3: Digital Twin Injection"]
    E --> F["Stage 4: Search CDC/WHO Guidelines (ChromaDB)"]
    F --> G["Stage 5: Rule-Based Drug Interaction Engine"]
    G --> H["Stage 6: Grounded AI Answer Synthesis"]
    H --> I["Stage 7: Non-Diagnostic Disclaimer & Sources"]
    I --> J["💬 Safe Answer Delivered with Citations & TTS Audio"]
```

1. **Stage 1: Intent Classification** — Determines what you are asking (Pharmacology, Emergency, Lifestyle, or Symptoms).
2. **Stage 2: Red-Flag Emergency Guardrail** — If you say *"severe chest pain"* or *"trouble breathing"*, it bypasses everything and immediately tells you to call emergency services.
3. **Stage 3: Digital Twin Binding** — Pulls your specific health background from the local database.
4. **Stage 4: Knowledge Retrieval** — Finds the top scientific medical guidelines matching your question.
5. **Stage 5: Drug Interaction Verification** — Validates drug safety against your current medications.
6. **Stage 6: Grounded Reasoning** — Synthesizes a personalized, clear educational answer with citations `[source.md]`.
7. **Stage 7: Disclaimer & Safety Boundaries** — Ensures the phrasing is educational (not diagnosing) and attaches mandatory disclaimers.

---

## 🗄️ Database Architecture (SQLite & PostgreSQL)

MedTwin AI is designed to support **both** beginner and enterprise database systems:

```
             ┌──────────────────────────────────────────────┐
             │            MedTwin AI Application            │
             └──────────────────────┬───────────────────────┘
                                    │ SQLAlchemy ORM
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
       ┌─────────────────────────┐     ┌─────────────────────────┐
       │   Path 1: SQLite (.db)  │     │   Path 2: PostgreSQL    │
       │   - Single local file   │     │   - Enterprise Server   │
       │   - Zero installation   │     │   - Installed on laptop │
       │   - Instant development │     │   - High concurrency    │
       └─────────────────────────┘     └─────────────────────────┘
```

### What Information is Stored in the Database?
* 👤 **`user_profiles`**: Patient name, age, allergies, chronic conditions, medications, and lifestyle notes.
* 💬 **`conversations`**: Chat session history linked to the specific patient.
* ✉️ **`chat_messages`**: Every question and AI answer, complete with citation sources and latency timestamps.
* 📚 **`documents`**: Ingested medical guideline articles (source titles, URLs, categories, chunk counts).
* 📈 **`query_logs`**: System telemetry (response times, cache hits vs. misses, emergency flags).

---

## 🎙️ Voice Dictation (Whisper Speech-to-Text)

Don't want to type? MedTwin AI includes a built-in microphone recorder powered by **OpenAI Whisper-tiny** running entirely locally on your CPU.
* Speak your symptoms into your microphone.
* Whisper transcribes your voice in real time.
* Toggle *"Auto-send to chat upon transcription"* for hands-free medical consultations.
* Click **🔊 Read Aloud** on any assistant answer to hear the response read back to you!

---

## ⚡ Two-Layer Fast Caching (0ms Repetitive Queries)

If a patient asks the same question twice, why wait for the AI to re-think?
* **Layer 1 (Exact Match)**: Computes a SHA-256 fingerprint of the query + patient profile. If asked again, returns the answer in **0 milliseconds**.
* **Layer 2 (Semantic Match)**: Compares the vector cosine similarity (threshold >= 0.92) to match conceptually identical questions.
* In the UI, cached answers display a proud **`⚡ Cached`** badge!

---

## 🚀 Step-by-Step Installation Guide (For Anyone)

### Prerequisites
* **Windows, macOS, or Linux**
* **Python 3.10+** (Tested on Python 3.12)
* **Git** installed on your laptop

### 1. Clone the Project
Open PowerShell or your terminal and run:
```bash
git clone https://github.com/DG-Modi/MedTwin-AI.git
cd MedTwin-AI
```

### 2. Create and Activate a Virtual Environment
```powershell
# Windows:
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Packages
```bash
pip install -r requirements.txt
pip install "psycopg[binary]"
```

### 4. Setup Database Connection (`.env`)
Create a `.env` file in the main folder (`D:\MedTwin-AI\.env`):

* **To use SQLite (Easiest - Ready instantly):**
  ```env
  DATABASE_URL=sqlite:///./medtwin.db
  ```

* **To use PostgreSQL (If installed on your laptop):**
  ```env
  DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/medtwin_db
  ```

### 5. Initialize the Database Tables
Run this one-liner to generate all database tables automatically:
```powershell
python -c "from app.db.session import init_db; init_db(); print('Database tables ready!')"
```

### 6. Ingest Clinical Knowledge Documents
Load the 8 authoritative medical guidelines from CDC, WHO, and NIH into your vector database:
```powershell
python -m pipeline.run
```

---

## 💻 How to Run the Project

MedTwin AI runs as a modern two-tier web application:

### Terminal 1: Start the FastAPI Backend
```powershell
python -m uvicorn app.api.main:app --reload --port 8000
```
* Interactive Swagger API documentation: **[http://localhost:8000/docs](http://localhost:8000/docs)**
* System Health status: **[http://localhost:8000/health](http://localhost:8000/health)**

### Terminal 2: Launch the Streamlit Web Application
```powershell
python -m streamlit run streamlit_app.py
```
* Opens automatically in your browser at: **[http://localhost:8501](http://localhost:8501)**

---

## 🧪 Automated Testing

MedTwin AI comes with automated tests verifying RAG retrieval, drug interactions, database isolation, and emergency escalation:

Run the test suite:
```powershell
python -m pytest
```

---

## ⚠️ Safety & Ethical Boundaries

> [!IMPORTANT]
> **Educational & Informational Tool Only**
> 
> MedTwin AI is an open-source educational clinical assistant and **NOT** a certified medical diagnostic device or healthcare provider.
> 
> * It does not prescribe medication or issue formal diagnoses.
> * In any real-world emergency (crushing chest pain, severe shortness of breath, sudden numbness, uncontrolled bleeding), immediately call emergency services (**911**, **112**, or your local emergency number) or proceed to the nearest emergency room.

---

## 👥 Authors & Open Source Credits
* Built with ❤️ by [DG-Modi](https://github.com/DG-Modi)
* Clinical guidelines curated from CDC, NIH MedlinePlus, WHO, and ADA open-access repositories.
* Open-source licensed under the [MIT License](LICENSE).
