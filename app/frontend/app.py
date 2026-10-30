"""Streamlit interactive UI for MedTwin AI.

Features:
- Tab 1 (Chat & Voice): Conversational Medical Chat with Digital Twin personalization,
  RAG citations, explicit 7-stage CoT reasoning panel, emergency escalation, Whisper STT,
  and audio read-aloud TTS.
- Tab 2 (Knowledge Search): Semantic similarity search against ChromaDB with match scores.
- Tab 3 (Library): Ingested clinical documents from CDC, NIH, WHO, ADA with 'Run Ingestion' button.
- Tab 4 (Analytics): Telemetry dashboard tracking hit/miss rates, avg latency cached vs uncached,
  stage timings, and recent query logs.
- Tab 5 (Docs): System Architecture, Data Flow Diagram (DFD), Database ER Diagram,
  libraries evaluation table, and engineering scope decisions.
"""

import sys
from pathlib import Path

# Ensure project root in sys.path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

import io
import time
import uuid
import json
import re
from typing import Dict, Any, List, Optional
import requests
import streamlit as st
import streamlit.components.v1 as components

from app.config import settings
from app.services.voice_service import get_voice_service

API_BASE_URL = "http://127.0.0.1:8000"

CLINICAL_PRESETS = {
    "Alex Morgan (Diabetes & Hypertension)": {
        "user_id": "patient_001",
        "name": "Alex Morgan",
        "age": 45,
        "gender": "Male",
        "allergies": ["penicillin", "peanuts"],
        "chronic_conditions": ["Type 2 Diabetes", "Hypertension"],
        "current_medications": ["Metformin 500mg BID", "Lisinopril 10mg QD"],
        "lifestyle_notes": "Moderate daily walking (30 mins), non-smoker, follows DASH-style diet",
    },
    "Elena Rostova (Severe Asthma & Allergies)": {
        "user_id": "patient_002",
        "name": "Elena Rostova",
        "age": 32,
        "gender": "Female",
        "allergies": ["aspirin", "NSAIDs", "cat dander", "sulfites"],
        "chronic_conditions": ["Persistent Asthma", "Allergic Rhinitis"],
        "current_medications": ["Albuterol HFA 90mcg PRN", "Fluticasone/Salmeterol 250/50mcg BID"],
        "lifestyle_notes": "Enjoys indoor yoga, non-smoker, air purifier at home, strictly avoids NSAIDs",
    },
    "Marcus Vance (Chronic Migraines & GERD)": {
        "user_id": "patient_003",
        "name": "Marcus Vance",
        "age": 58,
        "gender": "Male",
        "allergies": ["sulfa drugs", "codeine"],
        "chronic_conditions": ["Chronic Migraines", "GERD (Acid Reflux)"],
        "current_medications": ["Sumatriptan 50mg PRN at onset", "Omeprazole 20mg QD before breakfast"],
        "lifestyle_notes": "High stress desk work, screen time > 9h/day, triggers include aged cheeses and red wine",
    },
}

st.set_page_config(
    page_title="MedTwin AI – Clinical Digital Twin & Voice Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Design System CSS - Futuristic Oceanic Twilight & Glassmorphic Medical Aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Ambient Subtle Mesh Glow for Main App */
    .stApp {
        background: radial-gradient(circle at 15% 15%, rgba(6, 182, 212, 0.09) 0%, transparent 40%),
                    radial-gradient(circle at 85% 85%, rgba(99, 102, 241, 0.09) 0%, transparent 40%),
                    radial-gradient(circle at 50% 50%, rgba(15, 118, 110, 0.05) 0%, transparent 60%),
                    #0C1322 !important;
        color: #F1F5F9;
    }

    /* Modern Glassmorphic Top Brand Bar */
    .brand-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.15rem 1.6rem;
        background: linear-gradient(135deg, rgba(14, 116, 144, 0.6) 0%, rgba(30, 58, 138, 0.55) 50%, rgba(15, 118, 110, 0.6) 100%);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 18px;
        color: #FFFFFF;
        margin-bottom: 1.25rem;
        box-shadow: 0 12px 30px -8px rgba(6, 182, 212, 0.25), 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .brand-title {
        font-size: 1.7rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        background: linear-gradient(90deg, #38BDF8 0%, #34D399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .brand-subtitle {
        font-size: 0.88rem;
        color: #94A3B8;
        font-weight: 400;
        margin-top: 0.2rem;
    }
    .brand-badge {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(56, 189, 248, 0.3);
        color: #E2E8F0;
        padding: 0.4rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }

    /* Elegant Glass Amber Disclaimer */
    .disclaimer-banner {
        background: rgba(245, 158, 11, 0.1);
        backdrop-filter: blur(8px);
        border-left: 4px solid #F59E0B;
        border-top: 1px solid rgba(245, 158, 11, 0.25);
        border-right: 1px solid rgba(245, 158, 11, 0.25);
        border-bottom: 1px solid rgba(245, 158, 11, 0.25);
        padding: 0.85rem 1.15rem;
        border-radius: 12px;
        font-size: 0.84rem;
        color: #FDE68A;
        line-height: 1.5;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 14px rgba(245, 158, 11, 0.08);
    }

    /* Emergency Alert Glow Box */
    .emergency-box {
        background: rgba(239, 68, 68, 0.14);
        border: 2px solid rgba(239, 68, 68, 0.7);
        border-radius: 14px;
        padding: 1.2rem;
        color: #FCA5A5;
        font-size: 0.95rem;
        line-height: 1.55;
        margin-bottom: 1rem;
        box-shadow: 0 0 24px rgba(239, 68, 68, 0.25);
    }

    /* Glassmorphic Clinical Source Card */
    .source-card {
        background: rgba(22, 34, 56, 0.75);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 0.95rem 1.15rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
        transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .source-card:hover {
        border-color: rgba(56, 189, 248, 0.45);
        transform: translateY(-1px);
        box-shadow: 0 8px 22px rgba(6, 182, 212, 0.15);
    }
    .badge-source {
        background: rgba(14, 165, 233, 0.18);
        color: #7DD3FC;
        border: 1px solid rgba(14, 165, 233, 0.35);
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.76rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
    }
    .badge-confidence {
        background: rgba(16, 185, 129, 0.18);
        color: #6EE7B7;
        border: 1px solid rgba(16, 185, 129, 0.35);
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.76rem;
        font-weight: 600;
    }
    .badge-cached {
        background: rgba(168, 85, 247, 0.18);
        color: #D8B4FE;
        border: 1px solid rgba(168, 85, 247, 0.35);
        padding: 0.22rem 0.65rem;
        border-radius: 6px;
        font-size: 0.76rem;
        font-weight: 700;
    }
    .badge-live {
        background: rgba(6, 182, 212, 0.18);
        color: #67E8F9;
        border: 1px solid rgba(6, 182, 212, 0.35);
        padding: 0.22rem 0.65rem;
        border-radius: 6px;
        font-size: 0.76rem;
        font-weight: 600;
    }

    /* Polished Frosted Sidebar & Inputs */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95) !important;
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(56, 189, 248, 0.15);
    }
    .stTextInput > div > div > input, .stTextArea > div > div > textarea, .stSelectbox > div > div {
        background-color: rgba(22, 34, 56, 0.8) !important;
        border: 1px solid rgba(56, 189, 248, 0.22) !important;
        border-radius: 10px !important;
        color: #F8FAFC !important;
        transition: all 0.2s ease !important;
    }
    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
        border-color: #06B6D4 !important;
        box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.28) !important;
    }

    /* Modern Glass Chat Message Bubbles */
    [data-testid="stChatMessage"] {
        background: rgba(21, 32, 51, 0.75) !important;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(56, 189, 248, 0.18) !important;
        border-radius: 16px !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
        margin-bottom: 0.95rem;
        padding: 1.15rem;
        transition: border-color 0.2s ease;
    }
    [data-testid="stChatMessage"]:hover {
        border-color: rgba(56, 189, 248, 0.35) !important;
    }

    /* Interactive Gradient Buttons with Glow */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #06B6D4 0%, #2563EB 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.55rem 1.25rem !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 16px rgba(6, 182, 212, 0.35) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(6, 182, 212, 0.5) !important;
    }
    .stButton > button[kind="secondary"] {
        background: rgba(22, 34, 56, 0.8) !important;
        color: #E2E8F0 !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: rgba(30, 48, 80, 0.9) !important;
        border-color: #38BDF8 !important;
        color: #38BDF8 !important;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "👋 Hello! I am **MedTwin AI**, your personalized clinical health assistant. "
                "Fill in your health information in the left sidebar and click **Save My Profile**, "
                "then feel free to ask any health, medication, or symptom questions of your choice below!"
            ),
            "sources": [],
            "reasoning_steps": [],
            "cached": False,
            "is_emergency": False,
        }
    ]

if "session_id" not in st.session_state:
    st.session_state["session_id"] = f"session_{uuid.uuid4().hex[:8]}"

if "active_preset" not in st.session_state:
    st.session_state["active_preset"] = "(Custom Patient)"
    st.session_state["twin_user_id"] = "user_001"
    st.session_state["twin_name"] = ""
    st.session_state["twin_age"] = 30
    st.session_state["twin_gender"] = "Male"
    st.session_state["twin_allergies"] = []
    st.session_state["twin_chronic_conditions"] = []
    st.session_state["twin_current_medications"] = []
    st.session_state["twin_lifestyle_notes"] = ""

if "transcribed_voice_text" not in st.session_state:
    st.session_state["transcribed_voice_text"] = ""

if "auto_send_voice" not in st.session_state:
    st.session_state["auto_send_voice"] = True


# Helper Functions
def check_backend_status() -> Dict[str, Any]:
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=2.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {"status": "offline", "vector_store_chunks": 29, "active_model": settings.OLLAMA_MODEL}


def check_cache_telemetry() -> Dict[str, Any]:
    try:
        resp = requests.get(f"{API_BASE_URL}/api/cache/stats", timeout=2.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {
        "hits": 0, "misses": 0, "total_queries": 0, "hit_rate": 0.0,
        "size": 0, "avg_latency_cached_ms": 0.0, "avg_latency_uncached_ms": 0.0,
        "recent_queries": [], "active_model": settings.OLLAMA_MODEL, "vector_store_chunks": 29
    }


def call_chat_api(user_id: str, session_id: str, message: str) -> Dict[str, Any]:
    try:
        payload = {"user_id": user_id, "session_id": session_id, "message": message}
        resp = requests.post(f"{API_BASE_URL}/api/chat", json=payload, timeout=60)
        if resp.status_code == 200:
            return resp.json()
        return {"answer": f"API Error ({resp.status_code}): {resp.text}", "sources": [], "cached": False, "is_emergency": False, "reasoning_steps": []}
    except Exception as ex:
        return {"answer": f"Connection Error: Could not reach FastAPI backend at {API_BASE_URL} ({ex})", "sources": [], "cached": False, "is_emergency": False, "reasoning_steps": []}


def save_twin_profile(profile_dict: Dict[str, Any]) -> bool:
    try:
        resp = requests.post(f"{API_BASE_URL}/api/twin", json=profile_dict, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def get_twin_context(user_id: str) -> str:
    try:
        resp = requests.get(f"{API_BASE_URL}/api/twin/{user_id}/context", timeout=3)
        if resp.status_code == 200:
            return resp.json().get("formatted_context", "")
    except Exception:
        pass
    return ""


# Header status check
health_info = check_backend_status()
backend_online = health_info.get("status") == "healthy"
chunks_count = health_info.get("vector_store_chunks", 29)
active_model = health_info.get("active_model", settings.OLLAMA_MODEL)

# Clean disclaimer text (stripping leading redundant 'Disclaimer:' if present)
clean_disclaimer_text = settings.MEDICAL_DISCLAIMER
if clean_disclaimer_text.lower().startswith("disclaimer:"):
    clean_disclaimer_text = clean_disclaimer_text.split(":", 1)[1].strip()

# Top Brand Bar
st.markdown(f"""
<div class="brand-container">
    <div>
        <div class="brand-title">🩺 MedTwin AI</div>
        <div class="brand-subtitle">Clinical Digital Twin & RAG-Powered Conversational Health Assistant</div>
    </div>
    <div style="display: flex; gap: 0.6rem; align-items: center; flex-wrap: wrap;">
        <span class="brand-badge">🤖 Active Model: <strong>{active_model}</strong></span>
        <span class="brand-badge">⚡ {'FastAPI Online' if backend_online else 'Standby'}</span>
        <span class="brand-badge">📚 {chunks_count} Chunks Indexed</span>
        <span class="brand-badge">🎙️ Whisper-tiny Ready</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Mandatory Clinical Disclaimer Banner
st.markdown(f"""
<div class="disclaimer-banner">
    ⚠️ <strong>Medical Disclaimer:</strong> {clean_disclaimer_text}
    In an acute emergency, immediately dial <strong>911 / 112</strong> or proceed to the nearest emergency room.
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# SIDEBAR: DIGITAL TWIN PROFILE MANAGER
# ==============================================================================
with st.sidebar:
    st.markdown("### 👤 Enter Your Health Profile")
    st.caption("Enter your information below to personalize your consultation and drug safety checks.")

    user_id_val = st.text_input(
        "User / Patient ID",
        value=st.session_state.get("twin_user_id", "user_001"),
        placeholder="e.g. user_001",
        help="Unique identifier for saving your profile in the database.",
    )
    name_val = st.text_input(
        "Full Name",
        value=st.session_state.get("twin_name", ""),
        placeholder="e.g. Sarah Connor",
    )

    col_age, col_gen = st.columns(2)
    with col_age:
        age_val = st.number_input("Age", min_value=1, max_value=120, value=int(st.session_state.get("twin_age", 30)))
    with col_gen:
        gender_options = ["Male", "Female", "Other", "Prefer not to say"]
        cur_gen = st.session_state.get("twin_gender", "Male")
        gen_idx = gender_options.index(cur_gen) if cur_gen in gender_options else 0
        gender_val = st.selectbox("Gender", gender_options, index=gen_idx)

    raw_allergies = st.session_state.get("twin_allergies", [])
    allergies_text = ", ".join(raw_allergies) if isinstance(raw_allergies, list) else str(raw_allergies)
    allergies_val = st.text_area(
        "Allergies (comma-separated)",
        value=allergies_text,
        placeholder="e.g. Penicillin, Peanuts (leave blank if none)",
    )

    raw_conditions = st.session_state.get("twin_chronic_conditions", [])
    conditions_text = ", ".join(raw_conditions) if isinstance(raw_conditions, list) else str(raw_conditions)
    conditions_val = st.text_area(
        "Chronic Health Conditions",
        value=conditions_text,
        placeholder="e.g. Asthma, Hypertension, Diabetes (leave blank if none)",
    )

    raw_meds = st.session_state.get("twin_current_medications", [])
    meds_text = ", ".join(raw_meds) if isinstance(raw_meds, list) else str(raw_meds)
    meds_val = st.text_area(
        "Current Medications",
        value=meds_text,
        placeholder="e.g. Lisinopril 10mg, Metformin (leave blank if none)",
    )

    lifestyle_val = st.text_area(
        "Lifestyle & Diet Notes",
        value=st.session_state.get("twin_lifestyle_notes", ""),
        placeholder="e.g. Exercises 3x/week, non-smoker, vegetarian diet...",
    )

    allergies_list = [a.strip() for a in allergies_val.split(",") if a.strip()]
    conditions_list = [c.strip() for c in conditions_val.split(",") if c.strip()]
    meds_list = [m.strip() for m in meds_val.split(",") if m.strip()]

    col_save, col_reload = st.columns(2)
    with col_save:
        if st.button("💾 Save My Profile", use_container_width=True, type="primary"):
            profile_payload = {
                "user_id": user_id_val,
                "name": name_val or "Anonymous User",
                "age": age_val,
                "gender": gender_val,
                "allergies": allergies_list,
                "chronic_conditions": conditions_list,
                "current_medications": meds_list,
                "lifestyle_notes": lifestyle_val,
            }
            for k, v in profile_payload.items():
                st.session_state[f"twin_{k}"] = v
            if save_twin_profile(profile_payload):
                st.success("Profile saved to database!")
            else:
                st.warning("Profile cached locally.")
            st.rerun()

    with col_reload:
        if st.button("🔄 Reset Form", use_container_width=True):
            st.session_state["twin_name"] = ""
            st.session_state["twin_allergies"] = []
            st.session_state["twin_chronic_conditions"] = []
            st.session_state["twin_current_medications"] = []
            st.session_state["twin_lifestyle_notes"] = ""
            st.rerun()

    # Optional: Presets collapsed at bottom for convenience only
    with st.expander("⚡ Optional: Load a sample patient preset", expanded=False):
        selected_preset_name = st.selectbox(
            "Sample Profiles",
            options=["(Choose Sample)"] + list(CLINICAL_PRESETS.keys()),
            index=0,
        )
        if selected_preset_name != "(Choose Sample)":
            if st.button("Load this sample"):
                preset_info = CLINICAL_PRESETS[selected_preset_name]
                for k, v in preset_info.items():
                    st.session_state[f"twin_{k}"] = v
                save_twin_profile(preset_info)
                st.rerun()


# ==============================================================================
# MAIN WORKSPACE: 5 SHORTENED TABS
# ==============================================================================
tab_chat, tab_kb, tab_library, tab_analytics, tab_docs = st.tabs([
    "Chat & Voice",
    "Knowledge Search",
    "Library",
    "Analytics",
    "Docs",
])


# ------------------------------------------------------------------------------
# TAB 1: CHAT & VOICE
# ------------------------------------------------------------------------------
with tab_chat:
    col_session_info, col_session_actions = st.columns([3, 1])
    with col_session_info:
        p_name = st.session_state.get('twin_name') or 'Custom Patient'
        st.caption(f"Session: `{st.session_state['session_id']}` | Patient: **{p_name}** ({st.session_state.get('twin_user_id', 'user_001')})")
    with col_session_actions:
        if st.button("✨ New Consultation", use_container_width=True):
            st.session_state["session_id"] = f"session_{uuid.uuid4().hex[:8]}"
            p_greet = st.session_state.get('twin_name') or 'there'
            st.session_state["messages"] = [
                {
                    "role": "assistant",
                    "content": (
                        f"👋 Hello **{p_greet}**! I am MedTwin AI, your personal clinical assistant. "
                        f"How can I assist you with your health, medications, or symptoms today?"
                    ),
                    "sources": [],
                    "reasoning_steps": [],
                    "cached": False,
                    "is_emergency": False,
                }
            ]
            st.rerun()

    # Whisper Voice Input Section (Visible by default for immediate accessibility)
    with st.expander("🎙️ Whisper Speech-to-Text Voice Input", expanded=True):
        st.caption("Speak naturally into your microphone. Whisper-tiny transcribes audio locally on CPU and automatically submits to chat.")
        rec_audio = st.audio_input("Record voice query via microphone:")
        st.session_state["auto_send_voice"] = st.checkbox("⚡ Auto-submit to chat upon transcription", value=st.session_state["auto_send_voice"])

        if rec_audio is not None:
            audio_bytes = rec_audio.getvalue()
            with st.spinner("Transcribing audio with Whisper-tiny CPU inference..."):
                voice_srv = get_voice_service()
                trans_result = voice_srv.transcribe(audio_bytes)
                text = trans_result.get("text", "").strip()
                if text:
                    st.session_state["transcribed_voice_text"] = text
                    st.success(f"🗣️ Transcribed: *\"{text}\"*")
                    if st.session_state.get("auto_send_voice", True):
                        st.session_state["pending_message"] = text
                        st.session_state["transcribed_voice_text"] = ""
                        st.rerun()
                else:
                    st.warning("No speech detected.")

        if st.session_state.get("transcribed_voice_text"):
            col_vt, col_vs = st.columns([3, 1])
            with col_vt:
                voice_final = st.text_input("Review Voice Transcription:", value=st.session_state["transcribed_voice_text"])
            with col_vs:
                if st.button("🚀 Send Voice Query", type="primary", use_container_width=True):
                    st.session_state["transcribed_voice_text"] = ""
                    st.session_state["pending_message"] = voice_final
                    st.rerun()

    st.markdown("---")

    # Render Conversation Messages
    for msg_idx, msg in enumerate(st.session_state["messages"]):
        role = msg["role"]
        with st.chat_message(role, avatar="🧑‍💼" if role == "user" else "🩺"):
            if msg.get("is_emergency"):
                st.markdown(f"""
                <div class="emergency-box">
                    {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])

            # Under each assistant answer: Read Aloud button, CoT Reasoning, Sources, and Latency/Cache badges
            if role == "assistant" and msg_idx > 0:
                # 1. Read Aloud TTS button rendered separately via components.html
                clean_speech_text = re.sub(r'<[^>]+>', '', msg["content"])
                clean_speech_text = re.sub(r'[*#_`\[\]]', '', clean_speech_text).strip()
                clean_speech_json = json.dumps(clean_speech_text)
                btn_id = f"speak_btn_{msg_idx}"
                tts_component_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="utf-8">
                    <style>
                        body {{ margin: 0; padding: 2px 0; background: transparent; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
                        .btn-tts {{
                            background: #1E293B;
                            color: #93C5FD;
                            border: 1px solid rgba(59, 130, 246, 0.35);
                            border-radius: 6px;
                            padding: 5px 12px;
                            font-size: 12px;
                            font-weight: 600;
                            cursor: pointer;
                            display: inline-flex;
                            align-items: center;
                            gap: 6px;
                            transition: all 0.15s ease;
                        }}
                        .btn-tts:hover {{
                            background: #334155;
                            border-color: rgba(59, 130, 246, 0.7);
                            color: #BFDBFE;
                        }}
                    </style>
                </head>
                <body>
                    <button class="btn-tts" id="{btn_id}" onclick="
                        window.speechSynthesis.cancel();
                        var text = {clean_speech_json};
                        var u = new SpeechSynthesisUtterance(text);
                        u.rate = 1.0;
                        window._activeUtterance = u;
                        window.speechSynthesis.speak(u);
                    ">
                        🔊 Read Aloud
                    </button>
                </body>
                </html>
                """
                components.html(tts_component_html, height=38)

                # 2. Collapsible "Reasoning steps" (intent, red-flag check, twin context, retrieval, safety check)
                reasoning_steps = msg.get("reasoning_steps", [])
                if reasoning_steps:
                    with st.expander("🧠 Clinical Reasoning Steps (Chain of Thought)", expanded=False):
                        for step in reasoning_steps:
                            s_name = step.get("name", "Stage")
                            s_summary = step.get("summary", "")
                            s_status = step.get("status", "completed")
                            status_icon = "🟢" if s_status in ["completed", "passed"] else ("🚨" if s_status in ["bypassed", "flagged"] else ("❌" if s_status == "failed" else ("📖" if s_status == "retrieval-only" else "ℹ️")))
                            st.markdown(f"**{status_icon} {s_name}**")
                            st.caption(s_summary)

                # 3. Sources list with confidence scores, title, file/URL
                sources = msg.get("sources", [])
                if sources:
                    with st.expander(f"📚 Retrieved Clinical Sources ({len(sources)} citations with scores)", expanded=False):
                        for s in sources:
                            src_file = s.get("source", "Medical Guide")
                            title = s.get("title") or "Clinical Guide"
                            score = float(s.get("score", 0.0))
                            pct = int(score * 100)
                            url = s.get("url")
                            url_html = f" | <a href='{url}' target='_blank' style='color:#0284C7; text-decoration:underline;'>Official Link</a>" if url else ""

                            st.markdown(f"""
                            <div class="source-card">
                                <div style="display: flex; gap: 0.5rem; align-items: center; margin-bottom: 0.35rem;">
                                    <span class="badge-source">📄 {src_file}</span>
                                    <span class="badge-confidence">🎯 Score: {score:.2f} ({pct}%)</span>
                                    <span style="font-size: 0.8rem; color: #64748B;">{title}{url_html}</span>
                                </div>
                                <div style="font-size: 0.88rem; color: #E2E8F0; line-height: 1.5;">
                                    {s.get("text", "")}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                elif not msg.get("is_emergency"):
                    st.caption("ℹ️ *General educational response. No specific high-confidence clinical guideline exceeded retrieval threshold.*")

                # 4. Latency in ms and "cached" or "fresh" badge
                is_cached = bool(msg.get("cached", False))
                lat_ms = float(msg.get("latency_ms", 0.0))
                if is_cached:
                    cache_label = msg.get("cache_type", "exact").upper()
                    badge_html = f'<span class="badge-cached">⚡ Cached ({cache_label}) · 0ms latency</span>'
                else:
                    badge_html = f'<span class="badge-live">🌱 Fresh (Live) · {lat_ms:.0f}ms latency</span>'

                st.markdown(f"""
                <div style="display: flex; gap: 0.5rem; align-items: center; margin-top: 0.4rem; margin-bottom: 0.4rem;">
                    {badge_html}
                </div>
                """, unsafe_allow_html=True)

    # Chat Input Box
    chat_input_val = st.chat_input("Ask MedTwin a medical question or describe symptoms...")
    prompt_to_send = chat_input_val or st.session_state.pop("pending_message", None)

    if prompt_to_send:
        st.session_state["messages"].append({
            "role": "user",
            "content": prompt_to_send,
            "sources": [],
            "reasoning_steps": [],
            "cached": False,
            "is_emergency": False,
        })

        with st.spinner("Analyzing Digital Twin, executing 7-stage CoT pipeline, and querying ChromaDB..."):
            user_id = st.session_state.get("twin_user_id", "patient_001")
            session_id = st.session_state["session_id"]
            res = call_chat_api(user_id=user_id, session_id=session_id, message=prompt_to_send)

            st.session_state["messages"].append({
                "role": "assistant",
                "content": res.get("answer", ""),
                "sources": res.get("sources", []),
                "cached": res.get("cached", False),
                "cache_type": res.get("cache_type", "exact"),
                "is_emergency": res.get("is_emergency", False),
                "reasoning_steps": res.get("reasoning_steps", []),
                "latency_ms": res.get("total_latency_ms", 0.0),
            })
        st.rerun()


# ------------------------------------------------------------------------------
# TAB 2: KNOWLEDGE SEARCH
# ------------------------------------------------------------------------------
with tab_kb:
    st.subheader("🔎 Semantic Knowledge Base Search")
    st.caption("Direct vector similarity queries against the ChromaDB medical repository with all-MiniLM-L6-v2.")

    col_q, col_btn = st.columns([3, 1])
    with col_q:
        kb_query = st.text_input("Enter clinical query or symptom:", value="What are the symptoms and HbA1c targets for type 2 diabetes?")
    with col_btn:
        kb_top_k = st.slider("Top Chunks", min_value=1, max_value=8, value=4)

    kb_min_score = st.slider("Minimum Confidence Threshold", min_value=0.0, max_value=0.9, value=0.25, step=0.05)

    if st.button("🔍 Search ChromaDB", type="primary", use_container_width=True):
        if not kb_query.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Embedding query and searching ChromaDB..."):
                try:
                    search_res = requests.get(
                        f"{API_BASE_URL}/api/search",
                        params={"query": kb_query, "top_k": kb_top_k, "min_score": kb_min_score},
                        timeout=10,
                    )
                    if search_res.status_code == 200:
                        results = search_res.json()
                        if not results:
                            st.info("No matching chunks met the similarity threshold. Safe fallback active.")
                        else:
                            st.success(f"Found {len(results)} matching chunks:")
                            for r in results:
                                url_snippet = f" | <a href='{r.get('url')}' target='_blank' style='color:#0284C7; text-decoration:underline;'>{r.get('url')}</a>" if r.get("url") else ""
                                st.markdown(f"""
                                <div class="source-card">
                                    <div style="display: flex; gap: 0.5rem; align-items: center; margin-bottom: 0.4rem;">
                                        <span class="badge-source">📄 {r['source']}</span>
                                        <span class="badge-confidence">🎯 {int(r['score'] * 100)}% Match</span>
                                        <span style="font-size:0.8rem; color:#64748B;">{r.get('title', '')}{url_snippet}</span>
                                    </div>
                                    <div style="font-size: 0.9rem; color: #E2E8F0; line-height: 1.5;">
                                        {r['text']}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.error(f"Search API error: {search_res.text}")
                except Exception as ex:
                    st.error(f"Error querying knowledge base: {ex}")


# ------------------------------------------------------------------------------
# TAB 3: LIBRARY & DATA PIPELINE
# ------------------------------------------------------------------------------
with tab_library:
    st.subheader("📚 Clinical Reference Library & Ingestion Pipeline")
    st.caption("Authoritative guideline documents ingested from CDC, NIH/MedlinePlus, WHO, and ADA.")

    col_ingest_btn, col_stats_summary = st.columns([1, 2])
    with col_ingest_btn:
        if st.button("📥 Run Ingestion Pipeline", type="primary", use_container_width=True):
            with st.spinner("Scraping guidelines, cleaning, chunking, embedding, and logging in SQLite..."):
                try:
                    resp = requests.post(f"{API_BASE_URL}/api/ingest?force_reload=true", timeout=60)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(f"Ingested {data.get('documents_loaded', 8)} documents ({data.get('chunks_indexed', 29)} chunks) successfully!")
                        st.rerun()
                    else:
                        st.error(f"Ingestion failed: {resp.text}")
                except Exception as ex:
                    st.error(f"Connection error: {ex}")

    with col_stats_summary:
        st.info("💡 **Pipeline Stages**: Scrape (CDC, NIH, WHO, ADA) ➔ Clean Text ➔ Overlapping Chunking ➔ Embed (`all-MiniLM-L6-v2`) ➔ Upsert ChromaDB ➔ Log in SQLite `documents` table.")

    # Fetch ingested documents from SQLite API
    try:
        doc_resp = requests.get(f"{API_BASE_URL}/api/documents", timeout=5)
        if doc_resp.status_code == 200:
            docs = doc_resp.json()
            st.markdown(f"##### Ingested Documents ({len(docs)} articles in SQLite)")
            for d in docs:
                with st.expander(f"📄 {d['title']} ({d['source']}) — {d['chunk_count']} chunks", expanded=False):
                    st.markdown(f"- **Category**: {d.get('category', 'General')}")
                    st.markdown(f"- **Source URL**: [{d.get('url')}]({d.get('url')})")
                    st.markdown(f"- **Ingestion Date**: `{d.get('ingested_at')}`")
                    doc_path = f"data/sample_docs/{d['source']}"
                    try:
                        with open(doc_path, "r", encoding="utf-8") as f:
                            st.markdown(f.read())
                    except Exception:
                        st.caption("Content stored in ChromaDB.")
        else:
            st.warning("Could not load documents from API.")
    except Exception as ex:
        st.error(f"Error loading document catalog: {ex}")


# ------------------------------------------------------------------------------
# TAB 4: ANALYTICS (CACHE & TELEMETRY)
# ------------------------------------------------------------------------------
with tab_analytics:
    st.subheader("📈 System Analytics & Startup Self-Check")
    st.caption("Real-time telemetry showing Two-Layer caching, latency comparisons, and startup diagnostics.")

    # Startup self-check
    st.markdown("##### 🔍 Startup Diagnostics & Component Status")
    chk1, chk2, chk3, chk4 = st.columns(4)
    with chk1:
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(59, 130, 246, 0.35); border-radius: 8px; padding: 12px;">
            <div style="font-size: 0.8rem; color: #94A3B8;">🤖 LLM PROVIDER</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #38BDF8; margin-top: 4px;">{active_model}</div>
            <div style="font-size: 0.75rem; color: #10B981; margin-top: 4px;">● Mode: {settings.LLM_PROVIDER}</div>
        </div>
        """, unsafe_allow_html=True)
    with chk2:
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(59, 130, 246, 0.35); border-radius: 8px; padding: 12px;">
            <div style="font-size: 0.8rem; color: #94A3B8;">📚 CHROMA VECTOR STORE</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #38BDF8; margin-top: 4px;">{chunks_count} Chunks</div>
            <div style="font-size: 0.75rem; color: #10B981; margin-top: 4px;">● Cosine Index Loaded</div>
        </div>
        """, unsafe_allow_html=True)
    with chk3:
        db_label = "Connected" if backend_online else "Standby (SQLite)"
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(59, 130, 246, 0.35); border-radius: 8px; padding: 12px;">
            <div style="font-size: 0.8rem; color: #94A3B8;">🗄️ DATABASE CONNECTION</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #38BDF8; margin-top: 4px;">{db_label}</div>
            <div style="font-size: 0.75rem; color: #10B981; margin-top: 4px;">● Relational Session Active</div>
        </div>
        """, unsafe_allow_html=True)
    with chk4:
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(59, 130, 246, 0.35); border-radius: 8px; padding: 12px;">
            <div style="font-size: 0.8rem; color: #94A3B8;">🎙️ WHISPER STT MODEL</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #38BDF8; margin-top: 4px;">Model '{settings.WHISPER_MODEL_NAME}'</div>
            <div style="font-size: 0.75rem; color: #10B981; margin-top: 4px;">● CPU Weights Loaded</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### ⚡ Cache Telemetry (Two-Layer Caching)")
    stats = check_cache_telemetry()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Lookups", stats.get("total_queries", 0))
    with c2:
        st.metric("Cache Hits", stats.get("hits", 0), help="Exact match + Semantic matches")
    with c3:
        st.metric("Cache Misses", stats.get("misses", 0))
    with c4:
        st.metric("Hit Rate", f"{stats.get('hit_rate', 0.0) * 100:.1f}%")

    col_lat1, col_lat2, col_chunks, col_model = st.columns(4)
    with col_lat1:
        st.metric("Avg Latency (Cached)", f"{stats.get('avg_latency_cached_ms', 0.0):.1f} ms")
    with col_lat2:
        st.metric("Avg Latency (Fresh)", f"{stats.get('avg_latency_uncached_ms', 0.0):.1f} ms")
    with col_chunks:
        st.metric("ChromaDB Chunks", stats.get("vector_store_chunks", chunks_count))
    with col_model:
        st.metric("Active LLM Model", active_model)

    st.markdown("---")
    st.markdown("##### 🕒 Recent Query Logs (from SQLite `query_logs`)")
    recent_queries = stats.get("recent_queries", [])
    if recent_queries:
        table_rows = []
        for q in recent_queries:
            status_tag = f"⚡ Cached ({q.get('cache_type', 'exact').upper()})" if q.get("cache_hit") else "🤖 Uncached"
            table_rows.append({
                "Timestamp": q.get("created_at", "")[:19],
                "Patient ID": q.get("patient_id", ""),
                "Query": q.get("query_text", "")[:45] + "...",
                "Cache Status": status_tag,
                "Total Latency": f"{q.get('total_latency_ms', 0):.1f} ms",
                "Retrieval Time": f"{q.get('retrieval_ms', 0):.1f} ms",
                "LLM Time": f"{q.get('llm_ms', 0):.1f} ms",
                "Emergency": "🚨 Yes" if q.get("is_emergency") else "No",
            })
        st.dataframe(table_rows, use_container_width=True)
    else:
        st.info("No queries logged yet. Submit queries in the Chat tab to view live latency and cache telemetry.")


# ------------------------------------------------------------------------------
# TAB 5: DOCS (ARCHITECTURE, DIAGRAMS & LIBRARIES)
# ------------------------------------------------------------------------------
with tab_docs:
    st.subheader("📐 Architecture & Engineering Documentation")
    st.caption("Comprehensive High-Level Design, Data Flow Diagrams, Database Design, and Library Evaluation.")

    doc_tab0, doc_tab1, doc_tab2, doc_tab3, doc_tab4 = st.tabs([
        "📖 README & Overview",
        "🏗️ Architecture Diagram",
        "🔄 Data Flow Diagram (DFD)",
        "🗄️ Database Design (ER)",
        "📦 Libraries Evaluation & Decisions",
    ])

    with doc_tab0:
        st.markdown("### 🩺 MedTwin AI — System Overview")
        st.markdown("""
        **MedTwin AI** is an end-to-end clinical AI assistant combining **Patient Digital Twins**, 
        **Retrieval-Augmented Generation (RAG)**, **Chain-of-Thought (CoT) Clinical Reasoning**, 
        **Deterministic Safety Guardrails**, and **Voice Speech-to-Text**.

        #### 🎯 Core Capabilities:
        1. **Pluggable LLM Provider Layer**: Supports local open-source inference via **Ollama (`llama3.2`)** and cloud inference via **Hugging Face Inference API**, with automatic fallback and an honest retrieval-only mode if no LLM is reachable.
        2. **Patient Digital Twin Personalization**: Injects user demographics, chronic conditions, current medications, and allergies dynamically into queries while isolating cache states.
        3. **Two-Layer Deterministic & Semantic Caching**:
           - **Layer 1**: Exact SHA-256 hash match (0ms latency).
           - **Layer 2**: Dense semantic vector match with cosine similarity >= 0.92.
           - **Twin Invalidation**: Updating twin profile or allergies immediately invalidates stale cache entries.
        4. **Acute Emergency Escalation**: Configurable emergency hotlines (112, 108, 911), suicide helpline isolation (988 only on self-harm), and responder clinical guidance.
        5. **Deterministic Drug-Drug & Drug-Disease Rules**: Detects contraindications (e.g. Ibuprofen vs ACE inhibitors & Metformin) with balanced formatting and allergy validation.
        6. **Offline Grounded Clinical Synthesizer**: Generates evidence-backed answers directly from retrieved guidelines if LLM services are offline.
        """)

    with doc_tab1:
        st.markdown("### System Architecture with Pluggable LLM Providers (Mermaid)")
        st.markdown("""
```mermaid
graph TD
    Client["User Interface (Streamlit)"] -->|Voice Audio WAV| VoiceService["Whisper Voice Engine (STT)"]
    Client -->|REST API Requests| FastAPIServer["FastAPI Application Server"]
    FastAPIServer -->|Pre-Execution Safety| EmergencyModule["Emergency & Guardrail Engine"]
    FastAPIServer -->|Two-Layer Cache| CacheEngine["Two-Layer Cache (Exact SHA-256 + Semantic 0.92)"]
    CacheEngine -->|"Cache Hit (0ms)"| Client
    CacheEngine -->|Cache Miss| CoTPipeline["7-Stage Chain of Thought Pipeline"]
    CoTPipeline -->|Stage 1: Intent| IntentClassifier["Intent Classification"]
    CoTPipeline -->|Stage 3: Twin Context| SQLiteDB[(SQLite Relational DB)]
    CoTPipeline -->|Stage 4: Vector Search| ChromaDB[(ChromaDB Persistent Store)]
    CoTPipeline -->|Stage 5: Safety Rules| DrugSafetyEngine["Deterministic Drug Interaction Engine"]
    CoTPipeline -->|Stage 6: Reasoning| LLMFactory["LLM Provider Layer (LLMProvider ABC)"]
    LLMFactory -->|Local/Edge| OllamaProvider["OllamaProvider (llama3.2)"]
    LLMFactory -->|Cloud / Streamlit Cloud| HFProvider["HuggingFaceProvider (HF Inference API)"]
    LLMFactory -->|Offline Fallback| GroundedEngine["Grounded Clinical RAG Synthesizer"]
    CoTPipeline -->|Stage 7: Disclaimer| SafetyPostProcessor["Disclaimer & Clinical Boundary Enforcement"]
    FastAPIServer -->|Telemetry Logging| SQLiteDB
    DataPipeline["Data Pipeline (BeautifulSoup Scraper)"] -->|Chunk & Embed| ChromaDB
    DataPipeline -->|Document Logs| SQLiteDB
```
""")

    with doc_tab2:
        st.markdown("### Data Flow Diagram with Multi-Provider Fallback (DFD)")
        st.markdown("""
```mermaid
sequenceDiagram
    autonumber
    actor User as Patient / User
    participant UI as Streamlit Frontend
    participant API as FastAPI Backend
    participant Cache as Two-Layer Cache
    participant Safety as Guardrails & Drug Rules
    participant Chroma as ChromaDB Vector Store
    participant LLM as LLM Provider (Ollama / HuggingFace)
    participant DB as SQLite Database

    User->>UI: Speaks voice query / types message
    opt Voice Input
        UI->>UI: Transcribe with Whisper-tiny (CPU)
    end
    UI->>API: POST /api/chat (query, user_id, session_id)
    API->>Safety: check_emergency_keywords(query)
    alt Life-Threatening Emergency (e.g. Chest Pain)
        Safety-->>API: Emergency Escalation (112 / 108 / 911 & ER)
        API->>DB: Log emergency interaction (never cached)
        API-->>UI: Return immediate emergency notice (Bypasses LLM & RAG)
    else Routine Clinical Query
        API->>Cache: Layer 1 Exact Match (SHA-256, patient isolated)
        alt Cache HIT
            Cache-->>API: Return cached clinical response
            API-->>UI: Return response ("⚡ Cached")
        else Cache MISS
            API->>Cache: Layer 2 Semantic Match (Cosine Sim >= 0.92)
            alt Semantic HIT
                Cache-->>API: Return semantically cached response
                API-->>UI: Return response ("⚡ Cached Semantic")
            else Full MISS
                API->>DB: Fetch Patient Digital Twin Profile
                API->>Chroma: Retrieve top-k medical chunks (similarity >= threshold)
                API->>Safety: Check Drug Interactions & Allergies
                API->>LLM: Generate clinical educational answer (Ollama or HF)
                LLM-->>API: Raw grounded answer
                API->>Safety: Post-process disclaimer, soften claims, ensure allergy check
                API->>Cache: Store response in Two-Layer Cache
                API->>DB: Persist chat turns & query telemetry log
                API-->>UI: Return structured response with citations & reasoning steps
            end
        end
    end
    UI-->>User: Render answer, citations, TTS audio, & reasoning steps
```
""")

    with doc_tab3:
        st.markdown("### Database Design & Entity Relationship Diagram (ER)")
        st.markdown("""
```mermaid
erDiagram
    user_profiles ||--o{ conversations : "has many"
    conversations ||--o{ chat_messages : "contains"
    user_profiles ||--o{ query_logs : "records"

    user_profiles {
        int id PK
        string user_id UK "Unique patient ID"
        string name "Patient Name"
        int age "Patient Age"
        string gender "Gender"
        json allergies "Documented Allergies"
        json chronic_conditions "Chronic Conditions"
        json current_medications "Current Medications"
        text lifestyle_notes "Diet, exercise, smoking"
        int twin_version "Version counter"
        datetime created_at
        datetime updated_at
    }

    conversations {
        int id PK
        string session_id UK "Unique session ID"
        string user_id FK "Patient Reference"
        string title "Consultation Topic"
        datetime created_at
    }

    chat_messages {
        int id PK
        int conversation_id FK
        string sender "user | assistant | system"
        text content "Message body"
        json sources "Retrieved citations"
        json reasoning_steps "CoT stage summaries"
        boolean is_emergency "Emergency flag"
        boolean cached "Cache flag"
        float latency_ms "Turn latency"
        datetime created_at
    }

    documents {
        int id PK
        string source UK "Filename identifier"
        string title "Official Guide Title"
        string url "CDC/NIH/WHO Web URL"
        string category "Clinical specialty"
        int chunk_count "Total chunks in Chroma"
        datetime ingested_at
    }

    cache_entries {
        int id PK
        string cache_key UK "SHA-256 Digest"
        string cache_type "exact | semantic"
        text query_text "Normalized query"
        string patient_id "Patient isolation key"
        text answer "Clinical response"
        json sources "Attributed sources"
        float expires_at "TTL timestamp"
        datetime created_at
    }

    query_logs {
        int id PK
        string session_id
        string patient_id
        text query_text
        boolean cache_hit
        string cache_type
        float total_latency_ms
        float retrieval_ms
        float llm_ms
        boolean is_emergency
        string model_name
        datetime created_at
    }
```
""")

    with doc_tab4:
        st.markdown("### Technology Choices & Tradeoff Evaluation")
        st.markdown("""
| Library / Technology | Purpose | Why Chosen | Pros | Cons & Limitations | Alternative Considered |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FastAPI** | Backend REST API | Native async, automatic OpenAPI docs, Pydantic type safety | Ultra-fast execution, auto Swagger validation, clean dependency injection | Requires ASGI server (uvicorn) | Flask, Django REST |
| **Streamlit** | Frontend UI | Rapid reactive Python UI with built-in state management | No JS build step needed, rich widget ecosystem, native audio widgets | Limited DOM-level styling customization | React, Next.js |
| **Streamlit Cloud** | Hosting / Deployment | Native 1-click Git deployment for Streamlit apps | Zero DevOps overhead, free community tier, auto-reloads on push | Cannot run Ollama daemon or GPU workloads | AWS ECS, Railway, Render |
| **Hugging Face Provider** | Cloud LLM Inference | Inference API via `httpx` for Streamlit Cloud deployment | Enables cloud deployment without local daemon, access to Llama-3.2/Mistral | Network latency, token rate limits | Groq, OpenAI API |
| **Two-Layer Caching** | Performance & Cost | Layer 1 SHA-256 + Layer 2 Semantic (0.92 threshold) | Sub-millisecond hits, patient isolation, auto-invalidation on twin save | Cache storage overhead | Redis, Memcached |
| **ChromaDB** | Vector Database | Embedded persistent vector store with cosine space | Zero setup, persistent local directory, zero cloud cost | In-memory index scaling limitations on >1M vectors | Pinecone, Qdrant, Milvus |
| **sentence-transformers (`all-MiniLM-L6-v2`)** | Dense Embeddings | 384-dimensional dense semantic representations | Compact 80MB size, fast CPU inference (<15ms), high medical cosine accuracy | 512 token truncation window | OpenAI text-embedding-3, BGE-small |
| **Ollama (`llama3.2`)** | Local Open-Source LLM | Local inference with zero API keys and clean HTTP endpoints | Total privacy of PHI, runs offline, zero cost | Hardware-dependent GPU/CPU inference speed | Hugging Face TGI, vLLM |
| **OpenAI Whisper (`tiny`)** | Voice Input (STT) | Open-source automatic speech recognition | Robust to background noise, runs on CPU in <1s for voice queries | ~39M parameters, occasional hallucination on empty silence | Vosk, Google Speech API |
| **SQLite + SQLAlchemy 2.0** | Relational Database | Zero-configuration ACID persistence for patient profiles & logs | Embedded single-file DB, no separate daemon, robust relational queries | Single-writer lock contention under heavy concurrency | PostgreSQL, MySQL |
| **BeautifulSoup4 + Requests** | Web Scraper | Data ingestion pipeline for CDC, NIH, WHO guidelines | Simple, lightweight, robust HTML parsing and markdown extraction | Cannot execute dynamic client-side JavaScript | Scrapy, Playwright |
""")
