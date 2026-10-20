"""Streamlit interactive UI for MedTwin AI.

Allows testing the Digital Twin profile manager and ChromaDB RAG retrieval directly in the browser.
"""

import streamlit as st
import requests
from typing import Dict, Any, List

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="MedTwin AI - Medical Knowledge Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 0.85rem;
        border-radius: 4px;
        color: #92400E;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
    .card-chunk {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.85rem;
    }
    .source-badge {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .score-badge {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.82rem;
    }
</style>
""", unsafe_allow_html=True)


def check_backend_health() -> Dict[str, Any]:
    """Queries FastAPI /health endpoint."""
    try:
        res = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return {"status": "offline", "vector_store_chunks": 0}


# Header
st.markdown('<div class="main-header">🩺 MedTwin AI – Medical Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">RAG Retrieval & Digital Twin Context Personalization (MVP Phase)</div>', unsafe_allow_html=True)

# Mandatory Disclaimer
st.markdown("""
<div class="disclaimer-box">
    ⚠️ <strong>Medical Disclaimer:</strong> MedTwin AI is an educational medical-information assistant and NOT a diagnostic system.
    It does not provide diagnoses or prescribe medication. Consult a licensed physician for medical advice.
    In an emergency, dial 911 / 112 immediately.
</div>
""", unsafe_allow_html=True)

# Sidebar: Patient Digital Twin Profile Manager
st.sidebar.header("👤 Patient Digital Twin Profile")
st.sidebar.caption("Structured user context for educational personalization (Not diagnostic)")

# Session state initialization
if "patient_id" not in st.session_state:
    st.session_state["patient_id"] = "patient_001"
if "patient_name" not in st.session_state:
    st.session_state["patient_name"] = "Alex Morgan"
if "patient_age" not in st.session_state:
    st.session_state["patient_age"] = 45
if "patient_gender" not in st.session_state:
    st.session_state["patient_gender"] = "Male"
if "allergies_input" not in st.session_state:
    st.session_state["allergies_input"] = "penicillin, peanuts"
if "conditions_input" not in st.session_state:
    st.session_state["conditions_input"] = "Type 2 Diabetes, Hypertension"
if "medications_input" not in st.session_state:
    st.session_state["medications_input"] = "Metformin 500mg BID, Lisinopril 10mg QD"
if "lifestyle_input" not in st.session_state:
    st.session_state["lifestyle_input"] = "Moderate daily walking, non-smoker, DASH-style diet"

# Load profile button
if st.sidebar.button("🔄 Reload Profile from Backend"):
    try:
        r = requests.get(f"{API_BASE_URL}/api/twin/{st.session_state['patient_id']}", timeout=3)
        if r.status_code == 200:
            data = r.json()
            st.session_state["patient_name"] = data.get("name") or ""
            st.session_state["patient_age"] = data.get("age") or 45
            st.session_state["patient_gender"] = data.get("gender") or "Male"
            st.session_state["allergies_input"] = ", ".join(data.get("allergies") or [])
            st.session_state["conditions_input"] = ", ".join(data.get("chronic_conditions") or [])
            st.session_state["medications_input"] = ", ".join(data.get("current_medications") or [])
            st.session_state["lifestyle_input"] = data.get("lifestyle_notes") or ""
            st.sidebar.success("Loaded profile from SQLite database!")
        else:
            st.sidebar.info("No saved profile on server yet. Click Save to persist.")
    except Exception as e:
        st.sidebar.error(f"Cannot reach backend: {e}")

user_id = st.sidebar.text_input("User / Patient ID", value=st.session_state["patient_id"])
name = st.sidebar.text_input("Full Name", value=st.session_state["patient_name"])
col_age, col_gen = st.sidebar.columns(2)
with col_age:
    age = st.number_input("Age", min_value=1, max_value=120, value=st.session_state["patient_age"])
with col_gen:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0 if st.session_state["patient_gender"] == "Male" else 1)

allergies_str = st.sidebar.text_area("Allergies (comma-separated)", value=st.session_state["allergies_input"], help="e.g. penicillin, aspirin, latex")
conditions_str = st.sidebar.text_area("Chronic Conditions (comma-separated)", value=st.session_state["conditions_input"], help="e.g. Type 2 Diabetes, Asthma")
medications_str = st.sidebar.text_area("Current Medications (comma-separated)", value=st.session_state["medications_input"], help="e.g. Metformin 500mg BID")
lifestyle = st.sidebar.text_input("Lifestyle Notes", value=st.session_state["lifestyle_input"])

if st.sidebar.button("💾 Save Digital Twin to Database", use_container_width=True):
    allergies = [a.strip() for a in allergies_str.split(",") if a.strip()]
    conditions = [c.strip() for c in conditions_str.split(",") if c.strip()]
    medications = [m.strip() for m in medications_str.split(",") if m.strip()]

    payload = {
        "user_id": user_id,
        "name": name,
        "age": int(age),
        "gender": gender,
        "allergies": allergies,
        "chronic_conditions": conditions,
        "current_medications": medications,
        "lifestyle_notes": lifestyle,
    }

    try:
        res = requests.post(f"{API_BASE_URL}/api/twin", json=payload, timeout=5)
        if res.status_code == 200:
            st.sidebar.success("Profile saved successfully to SQLite!")
        else:
            st.sidebar.error(f"Save failed: {res.text}")
    except Exception as e:
        st.sidebar.error(f"Backend offline: {e}")

# Expandable Twin Prompt Context Preview
with st.sidebar.expander("🔍 View Digital Twin Prompt Context", expanded=False):
    try:
        ctx_res = requests.get(f"{API_BASE_URL}/api/twin/{user_id}/context", timeout=3)
        if ctx_res.status_code == 200:
            st.code(ctx_res.json()["formatted_context"], language="markdown")
        else:
            st.info("Save profile first to view formatted prompt context.")
    except Exception:
        st.info("Backend offline or profile not saved.")


# Main Section: System Health & RAG Explorer
col1, col2, col3 = st.columns([1, 1, 1])

health = check_backend_health()
backend_status = health.get("status", "offline")
chunks_count = health.get("vector_store_chunks", 0)

with col1:
    st.metric("FastAPI Backend", "Online 🟢" if backend_status == "healthy" else "Offline 🔴")
with col2:
    st.metric("ChromaDB Vector Store", f"{chunks_count} Chunks Indexed")
with col3:
    if st.button("📥 Re-index Knowledge Base"):
        with st.spinner("Ingesting and embedding sample medical docs..."):
            try:
                ingest_res = requests.post(f"{API_BASE_URL}/api/ingest?force_reload=true", timeout=60)
                if ingest_res.status_code == 200:
                    st.success(f"Indexed successfully: {ingest_res.json().get('chunks_indexed', 0)} chunks!")
                    st.rerun()
                else:
                    st.error(f"Ingest failed: {ingest_res.text}")
            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")

# RAG Search Explorer
st.subheader("🔎 Semantic RAG Knowledge Retrieval")
st.write("Test semantic search against the clinical knowledge base stored in ChromaDB.")

# Quick sample query pills
sample_queries = [
    "What is normal blood pressure and stage 1 hypertension?",
    "What are the symptoms and HbA1c target for type 2 diabetes?",
    "When is asthma an acute emergency?",
    "What are migraine warning signs and the SNOOP criteria?",
    "How to differentiate a common cold from the flu?",
]

selected_query = st.selectbox("💡 Choose a sample clinical query, or type below:", ["(Custom query)"] + sample_queries)

if selected_query != "(Custom query)":
    default_text = selected_query
else:
    default_text = "What are the common symptoms and dietary guidelines for diabetes?"

query_input = st.text_input("Enter a medical symptom, condition, or question:", value=default_text)

col_k, col_thresh = st.columns([1, 1])
with col_k:
    top_k = st.slider("Top K Chunks to Retrieve", min_value=1, max_value=6, value=3)
with col_thresh:
    threshold = st.slider("Minimum Similarity Confidence", min_value=0.0, max_value=0.9, value=0.25, step=0.05)

if st.button("🔍 Search ChromaDB Knowledge Base", type="primary", use_container_width=True):
    if not query_input.strip():
        st.warning("Please enter a search query.")
    else:
        with st.spinner("Embedding query and querying ChromaDB..."):
            try:
                res = requests.get(
                    f"{API_BASE_URL}/api/search",
                    params={"query": query_input, "top_k": top_k, "min_score": threshold},
                    timeout=10,
                )
                if res.status_code == 200:
                    results = res.json()
                    if not results:
                        st.info("No matching chunks met the confidence threshold. Try lowering the threshold.")
                    else:
                        st.success(f"Retrieved {len(results)} relevant medical chunks!")
                        for i, chunk in enumerate(results, 1):
                            st.markdown(f"""
                            <div class="card-chunk">
                                <div>
                                    <span class="source-badge">📄 {chunk['source']}</span>
                                    <span class="score-badge">🎯 Confidence: {int(chunk['score'] * 100)}%</span>
                                </div>
                                <div style="margin-top: 0.6rem; font-size: 0.95rem; line-height: 1.5; color: #1F2937;">
                                    {chunk['text']}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.error(f"Search failed: {res.text}")
            except Exception as e:
                st.error(f"Backend error: {e}")

st.markdown("---")

# Knowledge Base Document Inspector
with st.expander("📚 Browse Available Medical Knowledge Base Documents"):
    st.write("These clinical guides are stored in `data/sample_docs/` and indexed in ChromaDB:")
    doc_tabs = st.tabs([
        "Type 2 Diabetes",
        "Hypertension",
        "Asthma Management",
        "Migraines & Headaches",
        "Cold vs. Flu",
    ])
    filenames = [
        "type_2_diabetes.md",
        "hypertension.md",
        "asthma_management.md",
        "migraine_headaches.md",
        "common_cold_vs_flu.md",
    ]
    for tab, fname in zip(doc_tabs, filenames):
        with tab:
            try:
                with open(f"data/sample_docs/{fname}", "r", encoding="utf-8") as f:
                    content = f.read()
                st.markdown(content)
            except Exception as err:
                st.error(f"Could not read {fname}: {err}")
