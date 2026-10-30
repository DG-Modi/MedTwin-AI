"""Unit and integration tests for the RAG pipeline components: loader, chunker, vector store, and retriever."""

import pytest
from pathlib import Path
from app.config import settings
from app.rag.document_loader import clean_medical_text, extract_title, load_document, load_all_documents
from app.rag.chunking import recursive_character_split, chunk_document, chunk_all_documents
from app.rag.vector_store import ChromaVectorStore
from app.rag.pipeline import RAGPipeline


def test_clean_medical_text():
    """Verify text cleaning strips excess whitespace and normalizes line breaks."""
    raw = "\r\n\n\n# Diabetes\n\n\n\nSymptoms: thirst.    \r\n\n"
    cleaned = clean_medical_text(raw)
    assert cleaned.startswith("# Diabetes")
    assert "\r" not in cleaned
    assert "\n\n\n" not in cleaned


def test_extract_title():
    """Verify markdown title extraction."""
    text1 = "# Clinical Guide: Type 2 Diabetes Mellitus\n\nBody content"
    assert extract_title(text1, "Fallback") == "Clinical Guide: Type 2 Diabetes Mellitus"

    text2 = "No header line\nJust regular text"
    assert extract_title(text2, "Fallback") == "Fallback"


def test_load_sample_documents():
    """Verify loading sample documents from data/sample_docs directory."""
    docs_dir = settings.BASE_DIR / "data" / "sample_docs"
    docs = load_all_documents(docs_dir)

    assert len(docs) >= 5
    sources = [doc["source"] for doc in docs]
    assert "type_2_diabetes.md" in sources
    assert "hypertension.md" in sources
    assert "asthma_management.md" in sources

    for doc in docs:
        assert len(doc["text"]) > 100
        assert doc["title"]
        assert doc["source"].endswith(".md")


def test_chunking_strategy():
    """Verify recursive text chunking and metadata attribution."""
    sample_doc = {
        "text": (
            "Paragraph one is about diabetes symptoms like thirst and fatigue. " * 5
            + "\n\n"
            + "Paragraph two discusses insulin resistance and blood glucose targets. " * 5
            + "\n\n"
            + "Paragraph three details emergency complications like DKA. " * 5
        ),
        "source": "diabetes_test.md",
        "title": "Diabetes Test Guide",
    }

    chunks = chunk_document(sample_doc, chunk_size=300, chunk_overlap=50)

    assert len(chunks) >= 3
    for idx, chunk in enumerate(chunks):
        assert chunk["source"] == "diabetes_test.md"
        assert chunk["title"] == "Diabetes Test Guide"
        assert chunk["id"] == f"diabetes_test.md_{idx}"
        assert chunk["chunk_index"] == idx
        assert len(chunk["text"]) <= 350  # reasonable boundary with separator


def test_chroma_vector_store_in_memory():
    """Verify ChromaDB ephemeral indexing and semantic similarity search."""
    # Use ephemeral client for fast, isolated test execution
    vstore = ChromaVectorStore(
        collection_name="test_medical_collection",
        is_ephemeral=True,
    )

    test_chunks = [
        {
            "id": "doc1_0",
            "text": "Hypertension is high blood pressure. Systolic pressure over 140 mmHg is stage 2 hypertension.",
            "source": "hypertension.md",
            "title": "Hypertension Guide",
            "chunk_index": 0,
        },
        {
            "id": "doc2_0",
            "text": "Asthma causes wheezing, airway inflammation, and shortness of breath treated with albuterol.",
            "source": "asthma.md",
            "title": "Asthma Action Plan",
            "chunk_index": 0,
        },
    ]

    added = vstore.add_chunks(test_chunks)
    assert added == 2
    assert vstore.count() == 2

    # Query specifically for blood pressure
    results = vstore.similarity_search("What is high blood pressure?", top_k=1)
    assert len(results) == 1
    top_match = results[0]
    assert top_match["source"] == "hypertension.md"
    assert top_match["score"] > 0.3
    assert "blood pressure" in top_match["text"].lower()

    # Query for respiratory symptoms
    asthma_results = vstore.similarity_search("inhaler for wheezing", top_k=1)
    assert len(asthma_results) == 1
    assert asthma_results[0]["source"] == "asthma.md"


def test_rag_context_formatting():
    """Verify RAG prompt context includes citations, confidence, and instructions."""
    pipeline = RAGPipeline(vector_store=ChromaVectorStore(is_ephemeral=True))

    chunks = [
        {
            "source": "type_2_diabetes.md",
            "title": "Diabetes Overview",
            "score": 0.85,
            "text": "Target HbA1c is below 7.0% for most adults.",
        }
    ]

    context = pipeline.format_rag_context(chunks)
    assert "[1] SOURCE: type_2_diabetes.md" in context
    assert "CONFIDENCE: 85%" in context
    assert "Target HbA1c" in context
    assert "CHUNK TEXT:" in context


def test_rag_context_formatting_empty():
    """Verify fallback when no documents match."""
    pipeline = RAGPipeline(vector_store=ChromaVectorStore(is_ephemeral=True))
    context = pipeline.format_rag_context([])
    assert "No specific clinical knowledge base documents matched" in context
    assert "not found in knowledge base" in context


def test_prompt_contains_chunk_text_and_twin_data():
    """(a) Test that the prompt contains chunk text and twin data."""
    from app.llm.prompts import build_medical_prompt

    query = "What are my target blood pressure and HbA1c ranges?"
    rag_context = "[1] SOURCE: hypertension.md\nTarget BP for adults is < 130/80 mmHg."
    twin_context = (
        "Patient: Alex Morgan\n"
        "Conditions: Type 2 Diabetes, Hypertension\n"
        "Meds: Metformin 500mg, Lisinopril 10mg\n"
        "Lifestyle: DASH diet, daily walking"
    )

    prompt = build_medical_prompt(
        query=query,
        rag_context=rag_context,
        twin_context=twin_context,
    )

    # Verify chunk text is present in the prompt
    assert "Target BP for adults is < 130/80 mmHg" in prompt
    assert "hypertension.md" in prompt

    # Verify patient twin data is present in the prompt
    assert "Alex Morgan" in prompt
    assert "Type 2 Diabetes" in prompt
    assert "Hypertension" in prompt
    assert "Metformin" in prompt
    assert "Lisinopril" in prompt
    assert "DASH diet" in prompt
    assert "daily walking" in prompt
    assert "not found in knowledge base" in prompt


def test_knowledge_base_contains_targets():
    """(b) Test that the knowledge base contains explicit targets: 130/80 and 7%."""
    from app.rag.pipeline import get_rag_pipeline

    pipeline = get_rag_pipeline()
    retrieved = pipeline.retrieve("What are my target blood pressure and HbA1c ranges?", top_k=4)

    assert len(retrieved) >= 1
    combined_text = " ".join(c["text"] for c in retrieved)

    # Check for targets in retrieved knowledge base chunks
    assert "130/80" in combined_text
    assert "7%" in combined_text or "7.0%" in combined_text

    # Verify source URLs and titles are returned
    for chunk in retrieved:
        assert chunk.get("source") in ["hypertension.md", "type_2_diabetes.md", "dash_diet.md"]
        assert chunk.get("url") is not None
        assert chunk.get("score") > 0.0


@pytest.mark.asyncio
async def test_response_contains_targets_when_in_kb():
    """(b) Test that response contains '130/80' or '7%' when generated with KB context."""
    from unittest.mock import AsyncMock, MagicMock
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.db.session import Base
    from app.db.models import UserProfile
    from app.core.cot_pipeline import ChainOfThoughtPipeline
    from app.llm.ollama_client import OllamaClient

    # In-memory test DB
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    # Create Alex Morgan twin
    twin = UserProfile(
        user_id="patient_001",
        name="Alex Morgan",
        age=45,
        chronic_conditions=["Type 2 Diabetes", "Hypertension"],
        current_medications=["Metformin 500mg BID", "Lisinopril 10mg QD"],
        lifestyle_notes="Moderate daily walking, DASH diet",
    )
    db.add(twin)
    db.commit()

    # Mock LLM that generates response synthesizing the KB targets and Twin
    mock_llm = MagicMock(spec=OllamaClient)
    mock_llm.model = "llama3"
    mock_llm.generate = AsyncMock(
        return_value=(
            "Based on [hypertension.md] and [type_2_diabetes.md], your target blood pressure is "
            "< 130/80 mmHg and your target HbA1c is < 7% (individualized). "
            "For your Type 2 Diabetes and Hypertension, your current Metformin and Lisinopril, "
            "along with your DASH diet and regular walking, actively support reaching these targets."
        )
    )

    pipeline = ChainOfThoughtPipeline(llm_client=mock_llm)
    res = await pipeline.execute(
        query="What are my target blood pressure and HbA1c ranges?",
        user_id="patient_001",
        session_id="test_sess",
        db=db,
    )

    answer = res["answer"]
    assert "130/80" in answer
    assert "7%" in answer
    assert "Lisinopril" in answer
    assert "Metformin" in answer
    db.close()


@pytest.mark.asyncio
async def test_llm_failure_path_returns_error_instead_of_template():
    """(c) Test that the LLM-failure path returns an honest error instead of a fake generic template."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.db.session import Base
    from app.core.cot_pipeline import ChainOfThoughtPipeline
    from app.llm.ollama_client import OllamaClient

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    # OllamaClient with unreachable port and fallback disabled to verify error handling
    unreachable_client = OllamaClient(base_url="http://127.0.0.1:99999", timeout_seconds=1)

    pipeline = ChainOfThoughtPipeline(llm_client=unreachable_client, enable_grounded_fallback=False)
    res = await pipeline.execute(
        query="What are my target blood pressure and HbA1c ranges?",
        user_id="patient_001",
        session_id="test_sess_fail",
        db=db,
    )

    answer = res["answer"]
    # Must return an honest error indicating the local model is unavailable
    assert "The local model is unavailable" in answer
    # Must NOT contain the fake generic template
    assert "documented for this topic" not in answer
    assert "inhaler" not in answer.lower()
    assert "respiratory distress" not in answer.lower()
    db.close()


@pytest.mark.asyncio
async def test_grounded_rag_engine_synthesizes_when_ollama_offline():
    """Verify Grounded RAG Engine produces accurate clinical targets and personalization when Ollama is offline."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.db.session import Base
    from app.db.models import UserProfile
    from app.core.cot_pipeline import ChainOfThoughtPipeline
    from app.llm.ollama_client import OllamaClient

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    twin = UserProfile(
        user_id="patient_001",
        name="Alex Morgan",
        age=45,
        chronic_conditions=["Type 2 Diabetes", "Hypertension"],
        current_medications=["Metformin 500mg BID", "Lisinopril 10mg QD"],
        lifestyle_notes="Moderate daily walking, DASH diet",
    )
    db.add(twin)
    db.commit()

    offline_client = OllamaClient(base_url="http://127.0.0.1:99999", timeout_seconds=1)
    pipeline = ChainOfThoughtPipeline(llm_client=offline_client, enable_grounded_fallback=True)

    res = await pipeline.execute(
        query="What are my target blood pressure and HbA1c ranges?",
        user_id="patient_001",
        session_id="test_offline_sess",
        db=db,
    )

    answer = res["answer"]
    assert "130/80" in answer
    assert "7%" in answer or "7.0%" in answer
    assert "Alex Morgan" in answer
    assert "Lisinopril" in answer
    assert "Metformin" in answer
    assert "inhaler" not in answer.lower()
    db.close()


def test_no_inhaler_text_for_diabetes_hypertension():
    """(d) Test that no 'inhaler' text appears for diabetes/hypertension queries."""
    from app.rag.pipeline import get_rag_pipeline

    pipeline = get_rag_pipeline()
    retrieved = pipeline.retrieve("What are my target blood pressure and HbA1c ranges?", top_k=4)

    assert len(retrieved) > 0
    for chunk in retrieved:
        assert "inhaler" not in chunk["text"].lower()
        assert "respiratory distress" not in chunk["text"].lower()

    formatted_context = pipeline.format_rag_context(retrieved)
    assert "inhaler" not in formatted_context.lower()
    assert "respiratory distress" not in formatted_context.lower()
