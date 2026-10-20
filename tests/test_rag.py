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
    assert "MUST ATTRIBUTE SOURCES" in context
    assert "NEVER invent or hallucinate" in context


def test_rag_context_formatting_empty():
    """Verify fallback when no documents match."""
    pipeline = RAGPipeline(vector_store=ChromaVectorStore(is_ephemeral=True))
    context = pipeline.format_rag_context([])
    assert "No specific clinical knowledge base documents matched" in context
    assert "Clearly indicate that knowledge-base information is limited" in context
