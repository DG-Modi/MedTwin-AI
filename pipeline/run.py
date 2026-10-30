"""Executable data pipeline runner for MedTwin AI.

Executes:
1. Scrape / load authoritative clinical guidelines (CDC, NIH, WHO, ADA).
2. Clean text and parse markdown headers/sections.
3. Chunk with recursive character splitting and overlap.
4. Embed with sentence-transformers (all-MiniLM-L6-v2).
5. Upsert into ChromaDB vector store.
6. Log document metadata, URLs, categories, and chunk counts in SQLite `documents` table.

Can be run from command line:
    python -m pipeline.run
"""

import sys
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure project root in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from sqlalchemy.orm import Session
from app.config import settings
from app.db.session import SessionLocal, init_db
from app.db.models import Document, utc_now
from app.rag.chunking import chunk_document
from app.rag.vector_store import get_vector_store
from pipeline.scraper import scrape_medical_guidelines, MEDICAL_ARTICLES_CATALOG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pipeline.run")


def run_ingestion_pipeline(
    docs_dir: Optional[Path] = None,
    force_reload: bool = True,
    db: Optional[Session] = None,
) -> Dict[str, Any]:
    """Executes the complete end-to-end ingestion pipeline."""
    logger.info("Starting MedTwin AI Clinical Knowledge Base Ingestion Pipeline...")
    t0 = datetime.now(timezone.utc)

    # 1. Initialize DB tables
    init_db()

    target_dir = docs_dir or (settings.BASE_DIR / "data" / "sample_docs")
    target_dir.mkdir(parents=True, exist_ok=True)

    # 2. Scrape / materialize authoritative articles
    scraped_docs = scrape_medical_guidelines(target_dir=target_dir)
    logger.info(f"Loaded {len(scraped_docs)} clinical guideline articles.")

    # 3. Vector Store initialization
    vector_store = get_vector_store()
    if force_reload:
        logger.info("Resetting ChromaDB collection for clean re-indexing...")
        vector_store.reset()

    # 4. Chunk & Embed
    all_chunks = []
    chunk_counts_by_doc = {}

    for doc in scraped_docs:
        doc_meta = {
            "title": doc["title"],
            "source": doc["source"],
            "url": doc["url"],
            "category": doc["category"],
            "author": doc["author"],
        }
        chunks = chunk_document(
            document={"text": doc["text"], **doc_meta},
            chunk_size=500,
            chunk_overlap=100,
        )
        # Enrich chunk metadata with URL and category
        for c in chunks:
            c["url"] = doc["url"]
            c["category"] = doc["category"]
            c["title"] = doc["title"]
        all_chunks.extend(chunks)
        chunk_counts_by_doc[doc["source"]] = len(chunks)

    # 5. Upsert into ChromaDB
    indexed_count = vector_store.add_chunks(all_chunks)
    logger.info(f"Successfully upserted {indexed_count} chunks into ChromaDB.")

    # 6. Log in SQLite `documents` table
    own_session = False
    session = db
    if session is None:
        session = SessionLocal()
        own_session = True

    try:
        catalog_lookup = {item["source"]: item for item in MEDICAL_ARTICLES_CATALOG}
        for source_name, count in chunk_counts_by_doc.items():
            cat_info = catalog_lookup.get(source_name, {})
            existing = session.query(Document).filter(Document.source == source_name).first()
            if existing:
                existing.title = cat_info.get("title", existing.title)
                existing.url = cat_info.get("url", existing.url)
                existing.category = cat_info.get("category", existing.category)
                existing.chunk_count = count
                existing.ingested_at = utc_now()
            else:
                new_doc = Document(
                    source=source_name,
                    title=cat_info.get("title", source_name),
                    url=cat_info.get("url", ""),
                    category=cat_info.get("category", "General"),
                    chunk_count=count,
                    ingested_at=utc_now(),
                )
                session.add(new_doc)
        session.commit()
        logger.info(f"Logged {len(chunk_counts_by_doc)} documents in SQLite 'documents' table.")
    except Exception as ex:
        session.rollback()
        logger.error(f"Error persisting document metadata to SQLite: {ex}")
    finally:
        if own_session:
            session.close()

    duration_sec = (datetime.now(timezone.utc) - t0).total_seconds()
    last_run_iso = t0.isoformat()

    return {
        "status": "success",
        "documents_loaded": len(scraped_docs),
        "chunks_indexed": indexed_count,
        "duration_seconds": round(duration_sec, 2),
        "last_run": last_run_iso,
        "message": f"Successfully ingested {len(scraped_docs)} documents ({indexed_count} chunks) in {duration_sec:.2f}s",
    }


def main():
    """Command-line entrypoint."""
    result = run_ingestion_pipeline(force_reload=True)
    print("\n" + "=" * 60)
    print("[+] MedTwin AI Knowledge Base Ingestion Summary")
    print("=" * 60)
    print(f"Status:            {result['status']}")
    print(f"Documents Loaded:  {result['documents_loaded']}")
    print(f"Chunks Indexed:    {result['chunks_indexed']}")
    print(f"Duration:          {result['duration_seconds']}s")
    print(f"Last Run:          {result['last_run']}")
    print(f"Message:           {result['message']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
