"""Knowledge base ingestion and clinical document data pipeline for MedTwin AI."""

from pipeline.scraper import scrape_medical_guidelines, MEDICAL_ARTICLES_CATALOG
from pipeline.run import run_ingestion_pipeline

__all__ = ["scrape_medical_guidelines", "MEDICAL_ARTICLES_CATALOG", "run_ingestion_pipeline"]
