"""FastAPI application entrypoint for MedTwin AI."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db.session import init_db
from app.rag.pipeline import get_rag_pipeline
from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle handler."""
    # 1. Initialize SQLite database tables
    init_db()

    # 2. Automatically ingest medical knowledge base if ChromaDB is empty
    rag = get_rag_pipeline()
    if rag.vector_store.count() == 0:
        rag.ingest_knowledge_base()

    yield


app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Educational medical-information assistant and Digital Twin personalizer.\n\n"
        "**Clinical Disclaimer**: This application is strictly an educational tool and does NOT "
        "provide medical diagnoses, treatment prescriptions, or clinical decision support."
    ),
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api.main:app", host="127.0.0.1", port=8000, reload=True)
