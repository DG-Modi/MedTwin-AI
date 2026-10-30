"""Database session management and engine configuration."""

from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from app.config import settings

# SQLite connection args (check_same_thread=False allows FastAPI multi-threaded requests)
connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for obtaining a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Creates database tables if they do not exist, and runs lightweight schema upgrades."""
    # Import models here to ensure registration with Base metadata
    from sqlalchemy import text
    from app.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)

    # Ensure twin_version column exists in SQLite user_profiles table
    try:
        with engine.connect() as conn:
            columns = [row[1] for row in conn.execute(text("PRAGMA table_info(user_profiles)")).fetchall()]
            if columns and "twin_version" not in columns:
                conn.execute(text("ALTER TABLE user_profiles ADD COLUMN twin_version INTEGER DEFAULT 1"))
                conn.commit()
    except Exception:
        pass
