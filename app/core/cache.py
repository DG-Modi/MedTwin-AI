"""In-memory TTL query cache for MedTwin AI.

NOTE: This is deterministic query-based caching with user-context isolation,
rather than embedding-based semantic vector caching.
This provides fast, predictable, zero-overhead caching for identical clinical
queries within a patient's specific health context for this 20-hour MVP.
"""

import hashlib
import json
import re
import time
from typing import Any, Dict, Optional, Tuple
from app.config import settings


class CacheEntry:
    """Represents a cached item with an expiration timestamp."""

    __slots__ = ("value", "expires_at", "created_at")

    def __init__(self, value: Any, ttl_seconds: int) -> None:
        self.value = value
        self.created_at = time.time()
        self.expires_at = self.created_at + ttl_seconds

    def is_expired(self) -> bool:
        """Returns True if the current time exceeds expires_at."""
        return time.time() > self.expires_at


class QueryCache:
    """In-memory key-value cache with TTL expiration and performance hit/miss metrics."""

    def __init__(self, default_ttl: Optional[int] = None) -> None:
        self.default_ttl = default_ttl if default_ttl is not None else settings.CACHE_TTL_SECONDS
        self._store: Dict[str, CacheEntry] = {}
        self._hits: int = 0
        self._misses: int = 0

    def get(self, key: str) -> Optional[Any]:
        """Retrieves a cached value by key if it exists and has not expired."""
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None

        if entry.is_expired():
            # Evict expired entry
            del self._store[key]
            self._misses += 1
            return None

        self._hits += 1
        return entry.value

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Stores a value in the cache with the given or default TTL."""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        self._store[key] = CacheEntry(value=value, ttl_seconds=ttl)

    def delete(self, key: str) -> bool:
        """Deletes a key from the cache. Returns True if found, False otherwise."""
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> None:
        """Flushes all cached entries and resets statistics."""
        self._store.clear()
        self._hits = 0
        self._misses = 0

    @property
    def hits(self) -> int:
        """Total cache hits."""
        return self._hits

    @property
    def misses(self) -> int:
        """Total cache misses."""
        return self._misses

    def stats(self) -> Dict[str, Any]:
        """Returns cache telemetry: hit count, miss count, hit ratio, and active items."""
        total = self._hits + self._misses
        hit_ratio = round(self._hits / total, 4) if total > 0 else 0.0
        # Clean expired items on stats check
        self._purge_expired()
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total_lookups": total,
            "hit_ratio": hit_ratio,
            "active_entries": len(self._store),
        }

    def _purge_expired(self) -> None:
        """Removes expired entries from internal dictionary."""
        expired_keys = [k for k, v in self._store.items() if v.is_expired()]
        for k in expired_keys:
            del self._store[k]


def normalize_query_text(query: str) -> str:
    """Normalizes query text: lowercase, trimmed, and single whitespace."""
    if not query:
        return ""
    lowered = query.strip().lower()
    return re.sub(r"\s+", " ", lowered)


def generate_profile_hash(user_id: str, profile_dict: Optional[Dict[str, Any]] = None) -> str:
    """Generates a deterministic hash representing the user's Digital Twin context.

    Ensures that cached responses for one user's medical background (e.g. penicillin allergy)
    are NEVER returned to another user with different clinical parameters.
    """
    if not profile_dict:
        raw_sig = f"user:{user_id}:no_profile"
        return hashlib.sha256(raw_sig.encode("utf-8")).hexdigest()[:16]

    # Extract key clinical context elements deterministically
    allergies = sorted([str(a).strip().lower() for a in profile_dict.get("allergies", []) if a])
    conditions = sorted([str(c).strip().lower() for c in profile_dict.get("chronic_conditions", []) if c])
    medications = sorted([str(m).strip().lower() for m in profile_dict.get("current_medications", []) if m])
    age = profile_dict.get("age", "")
    gender = str(profile_dict.get("gender", "")).strip().lower()

    payload = {
        "user_id": str(user_id),
        "age": age,
        "gender": gender,
        "allergies": allergies,
        "conditions": conditions,
        "medications": medications,
    }

    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def generate_cache_key(
    query: str,
    user_id: str,
    user_profile: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
    version: Optional[str] = None,
) -> str:
    """Constructs a deterministic cache key uniting query, user context, model, and version."""
    norm_query = normalize_query_text(query)
    query_hash = hashlib.sha256(norm_query.encode("utf-8")).hexdigest()[:16]
    profile_hash = generate_profile_hash(user_id=user_id, profile_dict=user_profile)
    model_id = model or settings.OLLAMA_MODEL
    version_id = version or settings.APP_VERSION

    return f"medtwin:cache:{model_id}:{version_id}:{profile_hash}:{query_hash}"


# Application singleton cache instance
global_cache = QueryCache()


def get_cache() -> QueryCache:
    """Dependency / accessor for the application cache."""
    return global_cache
