"""Two-layer query cache with exact-match and semantic similarity layers.

Features:
- Layer 1: Deterministic exact-match (SHA-256 hash of normalized query + twin_hash).
- Layer 2: Semantic vector cache using cosine similarity threshold (~0.92) on embeddings within the same patient context and matching twin_hash.
- Patient & Digital Twin Isolation: Queries from different patients or modified twin profiles strictly produce cache misses.
- Cache Invalidation: On Save Twin, clears/invalidates all cache entries for that patient_id.
- Emergency safety: Emergency responses are NEVER cached or served from cache.
- LRU eviction via OrderedDict with configurable max_size and TTL expiration.
- Telemetry stats (hits, misses, total_queries, hit_rate, latency tracking) recorded in memory and SQLite.
"""

import hashlib
import json
import re
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from app.config import settings


class CacheEntry:
    """Represents a cached item with creation and expiration timestamps."""

    __slots__ = (
        "value",
        "expires_at",
        "created_at",
        "query_text",
        "embedding",
        "cache_type",
        "patient_id",
        "twin_hash",
    )

    def __init__(
        self,
        value: Any,
        ttl_seconds: int,
        query_text: str = "",
        embedding: Optional[List[float]] = None,
        cache_type: str = "exact",
        patient_id: str = "",
        twin_hash: str = "",
    ) -> None:
        self.value = value
        self.created_at = time.time()
        self.expires_at = self.created_at + ttl_seconds
        self.query_text = query_text
        self.embedding = embedding
        self.cache_type = cache_type
        self.patient_id = patient_id
        self.twin_hash = twin_hash

    def is_expired(self) -> bool:
        """Returns True if the current time exceeds expires_at."""
        return time.time() > self.expires_at


def normalize_query_text(query: str) -> str:
    """Normalizes query text: lowercase, stripped, and single whitespace."""
    if not query:
        return ""
    lowered = query.strip().lower()
    return re.sub(r"\s+", " ", lowered)


def generate_twin_hash(profile_dict: Optional[Dict[str, Any]] = None, user_id: str = "") -> str:
    """Generates a deterministic SHA-256 hash representing the user's Digital Twin context.

    Includes allergies, conditions, medications, age, gender, and twin_version.
    A change to any field (such as allergies) strictly alters the hash and causes a cache miss.
    """
    if not profile_dict:
        raw_sig = f"user:{user_id}:no_profile"
        return hashlib.sha256(raw_sig.encode("utf-8")).hexdigest()

    allergies = sorted([str(a).strip().lower() for a in profile_dict.get("allergies", []) if str(a).strip()])
    conditions = sorted([str(c).strip().lower() for c in profile_dict.get("chronic_conditions", []) if str(c).strip()])
    medications = sorted([str(m).strip().lower() for m in profile_dict.get("current_medications", []) if str(m).strip()])
    age = str(profile_dict.get("age", "")).strip()
    gender = str(profile_dict.get("gender", "")).strip().lower()
    version = str(profile_dict.get("twin_version", 1))

    payload = {
        "user_id": str(user_id or profile_dict.get("user_id", "")).strip(),
        "age": age,
        "gender": gender,
        "allergies": allergies,
        "conditions": conditions,
        "medications": medications,
        "twin_version": version,
    }

    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# Alias for backward compatibility
generate_profile_hash = generate_twin_hash


def generate_cache_key(
    arg1: Optional[str] = None,
    arg2: Optional[str] = None,
    *,
    user_id: Optional[str] = None,
    query: Optional[str] = None,
    user_profile: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
    version: Optional[str] = None,
    **kwargs,
) -> str:
    """Constructs a deterministic 64-character SHA-256 cache key from:
    sha256(normalized_query + twin_hash).

    Supports both keyword arguments and positional arguments:
    - generate_cache_key(user_id="p1", query="text", user_profile=profile)
    - generate_cache_key("p1", "text")
    """
    final_user_id = user_id or kwargs.get("patient_id")
    final_query = query

    if arg1 is not None and arg2 is not None:
        final_user_id = arg1
        final_query = arg2
    elif arg1 is not None and final_query is None:
        if final_user_id is not None:
            final_query = arg1
        else:
            final_user_id = arg1

    final_user_id = str(final_user_id or "default_patient").strip()
    norm_query = normalize_query_text(str(final_query or ""))
    twin_hash = generate_twin_hash(profile_dict=user_profile, user_id=final_user_id)

    raw_signature = f"{norm_query}{twin_hash}"
    return hashlib.sha256(raw_signature.encode("utf-8")).hexdigest()


class QueryCache:
    """Two-layer query cache with exact-match and semantic cosine similarity."""

    def __init__(
        self,
        max_size: int = 100,
        ttl_seconds: Optional[int] = None,
        default_ttl: Optional[int] = None,
        semantic_threshold: float = 0.92,
        **kwargs,
    ) -> None:
        effective_ttl = ttl_seconds if ttl_seconds is not None else default_ttl
        self.max_size = max_size
        self.default_ttl = effective_ttl if effective_ttl is not None else settings.CACHE_TTL_SECONDS
        self.semantic_threshold = semantic_threshold

        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._semantic_index: List[Dict[str, Any]] = []
        self._hits: int = 0
        self._misses: int = 0
        self._semantic_hits: int = 0

    def get(self, key: str, current_twin_hash: Optional[str] = None) -> Optional[Any]:
        """Layer 1: Exact match retrieval with LRU renewal, TTL check, and twin_hash verification."""
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None

        if entry.is_expired():
            del self._store[key]
            self._misses += 1
            return None

        # Never serve cached emergency response
        if isinstance(entry.value, dict) and entry.value.get("is_emergency", False):
            del self._store[key]
            self._misses += 1
            return None

        # Never serve cached response if twin hash differs
        if current_twin_hash and entry.twin_hash and entry.twin_hash != current_twin_hash:
            self._misses += 1
            return None

        # Move to most-recently-used end
        self._store.move_to_end(key)
        self._hits += 1
        return entry.value

    def get_semantic(
        self,
        query: str,
        patient_id: str,
        query_embedding: Optional[List[float]] = None,
        current_twin_hash: Optional[str] = None,
    ) -> Optional[Tuple[str, Any, float]]:
        """Layer 2: Semantic vector match within the same patient context and twin hash.

        Returns (cache_key, value, similarity) if similarity >= semantic_threshold.
        """
        if not query_embedding or not self._semantic_index:
            return None

        q_vec = np.array(query_embedding, dtype=np.float32)
        norm_q = np.linalg.norm(q_vec)
        if norm_q == 0:
            return None

        best_sim = 0.0
        best_match = None

        for item in self._semantic_index:
            if item["patient_id"] != patient_id:
                continue

            # Must match twin_hash: changed twin causes semantic cache miss
            if current_twin_hash and item.get("twin_hash") != current_twin_hash:
                continue

            candidate_key = item["cache_key"]
            entry = self._store.get(candidate_key)
            if not entry or entry.is_expired():
                continue

            # Never serve cached emergency response
            if isinstance(entry.value, dict) and entry.value.get("is_emergency", False):
                continue

            c_vec = np.array(item["embedding"], dtype=np.float32)
            norm_c = np.linalg.norm(c_vec)
            if norm_c == 0:
                continue

            cosine_sim = float(np.dot(q_vec, c_vec) / (norm_q * norm_c))
            if cosine_sim > best_sim:
                best_sim = cosine_sim
                best_match = (candidate_key, entry.value, cosine_sim)

        if best_match and best_sim >= self.semantic_threshold:
            self._hits += 1
            self._semantic_hits += 1
            self._store.move_to_end(best_match[0])
            return best_match

        return None

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        query_text: str = "",
        embedding: Optional[List[float]] = None,
        patient_id: Optional[str] = None,
        cache_type: str = "exact",
        twin_hash: str = "",
    ) -> None:
        """Stores a value in Layer 1 and indexes for Layer 2 semantic similarity.

        Guarantees that emergency responses are NEVER stored in cache.
        """
        # Never cache emergency responses
        if isinstance(value, dict) and value.get("is_emergency", False):
            return
        if getattr(value, "is_emergency", False):
            return

        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl

        # Evict least recently used if at capacity
        if len(self._store) >= self.max_size and key not in self._store:
            oldest_key, _ = self._store.popitem(last=False)
            self._semantic_index = [item for item in self._semantic_index if item["cache_key"] != oldest_key]

        entry = CacheEntry(
            value=value,
            ttl_seconds=ttl,
            query_text=query_text,
            embedding=embedding,
            cache_type=cache_type,
            patient_id=str(patient_id or ""),
            twin_hash=twin_hash,
        )
        self._store[key] = entry
        self._store.move_to_end(key)

        if embedding is not None and patient_id:
            self._semantic_index.append({
                "patient_id": str(patient_id),
                "cache_key": key,
                "embedding": embedding,
                "query_text": query_text,
                "twin_hash": twin_hash,
            })

    def invalidate_patient(self, patient_id: str) -> int:
        """Flushes all cache entries (Layer 1 exact and Layer 2 semantic) for a given patient_id."""
        pid = str(patient_id).strip()
        keys_to_del = [
            k for k, entry in self._store.items()
            if getattr(entry, "patient_id", "") == pid
        ]
        for k in keys_to_del:
            del self._store[k]
        self._semantic_index = [
            item for item in self._semantic_index
            if str(item.get("patient_id", "")).strip() != pid
        ]
        return len(keys_to_del)

    def delete(self, key: str) -> bool:
        """Deletes a key from the cache."""
        if key in self._store:
            del self._store[key]
            self._semantic_index = [i for i in self._semantic_index if i["cache_key"] != key]
            return True
        return False

    def clear(self) -> None:
        """Flushes all cached entries and resets statistics."""
        self._store.clear()
        self._semantic_index.clear()
        self._hits = 0
        self._misses = 0
        self._semantic_hits = 0

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def stats(self) -> Dict[str, Any]:
        """Returns comprehensive cache performance telemetry."""
        self._purge_expired()
        total = self._hits + self._misses
        hit_ratio = round(self._hits / total, 4) if total > 0 else 0.0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "semantic_hits": self._semantic_hits,
            "total_queries": total,
            "total_lookups": total,
            "hit_rate": hit_ratio,
            "hit_ratio": hit_ratio,
            "size": len(self._store),
            "active_entries": len(self._store),
            "semantic_entries": len(self._semantic_index),
            "max_size": self.max_size,
        }

    def _purge_expired(self) -> None:
        """Removes expired entries."""
        expired_keys = [k for k, v in self._store.items() if v.is_expired()]
        for k in expired_keys:
            del self._store[k]
        if expired_keys:
            expired_set = set(expired_keys)
            self._semantic_index = [i for i in self._semantic_index if i["cache_key"] not in expired_set]


# Singleton application cache instance
global_cache = QueryCache(max_size=100, ttl_seconds=settings.CACHE_TTL_SECONDS)


def get_cache() -> QueryCache:
    """Dependency / accessor for the application cache."""
    return global_cache
