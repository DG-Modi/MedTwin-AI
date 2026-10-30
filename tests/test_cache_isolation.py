"""Tests for QueryCache, patient data isolation, TTL expiration, and telemetry accuracy."""

import time
import pytest

from app.core.cache import QueryCache, generate_cache_key


def test_patient_cache_key_isolation():
    """Verify identical queries from different patients generate distinct cache keys.

    Crucial for clinical privacy: Patient A's personal medical context must NEVER leak to Patient B.
    """
    query = "What diet should I follow for my chronic condition?"
    key_patient_a = generate_cache_key(user_id="patient_001_diabetes", query=query)
    key_patient_b = generate_cache_key(user_id="patient_002_asthma", query=query)

    assert key_patient_a != key_patient_b
    assert isinstance(key_patient_a, str)
    assert len(key_patient_a) == 64  # SHA-256


def test_cache_query_normalization():
    """Verify cache keys normalize case and redundant whitespace."""
    q1 = "   What is normal blood pressure?   "
    q2 = "what is normal blood pressure?"
    k1 = generate_cache_key(user_id="p1", query=q1)
    k2 = generate_cache_key(user_id="p1", query=q2)

    assert k1 == k2


def test_cache_put_and_get(fresh_cache: QueryCache):
    """Verify storing and retrieving items from the cache."""
    key = generate_cache_key("p1", "test query")
    payload = {"answer": "Clinical recommendation", "sources": []}

    fresh_cache.set(key, payload)
    retrieved = fresh_cache.get(key)

    assert retrieved is not None
    assert retrieved["answer"] == "Clinical recommendation"


def test_cache_miss(fresh_cache: QueryCache):
    """Verify querying an uncached key returns None and increments miss counter."""
    key = generate_cache_key("p1", "uncached query")
    val = fresh_cache.get(key)
    assert val is None

    stats = fresh_cache.stats()
    assert stats["misses"] == 1
    assert stats["hits"] == 0
    assert stats["hit_rate"] == 0.0


def test_cache_hit_rate_telemetry(fresh_cache: QueryCache):
    """Verify hit rate telemetry calculation accuracy."""
    key = generate_cache_key("p1", "query_stat_test")
    fresh_cache.set(key, {"answer": "Cached answer"})

    # 1 Miss
    fresh_cache.get(generate_cache_key("p1", "non_existent"))
    # 1 Hit
    fresh_cache.get(key)

    stats = fresh_cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["total_queries"] == 2
    assert stats["hit_rate"] == 0.5


def test_cache_ttl_expiration():
    """Verify cache entries expire and are pruned after TTL expires."""
    cache = QueryCache(max_size=10, ttl_seconds=1)
    key = generate_cache_key("p1", "ttl_test_query")
    cache.set(key, {"data": "ephemeral"})

    # Immediate retrieval succeeds
    assert cache.get(key) is not None

    # Wait for TTL expiration (1.1s)
    time.sleep(1.1)

    # Retrieval after expiration returns None
    assert cache.get(key) is None


def test_cache_lru_eviction():
    """Verify cache respects max_size and evicts least recently accessed items."""
    cache = QueryCache(max_size=3, ttl_seconds=300)

    k1 = generate_cache_key("p1", "q1")
    k2 = generate_cache_key("p1", "q2")
    k3 = generate_cache_key("p1", "q3")
    k4 = generate_cache_key("p1", "q4")

    cache.set(k1, {"val": 1})
    cache.set(k2, {"val": 2})
    cache.set(k3, {"val": 3})

    # Access k1 so k2 becomes the least recently used
    cache.get(k1)

    # Insert k4, which should trigger eviction of k2
    cache.set(k4, {"val": 4})

    assert cache.get(k1) is not None
    assert cache.get(k2) is None  # Evicted
    assert cache.get(k3) is not None
    assert cache.get(k4) is not None


def test_cache_clear(fresh_cache: QueryCache):
    """Verify clear() purges all entries and resets statistics."""
    k1 = generate_cache_key("p1", "q1")
    fresh_cache.set(k1, {"val": 1})
    fresh_cache.get(k1)

    fresh_cache.clear()
    assert fresh_cache.get(k1) is None
    stats = fresh_cache.stats()
    assert stats["size"] == 0
