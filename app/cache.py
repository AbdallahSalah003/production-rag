"""
In Mem Cache with TTL for LLM response deduplication
In prod replace this with  Redis for:
- Persistence across restarts
- Shared cache across multiple instances
- Built in TTL management
"""

import hashlib
import time
from typing import Optional


class ResponseCache:
    """
    In-mem response cache with TTL
    """
    def __init__(self, ttl_seconds: int = 300):
        self.ttl = ttl_seconds
        self._cache: dict[str, dict] = {}
        self._hits: int  =0
        self._misses: int = 0

    def _make_key(self, query: str) -> str:
        """Create a cache key after normalization"""
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, query: str) -> Optional[str]:
        """Get cached res if it exists and hasn't expired"""
        key = self._make_key(query)
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                self._hits += 1
                return entry["response"]
            else:
                del self._cache[key]

        self._misses +=1
        return None 

    def set(self, query: str, response: str) -> None:
        """Cache a response"""
        key = self._make_key(query)
        self._cache[key] = {
            "response": response,
            "timestamp": time.time(),
            "query": query
        }

    @property
    def stats(self) -> dict:
        """Cache performance stats"""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "cached_entries": len(self._cache)
        }
