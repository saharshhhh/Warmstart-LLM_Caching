import json
import time
import redis
from dataclasses import dataclass
from typing import Optional
from src.cache_key import CacheContext, build_exact_key


@dataclass
class CacheResult:
    hit: bool
    answer: Optional[str] = None
    latency_ms: Optional[float] = None
    key: Optional[str] = None


class ExactMatchCache:
    def __init__(self, redis_client=None, ttl_seconds=3600):
        self.r = redis_client or redis.Redis(host="localhost", port=6379, decode_responses=True)
        self.ttl = ttl_seconds

    def get(self, ctx: CacheContext, query_text: str) -> CacheResult:
        key = build_exact_key(ctx, query_text)
        start = time.perf_counter()
        raw = self.r.get(key)
        latency_ms = (time.perf_counter() - start) * 1000

        if raw is None:
            return CacheResult(hit=False, key=key, latency_ms=latency_ms)

        data = json.loads(raw)
        return CacheResult(hit=True, answer=data["answer"], latency_ms=latency_ms, key=key)

    def set(self, ctx: CacheContext, query_text: str, answer: str):
        key = build_exact_key(ctx, query_text)
        payload = json.dumps({"answer": answer, "cached_at": time.time()})
        self.r.setex(key, self.ttl, payload)
        return key

    def flush(self):
        """For testing only -- clears everything."""
        self.r.flushdb()