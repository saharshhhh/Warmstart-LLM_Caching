import json
import time
import uuid
import numpy as np
from dataclasses import dataclass
from typing import Optional
from sentence_transformers import SentenceTransformer
from src.cache_key import CacheContext, build_semantic_namespace


@dataclass
class SemanticCacheResult:
    hit: bool
    answer: Optional[str] = None
    similarity: Optional[float] = None
    matched_text: Optional[str] = None
    latency_ms: Optional[float] = None
    namespace: Optional[str] = None


class SemanticCache:
    def __init__(self, redis_client, embedder=None, similarity_threshold=0.85, ttl_seconds=3600):
        self.r = redis_client
        self.model = embedder or SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.threshold = similarity_threshold
        self.ttl = ttl_seconds

    def _index_key(self, namespace: str) -> str:
        return f"{namespace}:idx"   # a Redis SET of entry ids, no expiry itself

    def _entry_key(self, namespace: str, entry_id: str) -> str:
        return f"{namespace}:entry:{entry_id}"  # individual entry, HAS expiry

    def get(self, ctx: CacheContext, query_text: str, log_all: list = None) -> SemanticCacheResult:
        namespace = build_semantic_namespace(ctx)
        start = time.perf_counter()

        entry_ids = self.r.smembers(self._index_key(namespace))
        if not entry_ids:
            latency_ms = (time.perf_counter() - start) * 1000
            return SemanticCacheResult(hit=False, namespace=namespace, latency_ms=latency_ms)

        query_emb = self.model.encode([query_text], normalize_embeddings=True)[0]

        best_sim, best_entry = -1.0, None
        stale_ids = []

        for eid in entry_ids:
            raw = self.r.get(self._entry_key(namespace, eid))
            if raw is None:
                stale_ids.append(eid)
                continue
            data = json.loads(raw)
            stored_emb = np.array(data["embedding"])
            sim = float(np.dot(query_emb, stored_emb))

            # NEW: log every comparison if a logging list was passed in
            if log_all is not None:
                log_all.append({
                    "query_text": query_text,
                    "matched_text": data["text"],
                    "matched_answer": data["answer"],
                    "similarity": sim,
                })

            if sim > best_sim:
                best_sim, best_entry = sim, data

        if stale_ids:
            self.r.srem(self._index_key(namespace), *stale_ids)

        latency_ms = (time.perf_counter() - start) * 1000

        if best_entry is not None and best_sim >= self.threshold:
            return SemanticCacheResult(
                hit=True, answer=best_entry["answer"], similarity=best_sim,
                matched_text=best_entry["text"], namespace=namespace, latency_ms=latency_ms
            )
        return SemanticCacheResult(
            hit=False, similarity=best_sim if best_entry else None,
            namespace=namespace, latency_ms=latency_ms
        )

    def set(self, ctx: CacheContext, query_text: str, answer: str):
        namespace = build_semantic_namespace(ctx)
        embedding = self.model.encode([query_text], normalize_embeddings=True)[0]

        entry_id = uuid.uuid4().hex[:12]
        payload = json.dumps({
            "text": query_text,
            "answer": answer,
            "embedding": embedding.tolist(),
            "cached_at": time.time(),
        })
        self.r.setex(self._entry_key(namespace, entry_id), self.ttl, payload)
        self.r.sadd(self._index_key(namespace), entry_id)
        return entry_id