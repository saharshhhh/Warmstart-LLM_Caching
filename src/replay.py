import json
import time
import random
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional

from src.cache_key import CacheContext
from src.exact_cache import ExactMatchCache
from src.semantic_cache import SemanticCache
from src.prompt_cache import ProviderPromptCache

FULL_CALL_COST_USD = 0.006
FULL_CALL_LATENCY_MS = 800


@dataclass
class ReplayRecord:
    query_id: str
    intent: str
    personalized: bool
    layer: str              # "exact" | "semantic" | "miss"
    cost_usd: float
    latency_ms: float
    similarity: Optional[float] = None
    served_answer: Optional[str] = None


def simulate_llm_answer(intent: str) -> str:
    canned = {
        "order_status": "Your order is currently in transit and should arrive within 3-5 business days.",
        "return_policy": "You have 30 days to return any item.",
        "shipping_canada": "Yes, we ship to Canada with standard international rates.",
        "change_address": "You can update your shipping address within 1 hour of placing the order.",
        "payment_methods": "We accept Visa, Mastercard, PayPal, and Apple Pay.",
        "unique": "Let me look into that specific product detail for you.",
    }
    return canned.get(intent, "I'm not sure, let me check.")


def run_replay(traffic, exact_cache: ExactMatchCache, semantic_cache: SemanticCache,
               prompt_cache: ProviderPromptCache, system_prompt_version: str):
    records = []
    now = time.time()

    for q in traffic:
        ctx = CacheContext(
            model="command-a-plus-05-2026", temperature=0.0, tool_set=("order_lookup", "return_policy_lookup"),
            system_prompt_version=system_prompt_version, customer_tier=q["customer_tier"],
            personalized=q["personalized"],
            customer_id=q["customer_id"] if q["personalized"] else None,
        )

        # Layer 1: exact match
        exact_result = exact_cache.get(ctx, q["text"])
        if exact_result.hit:
            records.append(ReplayRecord(
                query_id=q["query_id"], intent=q["intent"], personalized=q["personalized"],
                layer="exact", cost_usd=0.0, latency_ms=exact_result.latency_ms,
                served_answer=exact_result.answer
            ))
            continue

        # Layer 2: semantic match
        sem_result = semantic_cache.get(ctx, q["text"])
        if sem_result.hit:
            records.append(ReplayRecord(
                query_id=q["query_id"], intent=q["intent"], personalized=q["personalized"],
                layer="semantic", cost_usd=0.0, latency_ms=sem_result.latency_ms,
                similarity=sem_result.similarity,
                served_answer=sem_result.answer
            ))
            continue

        # Full miss -> simulate LLM call (with provider prompt caching applied)
        now += 0.01
        llm_est = prompt_cache.estimate_call(
            system_prompt_version=system_prompt_version,
            static_prefix_tokens=1200, dynamic_tokens=len(q["text"].split()) * 2,
            output_tokens=150, now=now
        )
        answer = simulate_llm_answer(q["intent"])

        exact_cache.set(ctx, q["text"], answer)
        semantic_cache.set(ctx, q["text"], answer)

        records.append(ReplayRecord(
            query_id=q["query_id"], intent=q["intent"], personalized=q["personalized"],
            layer="miss", cost_usd=llm_est.cost_usd, latency_ms=llm_est.latency_ms,
            served_answer=answer
        ))

    return records


def summarize(records):
    total = len(records)
    exact_hits = sum(1 for r in records if r.layer == "exact")
    semantic_hits = sum(1 for r in records if r.layer == "semantic")
    misses = sum(1 for r in records if r.layer == "miss")

    total_cost = sum(r.cost_usd for r in records)
    baseline_cost = total * FULL_CALL_COST_USD

    latencies = np.array([r.latency_ms for r in records])

    return {
        "total_queries": total,
        "exact_hit_rate": round(exact_hits / total, 4),
        "semantic_hit_rate": round(semantic_hits / total, 4),
        "overall_hit_rate": round((exact_hits + semantic_hits) / total, 4),
        "miss_rate": round(misses / total, 4),
        "total_cost_usd": round(total_cost, 4),
        "baseline_cost_usd": round(baseline_cost, 4),
        "cost_savings_pct": round(100 * (1 - total_cost / baseline_cost), 2),
        "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "baseline_p50_latency_ms": FULL_CALL_LATENCY_MS,
        "baseline_p95_latency_ms": FULL_CALL_LATENCY_MS,
    }