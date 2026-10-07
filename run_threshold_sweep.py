import json
import fakeredis
from src.cache_key import CacheContext
from src.exact_cache import ExactMatchCache
from src.semantic_cache import SemanticCache
from src.prompt_cache import ProviderPromptCache
from src.replay import simulate_llm_answer
from src.threshold_curve import sweep_thresholds

traffic = json.load(open("data/traffic.json"))

fake_redis = fakeredis.FakeStrictRedis(decode_responses=True)
exact_cache = ExactMatchCache(redis_client=fake_redis)
semantic_cache = SemanticCache(fake_redis, similarity_threshold=0.80)
prompt_cache = ProviderPromptCache()

comparison_log = []
traffic_by_query_text = {}

# Run the SAME replay logic, but capture every semantic comparison made,
# and build the query_text -> intent map as we go.
now_counter = 0
for q in traffic:
    ctx = CacheContext(
        model="gpt-4", temperature=0.0, tool_set=("order_lookup", "return_policy_lookup"),
        system_prompt_version="v1", customer_tier=q["customer_tier"],
        personalized=q["personalized"],
        customer_id=q["customer_id"] if q["personalized"] else None,
    )
    traffic_by_query_text[q["text"]] = q["intent"]

    exact_result = exact_cache.get(ctx, q["text"])
    if exact_result.hit:
        continue

    sem_result = semantic_cache.get(ctx, q["text"], log_all=comparison_log)
    if sem_result.hit:
        continue

    answer = simulate_llm_answer(q["intent"])
    exact_cache.set(ctx, q["text"], answer)
    semantic_cache.set(ctx, q["text"], answer)

curve = sweep_thresholds(comparison_log, traffic_by_query_text)

print(f"Captured {len(comparison_log)} semantic comparisons\n")
print(f"{'Threshold':<12}{'Hits':<8}{'False Hits':<12}{'False Hit Rate':<16}")
for row in curve:
    print(f"{row['threshold']:<12}{row['hits']:<8}{row['false_hits']:<12}{row['false_hit_rate']:<16}")

with open("reports/threshold_curve.json", "w") as f:
    json.dump(curve, f, indent=2)