# run_judge_check.py
import json
import fakeredis
from src.cache_key import CacheContext
from src.exact_cache import ExactMatchCache
from src.semantic_cache import SemanticCache
from src.replay import simulate_llm_answer
from src.judge_eval import judge_sample

traffic = json.load(open("data/traffic.json"))

fake_redis = fakeredis.FakeStrictRedis(decode_responses=True)
exact_cache = ExactMatchCache(redis_client=fake_redis)
semantic_cache = SemanticCache(fake_redis, similarity_threshold=0.80)

comparison_log = []
for q in traffic:
    ctx = CacheContext(
        model="command-a-plus-05-2026", temperature=0.0, tool_set=("order_lookup", "return_policy_lookup"),
        system_prompt_version="v1", customer_tier=q["customer_tier"],
        personalized=q["personalized"],
        customer_id=q["customer_id"] if q["personalized"] else None,
    )
    exact_result = exact_cache.get(ctx, q["text"])
    if exact_result.hit:
        continue
    sem_result = semantic_cache.get(ctx, q["text"], log_all=comparison_log)
    if sem_result.hit:
        continue
    answer = simulate_llm_answer(q["intent"])
    exact_cache.set(ctx, q["text"], answer)
    semantic_cache.set(ctx, q["text"], answer)

print("=== Judging at threshold 0.80 (production) ===")
result_80 = judge_sample(comparison_log, threshold=0.80, sample_size=12)
print(f"Sample size: {result_80['sample_size']}, judge false-hit rate: {result_80['judge_false_hit_rate']}")
for f in result_80["flagged"]:
    print(f"  FLAGGED: '{f['query']}' <- '{f['answer']}' | {f['judge_reason']}")

print("\n=== Judging at threshold 0.65 (edge of curve) ===")
result_65 = judge_sample(comparison_log, threshold=0.65, sample_size=20)
print(f"Sample size: {result_65['sample_size']}, judge false-hit rate: {result_65['judge_false_hit_rate']}")
for f in result_65["flagged"]:
    print(f"  FLAGGED: '{f['query']}' <- '{f['answer']}' | {f['judge_reason']}")

with open("reports/judge_results.json", "w") as f:
    json.dump({"threshold_0.80": result_80, "threshold_0.65": result_65}, f, indent=2)