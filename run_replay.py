import json
import fakeredis
from src.exact_cache import ExactMatchCache
from src.semantic_cache import SemanticCache
from src.prompt_cache import ProviderPromptCache
from src.replay import run_replay, summarize
from precision_eval import compute_ground_truth_false_hit_rate

traffic = json.load(open("data/traffic.json"))

fake_redis = fakeredis.FakeStrictRedis(decode_responses=True)
exact_cache = ExactMatchCache(redis_client=fake_redis)
semantic_cache = SemanticCache(fake_redis, similarity_threshold=0.80)
prompt_cache = ProviderPromptCache()

records = run_replay(traffic, exact_cache, semantic_cache, prompt_cache, system_prompt_version="v1")
summary = summarize(records)
records_dicts = [vars(r) for r in records]

precision = compute_ground_truth_false_hit_rate(records_dicts)

print(json.dumps(summary, indent=2))
print("\n--- PRECISION CHECK ---")
print(json.dumps(precision, indent=2, default=str))

with open("reports/replay_results.json", "w") as f:
    json.dump({"summary": summary, "precision": precision, "records": records_dicts}, f, indent=2, default=str)