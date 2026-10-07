import fakeredis
from src.cache_key import CacheContext
from src.exact_cache import ExactMatchCache
from src.invalidation import InvalidationManager

fake_redis = fakeredis.FakeStrictRedis(decode_responses=True)
cache = ExactMatchCache(redis_client=fake_redis)
inv = InvalidationManager()

def make_ctx():
    return CacheContext(
        model="gpt-4", temperature=0.0, tool_set=("return_policy_lookup",),
        system_prompt_version=inv.combined_version(),
        customer_tier="standard", personalized=False
    )

ctx1 = make_ctx()
cache.set(ctx1, "what's your return policy?", "You have 30 days to return any item.")
print("Before policy change:", cache.get(ctx1, "what's your return policy?"))

# Policy changes: 30 days -> 45 days
inv.on_policy_change()
ctx2 = make_ctx()
print("After policy change:", cache.get(ctx2, "what's your return policy?"))  # should MISS

cache.set(ctx2, "what's your return policy?", "You have 45 days to return any item.")
print("New answer cached:", cache.get(ctx2, "what's your return policy?"))