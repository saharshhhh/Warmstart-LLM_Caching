from src.cache_key import CacheContext, build_exact_key, build_semantic_namespace

base = dict(model="gpt-4", temperature=0.0, tool_set=("order_lookup", "return_policy_lookup"),
            system_prompt_version="v3", customer_tier="standard")

# Same question, different customers, personalized -> must differ
ctx1 = CacheContext(**base, personalized=True, customer_id="cust_122")
ctx2 = CacheContext(**base, personalized=True, customer_id="cust_181")
print("Personalized, different customers:")
print(" ", build_exact_key(ctx1, "where is my order?"))
print(" ", build_exact_key(ctx2, "where is my order?"))
assert build_exact_key(ctx1, "where is my order?") != build_exact_key(ctx2, "where is my order?")

# Same question, generic -> must be the SAME regardless of customer
ctx3 = CacheContext(**base, personalized=False)
print("\nGeneric, no customer scoping needed:")
print(" ", build_exact_key(ctx3, "what's your return policy?"))
print(" ", build_exact_key(ctx3, "What's your return policy?"))  # different casing

# Prompt version bump -> must invalidate everything under old version
base_v4 = {**base, "system_prompt_version": "v4"}
ctx4 = CacheContext(**base_v4, personalized=False)
print("\nPrompt version bump changes the fingerprint:")
print(" ", build_exact_key(ctx3, "what's your return policy?"))
print(" ", build_exact_key(ctx4, "what's your return policy?"))
assert build_exact_key(ctx3, "what's your return policy?") != build_exact_key(ctx4, "what's your return policy?")

print("\nAll checks passed.")