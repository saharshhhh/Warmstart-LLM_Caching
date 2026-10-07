from dataclasses import dataclass

# Rough, documented assumptions -- not claimed as exact provider numbers.
# Anthropic/OpenAI prompt caching typically discounts cached input tokens
# ~90% and cuts their processing-latency contribution sharply.
CACHED_PREFIX_DISCOUNT = 0.90
CACHED_PREFIX_LATENCY_REDUCTION = 0.80


@dataclass
class LLMCallEstimate:
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    prefix_was_cached: bool


class ProviderPromptCache:
    """
    Simulates provider-side prefix caching for the STATIC system block
    (system prompt + tool definitions). Tracks whether that prefix was
    "warm" (sent recently enough to still be cached provider-side) using
    a simple TTL, since we don't have a real billing API to query.
    """
    def __init__(self, prefix_ttl_seconds=300,
                 cost_per_1k_input=0.003, cost_per_1k_output=0.015,
                 base_latency_ms=800):
        self.prefix_ttl = prefix_ttl_seconds
        self.cost_per_1k_input = cost_per_1k_input
        self.cost_per_1k_output = cost_per_1k_output
        self.base_latency_ms = base_latency_ms
        self._warm_until = {}  # system_prompt_version -> timestamp

    def estimate_call(self, system_prompt_version: str, static_prefix_tokens: int,
                       dynamic_tokens: int, output_tokens: int, now: float) -> LLMCallEstimate:
        warm = self._warm_until.get(system_prompt_version, 0) > now
        self._warm_until[system_prompt_version] = now + self.prefix_ttl

        total_input = static_prefix_tokens + dynamic_tokens

        if warm:
            effective_input = static_prefix_tokens * (1 - CACHED_PREFIX_DISCOUNT) + dynamic_tokens
            latency = self.base_latency_ms * (1 - CACHED_PREFIX_LATENCY_REDUCTION)
        else:
            effective_input = total_input
            latency = self.base_latency_ms

        cost = (effective_input / 1000) * self.cost_per_1k_input + (output_tokens / 1000) * self.cost_per_1k_output

        return LLMCallEstimate(
            input_tokens=total_input, output_tokens=output_tokens,
            cost_usd=round(cost, 6), latency_ms=round(latency, 2),
            prefix_was_cached=warm
        )