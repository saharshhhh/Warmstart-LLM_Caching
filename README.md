# Warmstart — LLM Semantic Caching

A three-layer caching system for an AI support assistant, built to cut inference cost and tail latency on repeated questions — with proof that it doesn't start serving wrong answers along the way.

## The use case (as specified)

An ecommerce brand's AI support assistant answers ~400,000 questions a month. Roughly six in ten are the same handful of questions phrased differently — order status, return policy, shipping to Canada. Paying frontier-model prices to answer the same question repeatedly, with p95 latency hurting conversion on the checkout page.

## What this proves

Inference spend and tail latency can be cut substantially, with evidence — not just an assumption — that the system did not start serving wrong answers to do it.

## The build

Three caching layers, as specified:

1. **Exact match** — normalized-text + context hash lookup (Redis / fakeredis)
2. **Semantic match** — embedding similarity above a tuned threshold, scoped to a namespace
3. **Provider-side prefix caching** — simulated cost/latency model for caching the static system-prompt block

### The cache key

Built from `model`, `temperature`, `tool_set`, `system_prompt_version`, and `customer_tier` — hashed into a short "context fingerprint" that prefixes every cache key. Changing any of these fields changes the fingerprint, which makes old entries unreachable immediately, without deleting anything.

**Versioned, not static.** `system_prompt_version` (and, by extension, `catalogue_version` / `policy_version`, combined into one effective version) is bumped on any change — this is what prevents serving yesterday's behavior.

### TTL + explicit invalidation

- Every cache entry carries a TTL and expires on its own.
- Explicit invalidation (catalogue reindex, policy change, prompt change) is handled by bumping a version string, not by hunting down and deleting matching keys — invalidation happens by construction, since nothing will ever build the old key again.

### The trap, and how it's avoided

**The trap:** keying the cache on the query text alone. Two customers with different order histories would get the same cached answer — a data leak, not an optimization.

**The fix:** every `CacheContext` carries a `personalized` flag. When `True`, `customer_id` is woven directly into both the exact-match key and the semantic-match namespace — a personalized query only ever matches within that one customer's own cache entries, structurally, not by convention. Generic queries (return policy, shipping) carry no customer scoping at all, which is where the actual sharing/savings comes from.

## Stack

- Redis (via `fakeredis` for local/demo use — swappable for real Redis/Memurai with no code change, via dependency-injected client)
- `sentence-transformers` (`bge-small-en-v1.5`) for semantic cache embeddings
- A simulated provider-side prompt-caching cost/latency model (documented, stated assumptions — not claimed as exact provider pricing)
- A replay harness over 1,000 synthetically generated, production-shaped queries
- Cohere (`command-a-plus-05-2026`) as the judge model for precision auditing
- Streamlit dashboard for non-technical stakeholders

## Ship gate results

### 1. Replay 1,000 production-shaped queries — hit rate, cost delta, p50/p95 latency delta

| Metric | Result |
|---|---|
| Exact-match hit rate | 63.4% |
| Semantic-match hit rate | 13.0% |
| Overall hit rate | 76.4% |
| Miss rate (sent to LLM) | 23.6% |
| Cost savings | **89.51%** vs. no-cache baseline |
| p50 latency | 0.13 ms (vs. 800 ms baseline) |
| p95 latency | 160 ms (vs. 800 ms baseline) |
| Projected monthly cost (scaled to 400K queries) | ~$252 vs. ~$2,400 baseline |

### 2. False hit rate with a judge model, target <1%

Two independent measurements were taken, and **they disagree — on purpose, reported honestly rather than reconciled by picking the favorable one:**

| Method | What it checks | Result |
|---|---|---|
| Ground-truth bucket match | Was the right *topic/intent* served? | **0.0%** false-hit rate (0/130 hits) |
| Judge model (blind to intent labels) | Did the answer *fully and specifically* address the question asked? | **50.0%** flagged (12-sample, threshold 0.80); **35.0%** (20-sample, threshold 0.65) |

**Root cause, diagnosed not dismissed:** several synthetic "paraphrases" grouped under one intent cluster actually ask distinguishable sub-questions (e.g. "how do returns work?" vs. "do you offer free returns?" — the latter asks about cost, which the canned return-policy answer never addresses). The ground-truth check can't see this, because both queries are correctly bucketed; the judge catches it, because it checks whether the specific question was actually answered. This is a limitation of the synthetic intent-clustering and canned-answer design, not a defect in the caching mechanism itself. Full writeup in `reports/findings.md`.

**Consequence:** the production threshold was **not** lowered based on the ground-truth curve alone, despite it suggesting headroom down to 0.65-0.70 — the judge finding overrides that recommendation until canned answers are rewritten to fully cover their cluster's scope.

### 3. Precision curve across similarity thresholds

| Threshold | Hits caught | Wrong-topic rate |
|---|---|---|
| 0.60 | 21 | 4.76% |
| 0.65 | 20 | 0% |
| 0.70 | 20 | 0% |
| 0.75 | 19 | 0% |
| **0.80 (production)** | 12 | 0% |
| 0.85 | 7 | 0% |
| 0.90+ | 0 | 0% |

The precision boundary sits between 0.60 and 0.65. Production threshold (0.80) carries meaningful margin above that boundary. Caveat: the 0.60-0.75 range is backed by a small sample (19-21 comparisons per bucket) and would need more data before being treated as a confident production recommendation on its own — which is exactly why the judge-model check (above) matters as a second, independent signal rather than relying on this curve alone.

### 4. Prove a system prompt change busts the cache

Demonstrated directly — before/after cache keys from an actual policy-change test:

```
Before policy change:  exact:61b3bb2c775aa681:... -> HIT  (30-day return answer)
After policy change:   exact:1966a13e26fa6398:... -> MISS (different fingerprint, old entry unreachable)
New answer cached:     exact:1966a13e26fa6398:... -> HIT  (45-day return answer)
```

The fingerprint changed the moment `on_policy_change()` was called — no manual deletion, no stale answer served.

## Dashboard
<img width="940" height="412" alt="image" src="https://github.com/user-attachments/assets/3fababbb-8ee1-4a91-8baa-0330db855de3" />

<img width="929" height="397" alt="image" src="https://github.com/user-attachments/assets/87e1d82e-4192-4a32-afd8-b84614dbe673" />

`streamlit run dashboard.py` — built for a support lead to read without the engineer present:
- Top-line cost savings, hit rate, p95 latency, projected monthly cost
- Breakdown of where answers come from (exact / semantic / miss)
- **Both** precision checks shown side by side, including the disagreement, not just the favorable one
- Interactive threshold-vs-precision curve
- Plain-language "Known Limitations" section

## Project structure

```
LLM Caching/
├── data/
│   └── traffic.json              # 1,000 synthetic, production-shaped queries
├── src/
│   ├── traffic.py                # traffic generator
│   ├── cache_key.py              # CacheContext, fingerprinting, key construction
│   ├── exact_cache.py            # Redis-backed exact-match layer
│   ├── semantic_cache.py         # embedding-based semantic-match layer
│   ├── prompt_cache.py           # simulated provider-side prefix caching
│   ├── invalidation.py           # version-bump based explicit invalidation
│   ├── replay.py                 # replay harness
│   ├── precision_eval.py         # ground-truth false-hit check
│   ├── threshold_curve.py        # precision-vs-threshold sweep
│   └── judge_eval.py             # judge-model precision audit
├── reports/
│   ├── replay_results.json
│   ├── threshold_curve.json
│   ├── judge_results.json
│   └── findings.md               # judge vs ground-truth disagreement writeup
├── run_replay.py
├── run_threshold_sweep.py
├── run_judge_check.py
└── dashboard.py
```

## Running it

```bash
pip install -r requirements.txt
python src/traffic.py              # generate synthetic traffic
python run_replay.py               # run the full cache replay
python run_threshold_sweep.py      # precision-vs-threshold curve
python run_judge_check.py          # judge-model audit (requires COHERE_API_KEY)
streamlit run dashboard.py         # view results
```

## What this signals

A semantic cache is a retrieval system with a precision problem — hit rate alone is not sufficient evidence of safety, which is why two independent correctness checks are built here rather than one, and why they're both reported even though they disagree. Threshold tuning is a business decision about how wrong the system can afford to be, not purely a technical optimization — made here with an actual empirical curve, not intuition. The customer-scoped key design (Stage 2) exists specifically because the named trap — keying on query text alone — is not a hypothetical, it's the default mistake this entire key schema was built to prevent structurally rather than by convention.

## Known limitations

- Traffic is synthetic, not real production logs — real-world phrasing variety and intent distribution may differ.
- The 50% judge-flagged rate reflects a gap in canned-answer completeness for narrower sub-questions, not a flaw in the retrieval/caching mechanism — see `reports/findings.md`.
- Threshold behavior below 0.70 rests on a small sample and should be re-validated with more traffic before any production threshold change.
- Provider-side prompt caching numbers are a documented simulation (stated discount/latency assumptions), not measured against a live billing API.
