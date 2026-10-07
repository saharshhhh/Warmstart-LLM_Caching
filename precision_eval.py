from src.replay import simulate_llm_answer


def compute_ground_truth_false_hit_rate(records):
    """
    Exact check: a semantic hit is a FALSE hit if the served answer doesn't
    match what the query's OWN intent should have produced.
    """
    semantic_hits = [r for r in records if r["layer"] == "semantic"]
    false_hits = []

    for r in semantic_hits:
        correct_answer = simulate_llm_answer(r["intent"])
        if r["served_answer"] != correct_answer:
            false_hits.append(r)

    rate = len(false_hits) / len(semantic_hits) if semantic_hits else 0.0
    return {
        "semantic_hits": len(semantic_hits),
        "false_hits": len(false_hits),
        "false_hit_rate": round(rate, 4),
        "false_hit_examples": false_hits[:10],
    }