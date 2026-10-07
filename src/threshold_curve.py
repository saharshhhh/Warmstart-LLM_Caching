import numpy as np


def sweep_thresholds(comparison_log, traffic_by_query_text, thresholds=None):
    """
    comparison_log: list of {query_text, matched_text, matched_answer, similarity}
                     captured from SemanticCache.get(..., log_all=...)
    traffic_by_query_text: maps query_text -> its true intent (to check correctness)
    """
    if thresholds is None:
        thresholds = np.arange(0.60, 0.96, 0.05)

    from src.replay import simulate_llm_answer

    results = []
    for t in thresholds:
        hits = [c for c in comparison_log if c["similarity"] >= t]

        # for a given query, only count its BEST match at this threshold
        best_per_query = {}
        for c in hits:
            q = c["query_text"]
            if q not in best_per_query or c["similarity"] > best_per_query[q]["similarity"]:
                best_per_query[q] = c

        total_hits = len(best_per_query)
        false_hits = 0
        for q, c in best_per_query.items():
            true_intent = traffic_by_query_text.get(q)
            correct_answer = simulate_llm_answer(true_intent) if true_intent else None
            if c["matched_answer"] != correct_answer:
                false_hits += 1

        false_hit_rate = false_hits / total_hits if total_hits else 0.0
        results.append({
            "threshold": round(float(t), 2),
            "hits": total_hits,
            "false_hits": false_hits,
            "false_hit_rate": round(false_hit_rate, 4),
        })

    return results