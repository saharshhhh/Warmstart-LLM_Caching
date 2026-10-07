import json
import os
import time
import cohere
from dotenv import load_dotenv

load_dotenv()
co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

JUDGE_PROMPT = """You are auditing an AI customer support cache. A customer asked a
question, and the system served them an answer that was originally written for a
DIFFERENT (but supposedly similar) question. Your job is to judge whether reusing
this answer was appropriate.

Customer's actual question:
"{query}"

Answer that was served (originally written for a different question):
"{answer}"

Does this answer correctly and completely address the customer's actual question?
Respond with EXACTLY two lines, nothing else:
VERDICT: YES or NO
REASON: <one sentence>"""


def judge_one(query_text: str, matched_answer: str) -> dict:
    resp = co.chat(
        model="command-a-plus-05-2026",
        max_tokens=500,   # raised from 150 -- avoids truncated/empty responses
        messages=[{"role": "user",
                   "content": JUDGE_PROMPT.format(query=query_text, answer=matched_answer)}]
    )

    text = ""
    for block in resp.message.content:
        if getattr(block, "type", None) == "text":
            text = (block.text or "").strip()
            break

    if not text:
        return {
            "query": query_text, "answer": matched_answer,
            "judge_verdict": "unparseable", "judge_reason": "EMPTY RESPONSE",
            "raw": "",
        }

    text_upper = text.upper()
    lines = text.split("\n")
    verdict_line = next((l for l in lines if l.strip().upper().startswith("VERDICT:")), "")
    reason_line = next((l for l in lines if l.strip().upper().startswith("REASON:")), "")

    if verdict_line:
        verdict = "YES" in verdict_line.upper()
    else:
        verdict = text_upper.startswith("YES") or ("YES" in text_upper.split("\n")[0])

    reason = reason_line.split(":", 1)[1].strip() if ":" in reason_line else text[:150]

    # flag anything where the reason looks suspiciously like a parsing artifact
    if reason.upper() in ("VERDICT", "YES", "NO", ""):
        reason = f"[PARSE WARNING] raw={text!r}"

    return {
        "query": query_text,
        "answer": matched_answer,
        "judge_verdict": "appropriate" if verdict else "inappropriate",
        "judge_reason": reason,
        "raw": text,
    }


def judge_sample(comparison_log, threshold, sample_size=30, seed=42, delay_seconds=3.5):
    import random
    random.seed(seed)

    hits = [c for c in comparison_log if c["similarity"] >= threshold]
    best_per_query = {}
    for c in hits:
        q = c["query_text"]
        if q not in best_per_query or c["similarity"] > best_per_query[q]["similarity"]:
            best_per_query[q] = c

    candidates = list(best_per_query.values())
    sample = random.sample(candidates, min(sample_size, len(candidates)))

    results = []
    for i, c in enumerate(sample):
        verdict = judge_one(c["query_text"], c["matched_answer"])
        verdict["similarity"] = c["similarity"]
        verdict["matched_text"] = c["matched_text"]
        results.append(verdict)
        print(f"  [{i+1}/{len(sample)}] {verdict['judge_verdict']}: '{c['query_text'][:50]}'")
        if i < len(sample) - 1:
            time.sleep(delay_seconds)

    inappropriate = [r for r in results if r["judge_verdict"] == "inappropriate"]
    unparseable = [r for r in results if r["judge_verdict"] == "unparseable"]
    return {
        "threshold": threshold,
        "sample_size": len(results),
        "judge_false_hit_rate": round(len(inappropriate) / len(results), 4) if results else 0.0,
        "unparseable_count": len(unparseable),
        "flagged": inappropriate,
        "all_results": results,
    }