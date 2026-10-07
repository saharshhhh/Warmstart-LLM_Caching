import random
import json
from dataclasses import dataclass, asdict
from pathlib import Path

# A handful of "topic clusters" representing the ~60% repeated questions.
# Each cluster has a canonical intent, several paraphrasings, and whether
# the answer depends on a specific customer (personalized) or not (generic).
TOPIC_CLUSTERS = [
    {
        "intent": "order_status",
        "personalized": True,
        "paraphrases": [
            "Where is my order?",
            "Can you tell me the status of my order?",
            "My package hasn't arrived yet, what's going on?",
            "Track my order please",
            "When will my order get here?",
        ],
    },
    {
        "intent": "return_policy",
        "personalized": False,
        "paraphrases": [
            "How do returns work?",
            "What's your return policy?",
            "Can I return an item I don't like?",
            "How many days do I have to return something?",
            "Do you offer free returns?",
        ],
    },
    {
        "intent": "shipping_canada",
        "personalized": False,
        "paraphrases": [
            "Does this ship to Canada?",
            "Do you deliver to Canadian addresses?",
            "Can I order from Canada?",
            "Is international shipping to Canada available?",
        ],
    },
    {
        "intent": "change_address",
        "personalized": True,
        "paraphrases": [
            "Can I change my shipping address after ordering?",
            "I need to update my delivery address",
            "I typed the wrong address, can you fix it?",
        ],
    },
    {
        "intent": "payment_methods",
        "personalized": False,
        "paraphrases": [
            "What payment methods do you accept?",
            "Can I pay with PayPal?",
            "Do you take Apple Pay?",
        ],
    },
]

# Unique, one-off queries -- the "long tail" that should NOT cache-hit
UNIQUE_QUERIES = [
    "Does the blue jacket in size M run small?",
    "Is the espresso machine compatible with 220V outlets?",
    "Can I get a gift receipt for my mother's birthday present?",
    "What's the warranty on the leather boots?",
    "Is the standing desk motor loud when adjusting height?",
]


@dataclass
class SimulatedQuery:
    query_id: str
    customer_id: str
    customer_tier: str
    text: str
    intent: str
    personalized: bool


def generate_traffic(n=1000, repeat_rate=0.6, seed=42):
    """
    repeat_rate: fraction of traffic drawn from the repeated topic clusters
    (the "six in ten are the same handful of questions" from the spec).
    The remainder is long-tail unique queries.
    """
    random.seed(seed)
    tiers = ["standard", "standard", "standard", "premium"]  # premium is rarer
    queries = []

    n_repeated = int(n * repeat_rate)
    n_unique = n - n_repeated

    for i in range(n_repeated):
        cluster = random.choice(TOPIC_CLUSTERS)
        text = random.choice(cluster["paraphrases"])
        queries.append(SimulatedQuery(
            query_id=f"q_{i}",
            customer_id=f"cust_{random.randint(1, 300)}",
            customer_tier=random.choice(tiers),
            text=text,
            intent=cluster["intent"],
            personalized=cluster["personalized"],
        ))

    for i in range(n_unique):
        text = random.choice(UNIQUE_QUERIES)
        queries.append(SimulatedQuery(
            query_id=f"q_{n_repeated + i}",
            customer_id=f"cust_{random.randint(1, 300)}",
            customer_tier=random.choice(tiers),
            text=text,
            intent="unique",
            personalized=False,
        ))

    random.shuffle(queries)
    return queries


if __name__ == "__main__":
    traffic = generate_traffic(n=1000)
    Path("data").mkdir(exist_ok=True)
    with open("data/traffic.json", "w") as f:
        json.dump([asdict(q) for q in traffic], f, indent=2)

    from collections import Counter
    print(f"Generated {len(traffic)} queries")
    print("By intent:", Counter(q.intent for q in traffic))
    print("Personalized:", sum(q.personalized for q in traffic))