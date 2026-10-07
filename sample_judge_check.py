from src.judge_eval import judge_one

result = judge_one(
    "Where is my order?",
    "Your order is currently in transit and should arrive within 3-5 business days."
)
print(result["raw"])
print("---")
print(result["judge_verdict"])