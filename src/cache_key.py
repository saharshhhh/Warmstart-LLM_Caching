import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class CacheContext:
    """Everything that affects whether a cached answer is still valid."""
    model: str
    temperature: float
    tool_set: Tuple[str, ...]          # sorted tuple, e.g. ("order_lookup", "return_policy_lookup")
    system_prompt_version: str          # bump this on ANY prompt change
    customer_tier: str
    personalized: bool                  # does the answer depend on WHO is asking?
    customer_id: Optional[str] = None   # required if personalized=True


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation/extra whitespace -- NOT semantic,
    just enough to catch trivial formatting differences for exact match."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _context_fingerprint(ctx: CacheContext) -> str:
    """The part of the key that's shared by every query under this context --
    this is what makes a prompt/model/tool change invalidate EVERYTHING at once,
    without having to track down individual cache entries."""
    parts = [
        ctx.model,
        f"{ctx.temperature:.2f}",
        ",".join(sorted(ctx.tool_set)),
        ctx.system_prompt_version,
        ctx.customer_tier,
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def build_exact_key(ctx: CacheContext, query_text: str) -> str:
    """Key for the exact-match cache layer (Stage 3)."""
    if ctx.personalized and not ctx.customer_id:
        raise ValueError("personalized context requires customer_id")

    fingerprint = _context_fingerprint(ctx)
    norm_text = normalize_text(query_text)
    text_hash = hashlib.sha256(norm_text.encode()).hexdigest()[:16]

    if ctx.personalized:
        return f"exact:{fingerprint}:cust:{ctx.customer_id}:{text_hash}"
    else:
        return f"exact:{fingerprint}:{text_hash}"


def build_semantic_namespace(ctx: CacheContext) -> str:
    """Namespace for the semantic-match cache layer (Stage 4).
    Semantic search only happens WITHIN this namespace -- a premium
    customer's embeddings never get compared against standard customer's,
    and personalized queries only ever search within that one customer's
    own history."""
    if ctx.personalized and not ctx.customer_id:
        raise ValueError("personalized context requires customer_id")

    fingerprint = _context_fingerprint(ctx)

    if ctx.personalized:
        return f"semantic:{fingerprint}:cust:{ctx.customer_id}"
    else:
        return f"semantic:{fingerprint}"