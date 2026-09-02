from __future__ import annotations

import re

from .embeddings import EmbeddingProvider, cosine_similarity
from .models import FactVersion, MemoryClaim, normalize_key
from .store import MemoryStore

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class CandidateMatcher:
    """Generate likely update targets before asking an LLM for a relation.

    Matching is deliberately top-k rather than threshold-driven: exact slot
    matches win, then predicate/type overlap, then optional semantic similarity.
    """

    def __init__(self, store: MemoryStore, embedder: EmbeddingProvider | None = None, limit: int = 5):
        self.store = store
        self.embedder = embedder
        self.limit = limit

    def match(self, claim: MemoryClaim) -> list[FactVersion]:
        if claim.entity_id:
            active = self.store.list_active_facts(entity_id=claim.entity_id)
        else:
            active = self.store.list_active_facts(subject=claim.entity_mention)
        if not active:
            return []

        claim_pred = claim.slot or claim.predicate_text
        claim_tokens = set(_TOKEN_RE.findall(f"{claim_pred} {claim.value_text}".lower()))
        semantic_scores: dict[str, float] = {}
        if self.embedder is not None:
            query = f"{claim.entity_mention} {claim_pred} {claim.value_text}"
            qv = self.embedder.embed([query])[0]
            texts = [f"{f.entity_mention} {f.slot or f.predicate_text} {f.value_text}" for f in active]
            vectors = self.embedder.embed(texts)
            semantic_scores = {f.fact_id: cosine_similarity(qv, v) for f, v in zip(active, vectors)}

        def score(fact: FactVersion) -> tuple[float, float, float, float]:
            fact_pred = fact.slot or fact.predicate_text
            same_slot = 1.0 if claim.slot and fact.slot == claim.slot else 0.0
            same_predicate = 1.0 if normalize_key(fact_pred) == normalize_key(claim_pred) else 0.0
            same_type = 1.0 if fact.memory_type == claim.memory_type else 0.0
            fact_tokens = set(_TOKEN_RE.findall(f"{fact_pred} {fact.value_text}".lower()))
            lexical = len(claim_tokens & fact_tokens) / max(len(claim_tokens | fact_tokens), 1)
            semantic = semantic_scores.get(fact.fact_id, 0.0)
            # Tuple ordering gives deterministic priority without a fitted
            # scalar relevance formula.
            return (same_slot, same_predicate, same_type + lexical, semantic)

        ranked = sorted(active, key=score, reverse=True)
        # If the claim uses a typed slot, unrelated slots should never be passed
        # to the relation classifier merely because they share the entity.
        if claim.slot:
            slot_matches = [f for f in ranked if f.slot == claim.slot]
            if slot_matches:
                return slot_matches[: self.limit]
        # For free-text facts, require at least predicate/type affinity.
        filtered = [
            f for f in ranked
            if normalize_key(f.predicate_text) == normalize_key(claim.predicate_text)
            or f.memory_type == claim.memory_type
        ]
        return filtered[: self.limit]
