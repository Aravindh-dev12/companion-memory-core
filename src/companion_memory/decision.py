from __future__ import annotations

from .matching import CandidateMatcher
from .models import FactVersion, MemoryClaim, Modality, ResolutionAction, StateDecision
from .store import MemoryStore


class DecisionClassifier:
    """Candidate matching first, relation classification second."""

    def __init__(self, store: MemoryStore, provider, matcher: CandidateMatcher | None = None):
        self.store = store
        self.provider = provider
        self.matcher = matcher or CandidateMatcher(store)

    def decide(self, claim: MemoryClaim) -> StateDecision:
        candidates = self.matcher.match(claim)
        if claim.modality in {Modality.HEDGED, Modality.HYPOTHETICAL, Modality.REPORTED_BY_THIRD_PARTY}:
            exact = next((f for f in candidates if self._same_proposition(f, claim)), None)
            if exact:
                return StateDecision(action=ResolutionAction.IGNORE, reason="Equivalent qualified proposition already exists.", target_fact_id=exact.fact_id)
            return StateDecision(action=ResolutionAction.ADD, reason=f"Qualified {claim.modality.value} proposition is recorded without replacing asserted state.")
        if not candidates:
            return StateDecision(action=ResolutionAction.ADD, reason="No plausible active update target.")

        exact = next((f for f in candidates if self._same_proposition(f, claim)), None)
        if exact:
            return StateDecision(
                action=ResolutionAction.IGNORE,
                reason="Equivalent active proposition already exists.",
                target_fact_id=exact.fact_id,
            )

        history = self.store.list_fact_history(claim.fact_key)
        return self.provider.classify_transition(claim, candidates, history)

    @staticmethod
    def _same_proposition(fact: FactVersion, claim: MemoryClaim) -> bool:
        same_slot = (fact.slot or fact.predicate_text) == (claim.slot or claim.predicate_text)
        same_value = fact.value_text.strip().casefold() == claim.value_text.strip().casefold()
        same_modality = fact.modality == claim.modality and fact.polarity == claim.polarity
        return same_slot and same_value and same_modality
