from __future__ import annotations

from datetime import datetime, timezone

from .models import FactStatus, FactVersion, MemoryClaim, ResolutionAction, StateDecision
from .store import MemoryStore


class StateResolver:
    """Deterministic bitemporal state machine.

    Models may propose a relation and target. They never directly update rows.
    World-valid time (valid_from/valid_to) and system/transaction time
    (created_at/retired_at) are intentionally distinct.
    """

    def __init__(self, store: MemoryStore, *, temporal_resolution: bool = True):
        self.store = store
        self.temporal_resolution = temporal_resolution

    def apply(self, claim: MemoryClaim, decision: StateDecision) -> FactVersion | None:
        self.store.add_claim(claim)
        active = self.store.list_active_facts(entity_id=claim.entity_id) if claim.entity_id else self.store.list_active_facts(subject=claim.entity_mention)

        exact = next(
            (
                f for f in active
                if (f.slot or f.predicate_text) == (claim.slot or claim.predicate_text)
                and f.value_text.strip().casefold() == claim.value_text.strip().casefold()
                and f.modality == claim.modality
                and f.polarity == claim.polarity
            ),
            None,
        )
        if exact is not None:
            effective = StateDecision(
                action=ResolutionAction.IGNORE,
                reason="Equivalent proposition is already active; preserve evidence without duplicating state.",
                target_fact_id=exact.fact_id,
                confidence=1.0,
            )
            self.store.record_transition(claim.claim_id, effective, exact.fact_id, exact.fact_id)
            return exact

        if decision.action is ResolutionAction.IGNORE:
            self.store.record_transition(claim.claim_id, decision, decision.target_fact_id, None)
            return None

        if decision.action in {ResolutionAction.ADD, ResolutionAction.COEXIST}:
            fact = self._create_fact(claim)
            self.store.record_transition(claim.claim_id, decision, decision.target_fact_id, fact.fact_id)
            return fact

        target = self._select_target(active, decision)
        if target is None:
            fallback = StateDecision(
                action=ResolutionAction.ADD,
                reason=f"No valid active target exists for {decision.action.value}; safely recorded as new proposition.",
                confidence=decision.confidence,
            )
            fact = self._create_fact(claim)
            self.store.record_transition(claim.claim_id, fallback, None, fact.fact_id)
            return fact

        if not self.temporal_resolution and decision.action in {
            ResolutionAction.CORRECT,
            ResolutionAction.SUPERSEDE,
            ResolutionAction.TEMPORAL_TRANSITION,
            ResolutionAction.REFINE,
            ResolutionAction.WITHDRAW,
        }:
            ablated = StateDecision(
                action=ResolutionAction.COEXIST,
                reason="Temporal resolver disabled for ablation; conflicting proposition retained.",
                confidence=decision.confidence,
            )
            fact = self._create_fact(claim)
            self.store.record_transition(claim.claim_id, ablated, target.fact_id, fact.fact_id)
            return fact

        now = datetime.now(timezone.utc)
        action = decision.action

        if action is ResolutionAction.WITHDRAW:
            self.store.retire_fact(target.fact_id, FactStatus.WITHDRAWN, retired_at=now, valid_to=None)
            self.store.record_transition(claim.claim_id, decision, target.fact_id, None)
            return None

        if action is ResolutionAction.CORRECT:
            # Correction says the old proposition was not world-truth. Preserve
            # its old validity fields for audit rather than closing at "now".
            self.store.retire_fact(target.fact_id, FactStatus.CORRECTED, retired_at=now, valid_to=None)
            return self._replace(
                claim, decision, target,
                valid_from=claim.valid_from if claim.valid_from is not None else target.valid_from,
            )

        if action is ResolutionAction.TEMPORAL_TRANSITION:
            event_time = decision.event_time or claim.valid_from or now
            self.store.retire_fact(target.fact_id, FactStatus.SUPERSEDED, retired_at=now, valid_to=event_time)
            return self._replace(claim, decision, target, valid_from=event_time)

        if action is ResolutionAction.REFINE:
            self.store.retire_fact(target.fact_id, FactStatus.SUPERSEDED, retired_at=now, valid_to=target.valid_to)
            return self._replace(claim, decision, target, valid_from=target.valid_from)

        if action is ResolutionAction.SUPERSEDE:
            self.store.retire_fact(target.fact_id, FactStatus.SUPERSEDED, retired_at=now, valid_to=claim.valid_from or now)
            return self._replace(claim, decision, target, valid_from=claim.valid_from or now)

        raise ValueError(f"unsupported resolution action: {action}")

    def _replace(
        self,
        claim: MemoryClaim,
        decision: StateDecision,
        target: FactVersion,
        *,
        valid_from,
    ) -> FactVersion:
        fact = self._create_fact(claim, supersedes_fact_id=target.fact_id, valid_from=valid_from)
        self.store.record_transition(claim.claim_id, decision, target.fact_id, fact.fact_id)
        return fact

    @staticmethod
    def _select_target(active: list[FactVersion], decision: StateDecision) -> FactVersion | None:
        if decision.target_fact_id:
            for fact in active:
                if fact.fact_id == decision.target_fact_id:
                    return fact
            raise ValueError("decision target does not match an active fact")
        if len(active) == 1:
            return active[0]
        return None

    def _create_fact(
        self,
        claim: MemoryClaim,
        supersedes_fact_id: str | None = None,
        valid_from=None,
    ) -> FactVersion:
        fact = FactVersion(
            fact_key=claim.fact_key,
            entity_id=claim.entity_id,
            entity_mention=claim.entity_mention,
            entity_type=claim.entity_type,
            relation_to_user=claim.relation_to_user,
            slot=claim.slot,
            predicate_text=claim.predicate_text,
            value_text=claim.value_text,
            modality=claim.modality,
            polarity=claim.polarity,
            memory_type=claim.memory_type,
            source_claim_id=claim.claim_id,
            supersedes_fact_id=supersedes_fact_id,
            valid_from=claim.valid_from if valid_from is None else valid_from,
            valid_to=claim.valid_to,
            event_time_precision=claim.event_time_precision,
            confidence=claim.confidence,
            importance=claim.importance,
            expires_at=claim.expires_at,
        )
        self.store.add_fact(fact)
        return fact
