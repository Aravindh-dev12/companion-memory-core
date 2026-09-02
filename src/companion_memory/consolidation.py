from __future__ import annotations

from dataclasses import dataclass

from .entities import EntityResolver
from .models import (
    EntityType,
    MemoryClaim,
    MemoryType,
    Modality,
    ResolutionAction,
    SessionConsolidation,
    StateDecision,
)
from .resolver import StateResolver
from .retrieval import Retriever
from .store import MemoryStore


@dataclass(frozen=True)
class ConsolidationResult:
    session_id: str
    summary_fact_id: str | None
    inference_fact_ids: tuple[str, ...]
    skipped: bool = False


class SessionConsolidator:
    """Sleep-time/session-boundary memory work.

    Consolidation never rewrites the event ledger. It derives one shared-history
    episode for the user/companion pair and at most a few explicitly hedged,
    low-confidence inferences. Re-running a consolidated session is idempotent.
    """

    def __init__(self, store: MemoryStore, provider, resolver: StateResolver, entity_resolver: EntityResolver, retriever: Retriever):
        self.store = store
        self.provider = provider
        self.resolver = resolver
        self.entity_resolver = entity_resolver
        self.retriever = retriever

    def consolidate(self, session_id: str) -> ConsolidationResult:
        if self.store.is_session_consolidated(session_id):
            return ConsolidationResult(session_id=session_id, summary_fact_id=None, inference_fact_ids=(), skipped=True)
        events = self.store.list_session_events(session_id)
        if not events:
            self.store.record_session_consolidation(session_id, summary_fact_id=None, metadata={"empty": True})
            return ConsolidationResult(session_id=session_id, summary_fact_id=None, inference_fact_ids=(), skipped=True)

        facts = []
        for item in self.store.list_fact_evidence(include_inactive=True):
            if any((self.store.get_event(eid) and self.store.get_event(eid).session_id == session_id) for eid in item.source_event_ids):
                facts.append(item.fact)

        result: SessionConsolidation = self.provider.consolidate_session(session_id=session_id, events=events, facts=facts)
        source_event_id = events[-1].event_id
        source_ids = [event.event_id for event in events]
        summary_fact_id: str | None = None
        inference_fact_ids: list[str] = []

        if result.summary.strip():
            summary_claim = MemoryClaim(
                source_event_id=source_event_id,
                source_event_ids=source_ids,
                entity_mention="companion_user_pair",
                entity_type=EntityType.PAIR,
                memory_type=MemoryType.EPISODIC_EVENT,
                predicate_text=f"session_summary_{session_id}",
                value_text=result.summary.strip(),
                modality=Modality.ASSERTED,
                confidence=0.95,
                importance=0.70,
                evidence_text=result.summary.strip(),
            )
            self.entity_resolver.resolve_claim(summary_claim)
            summary_fact = self.resolver.apply(summary_claim, StateDecision(action=ResolutionAction.ADD, reason="Session consolidation shared-history episode."))
            if summary_fact:
                summary_fact_id = summary_fact.fact_id
                self.retriever.index_fact(summary_fact.fact_id)

        for candidate in result.inferences[:3]:
            if not candidate.memory_worthy:
                continue
            claim = MemoryClaim(
                source_event_id=source_event_id,
                source_event_ids=source_ids,
                entity_mention="user",
                entity_type=EntityType.SELF,
                memory_type=MemoryType.INFERENCE,
                predicate_text=candidate.predicate_text or candidate.predicate or "session_inference",
                value_text=candidate.value_text or candidate.value or "",
                modality=Modality.HEDGED,
                polarity=candidate.polarity,
                confidence=min(candidate.confidence, 0.69),
                importance=min(candidate.importance, 0.65),
                valid_from=candidate.valid_from,
                valid_to=candidate.valid_to,
                expires_at=candidate.expires_at,
                evidence_text=candidate.evidence_text or result.summary,
            )
            self.entity_resolver.resolve_claim(claim)
            fact = self.resolver.apply(claim, StateDecision(action=ResolutionAction.ADD, reason="Low-confidence session inference; never asserted as user truth."))
            if fact:
                inference_fact_ids.append(fact.fact_id)
                self.retriever.index_fact(fact.fact_id)

        self.store.record_session_consolidation(
            session_id,
            summary_fact_id=summary_fact_id,
            metadata={"inference_fact_ids": inference_fact_ids},
        )
        return ConsolidationResult(
            session_id=session_id,
            summary_fact_id=summary_fact_id,
            inference_fact_ids=tuple(inference_fact_ids),
        )
