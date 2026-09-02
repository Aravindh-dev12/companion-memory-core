from __future__ import annotations

from .models import Event, MemoryCandidate, MemoryClaim, MemoryExtraction, Speaker
from .store import MemoryStore


class MemoryExtractor:
    def __init__(self, provider, store: MemoryStore | None = None, *, window_size: int = 4, worthiness_threshold: float = 0.4):
        self.provider = provider
        self.store = store
        self.window_size = window_size
        self.worthiness_threshold = worthiness_threshold

    def extract(self, event: Event, context: list[Event] | None = None) -> list[MemoryClaim]:
        events = list(context or [])
        if not events or events[-1].event_id != event.event_id:
            events.append(event)
        events = events[-self.window_size :]

        # New providers accept a window; older/offline fixtures may still accept
        # a single Event. The compatibility fallback is intentional and local.
        try:
            result: MemoryExtraction = self.provider.extract_user_memories(events)
        except (TypeError, AttributeError):
            result = self.provider.extract_user_memories(event)

        claims: list[MemoryClaim] = []
        for candidate in result.candidates:
            if not candidate.memory_worthy or candidate.importance < self.worthiness_threshold:
                continue
            source = self._source_event(events, candidate) or event
            claim = self._to_claim(source, events, candidate)
            if self.store and self.store.claim_exists_for_event_value(source.event_id, claim.fact_key, claim.value_text):
                continue
            claims.append(claim)
        return claims

    @staticmethod
    def _source_event(events: list[Event], candidate: MemoryCandidate) -> Event | None:
        evidence = candidate.evidence_text.strip().casefold()
        user_events = [e for e in events if e.speaker is Speaker.USER]
        for event in reversed(user_events):
            text = event.text.strip().casefold()
            if evidence and (evidence in text or text in evidence):
                return event
        return user_events[-1] if user_events else None

    @staticmethod
    def _to_claim(source: Event, window: list[Event], candidate: MemoryCandidate) -> MemoryClaim:
        return MemoryClaim(
            source_event_id=source.event_id,
            source_event_ids=[e.event_id for e in window],
            memory_type=candidate.memory_type,
            entity_mention=candidate.entity_mention,
            entity_type=candidate.entity_type,
            relation_to_user=candidate.relation_to_user,
            slot=candidate.slot,
            predicate_text=candidate.predicate_text,
            value_text=candidate.value_text,
            modality=candidate.modality,
            polarity=candidate.polarity,
            confidence=candidate.confidence,
            importance=candidate.importance,
            valid_from=candidate.valid_from,
            valid_to=candidate.valid_to,
            event_time_text=candidate.event_time_text,
            event_time_precision=candidate.event_time_precision,
            expires_at=candidate.expires_at,
            evidence_text=candidate.evidence_text.strip() or source.text,
        )
