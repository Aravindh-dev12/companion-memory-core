from datetime import datetime, timezone
from pathlib import Path

from companion_memory.decision import DecisionClassifier
from companion_memory.embeddings import HashEmbeddingProvider
from companion_memory.entities import EntityResolver
from companion_memory.matching import CandidateMatcher
from companion_memory.models import (
    EntityType,
    Event,
    EventTimePrecision,
    MemoryClaim,
    MemoryType,
    Modality,
    ResolutionAction,
    Speaker,
    StateDecision,
)
from companion_memory.providers import HeuristicProvider
from companion_memory.resolver import StateResolver
from companion_memory.retrieval import Retriever
from companion_memory.store import MemoryStore


def _event(store, turn, text):
    event = Event(session_id="s", turn_id=turn, speaker=Speaker.USER, text=text)
    store.add_event(event)
    return event


def test_entity_channel_recovers_semantically_distant_constraint(tmp_path: Path):
    store = MemoryStore(tmp_path / "m.db")
    resolver = StateResolver(store)
    entity_resolver = EntityResolver(store, HashEmbeddingProvider())
    event = _event(store, 1, "My mum gets extremely anxious on long flights.")
    claim = MemoryClaim(
        source_event_id=event.event_id,
        memory_type=MemoryType.CONSTRAINT,
        entity_mention="mum",
        entity_type=EntityType.PERSON,
        relation_to_user="mother",
        predicate_text="flight_anxiety",
        value_text="anxious on long flights",
        confidence=1.0,
        importance=0.95,
        evidence_text=event.text,
    )
    entity_resolver.resolve_claim(claim)
    resolver.apply(claim, StateDecision(action=ResolutionAction.ADD, reason="new"))

    retriever = Retriever(store, embedder=HashEmbeddingProvider())
    results = retriever.retrieve("Would Bali be an easy surprise for Mom?")
    store.close()

    assert results
    assert results[0].fact.predicate_text == "flight_anxiety"
    assert results[0].entity_rank == 1
    assert "entity-channel" in results[0].retrieval_reason


def test_hedged_memory_cannot_replace_asserted_state(tmp_path: Path):
    store = MemoryStore(tmp_path / "m.db")
    resolver = StateResolver(store)
    matcher = CandidateMatcher(store)
    decider = DecisionClassifier(store, HeuristicProvider(), matcher)

    e1 = _event(store, 1, "I work at Acme")
    current = MemoryClaim(
        source_event_id=e1.event_id,
        memory_type=MemoryType.IDENTITY_FACT,
        entity_mention="user",
        entity_type=EntityType.SELF,
        slot="employer",
        predicate_text="employer",
        value_text="Acme",
        confidence=1.0,
        evidence_text=e1.text,
    )
    EntityResolver(store).resolve_claim(current)
    resolver.apply(current, StateDecision(action=ResolutionAction.ADD, reason="new"))

    e2 = _event(store, 2, "I might quit Acme after the review")
    hedged = MemoryClaim(
        source_event_id=e2.event_id,
        memory_type=MemoryType.IDENTITY_FACT,
        entity_mention="user",
        entity_type=EntityType.SELF,
        slot="employer",
        predicate_text="employer",
        value_text="none",
        modality=Modality.HEDGED,
        confidence=0.8,
        evidence_text=e2.text,
    )
    EntityResolver(store).resolve_claim(hedged)
    decision = decider.decide(hedged)
    resolver.apply(hedged, decision)

    active = store.list_active_facts(slot="employer")
    store.close()
    asserted = [f for f in active if f.modality is Modality.ASSERTED]
    qualified = [f for f in active if f.modality is Modality.HEDGED]
    assert decision.action is ResolutionAction.ADD
    assert [f.value_text for f in asserted] == ["Acme"]
    assert [f.value_text for f in qualified] == ["none"]


def test_bitemporal_transition_separates_world_and_recording_time(tmp_path: Path):
    store = MemoryStore(tmp_path / "m.db")
    resolver = StateResolver(store)
    er = EntityResolver(store)
    jan1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    jan10 = datetime(2026, 1, 10, tzinfo=timezone.utc)

    e1 = _event(store, 1, "My girlfriend is Maya")
    first = MemoryClaim(
        source_event_id=e1.event_id,
        memory_type=MemoryType.RELATIONSHIP_STATE,
        entity_mention="user",
        entity_type=EntityType.SELF,
        slot="partner",
        predicate_text="partner",
        value_text="Maya",
        valid_from=jan1,
        event_time_precision=EventTimePrecision.DAY,
        confidence=1.0,
        evidence_text=e1.text,
    )
    er.resolve_claim(first)
    old = resolver.apply(first, StateDecision(action=ResolutionAction.ADD, reason="new"))

    e2 = _event(store, 2, "Maya and I broke up last weekend")
    second = MemoryClaim(
        source_event_id=e2.event_id,
        memory_type=MemoryType.RELATIONSHIP_STATE,
        entity_mention="user",
        entity_type=EntityType.SELF,
        slot="partner",
        predicate_text="partner",
        value_text="none",
        valid_from=jan10,
        event_time_precision=EventTimePrecision.APPROXIMATE,
        confidence=1.0,
        evidence_text=e2.text,
    )
    er.resolve_claim(second)
    new = resolver.apply(
        second,
        StateDecision(
            action=ResolutionAction.TEMPORAL_TRANSITION,
            reason="world state changed",
            target_fact_id=old.fact_id,
            event_time=jan10,
        ),
    )
    old_after = store.get_fact(old.fact_id)
    store.close()

    assert old_after.valid_to == jan10
    assert old_after.retired_at is not None
    assert old_after.retired_at != old_after.valid_to
    assert new.valid_from == jan10


def test_candidate_matching_happens_before_relation_classification(tmp_path: Path):
    store = MemoryStore(tmp_path / "m.db")
    resolver = StateResolver(store)
    er = EntityResolver(store)
    for turn, value in [(1, "coffee"), (2, "tea")]:
        event = _event(store, turn, f"I like {value}")
        claim = MemoryClaim(
            source_event_id=event.event_id,
            memory_type=MemoryType.PREFERENCE,
            entity_mention="user",
            entity_type=EntityType.SELF,
            predicate_text="liked_beverage",
            value_text=value,
            confidence=1.0,
            evidence_text=event.text,
        )
        er.resolve_claim(claim)
        resolver.apply(claim, StateDecision(action=ResolutionAction.ADD, reason="new"))

    probe_event = _event(store, 3, "I don't like coffee anymore")
    probe = MemoryClaim(
        source_event_id=probe_event.event_id,
        memory_type=MemoryType.PREFERENCE,
        entity_mention="user",
        entity_type=EntityType.SELF,
        predicate_text="liked_beverage",
        value_text="coffee",
        modality=Modality.NEGATED,
        polarity=-1,
        confidence=1.0,
        evidence_text=probe_event.text,
    )
    er.resolve_claim(probe)
    candidates = CandidateMatcher(store).match(probe)
    store.close()
    assert {f.value_text for f in candidates} == {"coffee", "tea"}
