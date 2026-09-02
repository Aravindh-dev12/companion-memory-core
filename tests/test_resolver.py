from datetime import datetime, timezone
from pathlib import Path

from companion_memory.models import (
    Event,
    FactStatus,
    MemoryClaim,
    MemoryType,
    ResolutionAction,
    Speaker,
    StateDecision,
)
from companion_memory.resolver import StateResolver
from companion_memory.store import MemoryStore


def _event(store: MemoryStore, turn: int, text: str) -> Event:
    event = Event(session_id="s1", turn_id=turn, speaker=Speaker.USER, text=text)
    store.add_event(event)
    return event


def _claim(event: Event, predicate: str, value: str, kind=MemoryType.RELATIONSHIP_STATE) -> MemoryClaim:
    return MemoryClaim(
        source_event_id=event.event_id,
        memory_type=kind,
        subject="user",
        predicate=predicate,
        value=value,
        confidence=0.99,
        evidence_text=event.text,
    )


def test_temporal_transition_preserves_history(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    resolver = StateResolver(store)

    first_event = _event(store, 1, "My girlfriend is Maya")
    old_fact = resolver.apply(
        _claim(first_event, "partner", "Maya"),
        StateDecision(action=ResolutionAction.ADD, reason="new fact"),
    )

    second_event = _event(store, 2, "Maya and I broke up")
    second = _claim(second_event, "partner", "none")
    second.valid_from = datetime.now(timezone.utc)
    new_fact = resolver.apply(
        second,
        StateDecision(
            action=ResolutionAction.TEMPORAL_TRANSITION,
            reason="relationship state changed",
            target_fact_id=old_fact.fact_id,
        ),
    )

    history = store.list_fact_history("user::partner")
    store.close()

    assert len(history) == 2
    assert history[0].status is FactStatus.SUPERSEDED
    assert history[1].status is FactStatus.ACTIVE
    assert new_fact.supersedes_fact_id == old_fact.fact_id


def test_correction_marks_old_fact_corrected(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    resolver = StateResolver(store)
    e1 = _event(store, 1, "My sister's name is Mina")
    old = resolver.apply(
        _claim(e1, "sister_name", "Mina", MemoryType.IDENTITY_FACT),
        StateDecision(action=ResolutionAction.ADD, reason="new"),
    )
    e2 = _event(store, 2, "I typoed that — my sister's name is Nina, not Mina")
    new = resolver.apply(
        _claim(e2, "sister_name", "Nina", MemoryType.IDENTITY_FACT),
        StateDecision(action=ResolutionAction.CORRECT, reason="typo", target_fact_id=old.fact_id),
    )
    history = store.list_fact_history("user::sister_name")
    store.close()
    assert history[0].status is FactStatus.CORRECTED
    assert new.value == "Nina"
    assert new.supersedes_fact_id == old.fact_id


def test_coexistence_keeps_two_active_values(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    resolver = StateResolver(store)
    e1 = _event(store, 1, "I like coffee")
    resolver.apply(
        _claim(e1, "liked_beverage", "coffee", MemoryType.PREFERENCE),
        StateDecision(action=ResolutionAction.ADD, reason="new"),
    )
    e2 = _event(store, 2, "I like tea too")
    resolver.apply(
        _claim(e2, "liked_beverage", "tea", MemoryType.PREFERENCE),
        StateDecision(action=ResolutionAction.COEXIST, reason="additive"),
    )
    active = store.list_active_facts(fact_key="user::liked_beverage")
    store.close()
    assert {f.value for f in active} == {"coffee", "tea"}


def test_duplicate_evidence_does_not_duplicate_state(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    resolver = StateResolver(store)
    e1 = _event(store, 1, "I like coffee")
    resolver.apply(
        _claim(e1, "liked_beverage", "coffee", MemoryType.PREFERENCE),
        StateDecision(action=ResolutionAction.ADD, reason="new"),
    )
    e2 = _event(store, 2, "Still love coffee")
    resolver.apply(
        _claim(e2, "liked_beverage", "coffee", MemoryType.PREFERENCE),
        StateDecision(action=ResolutionAction.ADD, reason="new again"),
    )
    active = store.list_active_facts(fact_key="user::liked_beverage")
    transitions = store.transitions_for_event(e2.event_id)
    store.close()
    assert len(active) == 1
    assert transitions[-1].action is ResolutionAction.IGNORE


def test_no_temporal_ablation_retains_conflicting_active_state(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    resolver = StateResolver(store, temporal_resolution=False)
    e1 = _event(store, 1, "My girlfriend is Maya")
    old = resolver.apply(
        _claim(e1, "partner", "Maya"),
        StateDecision(action=ResolutionAction.ADD, reason="new"),
    )
    e2 = _event(store, 2, "Maya and I broke up")
    resolver.apply(
        _claim(e2, "partner", "none"),
        StateDecision(action=ResolutionAction.TEMPORAL_TRANSITION, reason="change", target_fact_id=old.fact_id),
    )
    active = store.list_active_facts(fact_key="user::partner")
    store.close()
    assert {f.value for f in active} == {"Maya", "none"}
