from pathlib import Path

from companion_memory.embeddings import HashEmbeddingProvider
from companion_memory.models import Event, MemoryClaim, MemoryType, ResolutionAction, Speaker, StateDecision
from companion_memory.resolver import StateResolver
from companion_memory.retrieval import Retriever
from companion_memory.store import MemoryStore


def _add(store, resolver, turn, text, predicate, value, action, target=None, kind=MemoryType.RELATIONSHIP_STATE):
    e = Event(session_id="s", turn_id=turn, speaker=Speaker.USER, text=text)
    store.add_event(e)
    c = MemoryClaim(
        source_event_id=e.event_id,
        memory_type=kind,
        subject="user",
        predicate=predicate,
        value=value,
        confidence=1.0,
        importance=0.9,
        evidence_text=text,
    )
    return resolver.apply(c, StateDecision(action=action, reason="test", target_fact_id=target))


def test_current_query_uses_retired_entity_as_state_bridge(tmp_path: Path):
    store = MemoryStore(tmp_path / "m.db")
    resolver = StateResolver(store)
    old = _add(store, resolver, 1, "My girlfriend is Maya", "partner", "Maya", ResolutionAction.ADD)
    _add(store, resolver, 2, "Maya and I broke up", "partner", "none", ResolutionAction.TEMPORAL_TRANSITION, old.fact_id)
    retriever = Retriever(store, embedder=HashEmbeddingProvider())
    results = retriever.retrieve("Am I still with Maya?")
    store.close()
    assert results
    assert results[0].fact.fact_key == "user::partner"
    assert results[0].fact.value == "none"
    assert all(m.fact.value != "Maya" for m in results if m.fact.fact_key == "user::partner")
    assert "state-closure" in results[0].retrieval_reason


def test_historical_query_can_retrieve_superseded_truth(tmp_path: Path):
    store = MemoryStore(tmp_path / "m.db")
    resolver = StateResolver(store)
    old = _add(store, resolver, 1, "My girlfriend is Maya", "partner", "Maya", ResolutionAction.ADD)
    _add(store, resolver, 2, "Maya and I broke up", "partner", "none", ResolutionAction.TEMPORAL_TRANSITION, old.fact_id)
    retriever = Retriever(store, embedder=HashEmbeddingProvider())
    results = retriever.retrieve("Who was I dating earlier?")
    store.close()
    assert any(m.fact.value == "Maya" and m.fact.status.value == "superseded" for m in results)


def test_irrelevant_active_facts_are_not_dumped(tmp_path: Path):
    store = MemoryStore(tmp_path / "m.db")
    resolver = StateResolver(store)
    for i, (pred, value) in enumerate([("occupation", "designer"), ("city", "Chennai"), ("dog_name", "Max")], 1):
        _add(store, resolver, i, f"fact {value}", pred, value, ResolutionAction.ADD, kind=MemoryType.IDENTITY_FACT)
    retriever = Retriever(store, embedder=HashEmbeddingProvider())
    results = retriever.retrieve("What music do I like?")
    store.close()
    assert results == []
