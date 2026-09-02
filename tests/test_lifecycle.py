from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from companion_memory.config import Settings
from companion_memory.factory import build_engine
from companion_memory.models import EntityType, Event, FactStatus, MemoryCandidate, MemoryType, Modality, RetrievalMode, SessionConsolidation, Speaker
from companion_memory.providers import HeuristicProvider


PERSONA = Path(__file__).resolve().parents[1] / "persona.yaml"


def _engine(tmp_path, **overrides):
    settings = replace(
        Settings(),
        db_path=tmp_path / "memory.sqlite3",
        persona_path=PERSONA,
        provider="heuristic",
        **overrides,
    )
    return build_engine(settings)


def test_cross_session_plan_becomes_open_loop(tmp_path):
    engine = _engine(tmp_path)
    try:
        engine.process_turn(session_id="s1", user_text="I have an interview Friday.")
        trace = engine.process_turn(session_id="s2", user_text="hi")
        assert trace.open_loops
        assert any("interview" in loop.summary.lower() for loop in trace.open_loops)
        assert "interview" in trace.final_response.lower()
    finally:
        engine.store.close()


def test_session_consolidation_creates_shared_history_episode_and_is_idempotent(tmp_path):
    engine = _engine(tmp_path)
    try:
        engine.process_turn(session_id="s1", user_text="I have an interview Friday.")
        engine.process_turn(session_id="s1", user_text="I'm nervous but I want to do well.")

        first = engine.consolidate_session("s1")
        assert first.summary_fact_id is not None
        summary = engine.store.get_fact(first.summary_fact_id)
        assert summary is not None
        assert summary.entity_mention == "companion_user_pair"
        assert summary.memory_type is MemoryType.EPISODIC_EVENT
        assert "interview" in summary.value_text.lower()

        second = engine.consolidate_session("s1")
        assert second.skipped is True

        retrieved = engine.retriever.retrieve("what did we talk about last time about my interview?", limit=12)
        assert any(item.fact.fact_id == first.summary_fact_id for item in retrieved)
    finally:
        engine.store.close()


def test_oracle_mode_exposes_active_and_superseded_versions(tmp_path):
    engine = _engine(tmp_path, retrieval_mode=RetrievalMode.ORACLE, temporal_filter=False, retrieval_limit=100)
    try:
        engine.process_turn(session_id="s1", user_text="My girlfriend is Maya.")
        engine.process_turn(session_id="s1", user_text="Maya and I broke up.")
        memories = engine.retriever.retrieve("relationship", limit=100)
        partner = [item.fact for item in memories if item.fact.fact_key == "user::partner"]
        assert any(f.status is FactStatus.SUPERSEDED and f.value_text == "Maya" for f in partner)
        assert any(f.status is FactStatus.ACTIVE and f.value_text == "none" for f in partner)
        assert all("oracle-full-store" in item.retrieval_reason for item in memories)
    finally:
        engine.store.close()


def test_auto_consolidates_previous_session_on_new_session_entry(tmp_path):
    engine = _engine(tmp_path, auto_consolidate_previous_sessions=True)
    try:
        engine.process_turn(session_id="old", user_text="I have an interview Friday.")
        assert not engine.store.is_session_consolidated("old")
        engine.process_turn(session_id="new", user_text="hi")
        assert engine.store.is_session_consolidated("old")
        pair_facts = [
            f for f in engine.store.list_active_facts()
            if f.entity_mention == "companion_user_pair" and f.memory_type is MemoryType.EPISODIC_EVENT
        ]
        assert pair_facts
    finally:
        engine.store.close()


def test_open_loop_closes_when_plan_becomes_episode(tmp_path):
    engine = _engine(tmp_path)
    try:
        engine.process_turn(session_id="s1", user_text="I have an interview Friday.")
        follow_up = engine.process_turn(session_id="s2", user_text="hi")
        assert any("interview" in loop.summary.lower() for loop in follow_up.open_loops)

        completed = engine.process_turn(session_id="s2", user_text="It went well.")
        assert any(decision.action.value == "temporal_transition" for decision in completed.decisions)
        active = engine.store.list_active_facts(fact_key="user::interview_plan")
        assert len(active) == 1
        assert active[0].memory_type is MemoryType.EPISODIC_EVENT

        later = engine.process_turn(session_id="s3", user_text="hi")
        assert not any("interview" in loop.summary.lower() for loop in later.open_loops)
    finally:
        engine.store.close()


def test_consolidation_inferences_are_forced_hedged_and_low_confidence(tmp_path, monkeypatch):
    def fake_consolidation(self, *, session_id, events, facts):
        return SessionConsolidation(
            summary="We talked about an upcoming interview and work pressure.",
            inferences=[
                MemoryCandidate(
                    memory_worthy=True,
                    memory_type=MemoryType.INFERENCE,
                    entity_mention="user",
                    entity_type=EntityType.SELF,
                    predicate_text="work_pressure",
                    value_text="under sustained work pressure",
                    modality=Modality.ASSERTED,
                    confidence=0.95,
                    importance=0.9,
                    evidence_text="interview and work discussion",
                )
            ],
        )

    monkeypatch.setattr(HeuristicProvider, "consolidate_session", fake_consolidation)
    engine = _engine(tmp_path)
    try:
        engine.process_turn(session_id="s1", user_text="I have an interview Friday.")
        result = engine.consolidate_session("s1")
        assert result.inference_fact_ids
        inference = engine.store.get_fact(result.inference_fact_ids[0])
        assert inference is not None
        assert inference.memory_type is MemoryType.INFERENCE
        assert inference.modality is Modality.HEDGED
        assert inference.confidence <= 0.69
        assert inference.importance <= 0.65
    finally:
        engine.store.close()


def test_companion_promise_becomes_cross_session_open_loop(tmp_path):
    engine = _engine(tmp_path)
    try:
        event = Event(
            session_id="s1",
            turn_id=1,
            speaker=Speaker.ASSISTANT,
            text="I'll ask you how the interview went when you're back.",
        )
        engine.store.add_event(event)
        claims = engine.commitment_extractor.extract(event)
        assert claims
        for claim in claims:
            engine.entity_resolver.resolve_claim(claim)
            decision = engine.decider.decide(claim)
            engine.resolver.apply(claim, decision)

        loops = engine.open_loop_manager.candidates(
            current_session_id="s2", user_text="hi", first_turn_in_session=True
        )
        assert any(loop.kind.value == "companion_promise" for loop in loops)
    finally:
        engine.store.close()


def test_coexisting_preference_retraction_targets_only_matching_value(tmp_path):
    engine = _engine(tmp_path)
    try:
        engine.process_turn(session_id="s", user_text="I really like coffee.")
        engine.process_turn(session_id="s", user_text="I've gotten into tea too.")
        changed = engine.process_turn(session_id="s", user_text="I don't like coffee anymore.")
        assert any(decision.action.value == "temporal_transition" for decision in changed.decisions)

        facts = engine.store.list_active_facts(fact_key="user::liked_beverage")
        assert any(f.value_text == "tea" and f.polarity == 1 for f in facts)
        assert any(f.value_text == "coffee" and f.polarity == -1 and f.modality is Modality.NEGATED for f in facts)
        assert not any(f.value_text == "coffee" and f.polarity == 1 for f in facts)
    finally:
        engine.store.close()
