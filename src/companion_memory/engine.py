from __future__ import annotations

from dataclasses import dataclass

from .consistency import ConsistencyFirewall
from .consolidation import ConsolidationResult, SessionConsolidator
from .decision import DecisionClassifier
from .entities import EntityResolver
from .extractor import MemoryExtractor
from .matching import CandidateMatcher
from .models import Event, MemoryClaim, Speaker, StateDecision, TurnTrace
from .persona import Persona, PersonaCommitmentExtractor
from .open_loops import OpenLoopManager
from .resolver import StateResolver
from .retrieval import Retriever
from .store import MemoryStore


@dataclass(frozen=True)
class EngineConfig:
    recent_event_limit: int = 8
    extraction_window: int = 4
    retrieval_limit: int = 12
    memory_enabled: bool = True
    full_history_context: bool = False
    open_loop_followups: bool = True
    auto_consolidate_previous_sessions: bool = False


class CompanionEngine:
    """Synchronous reference loop.

    The write path runs before retrieval so a disclosure such as "we broke up"
    cannot be answered using the just-retired relationship as current state.
    This is deliberately different from an eventually-consistent async design.
    """

    def __init__(
        self,
        *,
        store: MemoryStore,
        provider,
        persona: Persona,
        retriever: Retriever,
        resolver: StateResolver,
        consistency: ConsistencyFirewall,
        entity_resolver: EntityResolver | None = None,
        matcher: CandidateMatcher | None = None,
        config: EngineConfig | None = None,
    ):
        self.store = store
        self.provider = provider
        self.persona = persona
        self.retriever = retriever
        self.resolver = resolver
        self.consistency = consistency
        self.config = config or EngineConfig()
        self.entity_resolver = entity_resolver or EntityResolver(store, retriever.embedder)
        self.matcher = matcher or CandidateMatcher(store, retriever.embedder)
        self.extractor = MemoryExtractor(
            provider,
            store,
            window_size=self.config.extraction_window,
        )
        self.decider = DecisionClassifier(store, provider, self.matcher)
        self.commitment_extractor = PersonaCommitmentExtractor(provider)
        self.open_loop_manager = OpenLoopManager(store)
        self.consolidator = SessionConsolidator(
            store, provider, resolver, self.entity_resolver, retriever
        )
        self._consolidated_on_entry: set[str] = set()

    def consolidate_session(self, session_id: str) -> ConsolidationResult:
        return self.consolidator.consolidate(session_id)

    def consolidate_previous_sessions(self, *, exclude_session: str | None = None) -> list[ConsolidationResult]:
        return [
            self.consolidator.consolidate(session_id)
            for session_id in self.store.list_unconsolidated_sessions(exclude_session=exclude_session)
        ]

    def process_turn(self, *, session_id: str, user_text: str) -> TurnTrace:
        next_turn = self.store.next_turn_id(session_id)
        first_turn_in_session = next_turn == 1
        if self.config.auto_consolidate_previous_sessions and first_turn_in_session and session_id not in self._consolidated_on_entry:
            self.consolidate_previous_sessions(exclude_session=session_id)
            self._consolidated_on_entry.add(session_id)

        user_event = Event(
            session_id=session_id,
            turn_id=next_turn,
            speaker=Speaker.USER,
            text=user_text,
        )
        self.store.add_event(user_event)

        claims: list[MemoryClaim] = []
        decisions: list[StateDecision] = []
        if self.config.memory_enabled:
            extraction_context = self.store.list_recent_events(session_id, limit=self.config.extraction_window)
            claims = self.extractor.extract(user_event, extraction_context)
            for claim in claims:
                self.entity_resolver.resolve_claim(claim)
                decision = self.decider.decide(claim)
                decisions.append(decision)
                fact = self.resolver.apply(claim, decision)
                if fact is not None:
                    self.retriever.index_fact(fact.fact_id)

        retrieved = self.retriever.retrieve(user_text, limit=self.config.retrieval_limit) if self.config.memory_enabled else []
        open_loops = self.open_loop_manager.candidates(
            current_session_id=session_id,
            user_text=user_text,
            first_turn_in_session=first_turn_in_session,
        ) if (self.config.memory_enabled and self.config.open_loop_followups) else []
        recent = self.store.list_events() if self.config.full_history_context else self.store.list_recent_events(
            session_id, limit=self.config.recent_event_limit
        )
        draft = self.provider.generate_response(
            user_text=user_text,
            persona=self.persona,
            recent_events=recent,
            retrieved=retrieved,
            open_loops=open_loops,
        )
        verdict = self.consistency.check(
            user_text=user_text,
            draft=draft,
            persona=self.persona,
            retrieved=retrieved,
        )
        final = draft if verdict.consistent else (verdict.revised_response or draft)

        assistant_event = Event(
            session_id=session_id,
            turn_id=self.store.next_turn_id(session_id),
            speaker=Speaker.ASSISTANT,
            text=final,
        )
        self.store.add_event(assistant_event)

        if self.config.memory_enabled:
            for claim in self.commitment_extractor.extract(assistant_event):
                self.entity_resolver.resolve_claim(claim)
                decision = self.decider.decide(claim)
                self.resolver.apply(claim, decision)

        trace = TurnTrace(
            session_id=session_id,
            user_event_id=user_event.event_id,
            assistant_event_id=assistant_event.event_id,
            extracted_claims=claims,
            decisions=decisions,
            retrieved=retrieved,
            open_loops=open_loops,
            draft_response=draft,
            final_response=final,
            consistency=verdict,
        )
        self.store.add_trace(trace)
        return trace
