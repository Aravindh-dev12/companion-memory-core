from __future__ import annotations

import re
from datetime import datetime, timezone

from .models import FactStatus, MemoryType, OpenLoop, OpenLoopKind
from .store import MemoryStore

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_PROMISE_CUES = {"promise", "promised", "remind", "remember", "ask", "follow_up", "followup"}


class OpenLoopManager:
    """Derive follow-up candidates from durable memory without creating new truth.

    Open loops are a read-time projection over active goals/plans and companion
    promises. They are surfaced primarily when a user returns in a later session
    or when a due date has passed. Nothing is marked complete merely because the
    loop was surfaced; the normal extraction/resolution path owns state changes.
    """

    def __init__(self, store: MemoryStore):
        self.store = store

    def candidates(
        self,
        *,
        current_session_id: str,
        user_text: str,
        first_turn_in_session: bool,
        now: datetime | None = None,
        limit: int = 3,
    ) -> list[OpenLoop]:
        now = now or datetime.now(timezone.utc)
        q_tokens = set(_TOKEN_RE.findall(user_text.casefold()))
        loops: list[OpenLoop] = []

        for fact in self.store.list_active_facts():
            kind = self._kind(fact)
            if kind is None or fact.status is not FactStatus.ACTIVE:
                continue
            evidence = self.store.get_fact_evidence(fact.fact_id)
            if evidence is None:
                continue
            source_event = self.store.get_event(evidence.source_event_id)
            source_session = source_event.session_id if source_event else None
            due = bool(fact.expires_at and fact.expires_at <= now)
            cross_session = bool(source_session and source_session != current_session_id)
            relevant_now = self._query_relevant(fact, q_tokens)

            # Avoid nagging on the same turn a plan/goal is first disclosed.
            if not due and not cross_session and not relevant_now:
                continue
            # Proactive follow-up is mostly a return-session behavior. On later
            # turns in the same session, require a due or explicitly relevant loop.
            if not first_turn_in_session and not due and not relevant_now:
                continue

            priority = min(
                1.0,
                0.45 * max(fact.importance, 0.1)
                + (0.30 if due else 0.0)
                + (0.20 if cross_session else 0.0)
                + (0.15 if relevant_now else 0.0),
            )
            loops.append(
                OpenLoop(
                    fact_id=fact.fact_id,
                    kind=kind,
                    summary=self._summary(fact),
                    source_session_id=source_session,
                    due_at=fact.expires_at,
                    priority=priority,
                    reason=", ".join(
                        x for x in ["due" if due else "", "cross-session" if cross_session else "", "query-related" if relevant_now else ""] if x
                    ),
                )
            )

        loops.sort(key=lambda item: (item.priority, item.due_at or datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
        return loops[:limit]

    @staticmethod
    def _kind(fact) -> OpenLoopKind | None:
        if fact.memory_type is MemoryType.FUTURE_PLAN:
            return OpenLoopKind.USER_PLAN
        if fact.memory_type is MemoryType.GOAL:
            return OpenLoopKind.USER_GOAL
        if fact.memory_type is MemoryType.PERSONA_COMMITMENT:
            predicate_tokens = set(_TOKEN_RE.findall(fact.predicate_text.casefold()))
            value_tokens = set(_TOKEN_RE.findall(fact.value_text.casefold()))
            if predicate_tokens & _PROMISE_CUES or value_tokens & _PROMISE_CUES:
                return OpenLoopKind.COMPANION_PROMISE
        return None

    @staticmethod
    def _query_relevant(fact, q_tokens: set[str]) -> bool:
        if not q_tokens:
            return False
        fact_tokens = set(
            _TOKEN_RE.findall(
                f"{fact.entity_mention} {fact.relation_to_user or ''} {fact.slot or ''} {fact.predicate_text} {fact.value_text}".casefold()
            )
        )
        return len(q_tokens & fact_tokens) >= 2

    @staticmethod
    def _summary(fact) -> str:
        entity = fact.entity_mention if fact.entity_mention not in {"user", "companion"} else "you"
        if fact.memory_type is MemoryType.FUTURE_PLAN:
            return f"{entity} had a plan: {fact.value_text}"
        if fact.memory_type is MemoryType.GOAL:
            return f"{entity} had an unresolved goal: {fact.value_text}"
        return f"The companion committed to: {fact.value_text}"
