from __future__ import annotations

import re

from .models import ConsistencyVerdict, RetrievedMemory
from .persona import Persona
from .store import MemoryStore

_FIRST_PERSON_COMMITMENT = re.compile(
    r"\bI(?:\'ve| have| am|\'m)?\s+(?:love|like|hate|hated|prefer|always|never|think|believe|grew up|from|promised?|will remember)\b|\bI(?:\'ve| have)\s+(?:always|never)\s+(?:loved|liked|hated|preferred)\b",
    re.IGNORECASE,
)


class ConsistencyFirewall:
    def __init__(self, store: MemoryStore, provider, *, enabled: bool = True):
        self.store = store
        self.provider = provider
        self.enabled = enabled

    def check(self, *, user_text: str, draft: str, persona: Persona, retrieved: list[RetrievedMemory]) -> ConsistencyVerdict:
        if not self.enabled or not self._needs_check(draft, retrieved):
            return ConsistencyVerdict(consistent=True)
        commitments = self.store.list_active_facts(subject="companion")
        return self.provider.verify_response(
            user_text=user_text,
            draft=draft,
            persona=persona,
            retrieved=retrieved,
            active_commitments=commitments,
        )

    @staticmethod
    def _needs_check(draft: str, retrieved: list[RetrievedMemory]) -> bool:
        if _FIRST_PERSON_COMMITMENT.search(draft):
            return True
        low = draft.casefold()
        # A draft that actually uses retrieved user facts gets verified; mere
        # availability of memory does not double model calls on every turn.
        return any(
            m.fact.value_text and m.fact.value_text.casefold() in low
            for m in retrieved
            if m.fact.value_text.casefold() not in {"none", "unknown"}
        )
