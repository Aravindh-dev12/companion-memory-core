from __future__ import annotations

import re

from .models import ConsistencyVerdict, RetrievedMemory
from .persona import Persona
from .store import MemoryStore

_PERSONA_PRESSURE = re.compile(
    r"agree with me|always hated|always loved|tell me.{0,40}(?:like|hate|prefer)|ultra-corporate|\b\d{1,2}-point\b|productivity protocol",
    re.IGNORECASE,
)

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
        if not self.enabled or not self._needs_check(user_text, draft, retrieved):
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
    def _needs_check(user_text: str, draft: str, retrieved: list[RetrievedMemory]) -> bool:
        if _PERSONA_PRESSURE.search(user_text) or _FIRST_PERSON_COMMITMENT.search(draft):
            return True
        low = draft.casefold()
        return any(
            m.fact.value_text and m.fact.value_text.casefold() in low
            for m in retrieved
            if m.fact.value_text.casefold() not in {"none", "unknown"}
        )
