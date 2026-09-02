from __future__ import annotations

from datetime import datetime, timezone
from math import exp, log

from .models import FactVersion, MemoryType


# Half-life is deliberately a retrieval property, not a truth policy.
_HALF_LIFE_DAYS: dict[MemoryType, float | None] = {
    MemoryType.IDENTITY_FACT: None,
    MemoryType.PREFERENCE: None,
    MemoryType.RELATIONSHIP_STATE: None,
    MemoryType.GOAL: 180.0,
    MemoryType.FUTURE_PLAN: 60.0,
    MemoryType.EPISODIC_EVENT: 30.0,
    MemoryType.EMOTIONAL_STATE: 3.0,
    MemoryType.PERSONAL_VALUE: None,
    MemoryType.CONSTRAINT: None,
    MemoryType.INFERENCE: 14.0,
    MemoryType.PERSONA_COMMITMENT: None,
}


def retrieval_salience(fact: FactVersion, now: datetime | None = None) -> float:
    """Return [0,1] retrieval salience without modifying fact validity."""

    half_life = _HALF_LIFE_DAYS[fact.memory_type]
    if half_life is None:
        return 1.0
    now = now or datetime.now(timezone.utc)
    age_days = max((now - fact.created_at).total_seconds() / 86400.0, 0.0)
    return exp(-log(2) * age_days / half_life)
