from datetime import datetime, timedelta, timezone

from companion_memory.decay import retrieval_salience
from companion_memory.models import FactVersion, MemoryType


def _fact(kind: MemoryType, age_days: int) -> FactVersion:
    return FactVersion(
        fact_key="user::x",
        subject="user",
        predicate="x",
        value="v",
        memory_type=kind,
        source_claim_id="c",
        confidence=1.0,
        created_at=datetime.now(timezone.utc) - timedelta(days=age_days),
    )


def test_stable_identity_does_not_decay_by_age():
    assert retrieval_salience(_fact(MemoryType.IDENTITY_FACT, 1000)) == 1.0


def test_emotional_state_decays_faster_than_episode():
    emotional = retrieval_salience(_fact(MemoryType.EMOTIONAL_STATE, 10))
    episode = retrieval_salience(_fact(MemoryType.EPISODIC_EVENT, 10))
    assert emotional < episode
