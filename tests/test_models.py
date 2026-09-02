from companion_memory.models import ConsistencyVerdict, MemoryClaim, MemoryType


def test_fact_key_is_normalized():
    claim = MemoryClaim(
        source_event_id="evt_x",
        memory_type=MemoryType.PREFERENCE,
        subject=" User ",
        predicate=" Coffee_Order ",
        value="black",
        confidence=1.0,
        evidence_text="I drink black coffee",
    )
    assert claim.fact_key == "user::coffee_order"


def test_inconsistent_verdict_requires_revision():
    try:
        ConsistencyVerdict(consistent=False)
    except ValueError:
        pass
    else:
        raise AssertionError("expected validation error")
