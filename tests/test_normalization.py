from datetime import datetime, timezone

from companion_memory.models import EntityType, MemoryCandidate, MemoryType, Modality
from companion_memory.normalization import canonicalize_candidate, resolve_relative_event_time


def candidate(**kwargs):
    base = dict(
        memory_type=MemoryType.IDENTITY_FACT,
        confidence=0.9,
        importance=0.8,
        evidence_text="",
        predicate_text="described_as",
        value_text="x",
    )
    base.update(kwargs)
    return MemoryCandidate(**base)


def test_sister_name_is_canonicalized_to_user_state_key():
    c = candidate(
        entity_mention="Nina",
        entity_type=EntityType.PERSON,
        relation_to_user="sister",
        predicate_text="name",
        value_text="Nina",
        evidence_text="My sister's name is Nina.",
    )
    out = canonicalize_candidate(c, "My sister's name is Nina.")
    assert out.entity_mention == "user"
    assert out.slot == "sister_name"
    assert out.predicate_text == "sister_name"


def test_preferences_share_canonical_multi_value_key():
    c = candidate(
        memory_type=MemoryType.PREFERENCE,
        predicate_text="likes_coffee",
        value_text="coffee",
        evidence_text="I like coffee.",
    )
    out = canonicalize_candidate(c, "I like coffee.")
    assert out.predicate_text == "liked_beverage"
    assert out.slot is None


def test_relative_last_weekend_is_anchored_to_source_event_time():
    reference = datetime(2026, 9, 2, 9, 45, tzinfo=timezone.utc)
    resolved, precision = resolve_relative_event_time("last weekend", reference)
    assert resolved == datetime(2026, 8, 29, 0, 0, tzinfo=timezone.utc)
    assert precision.value == "approximate"


def test_hedged_quit_is_never_promoted_to_asserted_state():
    c = candidate(
        memory_type=MemoryType.FUTURE_PLAN,
        predicate_text="quit_job",
        value_text="job",
        modality=Modality.ASSERTED,
        evidence_text="I might quit after my performance review if it goes badly.",
    )
    out = canonicalize_candidate(c, c.evidence_text)
    assert out.predicate_text == "considering_quitting"
    assert out.modality is Modality.HEDGED
