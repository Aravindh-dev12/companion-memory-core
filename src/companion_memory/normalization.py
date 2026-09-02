from __future__ import annotations

import re
from datetime import datetime, timedelta

from .models import EntityType, EventTimePrecision, MemoryCandidate, MemoryType, Modality

_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def canonicalize_candidate(candidate: MemoryCandidate, source_text: str) -> MemoryCandidate:
    """Normalize high-value state slots without asking the model to own schema identity.

    The model extracts semantics; deterministic code maps common disclosures onto
    durable state keys used by matching, resolution, and evaluation.
    """
    text = " ".join(
        part for part in [source_text, candidate.evidence_text, candidate.slot or "", candidate.predicate_text]
        if part
    ).casefold()

    updates: dict[str, object] = {}

    if "sister" in text and "name" in text:
        updates.update(
            entity_mention="user",
            entity_type=EntityType.SELF,
            relation_to_user=None,
            slot="sister_name",
            predicate_text="sister_name",
        )
        name_match = re.search(r"name\s+is\s+([A-Z][a-z]+)", source_text)
        if name_match:
            updates["value_text"] = name_match.group(1)

    if candidate.memory_type is MemoryType.PREFERENCE and any(token in text for token in ("coffee", "tea", "beverage", "drink")):
        updates.update(
            entity_mention="user",
            entity_type=EntityType.SELF,
            relation_to_user=None,
            slot=None,
            predicate_text="liked_beverage",
        )

    if candidate.memory_type in {MemoryType.FUTURE_PLAN, MemoryType.EPISODIC_EVENT} and "interview" in text:
        updates.update(
            entity_mention="user",
            entity_type=EntityType.SELF,
            relation_to_user=None,
            slot=None,
            predicate_text="interview_plan",
        )
        if candidate.memory_type is MemoryType.FUTURE_PLAN:
            updates["value_text"] = "upcoming interview"

    if candidate.memory_type is MemoryType.RELATIONSHIP_STATE and any(
        token in text for token in ("girlfriend", "boyfriend", "partner", "broke up", "broken up")
    ):
        updates.update(
            entity_mention="user",
            entity_type=EntityType.SELF,
            relation_to_user=None,
            slot="partner",
            predicate_text="partner",
        )

    if any(token in text for token in ("might quit", "considering quitting", "consider quit")):
        updates.update(
            entity_mention="user",
            entity_type=EntityType.SELF,
            relation_to_user=None,
            slot=None,
            predicate_text="considering_quitting",
        )
        if candidate.modality is Modality.ASSERTED:
            updates["modality"] = Modality.HEDGED

    if updates:
        return candidate.model_copy(update=updates)
    return candidate


def resolve_relative_event_time(text: str | None, reference: datetime) -> tuple[datetime | None, EventTimePrecision | None]:
    """Resolve a small, auditable set of relative date phrases against source time.

    Unknown phrases stay unresolved rather than letting an LLM-supplied calendar
    date silently become world truth.
    """
    if not text:
        return None, None
    low = text.strip().casefold()
    base = reference.replace(hour=0, minute=0, second=0, microsecond=0)

    if "last weekend" in low:
        days_since_saturday = (base.weekday() - 5) % 7
        if base.weekday() in {5, 6}:
            days_since_saturday += 7
        return base - timedelta(days=days_since_saturday), EventTimePrecision.APPROXIMATE
    if re.search(r"\byesterday\b", low):
        return base - timedelta(days=1), EventTimePrecision.DAY
    if re.search(r"\btoday\b", low):
        return base, EventTimePrecision.DAY
    if re.search(r"\btomorrow\b", low):
        return base + timedelta(days=1), EventTimePrecision.DAY

    weekday_match = re.search(r"\b(?:(last|next|this)\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", low)
    if weekday_match:
        qualifier, name = weekday_match.groups()
        target = _WEEKDAYS[name]
        current = base.weekday()
        if qualifier == "last":
            delta = -((current - target) % 7 or 7)
        elif qualifier == "next":
            delta = (target - current) % 7 or 7
        else:
            delta = (target - current) % 7
        return base + timedelta(days=delta), EventTimePrecision.DAY

    return None, None
