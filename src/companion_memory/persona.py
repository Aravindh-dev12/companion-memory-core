from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from .models import EntityType, Event, MemoryClaim, MemoryExtraction, MemoryType, Modality


class Persona(BaseModel):
    name: str
    role: str
    voice: list[str] = Field(default_factory=list)
    core_values: list[str] = Field(default_factory=list)
    stable_preferences: dict[str, str] = Field(default_factory=dict)
    boundaries: list[str] = Field(default_factory=list)
    backstory: list[str] = Field(default_factory=list)

    def compact_context(self) -> str:
        return self.model_dump_json(indent=2)


def load_persona(path: str | Path) -> Persona:
    data = yaml.safe_load(Path(path).read_text())
    return Persona.model_validate(data)


class PersonaCommitmentExtractor:
    def __init__(self, provider):
        self.provider = provider

    def extract(self, event: Event) -> list[MemoryClaim]:
        result: MemoryExtraction = self.provider.extract_persona_commitments(event)
        claims: list[MemoryClaim] = []
        for candidate in result.candidates:
            if not candidate.memory_worthy:
                continue
            # Hard invariant: assistant self-memory cannot be rebound to user
            # memory even if a model emits malformed entity metadata.
            claims.append(
                MemoryClaim(
                    source_event_id=event.event_id,
                    source_event_ids=[event.event_id],
                    memory_type=MemoryType.PERSONA_COMMITMENT,
                    entity_mention="companion",
                    entity_type=EntityType.PERSON,
                    predicate_text=candidate.predicate_text or candidate.predicate or "commitment",
                    value_text=candidate.value_text or candidate.value or "",
                    modality=Modality.ASSERTED,
                    polarity=candidate.polarity,
                    confidence=candidate.confidence,
                    importance=candidate.importance,
                    valid_from=candidate.valid_from,
                    valid_to=candidate.valid_to,
                    event_time_text=candidate.event_time_text,
                    event_time_precision=candidate.event_time_precision,
                    expires_at=candidate.expires_at,
                    evidence_text=candidate.evidence_text.strip() or event.text,
                )
            )
        return claims
