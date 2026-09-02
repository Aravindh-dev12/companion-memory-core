from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
import re
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def normalize_key(value: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", value.strip().lower())
    return "_".join(tokens)


class Speaker(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class EntityType(StrEnum):
    SELF = "self"
    PERSON = "person"
    PLACE = "place"
    ORG = "org"
    PROJECT = "project"
    PET = "pet"
    PAIR = "pair"
    OTHER = "other"


class MemoryType(StrEnum):
    IDENTITY_FACT = "identity_fact"
    PREFERENCE = "preference"
    RELATIONSHIP_STATE = "relationship_state"
    GOAL = "goal"
    FUTURE_PLAN = "future_plan"
    EPISODIC_EVENT = "episodic_event"
    EMOTIONAL_STATE = "emotional_state"
    PERSONAL_VALUE = "personal_value"
    CONSTRAINT = "constraint"
    INFERENCE = "inference"
    PERSONA_COMMITMENT = "persona_commitment"


class Modality(StrEnum):
    ASSERTED = "asserted"
    HEDGED = "hedged"
    HYPOTHETICAL = "hypothetical"
    REPORTED_BY_THIRD_PARTY = "reported_by_third_party"
    NEGATED = "negated"


class EventTimePrecision(StrEnum):
    EXACT = "exact"
    DAY = "day"
    RANGE = "range"
    APPROXIMATE = "approximate"
    UNKNOWN = "unknown"


class ResolutionAction(StrEnum):
    ADD = "add"
    SUPERSEDE = "supersede"  # retained for backwards-compatible generic replacement
    CORRECT = "correct"
    TEMPORAL_TRANSITION = "temporal_transition"
    COEXIST = "coexist"
    REFINE = "refine"
    WITHDRAW = "withdraw"
    IGNORE = "ignore"


class FactStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    CORRECTED = "corrected"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"
    RETIRED = "retired"


class QueryIntent(StrEnum):
    CURRENT = "current"
    HISTORICAL = "historical"
    PERSONA = "persona"
    GENERAL = "general"


class RetrievalMode(StrEnum):
    HYBRID = "hybrid"
    STRUCTURED_ONLY = "structured_only"
    LEXICAL_ONLY = "lexical_only"
    SEMANTIC_ONLY = "semantic_only"
    ORACLE = "oracle"


class OpenLoopKind(StrEnum):
    USER_PLAN = "user_plan"
    USER_GOAL = "user_goal"
    COMPANION_PROMISE = "companion_promise"


class Event(BaseModel):
    event_id: str = Field(default_factory=lambda: new_id("evt"))
    session_id: str
    turn_id: int
    speaker: Speaker
    text: str
    event_time: datetime = Field(default_factory=utcnow)
    ingested_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Entity(BaseModel):
    entity_id: str = Field(default_factory=lambda: new_id("ent"))
    canonical_name: str
    entity_type: EntityType = EntityType.OTHER
    aliases: list[str] = Field(default_factory=list)
    relation_to_user: str | None = None
    first_seen_event_id: str | None = None
    last_seen_event_id: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    @property
    def normalized_aliases(self) -> set[str]:
        values = {normalize_key(self.canonical_name)}
        values.update(normalize_key(alias) for alias in self.aliases)
        if self.relation_to_user:
            values.add(normalize_key(self.relation_to_user))
        return values


class MemoryCandidate(BaseModel):
    """Provider-facing structured extraction schema.

    v2.1 is entity/proposition oriented. Legacy subject/predicate/value fields
    remain accepted so offline fixtures and older providers are still readable.
    The application, not the model, binds durable entity ids and provenance.
    """

    memory_worthy: bool = True
    memory_type: MemoryType

    entity_mention: str = "user"
    entity_type: EntityType = EntityType.SELF
    relation_to_user: str | None = None
    slot: str | None = None
    predicate_text: str = ""
    value_text: str = ""
    modality: Modality = Modality.ASSERTED
    polarity: int = Field(default=1, ge=-1, le=1)

    confidence: float = Field(ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    event_time_text: str | None = None
    event_time_precision: EventTimePrecision = EventTimePrecision.UNKNOWN
    expires_at: datetime | None = None
    evidence_text: str
    rationale: str = ""

    # Compatibility fields. They are synchronized with v2.1 fields.
    subject: str | None = None
    predicate: str | None = None
    value: str | None = None

    @model_validator(mode="after")
    def sync_legacy_fields(self) -> "MemoryCandidate":
        if self.subject and self.entity_mention == "user":
            self.entity_mention = self.subject
        if not self.subject:
            self.subject = self.entity_mention
        if self.predicate and not self.predicate_text:
            self.predicate_text = self.predicate
        if not self.predicate:
            self.predicate = self.predicate_text or self.slot or "described_as"
        if not self.predicate_text:
            self.predicate_text = self.predicate
        if self.value and not self.value_text:
            self.value_text = self.value
        if not self.value:
            self.value = self.value_text
        if not self.value_text:
            self.value_text = self.value or ""
        if self.slot:
            self.slot = normalize_key(self.slot)
        self.predicate_text = normalize_key(self.predicate_text)
        self.subject = normalize_key(self.subject or self.entity_mention)
        return self


class MemoryExtraction(BaseModel):
    candidates: list[MemoryCandidate] = Field(default_factory=list)
    ignored_summary: str = ""


class MemoryClaim(BaseModel):
    claim_id: str = Field(default_factory=lambda: new_id("clm"))
    source_event_id: str
    source_event_ids: list[str] = Field(default_factory=list)
    entity_id: str | None = None
    entity_mention: str = "user"
    entity_type: EntityType = EntityType.SELF
    relation_to_user: str | None = None
    memory_type: MemoryType
    slot: str | None = None
    predicate_text: str = ""
    value_text: str = ""
    modality: Modality = Modality.ASSERTED
    polarity: int = Field(default=1, ge=-1, le=1)
    confidence: float = Field(ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    event_time_text: str | None = None
    event_time_precision: EventTimePrecision = EventTimePrecision.UNKNOWN
    expires_at: datetime | None = None
    evidence_text: str
    created_at: datetime = Field(default_factory=utcnow)  # transaction/recorded time

    # Compatibility aliases used by existing CLI/tests.
    subject: str | None = None
    predicate: str | None = None
    value: str | None = None

    @model_validator(mode="after")
    def normalize_claim(self) -> "MemoryClaim":
        if not self.source_event_ids:
            self.source_event_ids = [self.source_event_id]
        if self.subject and self.entity_mention == "user":
            self.entity_mention = self.subject
        self.subject = normalize_key(self.subject or self.entity_mention)
        if self.predicate and not self.predicate_text:
            self.predicate_text = self.predicate
        if not self.predicate_text:
            self.predicate_text = self.slot or "described_as"
        self.predicate_text = normalize_key(self.predicate_text)
        self.predicate = normalize_key(self.predicate or self.predicate_text)
        if self.value and not self.value_text:
            self.value_text = self.value
        if not self.value_text:
            self.value_text = self.value or ""
        self.value = self.value_text
        if self.slot:
            self.slot = normalize_key(self.slot)
        return self

    @property
    def recorded_at(self) -> datetime:
        return self.created_at

    @property
    def state_key(self) -> str:
        mention = normalize_key(self.entity_mention or self.subject or "user")
        # Human-readable stable namespaces for the two canonical selves; entity
        # ids for everything else prevent two people with the same name from
        # colliding in state history/state-closure retrieval.
        anchor = mention if mention in {"user", "companion"} else (self.entity_id or mention)
        predicate = self.slot or self.predicate_text or self.predicate or "described_as"
        return f"{anchor}::{normalize_key(predicate)}"

    @property
    def fact_key(self) -> str:
        return self.state_key


class StateDecision(BaseModel):
    action: ResolutionAction
    reason: str
    target_fact_id: str | None = None
    event_time: datetime | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class FactVersion(BaseModel):
    fact_id: str = Field(default_factory=lambda: new_id("fact"))
    fact_key: str
    entity_id: str | None = None
    entity_mention: str = "user"
    entity_type: EntityType = EntityType.SELF
    relation_to_user: str | None = None
    slot: str | None = None
    predicate_text: str = ""
    value_text: str = ""
    modality: Modality = Modality.ASSERTED
    polarity: int = Field(default=1, ge=-1, le=1)
    memory_type: MemoryType
    status: FactStatus = FactStatus.ACTIVE
    source_claim_id: str
    supersedes_fact_id: str | None = None
    valid_from: datetime | None = None  # world-valid time
    valid_to: datetime | None = None
    event_time_precision: EventTimePrecision = EventTimePrecision.UNKNOWN
    confidence: float = Field(ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utcnow)
    retired_at: datetime | None = None  # transaction/system time
    expires_at: datetime | None = None

    # Compatibility aliases.
    subject: str | None = None
    predicate: str | None = None
    value: str | None = None

    @model_validator(mode="after")
    def normalize_fact(self) -> "FactVersion":
        self.subject = normalize_key(self.subject or self.entity_mention)
        if not self.entity_mention:
            self.entity_mention = self.subject
        if self.predicate and not self.predicate_text:
            self.predicate_text = self.predicate
        if not self.predicate_text:
            self.predicate_text = self.slot or "described_as"
        self.predicate_text = normalize_key(self.predicate_text)
        self.predicate = normalize_key(self.predicate or self.predicate_text)
        if self.value and not self.value_text:
            self.value_text = self.value
        if not self.value_text:
            self.value_text = self.value or ""
        self.value = self.value_text
        if self.slot:
            self.slot = normalize_key(self.slot)
        return self

    @property
    def recorded_at(self) -> datetime:
        return self.created_at


class FactEvidence(BaseModel):
    fact: FactVersion
    source_event_id: str
    source_event_ids: list[str] = Field(default_factory=list)
    evidence_text: str


class RetrievedMemory(BaseModel):
    fact: FactVersion
    source_event_id: str = ""
    evidence_text: str = ""
    entity_rank: int | None = None
    lexical_rank: int | None = None
    semantic_rank: int | None = None
    structured_score: float = 0.0  # retained for ablation/compatibility
    lexical_score: float = 0.0
    semantic_score: float = 0.0
    temporal_score: float = 0.0
    importance_score: float = 0.0
    salience_score: float = 0.0
    rrf_score: float = 0.0
    final_score: float = 0.0
    retrieval_reason: str = ""


class OpenLoop(BaseModel):
    fact_id: str
    kind: OpenLoopKind
    summary: str
    source_session_id: str | None = None
    due_at: datetime | None = None
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str = ""


class SessionConsolidation(BaseModel):
    summary: str
    inferences: list[MemoryCandidate] = Field(default_factory=list)


class EntailmentItem(BaseModel):
    proposition: str
    entailed: bool
    reason: str = ""


class EntailmentJudgment(BaseModel):
    items: list[EntailmentItem] = Field(default_factory=list)


class PersonaJudgment(BaseModel):
    self_contradiction: bool = False
    generic_drift_score: int = Field(ge=1, le=5)  # 1=strongly in voice, 5=generic/helpdesk drift
    voice_adherence_score: int = Field(ge=1, le=5)  # 1=poor, 5=strong
    reason: str = ""


class ConsistencyIssue(BaseModel):
    kind: str
    detail: str


class ConsistencyVerdict(BaseModel):
    consistent: bool
    issues: list[ConsistencyIssue] = Field(default_factory=list)
    revised_response: str | None = None

    @model_validator(mode="after")
    def revision_required_when_inconsistent(self) -> "ConsistencyVerdict":
        if not self.consistent and not (self.revised_response or "").strip():
            raise ValueError("inconsistent verdict requires revised_response")
        return self


class TransitionRecord(BaseModel):
    transition_id: str = Field(default_factory=lambda: new_id("txn"))
    claim_id: str
    action: ResolutionAction
    reason: str
    from_fact_id: str | None = None
    to_fact_id: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class TurnTrace(BaseModel):
    trace_id: str = Field(default_factory=lambda: new_id("trace"))
    session_id: str
    user_event_id: str
    assistant_event_id: str | None = None
    extracted_claims: list[MemoryClaim] = Field(default_factory=list)
    decisions: list[StateDecision] = Field(default_factory=list)
    retrieved: list[RetrievedMemory] = Field(default_factory=list)
    open_loops: list[OpenLoop] = Field(default_factory=list)
    draft_response: str = ""
    final_response: str = ""
    consistency: ConsistencyVerdict | None = None
    created_at: datetime = Field(default_factory=utcnow)
