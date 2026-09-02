from __future__ import annotations

import re
from dataclasses import dataclass

from .decay import retrieval_salience
from .embeddings import EmbeddingProvider, cosine_similarity
from .models import FactEvidence, FactStatus, MemoryType, QueryIntent, RetrievedMemory, RetrievalMode
from .store import MemoryStore

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_HISTORICAL_CUES = {"earlier", "before", "previously", "used", "was", "were", "back", "then", "past"}
_CURRENT_CUES = {"still", "now", "current", "currently", "anymore", "today"}
_QUERY_EXPANSIONS = {
    "dating": {"partner", "relationship"},
    "dated": {"partner", "relationship"},
    "relationship": {"partner"},
    "girlfriend": {"partner"},
    "boyfriend": {"partner"},
    "sister": {"sister_name"},
    "brother": {"brother_name"},
    "planning": {"planned_trip", "future_plan"},
    "planned": {"planned_trip", "future_plan"},
    "drink": {"liked_beverage", "preference"},
    "drinks": {"liked_beverage", "preference"},
    "beverage": {"liked_beverage", "preference"},
    "music": {"music", "preference"},
    "job": {"employer", "occupation", "job_title"},
    "work": {"employer", "occupation", "job_title"},
}


@dataclass(frozen=True)
class RetrievalPlan:
    intent: QueryIntent
    expanded_query: str
    memory_type_hint: MemoryType | None = None


class Retriever:
    """Entity + lexical + semantic retrieval fused with Reciprocal Rank Fusion.

    Channel ranks are fused without learned/manual relevance weights. Temporal
    validity is a hard filter; importance/decay are post-fusion priority terms,
    never truth mutation.
    """

    RRF_K = 60

    def __init__(
        self,
        store: MemoryStore,
        *,
        embedder: EmbeddingProvider | None = None,
        weights=None,  # compatibility: old callers may still pass weights
        mode: RetrievalMode = RetrievalMode.HYBRID,
        temporal_filter: bool = True,
    ):
        self.store = store
        self.embedder = embedder
        self.mode = mode
        self.temporal_filter = temporal_filter

    def infer_intent(self, query: str) -> QueryIntent:
        return self.plan(query).intent

    def plan(self, query: str) -> RetrievalPlan:
        tokens = set(_TOKEN_RE.findall(query.lower()))
        if tokens & _HISTORICAL_CUES:
            intent = QueryIntent.HISTORICAL
        elif tokens & _CURRENT_CUES:
            intent = QueryIntent.CURRENT
        else:
            intent = QueryIntent.GENERAL

        expanded = set(tokens)
        for token in list(tokens):
            expanded.update(_QUERY_EXPANSIONS.get(token, set()))

        hint = None
        if tokens & {"like", "likes", "love", "prefer", "preference", "drink", "drinks", "music"}:
            hint = MemoryType.PREFERENCE
        elif tokens & {"dating", "dated", "relationship", "partner", "girlfriend", "boyfriend"}:
            hint = MemoryType.RELATIONSHIP_STATE
        elif tokens & {"plan", "planning", "planned", "trip", "interview"}:
            hint = MemoryType.FUTURE_PLAN
        elif tokens & {"work", "job", "employer", "occupation", "sister", "brother", "name"}:
            hint = MemoryType.IDENTITY_FACT

        return RetrievalPlan(intent=intent, expanded_query=" ".join(sorted(expanded)), memory_type_hint=hint)

    def index_fact(self, fact_id: str) -> None:
        if self.embedder is None:
            return
        evidence = self.store.get_fact_evidence(fact_id)
        if evidence is None:
            return
        vector = self.embedder.embed([self._fact_text(evidence)])[0]
        self.store.upsert_embedding(fact_id, self.embedder.model_name, vector)

    def retrieve(self, query: str, *, limit: int = 6, intent: QueryIntent | None = None) -> list[RetrievedMemory]:
        if self.mode is RetrievalMode.ORACLE:
            evidence = self.store.list_fact_evidence(include_inactive=True)
            evidence.sort(key=lambda item: item.fact.created_at)
            return [
                RetrievedMemory(
                    fact=item.fact,
                    source_event_id=item.source_event_id,
                    evidence_text=item.evidence_text,
                    structured_score=1.0,
                    temporal_score=1.0,
                    importance_score=item.fact.importance,
                    salience_score=1.0,
                    rrf_score=1.0,
                    final_score=1.0,
                    retrieval_reason="oracle-full-store: all fact versions with status/modality visible to the reasoning model",
                )
                for item in evidence[-limit:]
            ]

        plan = self.plan(query)
        if intent is not None:
            plan = RetrievalPlan(intent=intent, expanded_query=plan.expanded_query, memory_type_hint=plan.memory_type_hint)

        include_inactive = plan.intent is QueryIntent.HISTORICAL or not self.temporal_filter
        evidence = self.store.list_fact_evidence(include_inactive=include_inactive)
        allowed: dict[str, FactEvidence] = {}
        for item in evidence:
            fact = item.fact
            if self.temporal_filter and plan.intent is not QueryIntent.HISTORICAL and fact.status is not FactStatus.ACTIVE:
                continue
            if plan.intent is QueryIntent.HISTORICAL and fact.status in {FactStatus.CORRECTED, FactStatus.WITHDRAWN}:
                continue
            allowed[fact.fact_id] = item

        ranks: dict[str, dict[str, int]] = {fid: {} for fid in allowed}
        notes: dict[str, list[str]] = {fid: [] for fid in allowed}

        if self.mode in {RetrievalMode.HYBRID, RetrievalMode.STRUCTURED_ONLY}:
            entity_facts: list[FactEvidence] = []
            for entity in self.store.find_entities_in_text(query, include_user=False):
                items = self.store.list_fact_evidence(
                    include_inactive=include_inactive,
                    entity_id=entity.entity_id,
                )
                items = [i for i in items if i.fact.status not in {FactStatus.CORRECTED, FactStatus.WITHDRAWN}]
                items.sort(key=lambda i: (i.fact.importance, i.fact.created_at), reverse=True)
                entity_facts.extend(items[:8])
            seen: set[str] = set()
            position = 0
            for item in entity_facts:
                if item.fact.fact_id in seen:
                    continue
                seen.add(item.fact.fact_id)
                position += 1
                allowed[item.fact.fact_id] = item
                ranks.setdefault(item.fact.fact_id, {})["entity"] = position
                notes.setdefault(item.fact.fact_id, []).append("entity-channel")

        if self.mode in {RetrievalMode.HYBRID, RetrievalMode.LEXICAL_ONLY}:
            lexical_rows = self.store.lexical_fact_search(
                plan.expanded_query,
                limit=max(limit * 5, 20),
                include_inactive=True,
            )
            lexical_position = 0
            for row in lexical_rows:
                fact = self.store.get_fact(row["fact_id"])
                if fact is None:
                    continue
                target_ids: list[str] = []
                if fact.fact_id in allowed:
                    target_ids = [fact.fact_id]
                elif self.temporal_filter and plan.intent is not QueryIntent.HISTORICAL and fact.status is not FactStatus.ACTIVE:
                    successors = self.store.list_active_facts(fact_key=fact.fact_key)
                    target_ids = [f.fact_id for f in successors]
                    for successor in successors:
                        ev = self.store.get_fact_evidence(successor.fact_id)
                        if ev:
                            allowed[successor.fact_id] = ev
                            ranks.setdefault(successor.fact_id, {})
                            notes.setdefault(successor.fact_id, []).append(f"state-closure:{fact.fact_id}")
                for target_id in target_ids:
                    lexical_position += 1
                    ranks.setdefault(target_id, {}).setdefault("lexical", lexical_position)

        if self.embedder is not None and self.mode in {RetrievalMode.HYBRID, RetrievalMode.SEMANTIC_ONLY}:
            semantic_pool = list(allowed.values())
            if plan.memory_type_hint is not None:
                semantic_pool = [e for e in semantic_pool if e.fact.memory_type == plan.memory_type_hint]
            self._ensure_embeddings(semantic_pool)
            embeddings = self.store.get_embeddings(self.embedder.model_name)
            query_vec = self.embedder.embed([plan.expanded_query])[0]
            scored = []
            for item in semantic_pool:
                vector = embeddings.get(item.fact.fact_id)
                if vector:
                    scored.append((cosine_similarity(query_vec, vector), item.fact.fact_id))
            scored.sort(reverse=True)
            for pos, (score, fact_id) in enumerate(scored[: max(limit * 4, 12)], start=1):
                ranks.setdefault(fact_id, {})["semantic"] = pos
                notes.setdefault(fact_id, []).append(f"semantic-cos={score:.3f}")

        if self.mode is RetrievalMode.STRUCTURED_ONLY:
            q_tokens = set(_TOKEN_RE.findall(plan.expanded_query))
            scored = []
            for fid, item in allowed.items():
                f_tokens = set(_TOKEN_RE.findall(
                    f"{item.fact.entity_mention} {item.fact.slot or ''} {item.fact.predicate_text} {item.fact.value_text}".lower()
                ))
                overlap = len(q_tokens & f_tokens)
                if overlap:
                    scored.append((overlap, fid))
            scored.sort(reverse=True)
            ranks = {fid: {"structured": pos} for pos, (_, fid) in enumerate(scored, start=1)}

        memories: list[RetrievedMemory] = []
        for fid, channel_ranks in ranks.items():
            if not channel_ranks:
                continue
            item = allowed.get(fid)
            if item is None:
                continue
            fact = item.fact
            temporal = self._temporal_score(fact, plan.intent)
            if temporal <= 0.0:
                continue

            rrf = sum(1.0 / (self.RRF_K + rank) for rank in channel_ranks.values())
            salience = retrieval_salience(fact)
            final = rrf * max(fact.importance, 0.05) * max(salience, 0.05) * temporal
            reasons = [f"rrf={rrf:.5f}"]
            reasons.extend(notes.get(fid, []))
            reasons.append(f"status={fact.status.value}")
            reasons.append(f"modality={fact.modality.value}")

            memories.append(
                RetrievedMemory(
                    fact=fact,
                    source_event_id=item.source_event_id,
                    evidence_text=item.evidence_text,
                    entity_rank=channel_ranks.get("entity"),
                    lexical_rank=channel_ranks.get("lexical"),
                    semantic_rank=channel_ranks.get("semantic"),
                    structured_score=(1.0 / channel_ranks["structured"]) if "structured" in channel_ranks else 0.0,
                    lexical_score=(1.0 / channel_ranks["lexical"]) if "lexical" in channel_ranks else 0.0,
                    semantic_score=(1.0 / channel_ranks["semantic"]) if "semantic" in channel_ranks else 0.0,
                    temporal_score=temporal,
                    importance_score=fact.importance,
                    salience_score=salience,
                    rrf_score=rrf,
                    final_score=final,
                    retrieval_reason=", ".join(reasons),
                )
            )

        memories.sort(key=lambda m: (m.final_score, m.rrf_score), reverse=True)
        return memories[:limit]

    def _ensure_embeddings(self, evidence_items: list[FactEvidence]) -> None:
        if self.embedder is None:
            return
        existing = self.store.get_embeddings(self.embedder.model_name)
        missing = [e for e in evidence_items if e.fact.fact_id not in existing]
        if not missing:
            return
        vectors = self.embedder.embed([self._fact_text(e) for e in missing])
        for item, vector in zip(missing, vectors):
            self.store.upsert_embedding(item.fact.fact_id, self.embedder.model_name, vector)

    @staticmethod
    def _fact_text(evidence: FactEvidence) -> str:
        f = evidence.fact
        return (
            f"entity={f.entity_mention}; relation={f.relation_to_user}; slot={f.slot}; "
            f"predicate={f.predicate_text}; value={f.value_text}; modality={f.modality.value}. "
            f"Evidence: {evidence.evidence_text}"
        )

    @staticmethod
    def _temporal_score(fact, intent: QueryIntent) -> float:
        if intent is QueryIntent.HISTORICAL:
            if fact.status is FactStatus.SUPERSEDED:
                return 1.0
            if fact.status is FactStatus.ACTIVE:
                return 0.55
            return 0.0
        return 1.0 if fact.status is FactStatus.ACTIVE else 0.0
