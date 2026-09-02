from __future__ import annotations

from .embeddings import EmbeddingProvider, cosine_similarity
from .models import Entity, EntityType, MemoryClaim, normalize_key
from .store import MemoryStore


_RELATION_ALIASES = {
    "mom": {"mom", "mum", "mother", "my_mom", "my_mum", "my_mother"},
    "dad": {"dad", "father", "my_dad", "my_father"},
    "sister": {"sister", "my_sister"},
    "brother": {"brother", "my_brother"},
    "partner": {"partner", "girlfriend", "boyfriend", "spouse", "wife", "husband"},
}


class EntityResolver:
    """Resolve conversational mentions to durable entities.

    Exact alias/relation matching is preferred because it is deterministic.
    Optional embedding similarity is only used as a tie-breaker before creating
    a new entity; there is deliberately no magic cosine threshold in the core.
    """

    def __init__(self, store: MemoryStore, embedder: EmbeddingProvider | None = None):
        self.store = store
        self.embedder = embedder

    def resolve_claim(self, claim: MemoryClaim) -> MemoryClaim:
        entity = self.resolve(
            mention=claim.entity_mention or claim.subject or "user",
            entity_type=claim.entity_type,
            relation_to_user=claim.relation_to_user,
            source_event_id=claim.source_event_id,
        )
        claim.entity_id = entity.entity_id
        claim.entity_mention = entity.canonical_name
        claim.subject = normalize_key(entity.canonical_name)
        # fact_key is computed dynamically, so rebinding the canonical entity
        # automatically moves the claim into the correct state namespace.
        return claim

    def resolve(
        self,
        *,
        mention: str,
        entity_type: EntityType,
        relation_to_user: str | None,
        source_event_id: str,
    ) -> Entity:
        normalized = normalize_key(mention)
        relation = normalize_key(relation_to_user) if relation_to_user else None

        if normalized in {"user", "me", "myself", "i"} or entity_type is EntityType.SELF:
            existing = self.store.find_entity_exact("user")
            if existing:
                return self.store.update_entity_aliases(existing.entity_id, [mention], source_event_id)
            return self.store.add_entity(
                Entity(
                    canonical_name="user",
                    entity_type=EntityType.SELF,
                    aliases=["me", "myself", "i"],
                    first_seen_event_id=source_event_id,
                    last_seen_event_id=source_event_id,
                )
            )

        exact = self.store.find_entity_exact(mention, relation)
        if exact:
            return self.store.update_entity_aliases(exact.entity_id, [mention], source_event_id)

        # Canonicalize common kinship references without requiring the model to
        # know a person's legal/name identity.
        canonical = mention.strip()
        if relation:
            for canonical_relation, aliases in _RELATION_ALIASES.items():
                if relation in aliases or normalized in aliases:
                    canonical = canonical_relation
                    break

        # Optional nearest-entity tie-breaker. We choose among the top entity of
        # the same broad type only when normalized lexical tokens overlap too;
        # this avoids silently merging unrelated people with similar embeddings.
        if self.embedder is not None:
            candidates = [e for e in self.store.list_entities() if e.entity_type == entity_type]
            if candidates:
                query_vec = self.embedder.embed([mention])[0]
                scored = []
                mention_tokens = set(normalized.split("_"))
                for entity in candidates:
                    entity_vec = self.embedder.embed([" ".join(sorted(entity.normalized_aliases))])[0]
                    overlap = bool(mention_tokens & set(normalize_key(entity.canonical_name).split("_")))
                    scored.append((cosine_similarity(query_vec, entity_vec), overlap, entity))
                scored.sort(key=lambda item: item[0], reverse=True)
                if scored[0][1]:
                    return self.store.update_entity_aliases(scored[0][2].entity_id, [mention], source_event_id)

        aliases = [mention]
        if relation:
            aliases.append(relation.replace("_", " "))
        return self.store.add_entity(
            Entity(
                canonical_name=canonical,
                entity_type=entity_type,
                aliases=aliases,
                relation_to_user=relation,
                first_seen_event_id=source_event_id,
                last_seen_event_id=source_event_id,
            )
        )
