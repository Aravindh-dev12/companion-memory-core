from __future__ import annotations

from .config import Settings
from .consistency import ConsistencyFirewall
from .embeddings import HashEmbeddingProvider, OpenAIEmbeddingProvider
from .engine import CompanionEngine, EngineConfig
from .entities import EntityResolver
from .matching import CandidateMatcher
from .persona import load_persona
from .providers import HeuristicProvider, OpenAIProvider
from .resolver import StateResolver
from .retrieval import Retriever
from .store import MemoryStore


def build_engine(settings: Settings) -> CompanionEngine:
    store = MemoryStore(settings.db_path)
    persona = load_persona(settings.persona_path)

    if settings.provider == "openai":
        provider = OpenAIProvider(
            extraction_model=settings.extraction_model,
            resolution_model=settings.resolution_model,
            response_model=settings.response_model,
            verification_model=settings.verification_model,
        )
        embedder = OpenAIEmbeddingProvider(settings.embedding_model)
    elif settings.provider == "heuristic":
        provider = HeuristicProvider()
        embedder = HashEmbeddingProvider()
    else:
        store.close()
        raise ValueError(f"unknown provider: {settings.provider}")

    retriever = Retriever(
        store,
        embedder=embedder,
        mode=settings.retrieval_mode,
        temporal_filter=settings.temporal_filter,
    )
    resolver = StateResolver(store, temporal_resolution=settings.temporal_resolution)
    firewall = ConsistencyFirewall(store, provider, enabled=settings.consistency_check)
    entity_resolver = EntityResolver(store, embedder)
    matcher = CandidateMatcher(store, embedder)
    return CompanionEngine(
        store=store,
        provider=provider,
        persona=persona,
        retriever=retriever,
        resolver=resolver,
        consistency=firewall,
        entity_resolver=entity_resolver,
        matcher=matcher,
        config=EngineConfig(
            recent_event_limit=settings.recent_event_limit,
            retrieval_limit=settings.retrieval_limit,
            memory_enabled=settings.memory_enabled,
            full_history_context=settings.full_history_context,
            open_loop_followups=settings.open_loop_followups,
            auto_consolidate_previous_sessions=settings.auto_consolidate_previous_sessions,
        ),
    )
