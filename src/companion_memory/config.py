from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .models import RetrievalMode


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class Settings:
    db_path: Path = Path("data/companion.sqlite3")
    persona_path: Path = Path("persona.yaml")
    provider: str = "openai"
    extraction_model: str = "gpt-5.6-luna"
    resolution_model: str = "gpt-5.6-luna"
    response_model: str = "gpt-5.6-terra"
    verification_model: str = "gpt-5.6-luna"
    oracle_model: str = "gpt-5.6-sol"
    embedding_model: str = "text-embedding-3-small"
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    consistency_check: bool = True
    temporal_filter: bool = True
    temporal_resolution: bool = True
    recent_event_limit: int = 8
    retrieval_limit: int = 12
    memory_enabled: bool = True
    full_history_context: bool = False
    open_loop_followups: bool = True
    auto_consolidate_previous_sessions: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            db_path=Path(os.getenv("CMC_DB_PATH", "data/companion.sqlite3")),
            persona_path=Path(os.getenv("CMC_PERSONA_PATH", "persona.yaml")),
            provider=os.getenv("CMC_PROVIDER", "openai"),
            extraction_model=os.getenv("CMC_EXTRACTION_MODEL", "gpt-5.6-luna"),
            resolution_model=os.getenv("CMC_RESOLUTION_MODEL", "gpt-5.6-luna"),
            response_model=os.getenv("CMC_RESPONSE_MODEL", "gpt-5.6-terra"),
            verification_model=os.getenv("CMC_VERIFICATION_MODEL", "gpt-5.6-luna"),
            oracle_model=os.getenv("CMC_ORACLE_MODEL", "gpt-5.6-sol"),
            embedding_model=os.getenv("CMC_EMBEDDING_MODEL", "text-embedding-3-small"),
            retrieval_mode=RetrievalMode(os.getenv("CMC_RETRIEVAL_MODE", "hybrid")),
            consistency_check=_bool_env("CMC_CONSISTENCY_CHECK", True),
            temporal_filter=_bool_env("CMC_TEMPORAL_FILTER", True),
            temporal_resolution=_bool_env("CMC_TEMPORAL_RESOLUTION", True),
            recent_event_limit=int(os.getenv("CMC_RECENT_EVENT_LIMIT", "8")),
            retrieval_limit=int(os.getenv("CMC_RETRIEVAL_LIMIT", "12")),
            open_loop_followups=_bool_env("CMC_OPEN_LOOP_FOLLOWUPS", True),
            auto_consolidate_previous_sessions=_bool_env("CMC_AUTO_CONSOLIDATE", False),
        )
