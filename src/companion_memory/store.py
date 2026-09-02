from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .models import (
    Entity, EntityType, Event, EventTimePrecision, FactEvidence, FactStatus,
    FactVersion, MemoryClaim, MemoryType, Modality, ResolutionAction,
    StateDecision, TransitionRecord, TurnTrace, normalize_key,
)

_SCHEMA = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY, started_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS session_consolidations (session_id TEXT PRIMARY KEY, summary_fact_id TEXT, consolidated_at TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}', FOREIGN KEY(session_id) REFERENCES sessions(session_id));
CREATE TABLE IF NOT EXISTS events (event_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, turn_id INTEGER NOT NULL, speaker TEXT NOT NULL, text TEXT NOT NULL, event_time TEXT NOT NULL, ingested_at TEXT NOT NULL, metadata_json TEXT NOT NULL, UNIQUE(session_id, turn_id), FOREIGN KEY(session_id) REFERENCES sessions(session_id));
CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(event_id UNINDEXED, text, tokenize='unicode61');
CREATE TABLE IF NOT EXISTS entities (entity_id TEXT PRIMARY KEY, canonical_name TEXT NOT NULL, normalized_name TEXT NOT NULL, entity_type TEXT NOT NULL, aliases_json TEXT NOT NULL, relation_to_user TEXT, first_seen_event_id TEXT, last_seen_event_id TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, FOREIGN KEY(first_seen_event_id) REFERENCES events(event_id), FOREIGN KEY(last_seen_event_id) REFERENCES events(event_id));
CREATE INDEX IF NOT EXISTS idx_entities_normalized ON entities(normalized_name);
CREATE INDEX IF NOT EXISTS idx_entities_relation ON entities(relation_to_user);
CREATE TABLE IF NOT EXISTS claims (claim_id TEXT PRIMARY KEY, source_event_id TEXT NOT NULL, source_event_ids_json TEXT NOT NULL, entity_id TEXT, entity_mention TEXT NOT NULL, entity_type TEXT NOT NULL, relation_to_user TEXT, memory_type TEXT NOT NULL, slot TEXT, predicate_text TEXT NOT NULL, value_text TEXT NOT NULL, fact_key TEXT NOT NULL, modality TEXT NOT NULL, polarity INTEGER NOT NULL, confidence REAL NOT NULL, importance REAL NOT NULL, valid_from TEXT, valid_to TEXT, event_time_text TEXT, event_time_precision TEXT NOT NULL, expires_at TEXT, evidence_text TEXT NOT NULL, created_at TEXT NOT NULL, FOREIGN KEY(source_event_id) REFERENCES events(event_id), FOREIGN KEY(entity_id) REFERENCES entities(entity_id));
CREATE TABLE IF NOT EXISTS fact_versions (fact_id TEXT PRIMARY KEY, fact_key TEXT NOT NULL, entity_id TEXT, entity_mention TEXT NOT NULL, entity_type TEXT NOT NULL, relation_to_user TEXT, slot TEXT, predicate_text TEXT NOT NULL, value_text TEXT NOT NULL, memory_type TEXT NOT NULL, modality TEXT NOT NULL, polarity INTEGER NOT NULL, status TEXT NOT NULL, source_claim_id TEXT NOT NULL, supersedes_fact_id TEXT, valid_from TEXT, valid_to TEXT, event_time_precision TEXT NOT NULL, confidence REAL NOT NULL, importance REAL NOT NULL, created_at TEXT NOT NULL, retired_at TEXT, expires_at TEXT, FOREIGN KEY(entity_id) REFERENCES entities(entity_id), FOREIGN KEY(source_claim_id) REFERENCES claims(claim_id), FOREIGN KEY(supersedes_fact_id) REFERENCES fact_versions(fact_id));
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(fact_id UNINDEXED, entity_name, slot, predicate_text, value_text, evidence_text, tokenize='unicode61');
CREATE TABLE IF NOT EXISTS fact_embeddings (fact_id TEXT NOT NULL, model TEXT NOT NULL, vector_json TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(fact_id, model), FOREIGN KEY(fact_id) REFERENCES fact_versions(fact_id));
CREATE TABLE IF NOT EXISTS transitions (transition_id TEXT PRIMARY KEY, claim_id TEXT NOT NULL, action TEXT NOT NULL, reason TEXT NOT NULL, from_fact_id TEXT, to_fact_id TEXT, created_at TEXT NOT NULL, FOREIGN KEY(claim_id) REFERENCES claims(claim_id), FOREIGN KEY(from_fact_id) REFERENCES fact_versions(fact_id), FOREIGN KEY(to_fact_id) REFERENCES fact_versions(fact_id));
CREATE TABLE IF NOT EXISTS turn_traces (trace_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, user_event_id TEXT NOT NULL, trace_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_events_session_turn ON events(session_id, turn_id);
CREATE INDEX IF NOT EXISTS idx_session_consolidations_time ON session_consolidations(consolidated_at);
CREATE INDEX IF NOT EXISTS idx_fact_versions_key_status ON fact_versions(fact_key, status);
CREATE INDEX IF NOT EXISTS idx_fact_versions_entity_status ON fact_versions(entity_id, status);
CREATE INDEX IF NOT EXISTS idx_fact_versions_slot_status ON fact_versions(slot, status);
CREATE INDEX IF NOT EXISTS idx_claims_event ON claims(source_event_id);
CREATE INDEX IF NOT EXISTS idx_transitions_claim ON transitions(claim_id);
CREATE INDEX IF NOT EXISTS idx_traces_session ON turn_traces(session_id, created_at);
"""


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _fts_query(text: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
    if not tokens:
        return '"__no_match__"'
    return " OR ".join(f'"{token}"' for token in tokens[:24])


class MemoryStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
    def close(self): self.conn.close()

    def ensure_session(self, session_id: str) -> None:
        self.conn.execute("INSERT OR IGNORE INTO sessions(session_id, started_at) VALUES (?, ?)", (session_id, datetime.now(timezone.utc).isoformat()))
        self.conn.commit()

    def next_turn_id(self, session_id: str) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(turn_id), 0) AS max_turn FROM events WHERE session_id = ?", (session_id,)).fetchone()
        return int(row["max_turn"]) + 1

    def list_sessions(self) -> list[str]:
        return [r["session_id"] for r in self.conn.execute("SELECT session_id FROM sessions ORDER BY started_at").fetchall()]

    def list_unconsolidated_sessions(self, *, exclude_session: str | None = None) -> list[str]:
        args: list[object] = []
        where = "WHERE sc.session_id IS NULL"
        if exclude_session is not None:
            where += " AND s.session_id != ?"; args.append(exclude_session)
        rows = self.conn.execute(f"""SELECT s.session_id FROM sessions s LEFT JOIN session_consolidations sc ON sc.session_id=s.session_id {where} AND EXISTS (SELECT 1 FROM events e WHERE e.session_id=s.session_id) ORDER BY s.started_at""", args).fetchall()
        return [r["session_id"] for r in rows]

    def is_session_consolidated(self, session_id: str) -> bool:
        return self.conn.execute("SELECT 1 FROM session_consolidations WHERE session_id=?", (session_id,)).fetchone() is not None

    def record_session_consolidation(self, session_id: str, *, summary_fact_id: str | None, metadata: dict | None = None) -> None:
        self.ensure_session(session_id)
        with self.conn:
            self.conn.execute("""INSERT INTO session_consolidations(session_id, summary_fact_id, consolidated_at, metadata_json) VALUES (?, ?, ?, ?) ON CONFLICT(session_id) DO UPDATE SET summary_fact_id=excluded.summary_fact_id, consolidated_at=excluded.consolidated_at, metadata_json=excluded.metadata_json""", (session_id, summary_fact_id, datetime.now(timezone.utc).isoformat(), json.dumps(metadata or {})))

    def list_session_events(self, session_id: str) -> list[Event]:
        return [self._row_to_event(r) for r in self.conn.execute("SELECT * FROM events WHERE session_id=? ORDER BY turn_id", (session_id,)).fetchall()]

    def add_event(self, event: Event) -> None:
        self.ensure_session(event.session_id)
        with self.conn:
            self.conn.execute("INSERT INTO events(event_id,session_id,turn_id,speaker,text,event_time,ingested_at,metadata_json) VALUES (?,?,?,?,?,?,?,?)", (event.event_id,event.session_id,event.turn_id,event.speaker.value,event.text,_iso(event.event_time),_iso(event.ingested_at),json.dumps(event.metadata,sort_keys=True)))
            self.conn.execute("INSERT INTO events_fts(event_id,text) VALUES (?,?)", (event.event_id,event.text))

    def list_recent_events(self, session_id: str, limit: int = 8) -> list[Event]:
        rows = self.conn.execute("SELECT * FROM events WHERE session_id=? ORDER BY turn_id DESC LIMIT ?", (session_id,limit)).fetchall()
        return [self._row_to_event(r) for r in reversed(rows)]

    def list_events(self, limit: int = 10000) -> list[Event]:
        return [self._row_to_event(r) for r in self.conn.execute("SELECT * FROM events ORDER BY ingested_at,session_id,turn_id LIMIT ?", (limit,)).fetchall()]

    def get_event(self, event_id: str) -> Event | None:
        row=self.conn.execute("SELECT * FROM events WHERE event_id=?",(event_id,)).fetchone(); return self._row_to_event(row) if row else None

    def add_entity(self, entity: Entity) -> Entity:
        aliases=sorted({a.strip() for a in entity.aliases if a.strip()})
        with self.conn:
            self.conn.execute("INSERT INTO entities(entity_id,canonical_name,normalized_name,entity_type,aliases_json,relation_to_user,first_seen_event_id,last_seen_event_id,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)", (entity.entity_id,entity.canonical_name,normalize_key(entity.canonical_name),entity.entity_type.value,json.dumps(aliases),normalize_key(entity.relation_to_user) if entity.relation_to_user else None,entity.first_seen_event_id,entity.last_seen_event_id,_iso(entity.created_at),_iso(entity.updated_at)))
        entity.aliases=aliases; return entity

    def update_entity_aliases(self, entity_id: str, aliases: list[str], last_seen_event_id: str | None=None) -> Entity:
        entity=self.get_entity(entity_id)
        if entity is None: raise ValueError(f"entity not found: {entity_id}")
        merged=sorted({*entity.aliases,*(a.strip() for a in aliases if a.strip())}); now=datetime.now(timezone.utc)
        with self.conn: self.conn.execute("UPDATE entities SET aliases_json=?, last_seen_event_id=COALESCE(?,last_seen_event_id), updated_at=? WHERE entity_id=?", (json.dumps(merged),last_seen_event_id,_iso(now),entity_id))
        refreshed=self.get_entity(entity_id); assert refreshed is not None; return refreshed

    def get_entity(self, entity_id: str) -> Entity | None:
        row=self.conn.execute("SELECT * FROM entities WHERE entity_id=?",(entity_id,)).fetchone(); return self._row_to_entity(row) if row else None

    def list_entities(self) -> list[Entity]:
        return [self._row_to_entity(r) for r in self.conn.execute("SELECT * FROM entities ORDER BY created_at").fetchall()]

    def find_entity_exact(self, mention: str, relation_to_user: str | None=None) -> Entity | None:
        needle=normalize_key(mention); relation=normalize_key(relation_to_user) if relation_to_user else None
        for entity in self.list_entities():
            aliases=entity.normalized_aliases
            if needle in aliases or (relation and relation in aliases): return entity
        return None

    def find_entities_in_text(self, text: str, *, include_user: bool=False) -> list[Entity]:
        normalized=normalize_key(text); padded=f"_{normalized}_"; matches=[]
        for entity in self.list_entities():
            if entity.entity_type is EntityType.SELF and not include_user: continue
            aliases=sorted(entity.normalized_aliases,key=len,reverse=True)
            if any(f"_{a}_" in padded or normalized==a for a in aliases if a): matches.append(entity)
        return matches

    def add_claim(self, claim: MemoryClaim) -> None:
        with self.conn:
            self.conn.execute("""INSERT INTO claims(claim_id,source_event_id,source_event_ids_json,entity_id,entity_mention,entity_type,relation_to_user,memory_type,slot,predicate_text,value_text,fact_key,modality,polarity,confidence,importance,valid_from,valid_to,event_time_text,event_time_precision,expires_at,evidence_text,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (claim.claim_id,claim.source_event_id,json.dumps(claim.source_event_ids),claim.entity_id,claim.entity_mention,claim.entity_type.value,normalize_key(claim.relation_to_user) if claim.relation_to_user else None,claim.memory_type.value,claim.slot,claim.predicate_text,claim.value_text,claim.fact_key,claim.modality.value,claim.polarity,claim.confidence,claim.importance,_iso(claim.valid_from),_iso(claim.valid_to),claim.event_time_text,claim.event_time_precision.value,_iso(claim.expires_at),claim.evidence_text,_iso(claim.created_at)))

    def claim_exists_for_event_value(self, event_id: str, fact_key: str, value: str) -> bool:
        needle=value.strip().casefold()
        for row in self.conn.execute("SELECT source_event_ids_json,value_text FROM claims WHERE fact_key=?",(fact_key,)).fetchall():
            if event_id in json.loads(row["source_event_ids_json"]) and row["value_text"].strip().casefold()==needle: return True
        return False

    def get_claim(self, claim_id: str) -> MemoryClaim | None:
        row=self.conn.execute("SELECT * FROM claims WHERE claim_id=?",(claim_id,)).fetchone(); return self._row_to_claim(row) if row else None

    def add_fact(self, fact: FactVersion) -> None:
        claim=self.get_claim(fact.source_claim_id)
        if claim is None: raise ValueError(f"source claim not found: {fact.source_claim_id}")
        with self.conn:
            self.conn.execute("""INSERT INTO fact_versions(fact_id,fact_key,entity_id,entity_mention,entity_type,relation_to_user,slot,predicate_text,value_text,memory_type,modality,polarity,status,source_claim_id,supersedes_fact_id,valid_from,valid_to,event_time_precision,confidence,importance,created_at,retired_at,expires_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (fact.fact_id,fact.fact_key,fact.entity_id,fact.entity_mention,fact.entity_type.value,normalize_key(fact.relation_to_user) if fact.relation_to_user else None,fact.slot,fact.predicate_text,fact.value_text,fact.memory_type.value,fact.modality.value,fact.polarity,fact.status.value,fact.source_claim_id,fact.supersedes_fact_id,_iso(fact.valid_from),_iso(fact.valid_to),fact.event_time_precision.value,fact.confidence,fact.importance,_iso(fact.created_at),_iso(fact.retired_at),_iso(fact.expires_at)))
            self.conn.execute("INSERT INTO facts_fts(fact_id,entity_name,slot,predicate_text,value_text,evidence_text) VALUES (?,?,?,?,?,?)", (fact.fact_id,fact.entity_mention,fact.slot or "",fact.predicate_text,fact.value_text,claim.evidence_text))

    def get_fact(self, fact_id: str) -> FactVersion | None:
        row=self.conn.execute("SELECT * FROM fact_versions WHERE fact_id=?",(fact_id,)).fetchone(); return self._row_to_fact(row) if row else None

    def get_fact_evidence(self, fact_id: str) -> FactEvidence | None:
        row=self.conn.execute("SELECT f.*,c.source_event_id,c.source_event_ids_json,c.evidence_text FROM fact_versions f JOIN claims c ON c.claim_id=f.source_claim_id WHERE f.fact_id=?",(fact_id,)).fetchone()
        if not row: return None
        return FactEvidence(fact=self._row_to_fact(row),source_event_id=row["source_event_id"],source_event_ids=json.loads(row["source_event_ids_json"]),evidence_text=row["evidence_text"])

    def list_active_facts(self, fact_key: str|None=None, subject: str|None=None, entity_id: str|None=None, slot: str|None=None) -> list[FactVersion]:
        where=["status = ?"]; args:list[object]=[FactStatus.ACTIVE.value]
        if fact_key is not None: where.append("fact_key = ?"); args.append(fact_key)
        if subject is not None: where.append("LOWER(entity_mention) = ?"); args.append(subject.strip().lower())
        if entity_id is not None: where.append("entity_id = ?"); args.append(entity_id)
        if slot is not None: where.append("slot = ?"); args.append(normalize_key(slot))
        return [self._row_to_fact(r) for r in self.conn.execute(f"SELECT * FROM fact_versions WHERE {' AND '.join(where)} ORDER BY created_at",args).fetchall()]

    def list_facts_for_entity(self, entity_id: str, *, include_inactive: bool=False) -> list[FactVersion]:
        if include_inactive: rows=self.conn.execute("SELECT * FROM fact_versions WHERE entity_id=? ORDER BY created_at",(entity_id,)).fetchall()
        else: rows=self.conn.execute("SELECT * FROM fact_versions WHERE entity_id=? AND status=? ORDER BY created_at",(entity_id,FactStatus.ACTIVE.value)).fetchall()
        return [self._row_to_fact(r) for r in rows]

    def get_active_fact(self, fact_key: str) -> FactVersion | None:
        active=self.list_active_facts(fact_key=fact_key); return active[-1] if active else None

    def list_fact_history(self, fact_key: str) -> list[FactVersion]:
        return [self._row_to_fact(r) for r in self.conn.execute("SELECT * FROM fact_versions WHERE fact_key=? ORDER BY created_at",(fact_key,)).fetchall()]

    def list_fact_evidence(self, *, include_inactive: bool=False, subject: str|None=None, entity_id: str|None=None) -> list[FactEvidence]:
        where=[]; args:list[object]=[]
        if not include_inactive: where.append("f.status = ?"); args.append(FactStatus.ACTIVE.value)
        if subject is not None: where.append("LOWER(f.entity_mention) = ?"); args.append(subject.strip().lower())
        if entity_id is not None: where.append("f.entity_id = ?"); args.append(entity_id)
        where_sql=f"WHERE {' AND '.join(where)}" if where else ""
        rows=self.conn.execute(f"SELECT f.*,c.source_event_id,c.source_event_ids_json,c.evidence_text FROM fact_versions f JOIN claims c ON c.claim_id=f.source_claim_id {where_sql} ORDER BY f.created_at",args).fetchall()
        return [FactEvidence(fact=self._row_to_fact(r),source_event_id=r["source_event_id"],source_event_ids=json.loads(r["source_event_ids_json"]),evidence_text=r["evidence_text"]) for r in rows]

    def retire_fact(self, fact_id: str, status: FactStatus, retired_at: datetime|None=None, valid_to: datetime|None=None) -> None:
        if status is FactStatus.ACTIVE: raise ValueError("retire_fact requires a non-active status")
        now=retired_at or datetime.now(timezone.utc)
        with self.conn: self.conn.execute("UPDATE fact_versions SET status=?, retired_at=?, valid_to=CASE WHEN ? IS NULL THEN valid_to ELSE ? END WHERE fact_id=?",(status.value,_iso(now),_iso(valid_to),_iso(valid_to),fact_id))

    def lexical_event_search(self, query: str, limit: int=5):
        return self.conn.execute("SELECT event_id,text,bm25(events_fts) AS rank FROM events_fts WHERE events_fts MATCH ? ORDER BY rank LIMIT ?",(_fts_query(query),limit)).fetchall()

    def lexical_fact_search(self, query: str, *, limit: int=12, include_inactive: bool=False):
        status_clause="" if include_inactive else "AND f.status = 'active'"
        return self.conn.execute(f"SELECT f.fact_id,f.status,bm25(facts_fts) AS rank FROM facts_fts JOIN fact_versions f ON f.fact_id=facts_fts.fact_id WHERE facts_fts MATCH ? {status_clause} ORDER BY rank LIMIT ?",(_fts_query(query),limit)).fetchall()

    def upsert_embedding(self, fact_id: str, model: str, vector: list[float]) -> None:
        with self.conn: self.conn.execute("INSERT INTO fact_embeddings(fact_id,model,vector_json,created_at) VALUES (?,?,?,?) ON CONFLICT(fact_id,model) DO UPDATE SET vector_json=excluded.vector_json, created_at=excluded.created_at",(fact_id,model,json.dumps(vector),datetime.now(timezone.utc).isoformat()))

    def get_embeddings(self, model: str) -> dict[str,list[float]]:
        return {r["fact_id"]:json.loads(r["vector_json"]) for r in self.conn.execute("SELECT fact_id,vector_json FROM fact_embeddings WHERE model=?",(model,)).fetchall()}

    def record_transition(self, claim_id: str, decision: StateDecision, from_fact_id: str|None, to_fact_id: str|None) -> TransitionRecord:
        record=TransitionRecord(claim_id=claim_id,action=decision.action,reason=decision.reason,from_fact_id=from_fact_id,to_fact_id=to_fact_id)
        with self.conn: self.conn.execute("INSERT INTO transitions(transition_id,claim_id,action,reason,from_fact_id,to_fact_id,created_at) VALUES (?,?,?,?,?,?,?)",(record.transition_id,record.claim_id,record.action.value,record.reason,record.from_fact_id,record.to_fact_id,_iso(record.created_at)))
        return record

    def transitions_for_event(self, event_id: str) -> list[TransitionRecord]:
        rows=self.conn.execute("SELECT DISTINCT t.* FROM transitions t JOIN claims c ON c.claim_id=t.claim_id WHERE c.source_event_id=? OR c.source_event_ids_json LIKE ? ORDER BY t.created_at",(event_id,f'%"{event_id}"%')).fetchall()
        return [self._row_to_transition(r) for r in rows]

    def add_trace(self, trace: TurnTrace) -> None:
        with self.conn: self.conn.execute("INSERT OR REPLACE INTO turn_traces(trace_id,session_id,user_event_id,trace_json,created_at) VALUES (?,?,?,?,?)",(trace.trace_id,trace.session_id,trace.user_event_id,trace.model_dump_json(),_iso(trace.created_at)))

    def list_traces(self, session_id: str|None=None, limit: int=20) -> list[TurnTrace]:
        if session_id: rows=self.conn.execute("SELECT trace_json FROM turn_traces WHERE session_id=? ORDER BY created_at DESC LIMIT ?",(session_id,limit)).fetchall()
        else: rows=self.conn.execute("SELECT trace_json FROM turn_traces ORDER BY created_at DESC LIMIT ?",(limit,)).fetchall()
        return [TurnTrace.model_validate_json(r["trace_json"]) for r in reversed(rows)]

    @staticmethod
    def _row_to_event(r):
        return Event(event_id=r["event_id"],session_id=r["session_id"],turn_id=r["turn_id"],speaker=r["speaker"],text=r["text"],event_time=_dt(r["event_time"]),ingested_at=_dt(r["ingested_at"]),metadata=json.loads(r["metadata_json"]))

    @staticmethod
    def _row_to_entity(r):
        return Entity(entity_id=r["entity_id"],canonical_name=r["canonical_name"],entity_type=EntityType(r["entity_type"]),aliases=json.loads(r["aliases_json"]),relation_to_user=r["relation_to_user"],first_seen_event_id=r["first_seen_event_id"],last_seen_event_id=r["last_seen_event_id"],created_at=_dt(r["created_at"]),updated_at=_dt(r["updated_at"]))

    @staticmethod
    def _row_to_claim(r):
        return MemoryClaim(claim_id=r["claim_id"],source_event_id=r["source_event_id"],source_event_ids=json.loads(r["source_event_ids_json"]),entity_id=r["entity_id"],entity_mention=r["entity_mention"],entity_type=EntityType(r["entity_type"]),relation_to_user=r["relation_to_user"],memory_type=MemoryType(r["memory_type"]),slot=r["slot"],predicate_text=r["predicate_text"],value_text=r["value_text"],modality=Modality(r["modality"]),polarity=r["polarity"],confidence=r["confidence"],importance=r["importance"],valid_from=_dt(r["valid_from"]),valid_to=_dt(r["valid_to"]),event_time_text=r["event_time_text"],event_time_precision=EventTimePrecision(r["event_time_precision"]),expires_at=_dt(r["expires_at"]),evidence_text=r["evidence_text"],created_at=_dt(r["created_at"]))

    @staticmethod
    def _row_to_fact(r):
        return FactVersion(fact_id=r["fact_id"],fact_key=r["fact_key"],entity_id=r["entity_id"],entity_mention=r["entity_mention"],entity_type=EntityType(r["entity_type"]),relation_to_user=r["relation_to_user"],slot=r["slot"],predicate_text=r["predicate_text"],value_text=r["value_text"],memory_type=MemoryType(r["memory_type"]),modality=Modality(r["modality"]),polarity=r["polarity"],status=FactStatus(r["status"]),source_claim_id=r["source_claim_id"],supersedes_fact_id=r["supersedes_fact_id"],valid_from=_dt(r["valid_from"]),valid_to=_dt(r["valid_to"]),event_time_precision=EventTimePrecision(r["event_time_precision"]),confidence=r["confidence"],importance=r["importance"],created_at=_dt(r["created_at"]),retired_at=_dt(r["retired_at"]),expires_at=_dt(r["expires_at"]))

    @staticmethod
    def _row_to_transition(r):
        return TransitionRecord(transition_id=r["transition_id"],claim_id=r["claim_id"],action=ResolutionAction(r["action"]),reason=r["reason"],from_fact_id=r["from_fact_id"],to_fact_id=r["to_fact_id"],created_at=_dt(r["created_at"]))
