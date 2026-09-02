# ADR-001 — Event-sourced bitemporal state in SQLite

## Status
Accepted.

## Context
The assessment requires process-restart persistence, contradiction handling, historical recall, and inspectability. A vector store can retrieve evidence but cannot represent whether a proposition is currently valid. Destructive overwrite loses history.

## Decision
Keep raw conversation as an immutable event ledger and derive versioned fact state with two time axes:

- world-valid time: `valid_from` / `valid_to`,
- transaction time: `created_at` / `retired_at`.

Use SQLite because the prototype needs transactions, FTS5, deterministic tests, and one inspectable file—not distributed infrastructure.

## Consequences
- corrections can differ from temporal transitions,
- historical questions remain answerable,
- state can be rebuilt/reasoned about from evidence,
- graph-scale traversal is intentionally deferred.
