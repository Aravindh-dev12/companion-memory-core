# ADR-006 — State-first evaluation with narrow semantic judges

## Status
Accepted.

## Context
`It felt consistent` is not evidence, but a general LLM judge introduces its own bias. Many memory failures have deterministic ground truth.

## Decision
Use scripted adversarial conversations and inspect SQL/state transitions first. Measure deterministic properties such as active state, prior fact status, modality, retrieval, restart persistence, and stale-memory suppression. Use an LLM only for narrow entailment or persona-style checks where string/state assertions are insufficient.

## Consequences
- failures can be attributed to extraction, matching, resolution, retrieval, generation, or verification,
- judge bias affects fewer metrics,
- the oracle row can isolate retrieval/state loss from generation loss,
- the small scenario suite remains a prototype benchmark, not a claim of statistical generality.
