# ADR-004 — Persona commitments are durable state

## Status
Accepted.

## Context
A static persona prompt does not prevent the model from creating contradictory first-person claims later. Once the companion says `I love rainy evenings`, that claim becomes a future consistency obligation.

## Decision
Keep a canonical `persona.yaml` constitution and extract durable assistant first-person commitments into the same versioned memory substrate under the `companion` entity. Run a gated consistency firewall only when a draft creates/uses consistency-sensitive claims.

## Consequences
- persona drift is inspectable and ablatable,
- the verifier does not double cost on every turn,
- stable opinions/promises can survive process restarts,
- commitment reconciliation remains conservative rather than a full belief-revision research project.
