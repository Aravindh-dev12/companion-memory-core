# Proposed GitHub issue

**Title:** Build assessment core loop: entity-anchored bitemporal memory + persona consistency

## Research question
Can a small companion architecture preserve current vs historical user state, avoid false memories from uncertain disclosures, retrieve semantically distant constraints, and remain persona-consistent over long multi-session conversations?

## Branch contract
Implement on `feat/assessment-core-loop`; keep `main` as the review/base branch.

## Primary architecture
**Memory Ledger + Living State v2.1**

- immutable event ledger,
- entity registry + aliases/relations,
- 3-4 turn windowed proposition extraction,
- modality (`asserted`, `hedged`, `hypothetical`, third-party, negated),
- candidate update-target matching before relation classification,
- deterministic bitemporal writes,
- entity + FTS5/BM25 + semantic retrieval fused with RRF,
- persona constitution + commitment ledger,
- gated consistency firewall.

## Acceptance criteria
- memory survives process restart/session boundaries,
- old evidence is never deleted when current state changes,
- correction differs operationally from temporal transition,
- hedged/hypothetical disclosures cannot silently overwrite asserted state,
- unrelated same-entity facts are not forced into contradiction classification,
- current queries suppress stale facts while historical queries can recover superseded truth,
- entity anchoring recovers the Mom/flight-anxiety -> Bali semantic-distance case,
- persona commitments survive 50+ turn pressure tests,
- deterministic tests run without API credentials,
- scripted eval emits numeric results and traces,
- controlled `no_temporal` ablation fails coexistence-retraction/correction/relationship/open-loop-closure scenarios as predicted,
- README documents rejected designs, stretch cut line, and known limitations.

## Current local evidence
- 34/34 deterministic unit/integration tests pass,
- 11/11 offline heuristic scenarios pass,
- full offline check pass rate: 100%,
- `no_temporal`: 63.6% scenario pass / 79.5% check pass after value-specific preference retraction was added; coexistence retraction + correction + relationship transition + open-loop closure fail.

The heuristic provider is plumbing evidence only; final model-quality results must come from a pinned hosted model configuration.

## Implemented after the core cut line
Open-loop follow-ups, idempotent session consolidation, shared-history episodes, oracle plumbing, and narrow semantic/persona judges were added only after the required core loop and temporal ablation were stable.

## Remaining non-goals
UI, auth, billing, multimodal features, production-scale infra, learned query ranking/planning, and background scheduling/latency optimization.
