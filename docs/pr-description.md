# Proposed pull request

**Title:** Implement research-grade companion memory core loop

## Summary
This PR implements the primary assessment architecture: **Memory Ledger + Living State v2.1**.

Instead of treating memory as a vector bag, the system separates immutable conversational evidence, durable entities, qualified propositions, and bitemporal current state. It matches update targets before relation classification, fuses entity/BM25/semantic retrieval with RRF, tracks assistant persona commitments, and records component-level traces for evaluation.

## Key changes
- append-only SQLite event ledger
- Pydantic-validated 3-4-turn windowed extraction schema
- entity registry + alias/relation resolution
- modality-aware propositions (asserted/hedged/hypothetical/third-party/negated)
- candidate target matching before relation classification
- typed transitions: ADD / CORRECT / TEMPORAL_TRANSITION / SUPERSEDE / COEXIST / REFINE / WITHDRAW / IGNORE
- deterministic bitemporal state mutation with multi-event provenance
- entity + FTS5/BM25 + optional embeddings fused with Reciprocal Rank Fusion
- state-closure retrieval for retired entity references
- type-aware retrieval decay
- persona constitution + commitment extraction
- gated consistency firewall
- persistent CLI and inspection commands
- 11 adversarial evaluation scenarios including 50+ turn, cross-session, entity-distance, modality, and continuity-lifecycle cases
- open-loop follow-ups for unresolved goals/plans and companion promises
- idempotent session consolidation + shared-history `companion_user_pair` episodes
- oracle evaluation mode over the full versioned store with a stronger configurable response model
- narrow entailment and persona-rubric judging only for semantic/style checks
- baseline/ablation switches
- GitHub Actions CI with credential-free tests and smoke eval

## Current evidence
- 33 deterministic unit/integration tests passing
- 11/11 offline heuristic smoke scenarios passing
- disabling temporal resolution breaks correction, relationship-transition, and plan-to-episode open-loop closure scenarios as expected

The heuristic provider is **not** presented as final model-quality evidence. Real submission numbers must be generated with a pinned OpenAI model configuration and saved with failures.

## Research rationale
The architecture follows the assessment requirement that old facts be updated/retired rather than merely appended. The key hypothesis is that temporal state validity is orthogonal to semantic similarity: the most semantically similar memory can be the most dangerous one if it is stale.

## Known limitations
- extraction and transition classification remain model-dependent,
- RRF removes channel-weight tuning, but post-fusion priority heuristics are not learned,
- semantic vectors are brute-force compared,
- scenario suite is intentionally compact,
- persona commitment reconciliation is conservative,
- implicit entity resolution and latent constraints without an explicit entity cue remain difficult,
- hosted-model benchmark numbers are not generated in this environment because no API credential is available; offline heuristic numbers are plumbing evidence only,
- learned query planning/ranking remains intentionally out of scope.
