# Assessment execution plan

The assessment itself budgets roughly 18 focused hours and explicitly prioritizes the core loop. This plan protects that priority.

## P0 — Research contract and deterministic substrate
Status: complete.

Deliverables:
- architecture thesis
- project `SKILL.md`
- SQLite schema
- typed memory models
- deterministic state-transition semantics
- initial tests
- literature/design notes

Exit criterion: history can be persisted and state versions can change without historical deletion.

## P1 — End-to-end core loop
Status: complete.

Built:
- provider adapter
- structured memory extractor
- structured update/contradiction classifier
- memory-worthiness gate
- chat loop with process-restart persistence
- debug traces showing extraction, resolution, and retrieval

Exit criterion: a scripted multi-session conversation persists a fact, updates it later, and retrieves current vs historical truth correctly.

## P2 — Retrieval quality
Status: complete.

Built:
- FTS5 lexical candidates
- embedding adapter
- hybrid candidate fusion/reranking
- temporal filtering
- query-intent routing only if it improves measured results

Exit criterion: retrieval tests distinguish direct recall, exact-name recall, semantic recall, and stale-fact suppression.

## P3 — Persona consistency
Status: complete.

Built:
- persona constitution injection
- assistant first-person commitment extraction
- commitment store
- optional consistency firewall

Exit criterion: 50+ turn scripted pressure test and an ablation with firewall on/off.

## P4 — Evaluation harness
Status: complete for the compact scripted harness; real hosted-model results still require credentials.

Built only after P1 was solid:
- scenario runner
- hidden canonical world state
- deterministic factual scorers
- optional LLM judge for tone/persona dimensions
- baselines and ablations
- JSONL traces + summary table

Exit criterion: numeric results, failure categories, and at least one honest failure example.

## P5 — Submission hardening
Status: in progress until GitHub push/PR and a pinned hosted-model run are complete.

- README as a compact research paper
- architecture diagram
- "tried and abandoned" section
- known limitations
- one-command demo/eval
- 15–20 minute walkthrough script

## P6 — Continuity lifecycle (added after core stability)
Status: implemented and deterministically tested.

Built:
- return-session open-loop follow-ups for active goals/plans and companion promises,
- idempotent session consolidation,
- shared-history `companion_user_pair` episode,
- at most three hedged low-confidence inferences,
- oracle eval mode over the full versioned store using a stronger configurable response model,
- narrow entailment/persona judges for semantic-only eval checks.

## Scope guard
Do not spend assessment time on UI, auth, deployment, multi-user support, production-scale vector infrastructure, or visual polish before the above is complete.
