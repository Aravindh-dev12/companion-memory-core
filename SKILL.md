# SKILL.md — Companion Memory Core v2.1

## Mission
Build the smallest inspectable system that convincingly demonstrates persistent long-horizon companion memory and persona consistency.

The core thesis is:

> **Memory is temporal state management with selective retrieval. The LLM interprets language; deterministic code owns state.**

This file is the governing implementation contract.

## Assessment priorities
In order:

1. persistence across process restarts,
2. memory-worthy extraction,
3. contradiction/update handling,
4. selective relevant retrieval,
5. 50+ turn persona consistency,
6. evaluation only after the core loop is sound.

Never sacrifice the core loop for UI, infrastructure, broad framework work, or an oversized benchmark harness.

## Canonical architecture

```text
immutable events
  -> windowed CandidateFact extraction
  -> entity resolution
  -> candidate target matching
  -> relation classification
  -> deterministic bitemporal transition
  -> entity + BM25 + semantic retrieval
  -> RRF
  -> temporal validity filter
  -> persona-aware generation
  -> gated consistency firewall
  -> response
  -> persona commitment extraction
```

### Critical ordering invariant
User-memory updates happen **before** retrieval/generation on the same turn. A newly disclosed breakup/job change/correction must affect that response immediately.

## 1. Event Ledger
Raw dialogue is append-only evidence. Never rewrite/delete source events because current belief changes.

## 2. Entity Registry
Resolve durable conversational entities (`user`, people, places, orgs, projects, pets). Preserve canonical names, aliases, and `relation_to_user`.

Resolution priority:
1. exact canonical/alias/relation match,
2. conservative optional semantic tie-breaker,
3. create new entity.

Do not merge people using embedding similarity alone.

## 3. Proposition Store
A memory proposition contains:
- entity,
- typed `slot` when one fits,
- free-text predicate otherwise,
- value,
- memory type,
- modality,
- polarity,
- confidence/importance,
- valid time,
- transaction time,
- provenance.

Typed slots should be limited to high-value mutable state (`partner`, `relationship_status`, `employer`, `job_title`, `location`, `lives_with`).

## 4. Modality
Preserve:
- `asserted`
- `hedged`
- `hypothetical`
- `reported_by_third_party`
- `negated`

A qualified proposition may be stored for future usefulness but must not silently replace asserted state or be rendered as certain truth.

## 5. Bitemporal semantics
Separate:
- **world time** — `valid_from`, `valid_to`,
- **system time** — `recorded_at`/`created_at`, `retired_at`.

`CORRECT` is not `TEMPORAL_TRANSITION`.

Supported actions:
- `ADD`
- `CORRECT`
- `TEMPORAL_TRANSITION`
- `COEXIST`
- `REFINE`
- `WITHDRAW`
- `SUPERSEDE`
- `IGNORE`

The model may choose a relation; application code performs the mutation.

## 6. Candidate matching before classification
Never ask the model to classify against the entire memory store.

For each candidate:
1. resolve entity,
2. select <=5 plausible active targets,
3. prioritize same typed slot,
4. then same predicate/type/semantic affinity,
5. if no plausible target, `ADD`,
6. otherwise classify relation + explicit `target_fact_id`.

Do not force an unrelated same-entity fact into the classifier. Do not depend on an unexplained cosine threshold.

## 7. Windowed extraction
Extract over the last 3-4 turns so cross-turn disclosures can be resolved. Track all contributing `source_event_ids` and deduplicate overlapping windows.

Memory-worthiness is based on future utility, persistence, specificity, explicitness, and state-change value. Thresholds are prototype configuration, not learned truth.

## 8. Retrieval
Proposed retrieval channels:
- entity channel,
- SQLite FTS5/BM25,
- semantic fact embeddings.

Fuse channel ranks with **Reciprocal Rank Fusion** (`k=60`). Do not use a hand-tuned semantic-vs-lexical weight formula without data to tune it.

Hard rules:
- active facts only by default,
- superseded facts only for historical intent,
- corrected/withdrawn facts are evidence, not historical world truth,
- stale lexical hits can bridge to the active version of the same state key,
- do not dump the full user entity into context.

Decay changes retrieval priority, never truth.

## 9. Persona
`persona.yaml` is the constitution: voice, values, backstory, preferences, boundaries.

Durable first-person assistant claims become `persona_commitment` memories. Store stable stance/reason when available.

## 10. Gated consistency firewall
Run verification only when a draft appears to:
- create a durable first-person commitment, or
- actually use retrieved user memory.

Check current/historical state, unsupported user facts, persona constitution, and active commitments. Keep the firewall switchable for ablation.

## 11. Unknown is valid
Absence of memory must remain unknown. Do not infer names, histories, outcomes, or certainty that the user never disclosed.

## 12. Provider-state isolation
The application owns durable memory. OpenAI Responses calls use `store=False`; no hidden provider conversation state may substitute for the memory architecture.

## 13. Evaluation
Prefer scripted adversarial scenarios with SQL-checkable state over generated benchmark complexity.

Required families:
1. long-delay recall,
2. temporal relationship/state transition,
3. correction,
4. coexistence,
5. cross-session restart,
6. distractors/entity ambiguity,
7. abstention,
8. semantic-distance/entity recall,
9. modality guard,
10. persona pressure/contradiction.

Measure when feasible:
- extraction precision/recall,
- modality accuracy,
- candidate-target accuracy,
- relation accuracy,
- state-registry accuracy,
- retrieval Recall@K,
- stale-fact rate,
- temporal QA,
- false-memory rate,
- persona contradiction/drift.

Use LLM judging only for semantic/tone dimensions that cannot be deterministic; disclose judge limitations.

## 14. Headline baselines
Keep the main comparison interpretable:
- no persistent memory,
- full-history context,
- naive vector bag,
- proposed system,
- minus temporal resolver,
- minus consistency firewall.

Lexical/semantic/structured-only modes may remain diagnostic ablations.

## 15. Continuity lifecycle and remaining cut line
The core became reliable first; the following are now implemented and tested:
- open-loop follow-ups for goals/plans/companion promises,
- idempotent session consolidation,
- `companion_user_pair` shared-history episodes,
- hedged low-confidence consolidation inferences,
- oracle evaluation mode over the full versioned store.

Remaining cut line:
- learned query planner/ranker,
- production background scheduling/latency optimization,
- graph infrastructure or vector-database scaling.

Do not let these remaining stretch directions destabilize the core loop.

## Engineering principles
- transparent Python,
- SQLite over unnecessary infrastructure,
- Pydantic structured model outputs,
- immutable provenance,
- deterministic state transitions,
- inspectable traces,
- unit tests before subjective evaluation,
- no large memory framework that hides the assessment logic.

## Anti-patterns
- transcript = memory,
- vector DB = truth store,
- append contradictory facts forever,
- delete old evidence,
- store every sentence,
- full-memory prompt dump,
- embeddings-only retrieval,
- static persona prompt as the only persona mechanism,
- LLM-judge-only evaluation,
- UI/deployment work before the core loop.

## Research anchors
Use these as baselines, not novelty claims:
- LongMemEval — extraction, multi-session reasoning, temporal reasoning, updates, abstention.
- LoCoMo / LoCoMo-Plus — long dialogue and semantically distant cognitive memory.
- Mem0 — explicit memory update loop.
- Zep / Graphiti — temporal/bitemporal knowledge representation.
- A-MEM — evolving memory organization.
- MemGPT / Letta — actively managed memory and offline memory work.
- persona-drift literature — long-horizon identity consistency.

The differentiation of this project is the **combination**: entity-anchored bitemporal user state + modality-aware memory formation + RRF retrieval + explicit persona commitment state in one inspectable prototype.

## Definition of done
A reviewer can:
1. install and run from README,
2. restart the process and retain memory,
3. inspect entities/facts/history in SQLite,
4. see correction vs temporal transition differ,
5. verify hedged memory stays qualified,
6. inspect RRF/entity retrieval traces,
7. run deterministic tests and scripted evals,
8. see at least one controlled ablation fail where predicted,
9. understand known weaknesses and cut features,
10. follow a 15-20 minute demo without UI polish,
11. see a return-session open-loop follow-up and inspect a shared-history consolidation episode.
