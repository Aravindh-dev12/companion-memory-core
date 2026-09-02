# Companion Memory Core

A research prototype for **persistent, temporally correct memory and persona consistency in AI companions**.

> **Research thesis:** long-term companion memory is a **state-management problem with retrieval**, not a vector-search problem with a prompt.

The project is intentionally small and inspectable. There is no UI, auth layer, billing, multimodal stack, or production-scale infrastructure. The system is designed to make the memory loop auditable: what was extracted, why state changed, what was retrieved, and whether the final response contradicted memory or persona.

## Why this architecture

A naive companion memory stack often becomes:

```text
conversation -> embeddings -> vector DB -> top-k memories -> prompt
```

That can retrieve a highly similar fact that is no longer true. This project instead separates **historical evidence**, **entities**, **qualified propositions**, and **current belief state**:

```text
user turn
   |
   +--> immutable event ledger
   |
   +--> windowed CandidateFact extraction (last 3-4 turns)
   |        -> entity resolution
   |        -> candidate target matching
   |        -> relation decision
   |        -> deterministic bitemporal write
   |
   +--> retrieval: entity channel + FTS5/BM25 + optional embeddings
   |        -> Reciprocal Rank Fusion (RRF)
   |        -> current/historical validity filter
   |
   +--> persona-aware response draft
            -> gated consistency firewall
            -> response
            -> persona-commitment extraction
```
### Invariant: history is immutable; beliefs are versioned

If a user says:

```text
"My girlfriend is Maya."
```

and later:

```text
"Maya and I broke up."
```

we do not delete the first disclosure and we do not keep both facts as equally current. The old `user::partner=Maya` fact becomes `superseded`, its validity window is closed, and a new active version is written. Historical questions can still recover Maya; current-state questions should not.

## What is implemented

- SQLite persistence across process restarts
- append-only event ledger with raw conversational evidence
- explicit memory-worthiness extraction policy
- Pydantic-validated **3-4 turn windowed** structured extraction
- entity registry with aliases and relation-to-user metadata
- entity resolution before state matching
- proposition modality (`asserted`, `hedged`, `hypothetical`, third-party, negated)
- candidate target matching **before** contradiction/update classification
- typed contradiction/update semantics (`ADD`, `CORRECT`, `TEMPORAL_TRANSITION`, `COEXIST`, `REFINE`, `WITHDRAW`, `SUPERSEDE`, `IGNORE`)
- deterministic state mutation after model classification
- bitemporal fact history: world-valid time separated from system recording/retirement time
- versioned fact history with multi-event provenance
- coexistence for additive facts such as multiple preferences
- entity-anchored retrieval for semantically distant cues
- FTS5/BM25 lexical retrieval
- optional OpenAI embeddings and cosine semantic retrieval over facts
- **Reciprocal Rank Fusion** instead of hand-tuned semantic/lexical weights
- temporal validity filtering, state-closure bridging, and type-aware retrieval decay
- retrieval traces with per-channel ranks/RRF reasons
- persona constitution (`persona.yaml`)
- assistant persona-commitment ledger
- open-loop follow-ups for unresolved goals/plans and companion promises on later sessions
- idempotent session consolidation into shared-history `companion_user_pair` episodes
- bounded hedged session inferences (`confidence < 0.7`) that never become asserted truth
- **gated** consistency firewall (only when a draft creates/uses consistency-sensitive claims)
- offline heuristic provider for deterministic plumbing tests
- real OpenAI Responses API provider for extraction, response generation, and verification
- 10-scenario adversarial YAML evaluation suite (50+ turn, cross-session, contradiction, entity recall, modality, distractors, abstention, persona pressure)
- baselines/ablations: no-memory, full-history, vector-bag, structured/lexical/semantic-only, no-temporal, no-firewall
- oracle mode: full versioned memory store + `gpt-5.6-sol` response model for isolating retrieval/state loss
- JSON/Markdown evaluation outputs and per-turn traces

## Repository map

```text
src/companion_memory/
  config.py          runtime settings
  models.py          typed memory/state/trace schemas
  store.py           SQLite event ledger, fact versions, FTS5, embeddings, traces
  extractor.py       windowed memory-worthiness extraction
  entities.py        entity resolution and alias binding
  matching.py        candidate update-target generation
  decision.py        relation classification after matching
  resolver.py        deterministic state transition semantics
  retrieval.py       entity/BM25/semantic retrieval + RRF + temporal filtering
  persona.py         persona constitution + commitment extraction
  open_loops.py      unresolved plan/goal/promise follow-up projection
  consolidation.py   session summaries, shared-history episodes, hedged inferences
  consistency.py     optional contradiction firewall
  providers.py       OpenAI Responses API adapter
  engine.py          end-to-end companion loop
  evaluation.py      scenario runner and ablations
  cli.py             terminal interface

eval/scenarios/      adversarial memory/persona tests
research/            literature and design implications
docs/                architecture, evaluation matrix, execution plan, demo script
tests/               deterministic component tests
SKILL.md              governing research/engineering contract
```

## Install

Python 3.11+.

### Offline/core tests only

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

### Real LLM-backed companion

```bash
pip install -e '.[all]'
cp .env.example .env
export OPENAI_API_KEY='...'
```

The OpenAI adapter uses the Responses API and Pydantic structured-output parsing. Provider-side response storage is disabled (`store=False`) because the application owns the durable memory state.

## Run the companion

```bash
companion-memory chat --session demo --provider openai --trace
```

Stop the process, run the same command again, and reuse the same session or ask about a remembered fact. The SQLite store survives the restart.

For a no-API smoke test:

```bash
companion-memory chat --session demo --provider heuristic --trace
```

### Continuity lifecycle

```bash
# inspect unresolved goals/plans/promises that can become natural follow-ups
companion-memory loops --session next-session --db data/companion.sqlite3

# consolidate a finished session into a shared-history episode
companion-memory consolidate --session demo --provider openai

# or automatically consolidate older sessions when a new session starts
companion-memory chat --session next-session --provider openai --auto-consolidate
```

Consolidation is idempotent per session. It never rewrites raw events; it derives a `companion_user_pair` episode and at most three hedged low-confidence inferences.

The heuristic provider is deliberately limited. It validates persistence/state/evaluation plumbing; it is **not** the submission-quality companion model.

## Inspect memory directly

```bash
companion-memory state
companion-memory inspect user::partner
companion-memory traces --session demo
```

This inspectability is intentional: a reviewer can see exactly when a fact became superseded/corrected and which fact version it replaced.

## Run evaluation

```bash
# offline plumbing run
PYTHONPATH=src python eval/run.py --provider heuristic

# real model
OPENAI_API_KEY=... PYTHONPATH=src python eval/run.py --provider openai

# preserve long-horizon turn distances (e.g. the persona test reaches turn 56)
OPENAI_API_KEY=... PYTHONPATH=src python eval/run.py --provider openai --preserve-turn-distance

# optional oracle row: full versioned memory store + flagship reasoning model
OPENAI_API_KEY=... PYTHONPATH=src python eval/run.py --provider openai --ablation oracle --preserve-turn-distance
```

Baselines and ablations:

```bash
# deliberately naive baselines
PYTHONPATH=src python eval/run.py --provider openai --ablation no_memory --preserve-turn-distance
PYTHONPATH=src python eval/run.py --provider openai --ablation full_history --preserve-turn-distance
PYTHONPATH=src python eval/run.py --provider openai --ablation vector_bag --preserve-turn-distance

# component ablations
PYTHONPATH=src python eval/run.py --provider openai --ablation structured_only --preserve-turn-distance
PYTHONPATH=src python eval/run.py --provider openai --ablation lexical_only --preserve-turn-distance
PYTHONPATH=src python eval/run.py --provider openai --ablation semantic_only --preserve-turn-distance
PYTHONPATH=src python eval/run.py --provider openai --ablation no_temporal --preserve-turn-distance
PYTHONPATH=src python eval/run.py --provider openai --ablation no_firewall --preserve-turn-distance
```

Each run writes:

- `results.<ablation>.json`
- `results.<ablation>.md`
- a SQLite database per scenario
- extraction / transition / retrieval / verification traces

The current offline heuristic run should be treated as an engineering smoke test, not as headline benchmark evidence. Final submission numbers should come from a pinned real model configuration and include failures.

### Current reproducible engineering evidence

```text
34 / 34 deterministic unit/integration tests pass
11 / 11 offline heuristic scenarios pass
100% offline smoke checks pass

no_temporal ablation:
  scenario pass rate = 63.6%
  check pass rate    = 79.5%
  failures           = coexistence retraction + correction + plan closure + relationship transition
```

These numbers are useful because the temporal ablation fails exactly where the architecture predicts. They are **not** a substitute for the final pinned-model experiment. See [`eval/results/offline_smoke.md`](eval/results/offline_smoke.md).

## Evaluation philosophy

The harness deliberately separates failure sources:

| Layer | Example metric |
|---|---|
| extraction | memory candidate precision/recall + modality accuracy |
| entity/matching | entity resolution + target candidate accuracy |
| state resolution | transition classification accuracy |
| persistence | cross-restart pass rate |
| retrieval | Recall@K / MRR / stale-fact retrieval rate |
| temporal reasoning | current-vs-historical state accuracy |
| abstention | false-memory rate |
| final response | factual consistency |
| persona | contradiction/drift rate |

Subjective LLM judging should be used mainly for tone/persona dimensions. Deterministic ground truth is preferred whenever a scenario has a known state transition.

## Research grounding

The design is informed by, but intentionally does not clone:

- **LongMemEval (ICLR 2025):** separates information extraction, multi-session reasoning, temporal reasoning, knowledge updates, and abstention.
- **LoCoMo (ACL 2024):** demonstrates difficulty of very long multi-session temporal/causal dialogue memory.
- **LoCoMo-Plus (ACL 2026):** tests cognitive memory under cue-trigger semantic disconnect and latent constraints.
- **A-MEM (NeurIPS 2025):** motivates memory evolution rather than static store/search.
- **Zep / Graphiti (2025):** motivates temporally aware knowledge representation and historical relationships.
- **MemGPT / Letta:** motivates actively managed memory rather than transcript-as-memory.

See [`research/literature.md`](research/literature.md) for the design implication drawn from each source.

## Tried / rejected design directions

### Vector-only memory
Rejected as the primary representation. Similarity cannot express whether a fact is still valid.

### Whole-history prompting
Useful as a baseline, but rejected as the architecture because it confounds persistence, retrieval, reasoning, and cost.

### Delete/overwrite on contradiction
Rejected because it destroys historical truth and makes questions such as “who was I dating before?” impossible to answer reliably.

### Graph database first
A temporal graph is compelling at larger scale, but SQLite is a better assessment choice: one file, transactions, FTS5, transparent schema, deterministic tests, and no infrastructure distraction.

### Static system-prompt-only persona
Rejected as insufficient. Assistant-generated first-person claims can create future consistency obligations, so persona commitments are tracked separately.

## Known limitations

- Extraction/update decisions still depend on model quality in OpenAI mode; deterministic mutation prevents state corruption semantics, but a wrong classifier can still choose the wrong transition.
- Hybrid channel fusion is parameter-free RRF, but post-fusion importance/decay remain prototype heuristics rather than learned calibration.
- Semantic embeddings are stored as JSON vectors and brute-force compared; this is intentional for prototype inspectability, not scale.
- Persona commitment conflicts are verified at response time, but commitment reconciliation itself is still conservative/deduplicative rather than a full temporal state machine.
- The evaluation suite is intentionally small; results should be reported with confidence intervals only after expanding scenario count.
- Entity retrieval improves semantically distant recall when the entity is explicit, but implicit/unresolved latent constraints remain difficult.
- Open loops, session consolidation/shared-history episodes, and oracle plumbing are implemented and deterministically tested. Their **model-quality value** still needs a pinned real-model run before they should be used as headline benchmark evidence.
- Learned query planning/ranking remains intentionally out of scope; the current planner is transparent and deterministic.

## Governing build contract

Read [`SKILL.md`](SKILL.md). It defines the invariants, hypotheses, evaluation standards, anti-patterns, and definition of done for this assessment.
