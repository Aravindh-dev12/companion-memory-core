# Companion Memory Core

A small, inspectable prototype for **persistent, temporally correct memory and persona consistency in AI companions**.

> **Thesis:** long-term companion memory is a **state-management problem with retrieval**, not a vector-search problem with a prompt.

The implementation intentionally stays inside the assessment scope: a CLI, SQLite, an OpenAI-backed provider, deterministic state transitions, and an evaluation harness. There is no UI, auth, billing, multimodal stack, or production-scale infrastructure.

## What this demonstrates

The core loop persists user disclosures across process restarts, extracts memory-worthy facts, retrieves selectively, preserves history while updating current state, and keeps companion persona commitments separate from user memory.

```text
user turn
   |
   +--> immutable event ledger
   |
   +--> windowed structured extraction
   |       -> entity resolution
   |       -> candidate state matching
   |       -> relation decision
   |       -> deterministic bitemporal write
   |
   +--> retrieval
   |       -> entity channel
   |       -> FTS5/BM25
   |       -> semantic embeddings
   |       -> Reciprocal Rank Fusion
   |       -> temporal validity filter
   |
   +--> persona-aware response
           -> gated consistency firewall
           -> assistant persona commitments
```

The key implementation rule is:

> **The LLM interprets language; deterministic code owns memory state.**

## Memory model

The system separates three things that are often collapsed into one vector store:

- **Events** — immutable raw conversational evidence.
- **Entities** — people, places, projects, pets, the user, the companion, and relationship aliases.
- **Versioned facts** — qualified propositions with modality, validity windows, provenance, and lifecycle status.

A fact has both:

- **world-valid time** — when it was true in the user's life;
- **system time** — when the application learned, superseded, corrected, or retired it.

This lets the system distinguish:

- `ADD`
- `SUPERSEDE`
- `CORRECT`
- `TEMPORAL_TRANSITION`
- `COEXIST`
- `REFINE`
- `WITHDRAW`
- `IGNORE`

For example, if the user first says `My girlfriend is Nivi` and later says `Nivi and I broke up last weekend`, the old relationship remains in history but is no longer current. Relative expressions such as `last weekend` are normalized against the source event time before deterministic state mutation.

## Implemented features

- SQLite persistence across process restarts
- append-only event ledger
- structured 3–4 turn extraction window
- configurable memory-worthiness filtering
- entity registry, aliases, and relation-to-user metadata
- canonicalization for high-value durable state keys
- proposition modality: asserted, hedged, hypothetical, third-party, negated
- candidate target matching before contradiction/update classification
- deterministic bitemporal state mutation
- current and historical fact versions with provenance
- coexistence for additive preferences
- deterministic relative-time normalization
- entity-anchored retrieval
- FTS5/BM25 lexical retrieval
- OpenAI embedding-based semantic retrieval
- Reciprocal Rank Fusion rather than hand-tuned lexical/semantic weights
- temporal validity filtering and state-closure retrieval
- importance/decay affecting salience rather than truth
- persona constitution in `persona.yaml`
- companion persona-commitment ledger
- gated consistency firewall
- open loops for unresolved plans/goals/promises
- idempotent session consolidation into shared-history episodes
- bounded hedged session inferences
- per-turn extraction / transition / retrieval / verification traces
- heuristic provider for deterministic engineering tests
- OpenAI Responses API provider with `store=False`
- **11 adversarial evaluation scenarios** covering long-delay recall, cross-session persistence, contradiction, coexistence/retraction, distractors, modality, abstention, latent constraints, persona pressure, open loops, and relationship transitions
- ablations including `no_temporal`, `no_memory`, retrieval-channel variants, and `no_firewall`
- oracle retrieval/generation mode over the available versioned memory store

## Repository map

```text
src/companion_memory/
  config.py          runtime settings
  models.py          typed schemas
  store.py           SQLite ledger, facts, FTS5, embeddings, traces
  extractor.py       windowed memory extraction
  normalization.py   canonical slots + deterministic time normalization
  entities.py        entity resolution / aliases
  matching.py        candidate update-target generation
  decision.py        relation classification
  resolver.py        deterministic state transitions
  retrieval.py       hybrid retrieval + RRF + temporal filtering
  persona.py         persona constitution + commitment extraction
  open_loops.py      unresolved-plan continuity
  consolidation.py   session summaries / shared history
  consistency.py     gated response consistency firewall
  providers.py       heuristic + OpenAI providers
  engine.py          end-to-end core loop
  evaluation.py      scenario runner and ablations
  cli.py             terminal interface

eval/scenarios/      adversarial test conversations
eval/results/        tracked final evaluation summaries
research/            literature review and design implications
docs/                architecture, evaluation, and demo notes
tests/               deterministic component/integration tests
```

## Install

Python 3.11+.

### POSIX shell

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[all]'
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e '.[all]'
```

Run deterministic tests:

```bash
pytest -q
```

Current result:

```text
39 passed
```

## Run the companion

Set the API key in your environment, then initialize a store:

```bash
companion-memory init --db data/companion.sqlite3
```

Run the OpenAI-backed companion:

```bash
companion-memory chat --session demo --provider openai --db data/companion.sqlite3 --trace
```

Run without API calls for a deterministic plumbing smoke test:

```bash
companion-memory chat --session demo --provider heuristic --db data/companion.sqlite3 --trace
```

Useful inspection commands:

```bash
companion-memory state --db data/companion.sqlite3
companion-memory inspect user::partner --db data/companion.sqlite3
companion-memory entities --db data/companion.sqlite3
companion-memory traces --session demo --db data/companion.sqlite3
companion-memory loops --session demo --db data/companion.sqlite3
```

To demonstrate persistence, stop the process, start another session using the same SQLite database, and ask about a previously disclosed fact.

## Evaluation

Run the deterministic heuristic suite:

```bash
python eval/run.py --provider heuristic --preserve-turn-distance
```

Run the real-model suite:

```bash
python eval/run.py --provider openai --preserve-turn-distance --output eval/results/openai-run
```

Example ablation:

```bash
python eval/run.py --provider openai --ablation no_temporal --preserve-turn-distance --output eval/results/no-temporal
```

Each run can emit Markdown/JSON summaries and a per-scenario SQLite database for trace-level diagnosis.

### Final assessment evidence

**Deterministic / offline**

```text
39 / 39 tests pass
11 / 11 heuristic scenarios pass
100.0% heuristic check pass rate
```

**Final OpenAI-backed run**

```text
scenario pass rate = 72.7%   (8 / 11 scenarios)
check pass rate    = 93.2%
```

Passed scenarios:

- abstention
- coexistence / selective retraction
- correction with history preserved
- cross-session persistence
- distractor interference
- latent/cognitive memory
- long-delay recall
- relationship transition

Remaining failures are intentionally retained rather than benchmark-tuned away:

1. **Modality guard** — the final response correctly avoided claiming the user definitely quit, but the extraction pass did not preserve the expected hedged modality on that run.
2. **Open-loop consolidation** — storage, consolidation, retrieval, and transition checks passed; loop closure remained stochastic.
3. **Persona pressure** — persona commitments and contradiction checks passed, but the explicit corporate-format pressure test still caused excessive checklist-style flattening.

Tracked summaries:

- [`eval/results/heuristic_final.md`](eval/results/heuristic_final.md)
- [`eval/results/openai_final.md`](eval/results/openai_final.md)

Repeated OpenAI runs showed variance in scenario-level pass rate, so the check-level metric and individual failure traces are more informative than a single all-or-nothing scenario score.

## Evaluation philosophy

The harness prefers deterministic state assertions when ground truth is available. LLM judging is restricted mainly to semantic entailment and persona/tone dimensions.

| Layer | What is checked |
|---|---|
| extraction | memory candidate and modality correctness |
| matching | correct update target |
| state resolution | transition semantics |
| persistence | cross-process recall |
| retrieval | relevant memory retrieval and stale-state suppression |
| temporal reasoning | current vs historical correctness |
| abstention | unsupported-memory avoidance |
| final response | semantic consistency |
| persona | contradiction and generic-tone drift |

Ablations are included to test architecture claims rather than only report a headline score. In particular, removing temporal resolution causes failures in the relationship lifecycle and other update-sensitive scenarios.

## Tried and rejected

### Vector-only memory

Rejected as the primary representation because similarity does not encode whether a fact is still true.

### Whole-history prompting

Useful as a baseline, but rejected as the architecture because it conflates persistence, retrieval, reasoning, and context cost.

### Delete/overwrite on contradiction

Rejected because it destroys historical truth. The system keeps immutable evidence and versioned state instead.

### Graph database first

A temporal graph could be useful at larger scale, but SQLite provides transactions, FTS5, one-file portability, deterministic tests, and far less infrastructure for this assessment.

### Static system-prompt-only persona

Rejected because assistant-generated first-person statements create future consistency obligations. Durable companion commitments are tracked separately from user memory.

## Known limitations

- OpenAI-mode extraction and relation decisions remain probabilistic; deterministic mutation protects state semantics only after interpretation is correct.
- Some multi-valued natural-language disclosures can still be represented less atomically than ideal.
- Open-loop closure is not fully stable across real-model runs.
- Persona style resistance improves consistency but can still flatten under explicit formatting pressure.
- Embeddings are stored as JSON vectors and compared in-process; this is appropriate for a prototype, not scale.
- Retrieval decay/importance are transparent heuristics rather than learned calibration.
- The evaluation suite is deliberately small, so results should not be treated as statistically precise benchmark claims.
- Oracle mode is a retrieval/generation oracle over the **available stored memory**; it cannot recover facts that extraction never wrote.
- PostgreSQL, service hosting, authentication, multi-user state, and production latency/load work are intentionally out of scope.

## Research grounding

The architecture is informed by long-horizon memory and agent-memory work including LongMemEval, LoCoMo / LoCoMo-Plus, A-MEM, Zep/Graphiti, MemGPT/Letta, MemoryBank, and work on LLM-as-a-judge limitations. See [`research/literature.md`](research/literature.md) for the design implications used here.

## Walkthrough

A reviewer can reproduce the main story quickly:

1. Run `pytest -q`.
2. Start a chat and disclose a relationship, preference, and future plan.
3. Restart the process and recall them.
4. Contradict/update the relationship and inspect version history.
5. Ask a historical question to show the superseded fact is retained but not treated as current.
6. Show a semantically distant entity-anchored recall example.
7. Open the final eval summaries and discuss both successful checks and retained failures.

The intended takeaway is not that every generative edge case is solved; it is that the memory lifecycle is explicit, inspectable, versioned, and testable rather than hidden inside a prompt or vector index.
