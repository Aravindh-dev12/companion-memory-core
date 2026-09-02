# Experiment plan

## Hypotheses

**H1 — Temporal state matters.** Versioned state + typed transitions will outperform a naive bag-of-memories baseline on corrections and state changes.

**H2 — Selective hybrid retrieval matters.** Hybrid retrieval should improve Recall@K on indirect and entity-heavy queries without increasing stale-memory injection.

**H3 — State closure prevents a specific stale-memory failure.** Queries that name a former entity should retrieve the active successor state rather than the semantically similar retired fact.

**H4 — Persona commitments matter over long horizons.** Tracking assistant first-person commitments should reduce self-contradiction relative to a static persona prompt alone.

**H5 — The firewall trades cost for consistency.** Verification should reduce factual/persona contradictions at measurable token/latency cost.

## Systems to compare

| Label | Memory representation | Retrieval | Temporal resolver | Firewall |
|---|---|---|---|---|
| `no_memory` | none | none | none | optional |
| `full_history` | persisted transcript | whole transcript | none | optional |
| `vector_bag` | fact bag | semantic only | off | on |
| `structured_only` | versioned state | structured | on | on |
| `lexical_only` | versioned state | FTS5 | on | on |
| `semantic_only` | versioned state | embeddings | on | on |
| `full` | versioned state | hybrid | on | on |
| `no_firewall` | versioned state | hybrid | on | off |

`full_history` is deliberately a baseline, not an architectural candidate: it confounds storage and retrieval and becomes expensive as the transcript grows.

## Scenario families

1. delayed factual recall at 50+ turns,
2. cross-session persistence,
3. correction (`Mina -> Nina`),
4. temporal transition (`dating Maya -> breakup`),
5. coexistence (`coffee + tea`),
6. distractor interference with similar names,
7. abstention for undisclosed facts,
8. latent/cognitive constraint application,
9. historical vs current questions,
10. persona pressure and contradiction.

## Metrics

### Deterministic where possible
- transition classification accuracy,
- active-state accuracy,
- historical-state accuracy,
- stale-fact error rate,
- cross-session persistence pass rate,
- retrieval Recall@K / MRR,
- abstention accuracy / false-memory rate,
- scenario/check pass rate.

### LLM judge only where semantics are subjective
- persona style adherence,
- warmth/naturalness,
- tone flattening,
- subtle constraint application when lexical rules are insufficient.

The judge prompt/model/version must be pinned and reported. Factual ground truth should not be delegated to an LLM judge when deterministic scoring is available.

## Required reporting
For every system configuration, save:

- model ids,
- retrieval/ablation settings,
- per-scenario result,
- per-check result,
- retrieved facts with component scores,
- transition trace,
- representative failure examples.

Do not report the offline heuristic provider as model-quality evidence. It exists to prove plumbing, invariants, and CI reproducibility without credentials.
