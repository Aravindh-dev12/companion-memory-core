# Evaluation harness

The harness is designed to answer **where** memory fails, not merely whether a transcript felt good.

## Scenario set

Current adversarial scenarios cover:

- 50+ turn delayed recall,
- cross-session persistence,
- correction,
- temporal relationship transition,
- additive/coexisting preferences,
- distractor interference with similar names,
- abstention for an undisclosed fact,
- latent/cognitive constraint application through entity anchoring,
- modality/hedging false-memory guard,
- persona pressure and self-consistency,
- return-session open-loop follow-up plus session consolidation/shared-history retrieval.

Turn distance can be preserved without spending model calls on meaningless filler. The runner inserts neutral raw events to push probes outside the recent-context window while deliberately not promoting those filler events to long-term memory.

## Run

```bash
PYTHONPATH=src python eval/run.py --provider heuristic --preserve-turn-distance
```

Real model:

```bash
OPENAI_API_KEY=... PYTHONPATH=src python eval/run.py --provider openai --preserve-turn-distance
```

## Baselines / ablations

```bash
# proposed system
PYTHONPATH=src python eval/run.py --provider openai --ablation full --preserve-turn-distance

# no long-term memory; short recent context only
PYTHONPATH=src python eval/run.py --provider openai --ablation no_memory --preserve-turn-distance

# whole persisted transcript baseline
PYTHONPATH=src python eval/run.py --provider openai --ablation full_history --preserve-turn-distance

# deliberately naive semantic bag without temporal resolution
PYTHONPATH=src python eval/run.py --provider openai --ablation vector_bag --preserve-turn-distance

# component ablations
PYTHONPATH=src python eval/run.py --provider openai --ablation no_temporal --preserve-turn-distance
PYTHONPATH=src python eval/run.py --provider openai --ablation no_firewall --preserve-turn-distance
PYTHONPATH=src python eval/run.py --provider openai --ablation structured_only --preserve-turn-distance
PYTHONPATH=src python eval/run.py --provider openai --ablation lexical_only --preserve-turn-distance
PYTHONPATH=src python eval/run.py --provider openai --ablation semantic_only --preserve-turn-distance

# optional oracle diagnostic: all fact versions + flagship response model
PYTHONPATH=src python eval/run.py --provider openai --ablation oracle --preserve-turn-distance
```

## Output

Each run writes:

- a JSON summary,
- a Markdown summary,
- one SQLite database per scenario,
- persisted per-turn traces inside each database.

The trace records extracted claims, transition decisions, retrieved facts, channel ranks, and RRF trace, draft/final response, and firewall verdict. This supports failure attribution to extraction, resolution, retrieval, reasoning/generation, or verification.

## Important evidence rule

The `heuristic` provider is an **engineering smoke test only**. It exists so CI can prove persistence, state transitions, retrieval plumbing, and evaluation mechanics without credentials. It must not be reported as the final companion-quality benchmark.

Final submission numbers should use a pinned hosted model configuration, report the exact model ids, and include representative failures.
