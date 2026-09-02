# Offline engineering smoke results

These results use the deterministic `heuristic` provider and hashing embeddings. They validate architecture plumbing and evaluation mechanics; they are **not** model-quality evidence.

Run command:

```bash
PYTHONPATH=src python eval/run.py --provider heuristic --preserve-turn-distance
```

## Full pipeline

- Unit/integration tests: **34 / 34 passed**
- Evaluation scenarios: **11 / 11 passed**
- Evaluation checks: **100% passed**

Scenario families include delayed recall, cross-session persistence, correction, temporal transition, coexistence, distractor interference, abstention, entity-anchored latent constraint use, modality/hedging, persona pressure, and open-loop + consolidation/shared-history continuity.

## Temporal-resolution ablation

```bash
PYTHONPATH=src python eval/run.py --provider heuristic --ablation no_temporal --preserve-turn-distance
```

- Scenario pass rate: **63.6%**
- Check pass rate: **79.5%**
- Expected failures: value-specific preference retraction, correction, relationship temporal-transition, and plan-to-episode open-loop closure scenarios.

This is useful as a deterministic sanity check that the evaluation suite actually detects removal of the state-resolution mechanism.

## Other offline baselines

`vector_bag`, `no_memory`, `full_history`, and `oracle` can also be run offline, but their absolute numbers are not meaningful because the heuristic response generator is intentionally not a general language model. These configurations should be compared using the real provider before submission.
