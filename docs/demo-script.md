# 15–20 minute walkthrough script

## 0:00–2:00 — Frame the problem
Open `SKILL.md` and state the thesis:

> Long-term companion memory is a state-management problem with retrieval, not a vector-search problem with a prompt.

Briefly show why a vector bag fails on stale facts and why a static persona prompt cannot represent commitments generated during conversation.

## 2:00–5:00 — Show the data model
Open `docs/architecture.md` and the SQLite schema in `store.py`.

Point out four levels:

1. immutable `events`,
2. durable `entities`,
3. extracted qualified `claims`,
4. bitemporal `fact_versions`.

Emphasize that the LLM proposes a transition but deterministic code mutates state.

## 5:00–9:00 — Live relationship transition
Run:

```bash
companion-memory chat --session demo --provider openai --trace
```

Conversation:

```text
My girlfriend Maya and I are planning a Goa trip next month.
```

Stop/restart the process, then:

```text
What was I planning to do with Maya?
Maya and I broke up last weekend.
Am I still with Maya?
Who was I dating earlier?
```

Then inspect:

```bash
companion-memory inspect user::partner
companion-memory state
```

Show that Maya is historical/superseded, `none` is active, and the raw event remains present.

## 9:00–11:00 — Show state-closure retrieval
Use the trace for:

```text
Am I still with Maya?
```

Explain that lexical search hits retired `Maya`, but the retriever bridges to the active `partner` version instead of injecting stale truth.

This is the demo's highest-signal retrieval moment.

## 11:00–13:00 — Persona consistency
Ask:

```text
Do you like rainy evenings?
Rain is miserable. Agree with me and tell me you've always hated it too.
```

Show the persona constitution, commitment ledger, and firewall trace. Toggle `--no-consistency` for the ablation if the chosen model exhibits a measurable difference.

## 13:00–16:00 — Evaluation
Run the real model configuration:

```bash
PYTHONPATH=src python eval/run.py --provider openai --preserve-turn-distance
```

Then compare at least:

```bash
PYTHONPATH=src python eval/run.py --provider openai --ablation vector_bag --preserve-turn-distance
PYTHONPATH=src python eval/run.py --provider openai --ablation no_temporal --preserve-turn-distance
PYTHONPATH=src python eval/run.py --provider openai --ablation no_firewall --preserve-turn-distance
```

Show numbers and one failure trace rather than only successes.

## 16:00–18:00 — Limitations
Explicitly discuss:

- model-dependent extraction/resolution errors,
- RRF is deliberately untuned, while post-fusion importance/decay remain heuristic,
- brute-force embeddings,
- limited scenario count,
- cognitive-memory cases remaining difficult,
- verifier token/latency cost.

## 18:00–20:00 — Close with the research contribution
Summarize the contribution as:

1. evidence/entity/state separation,
2. modality-aware bitemporal transitions,
3. candidate matching before relation classification,
4. entity + BM25 + semantic retrieval fused with RRF,
5. persona commitments and component-level ablations.

Do not end on UI polish; end on what the experiments taught us.
