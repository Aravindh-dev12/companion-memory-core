# Research notes and design implications

## LongMemEval — Wu et al., ICLR 2025
Paper: https://arxiv.org/abs/2410.10813

Key finding used here: long-term interactive memory is not one metric. The benchmark explicitly separates information extraction, multi-session reasoning, temporal reasoning, knowledge updates, and abstention. It also reports large degradation in sustained interaction and studies indexing/retrieval/reading design choices.

Design implication: our evaluation must isolate these abilities instead of reporting a single subjective "memory score".

## LoCoMo — Maharana et al., ACL 2024
Paper: https://aclanthology.org/2024.acl-long.747/

Key finding used here: very long multi-session conversations stress temporal and causal reasoning; long context and RAG help but remain below human performance.

Design implication: create delayed probes, distractors, temporal questions, and multi-session restarts rather than short local recall tests.

## LoCoMo-Plus — Li et al., ACL 2026
Paper: https://aclanthology.org/2026.acl-long.1150/

Key finding used here: realistic memory sometimes requires applying latent user constraints when the later cue has weak lexical/semantic overlap with the original disclosure.

Design implication: include "cognitive memory" cases where relevance is indirect, not merely paraphrased factual recall.

## MemGPT — Packer et al., 2023
Paper: https://arxiv.org/abs/2310.08560

Key contribution: hierarchical/virtual context management for context beyond the model's immediate window.

Design implication: memory should be an actively managed resource rather than a transcript synonym. We do not copy its OS abstraction because this assessment benefits more from transparent state transitions.

## Mem0 — Chhikara et al., 2025
Paper: https://arxiv.org/abs/2504.19413

Key contribution: dynamic extraction/consolidation/retrieval of salient memories, plus a graph-enhanced variant; evaluates against LoCoMo and emphasizes efficiency versus full-context approaches.

Design implication: extraction and consolidation are first-class operations. We deliberately keep our transition semantics inspectable instead of hiding them behind a memory service.

## Zep / Graphiti — Rasmussen et al., 2025
Paper: https://arxiv.org/abs/2501.13956

Key contribution: temporally aware knowledge graph that retains historical relationships and supports changing facts over time.

Design implication: model validity and provenance explicitly. For this assessment we use versioned SQLite facts rather than a graph database because the data volume is tiny and inspectability matters more than graph-scale infrastructure.

## Dynamic Persona Coherence — Qi et al., ACL 2026
Paper: https://aclanthology.org/2026.acl-long.1336/

Key contribution: separates stable identity from adaptive persona state and uses a closed-loop critic/corrector approach to mitigate drift.

Design implication: avoid treating personality as one monolithic static prompt; separate canonical identity from adaptive state and evaluate drift over long conversations.

## PersonaForge — Tong & Zou, Findings ACL 2026
Paper: https://aclanthology.org/2026.findings-acl.386/

Key contribution: explicit falsifiable claims and long-dialogue persona evaluation; reports reduced drift with a structured personality architecture and selective dual-process mechanism.

Design implication: make our persona-consistency claims falsifiable and ablate the consistency firewall rather than claiming that persona prompting alone is sufficient.

## What we are *not* doing
This repo is not a clone of Mem0, Zep, Letta/MemGPT, or a LangChain memory wrapper. Their results motivate specific design questions. The assessment implementation stays intentionally small so the reviewer can inspect each memory transition and trace failures to a component.

## A-MEM — Xu et al., NeurIPS 2025
Paper: https://arxiv.org/abs/2502.12110

Key contribution: memory is dynamically organized and evolves as new evidence arrives; new memories can update contextual representations and links among prior memories.

Design implication: treat memory evolution as a first-class operation rather than only append/search. Our divergence is intentional: raw evidence stays immutable, while derived state can evolve. This gives us a clean audit trail and makes update correctness easier to evaluate.

## Memory-mechanism surveys — 2025/2026
ACM survey: https://doi.org/10.1145/3748302
Graph-memory survey: https://arxiv.org/abs/2602.05665

Useful synthesis: modern agent-memory work can be viewed across source/form/operation dimensions or as a lifecycle spanning extraction, storage, retrieval, and evolution.

Design implication: our evaluation and README should cover the entire memory lifecycle. Retrieval quality alone is not evidence that the memory system is correct.

## OpenAI Responses API + current model split — 2026 implementation choice
Docs: https://developers.openai.com/api/docs/models
SDK: https://github.com/openai/openai-python

Current implementation uses the Responses API and SDK Pydantic parsing for structured outputs. The default configuration uses `gpt-5.6-luna` for high-volume extraction/resolution/verification, `gpt-5.6-terra` for conversational response generation, and `text-embedding-3-small` for semantic retrieval. All model ids remain environment-configurable.

Design implication: the provider is an interchangeable inference component, not the memory store. Calls use `store=False`, and the application supplies only explicitly selected context. This prevents provider-side conversation persistence from becoming an undeclared substitute for the memory architecture.

## Sleep-time Compute — Lin et al., 2025
Paper: https://arxiv.org/abs/2504.13171

Key contribution: useful computation can be moved outside the immediate query path and amortized across later queries.

Design implication: session consolidation is treated as bounded offline/transition-time work rather than forcing every operation into the response-critical path. We do **not** claim the paper specifically prescribes conversational summaries; it motivates the broader separation of online response work from offline context processing.

## MemoryBank — Zhong et al., AAAI 2024
Paper: https://ojs.aaai.org/index.php/AAAI/article/view/29946

Key contribution: long-term companion memory with reinforcement/forgetting inspired by the Ebbinghaus curve.

Design implication: forgetting is useful for **access priority**, but this prototype deliberately does not let age decay semantic truth. Stable identity/preferences remain true until contradicted; only retrieval salience decays for short-lived memory types.

## LLM-as-a-Judge position bias — Shi et al., IJCNLP-AACL 2025
Paper: https://aclanthology.org/2025.ijcnlp-long.18/

Key finding: judge behavior exhibits systematic position bias rather than purely random noise.

Design implication: deterministic SQL/state assertions are primary. LLM judging is restricted to narrow entailment/persona checks, and its limitations are disclosed rather than treating a single holistic score as ground truth.
