# Architecture: Memory Ledger + Living State v2.1

## Entity-Anchored Bitemporal Memory for Long-Horizon AI Companions

## Research question
Can a small, inspectable companion remain temporally correct and persona-consistent across long, multi-session conversations without treating either the transcript or a vector index as the memory itself?

## Thesis
**Long-term companion memory is a state-management problem with retrieval, not a retrieval problem with a prompt.**

Operationally:

> Memory is a database with an LLM on the write path and an LLM on the read path. The LLM interprets language; deterministic code owns memory state.

The design combines explicit memory operations, bitemporal state, entity anchoring, selective retrieval, and persona commitments in one SQLite prototype.

```text
USER TURN
   |
   +--> append immutable EVENT
   |
   +--> WINDOWED EXTRACTION (last 3-4 turns)
   |        |
   |        v
   |   CandidateFact[]
   |        |
   |        v
   |   ENTITY RESOLUTION
   |        |
   |        v
   |   CANDIDATE MATCHING (top-k, no forced threshold)
   |        |
   |        v
   |   RELATION CLASSIFIER
   |        |
   |        v
   |   DETERMINISTIC BITEMPORAL WRITE
   |
   +--> QUERY PLANNER
   |        |
   |        v
   |   entity channel + FTS5/BM25 + cosine
   |        |
   |        v
   |       RRF
   |        |
   |        v
   |   validity filter + priority rerank
   |        |
   |        v
   |   CONTEXT ASSEMBLY
   |        |
   |        v
   +--> GENERATE DRAFT
            |
            v
      gated CONSISTENCY FIREWALL
            |
            v
         RESPONSE
            |
            v
   PERSONA COMMITMENT EXTRACTION
```

### Why the write path runs before generation
A purely asynchronous write path looks attractive for latency, but it creates a correctness bug on state-changing turns. If the user says, "Maya and I broke up," retrieval must not still treat `partner=Maya` as current while generating that same turn's response. This reference implementation therefore applies the memory transition synchronously before retrieval/generation. Production latency engineering is intentionally out of scope.

---

## 1. Immutable Event Ledger
Every user and assistant turn is append-only evidence:

- `event_id`
- `session_id`
- `turn_id`
- `speaker`
- raw text
- event timestamp
- ingestion timestamp
- metadata

State changes never rewrite the source conversation. This is event sourcing: the current registry is a materialized view derived from immutable evidence.

**Invariant:** history is immutable; beliefs are versioned.

---

## 2. Entity Registry
Flat `subject/predicate/value` triples are too weak for real conversational memory. v2.1 gives people, places, organizations, projects, pets, and the user stable entity identities.

```text
entities
  entity_id
  canonical_name
  entity_type
  aliases[]
  relation_to_user
  first_seen_event_id
  last_seen_event_id
```

Examples:

```text
user
mom  aliases=[mum, mother] relation_to_user=mother
Maya relation_to_user=partner
Google entity_type=org
```

Entity resolution uses:

1. exact canonical/alias/relation match,
2. conservative optional embedding tie-breaker when lexical identity overlaps,
3. otherwise create a new entity.

It deliberately does **not** merge two people solely because an embedding is similar.

### Why entities matter for retrieval
A later cue may be semantically distant from the original fact:

```text
T4:  "My mum gets extremely anxious on long flights."
...
T52: "Would Bali be an easy surprise for Mom?"
```

The query explicitly mentions the `mom` entity, so the entity channel can retrieve active facts about her even if lexical/semantic similarity is weak.

---

## 3. Proposition / Fact Model
Facts are atomic propositions anchored to an entity.

```text
fact_id
entity_id
entity_mention
slot?                 # typed state slot where one fits
predicate_text        # free-text schema otherwise
value_text
memory_type
modality
polarity
confidence
importance
valid_from
valid_to
recorded_at           # represented by created_at in code
retired_at
status
supersedes_fact_id
source_event_ids
expires_at
```

Typed slots are reserved for high-value state where deterministic replacement is useful, for example:

- `partner`
- `relationship_status`
- `employer`
- `job_title`
- `location`
- `lives_with`

Everything else can remain a normalized free-text predicate.

---

## 4. Modality Prevents False Memory Formation
The memory system preserves how strongly a proposition was asserted:

- `asserted`
- `hedged`
- `hypothetical`
- `reported_by_third_party`
- `negated`

Examples:

```text
"I quit."                         -> asserted
"I might quit."                   -> hedged
"If the review goes badly, I'll quit." -> hypothetical
"My manager thinks I should quit."     -> reported_by_third_party
"I didn't quit."                  -> negated
```

A hedged/hypothetical/third-party proposition may be remembered because it can matter later, but it is not allowed to silently replace asserted current state. The response layer receives modality explicitly and must keep qualified facts qualified.

---

## 5. Bitemporal State
The system separates two clocks.

### World / valid time
When was the proposition true in the user's world?

- `valid_from`
- `valid_to`

### System / transaction time
When did the companion learn or retire the proposition?

- `recorded_at` (`created_at` in the implementation)
- `retired_at`

This makes correction and temporal update operationally different.

| New disclosure | Old fact | New fact |
|---|---|---|
| `CORRECT` | `corrected`, retired now; do **not** invent a world-valid end | inherits original validity origin |
| `TEMPORAL_TRANSITION` | `superseded`, `valid_to=event_time`, `retired_at=now` | `valid_from=event_time` |
| `COEXIST` | unchanged | added |
| `REFINE` | superseded without creating a new time epoch | inherits original `valid_from` |
| `WITHDRAW` | `withdrawn` | none |
| `SUPERSEDE` | generic replacement | new current version |
| `IGNORE` | unchanged | none |
| `ADD` | unchanged | added |

### Example

```text
T3:  "My girlfriend is Maya."
     -> user::partner = Maya [ACTIVE]

T35: "Maya and I broke up last weekend."
     -> Maya fact [SUPERSEDED, valid_to ~= last weekend, retired_at = T35 ingestion]
     -> user::partner = none [ACTIVE]
```

The system can answer both:

- "Am I still with Maya?" using active state.
- "Who was I dating earlier?" using superseded world-truth.

Corrected and withdrawn facts remain evidence but are not treated as historical world-truth.

---

## 6. Contradiction Handling Is Matching First
The hard problem is not only choosing `CORRECT` vs `TRANSITION`; it is identifying **which existing fact** the new disclosure changes.

```text
candidate proposition
    -> resolve entity
    -> generate <=5 active candidates
         exact slot first
         exact predicate next
         same memory type / lexical affinity
         optional semantic rank
    -> relation classifier
         {action, target_fact_id, event_time?, reason}
    -> deterministic state mutation
```

Candidate generation is top-k rather than threshold-based. The prototype deliberately avoids an unexplained cosine cutoff such as `0.75`.

If no plausible candidate exists, the result is `ADD`; unrelated facts about the same entity are not forced into the classifier.

---

## 7. Windowed Extraction
Facts can span turns:

```text
assistant: "Who is Maya?"
user:      "My girlfriend."
```

Extraction sees the last 3-4 turns rather than only the latest message. Every claim records the contributing `source_event_ids`, and repeated extraction from overlapping windows is deduplicated against already-recorded evidence/state.

Memory-worthiness remains explicit. The model estimates future conversational utility; the prototype's threshold is configuration, not a learned scientific constant.

---

## 8. Retrieval: Entity + BM25 + Semantic -> RRF
The proposed read path retrieves over **facts**, not raw conversation turns.

Channels:

1. **Entity channel** — active facts for explicitly mentioned non-user entities.
2. **Lexical channel** — SQLite FTS5/BM25 over entity/predicate/value/evidence.
3. **Semantic channel** — cosine similarity over fact embeddings when enabled.

The channels are fused with Reciprocal Rank Fusion:

```text
RRF(d) = sum(1 / (60 + rank_channel(d)))
```

No hand-tuned semantic-vs-lexical relevance weights are needed.

### Validity is a hard filter
Current queries default to active facts. Historical queries may include superseded facts. Corrected/withdrawn evidence is not treated as world history.

### State closure
A current query may mention an old entity/value:

```text
"Am I still with Maya?"
```

Lexical search naturally finds the retired `partner=Maya` fact. Rather than injecting stale state, the retriever uses that hit as a bridge to the active fact with the same state key (`partner=none`).

### Decay
Decay affects retrieval priority, not truth. Stable identity/preferences/values do not become false merely because they are old. Short-lived states and episodes can lose salience.

---

## 9. Persona Constitution + Commitment Ledger
`persona.yaml` defines stable canonical identity:

- voice
- values
- stable preferences
- boundaries
- backstory

Assistant-generated first-person claims can create new consistency obligations. Those are extracted as `persona_commitment` facts under the `companion` entity.

Example:

```text
"I like rainy evenings; they make everything feel quieter."
```

becomes a future obligation to preserve both the stance and, where available, its reason.

---

## 10. Gated Consistency Firewall
A verifier on every turn doubles model cost and can flatten voice. v2.1 only invokes it when the draft appears to create a durable first-person commitment or actually uses retrieved memory.

The verifier checks:

- contradiction with active user state,
- historical fact presented as current,
- unsupported invented user history,
- contradiction with persona constitution,
- contradiction with active persona commitments,
- severe generic-assistant flattening.

It can be disabled for ablation.

---

## 11. Provider State Is Not Memory State
All OpenAI Responses API calls use `store=False`. The application chooses every context item sent to the model; provider-side conversation storage is not an undeclared memory subsystem.

The default model roles are deliberately split:

- extraction/resolution/verification: lower-cost model,
- final companion generation: stronger balanced model,
- embeddings: separate embedding model.

Model IDs remain configurable through environment variables.

---

## 12. Why SQLite
SQLite is the assessment-appropriate choice:

- durable across process restarts,
- inspectable single file,
- transactional state changes,
- FTS5 built in,
- deterministic tests,
- no infrastructure distraction.

A graph database becomes more attractive when graph traversal itself is the dominant workload. That is not necessary to prove the assessment's core memory hypothesis.

---

## 13. Evaluation Strategy
Use hand-authored adversarial YAML scenarios with deterministic state checkpoints rather than an elaborate dialogue generator.

Current scenario families include:

1. long-delay recall,
2. relationship transition,
3. correction,
4. coexistence,
5. cross-session persistence,
6. distractor interference,
7. abstention,
8. semantic-distance/entity recall,
9. modality/hedging,
10. persona drift/pressure,
11. open-loop follow-up + session consolidation/shared-history retrieval.

State and retrieval assertions are preferred over subjective judging. LLM judging is reserved for dimensions such as persona tone or semantic entailment where deterministic checks are insufficient.

### Primary hypotheses
- **H1:** bitemporal state improves current-state accuracy after corrections/transitions.
- **H2:** entity-anchored retrieval improves semantically distant recall.
- **H3:** modality tagging reduces false-memory formation from hedged/hypothetical/attributed statements.
- **H4:** persona commitments + gated verification reduce self-contradiction under rephrasing/topic pressure.
- **H5:** oracle-vs-proposed gap can separate retrieval/state failures from generation failures when the oracle is run.

### Baselines / ablations
Keep the comparison small and interpretable:

- no persistent memory,
- full-history context,
- vector-bag/raw semantic baseline,
- proposed system,
- proposed minus temporal resolver,
- proposed minus firewall.

Additional lexical/semantic/structured-only modes exist for component diagnosis but are not required to headline the submission.

### Oracle row
The oracle mode is now implemented as a diagnostic, not a production retrieval policy: the evaluator passes the **full versioned fact store** (including status/modality) to a configurable stronger response model (`gpt-5.6-sol` by default). This isolates how much error remains after retrieval selection is removed. Oracle reads do not disable write-time temporal resolution.

---

## 14. Continuity Lifecycle: Open Loops + Consolidation
These began as stretch features and were added only after the core loop and ablation suite were stable.

### Open loops
Active `future_plan`, `goal`, and companion `promise_to_user` facts are projected into follow-up candidates. A loop is surfaced when it is due, relevant to the current query, or the user returns in a later session. It is **not** marked complete merely because it was surfaced; ordinary extraction/resolution owns the state change. This preserves the distinction between "remembering a plan" and "knowing how it turned out."

### Session consolidation
A session can be consolidated explicitly or older sessions can be consolidated on entry to a new session. Consolidation is idempotent and derives:

- one `companion_user_pair` episodic shared-history summary,
- at most three `user` inferences forced to `hedged` modality and confidence `< 0.7`.

The event ledger remains immutable. Consolidation adds derived memory; it never rewrites source dialogue.

### Remaining stretch
Learned query planning/ranking remains deliberately unimplemented because the assessment does not provide enough labeled data to justify training/tuning it.

---

## Failure Decomposition
Every trace preserves enough structure to label a failure as one of:

- extraction miss,
- wrong modality,
- entity resolution error,
- candidate matching miss,
- bad relation classification,
- deterministic transition bug,
- stale-state retrieval,
- lexical/semantic/entity retrieval miss,
- temporal reasoning error,
- response hallucination,
- persona contradiction,
- verifier false positive/negative.

This decomposition is what makes the prototype researchable rather than merely demoable.
