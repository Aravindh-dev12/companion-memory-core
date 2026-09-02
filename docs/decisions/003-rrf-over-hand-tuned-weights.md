# ADR-003 — Reciprocal Rank Fusion instead of hand-tuned retrieval weights

## Status
Accepted.

## Context
The assessment offers no labeled relevance set large enough to justify tuning semantic-vs-lexical coefficients. Arbitrary weights would look precise without evidence.

## Decision
Retrieve through entity, FTS5/BM25, and semantic channels; fuse channel ranks with Reciprocal Rank Fusion (`k=60`). Apply temporal validity as a hard filter and importance/decay only as post-fusion priority.

## Consequences
- no unexplained channel coefficients,
- each channel can be ablated independently,
- stale facts cannot become current merely by scoring highly,
- post-fusion salience remains a prototype heuristic and is documented as such.
