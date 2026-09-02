# ADR-005 — Open loops and consolidation are derived continuity memory

## Status
Accepted after core-loop stability.

## Context
Fact recall alone can feel like a CRM. A companion should remember unfinished plans and shared history, but continuity features must not invent outcomes or rewrite source evidence.

## Decision
Project active goals, future plans, and companion promises into **open-loop** follow-up candidates. Surface them primarily on later sessions, when due, or when directly relevant. A follow-up does not complete a loop; normal extraction/resolution must observe an outcome. A plan can transition into an episodic event when it happens.

Session consolidation is idempotent and adds:

- one `companion_user_pair` episodic summary,
- at most three forced-hedged, confidence `< 0.7` inferences.

The event ledger is never rewritten.

## Consequences
- the companion can naturally ask how an interview went,
- completed plans stop nagging after plan→episode transition,
- shared-history recall survives restarts,
- inferred state is visibly less certain than disclosed truth.
