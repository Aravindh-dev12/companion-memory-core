# ADR-002 — Entity-anchored propositions with modality

## Status
Accepted.

## Context
Flat `subject/predicate/value` triples fail on conversational references such as `mom`, `my mother`, named people, and uncertain disclosures such as `I might quit`.

## Decision
Resolve durable entities before state matching. Facts preserve typed slots only for high-value mutable state and otherwise use normalized free-text predicates. Every proposition carries modality (`asserted`, `hedged`, `hypothetical`, `reported_by_third_party`, `negated`) and polarity.

## Consequences
- semantically distant cues can retrieve by entity,
- hedged/third-party statements cannot silently become asserted truth,
- entity mistakes become a separately measurable failure class,
- two people with the same name are not merged solely by embedding similarity.
