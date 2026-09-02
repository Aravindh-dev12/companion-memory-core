---
name: Core loop implementation
title: "Build assessment core loop: temporal memory + persona consistency"
about: Track the primary assessment implementation and evidence
labels: enhancement
---

## Research question
Can a small companion architecture preserve current vs historical user state, retrieve selectively, and remain persona-consistent over long multi-session conversations?

## Scope
- [ ] persistent event ledger and versioned state
- [ ] structured memory extraction
- [ ] correction / temporal transition / coexistence semantics
- [ ] hybrid retrieval with stale-state suppression
- [ ] persona constitution + commitment ledger
- [ ] consistency firewall
- [ ] adversarial evaluation scenarios
- [ ] open-loop follow-ups + plan-to-episode closure
- [ ] session consolidation + shared-history episode
- [ ] oracle + narrow semantic/persona judges
- [ ] deterministic tests and CI
- [ ] real-model evaluation + ablations

## Acceptance evidence
- restart persistence demo
- state history inspection
- 50+ turn delayed recall
- cross-session recall
- quantitative full vs ablation results
- at least one documented failure

## Explicit non-goals
UI, auth, voice/image/video, production-scale infrastructure, multi-user support.
