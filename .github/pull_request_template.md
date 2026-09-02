## Research question
What specific memory/persona failure is this change intended to reduce?

## Architecture change
Describe the state, retrieval, persona, or evaluation mechanism changed by this PR.

## Evidence
- [ ] deterministic tests
- [ ] offline evaluation smoke
- [ ] real-model evaluation when applicable
- [ ] ablation updated when applicable

Include before/after numbers or a concrete failure trace when possible.

## Invariants checked
- [ ] raw evidence remains immutable
- [ ] updates do not leave stale singular facts active
- [ ] corrected facts are not treated as historical truth
- [ ] retrieval is selective
- [ ] no provider-side conversation state substitutes for application memory

## Known limitations
List what this PR does not solve.
