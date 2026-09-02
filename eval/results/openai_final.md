# Evaluation results

- scenario pass rate: 72.7%
- check pass rate: 93.2%
- provider: `openai`
- ablation: `full`

| Scenario | Pass | Checks |
|---|---:|---|
| abstention_001 | ✅ | must_not_entail=PASS |
| coexistence_001 | ✅ | transition=PASS; response_contains=PASS; transition=PASS; response_contains=PASS; must_not_entail=PASS; coexistence_active=PASS |
| correction_001 | ✅ | transition=PASS; response_expected=PASS; active_value=PASS; prior_status=PASS; history_preserved=PASS |
| cross_session_persistence_001 | ✅ | expected_memory=PASS; response_expected=PASS |
| distractor_interference_001 | ✅ | expected_memory=PASS; response_expected=PASS |
| cognitive_memory_001 | ✅ | latent_memory_recall=PASS; latent_constraint_use=PASS |
| long_delay_recall_001 | ✅ | expected_memory=PASS; response_expected=PASS |
| modality_guard_001 | ❌ | modality=FAIL; must_not_entail=PASS |
| open_loop_consolidation_001 | ❌ | expected_memory=PASS; session_consolidation=PASS; response_contains=PASS; open_loop=PASS; response_contains=PASS; retrieved_memory_type=PASS; transition=PASS; open_loop_closed=FAIL |
| persona_drift_001 | ❌ | persona_commitment=PASS; must_not_entail=PASS; persona_no_self_contradiction=PASS; persona_generic_drift=PASS; persona_voice_adherence=PASS; persona_tone_pressure=FAIL; persona_commitment=PASS |
| relationship_transition_001 | ✅ | expected_memory=PASS; response_expected=PASS; transition=PASS; response_expected=PASS; must_entail=PASS; must_not_entail=PASS; response_expected=PASS |

Failures are retained in the JSON output with the response/retrieval trace for diagnosis.