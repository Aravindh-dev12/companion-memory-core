# Evaluation matrix

| Research question | Scenario | Primary metric | Failure being isolated |
|---|---|---|---|
| Does memory survive process restart? | cross-session recall | persistence pass rate | persistence/storage |
| Does extraction capture durable facts? | disclosure set | precision / recall | extraction |
| Does the system stop using stale state? | breakup/job move/plan cancellation | stale-fact error rate | state resolution + validity |
| Can it distinguish correction from change? | name correction vs relationship change | transition classification accuracy | resolver |
| Can old truth still be recalled historically? | "who was I dating before?" | historical recall accuracy | temporal reasoning |
| Can it recall exact entities amid distractors? | similar names/projects | Recall@K / MRR | retrieval |
| Can it apply a latent constraint? | semantic-distance cue/trigger | constraint-consistency accuracy | cognitive memory |
| Can it say "I don't know"? | undisclosed facts | abstention accuracy | hallucinated memory |
| Does persona survive topic pressure? | 50+ turn mixed-topic script | persona contradiction rate | persona drift |
| Does the verifier actually help? | firewall ablation | delta in contradiction rate | consistency layer |

## Reporting rule
For every final-answer failure, preserve the component trace so we can label whether the failure came from extraction, resolution, retrieval, reasoning, generation, or verification.
