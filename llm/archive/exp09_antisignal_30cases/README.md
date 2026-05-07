# Exp09: Anti-signal Dataset — 30 cases

## Setup
- **Model**: DeepSeek-chat
- **Dataset**: `mw_dataset_antisignal.json` (best 60-70%, anti-signal 0-10%, mid 40-60%)
- **Run config**: 30 cases, 100 steps each
- **Prompts**: 4 v2 +hint prompts (weather/online × note/nonote)

## Results

| Prompt | Regret | ±Std |
|--------|--------|------|
| online +note | 11.3 | 10.3 |
| weather +note | 13.3 | 6.9 |
| online -note | 17.5 | 7.6 |
| weather -note | 21.9 | 7.0 |

### Baselines
| Baseline | Regret | ±Std |
|----------|--------|------|
| FTL | 2.7 | 2.1 |
| MW | 5.9 | 2.7 |
| FPW | 11.7 | 3.9 |
| MajVote | 25.2 | 5.0 |
| Random | 15.1 | 5.6 |

## Key Observations
- **Note vs nonote gap is LARGE again** (~6-8 regret points), even though best expert is only 60-70%
- Anti-signal expert (0-10% acc) is a strong negative signal — note helps LLM learn to avoid/invert it
- Without note, LLM is worse than Random (17-22 vs 15) — anti-signal expert poisons majority voting
- MajVote catastrophically bad (25.2) — anti-signal expert corrupts uniform weighting
- FTL excels (2.7) — quickly identifies and follows best expert, ignoring anti-signal
- Note benefit is NOT just about expert accuracy gap — it's about having a *confusing* expert that needs to be explicitly tracked and avoided

## Plot
See `regret_note_vs_nonote.png`
