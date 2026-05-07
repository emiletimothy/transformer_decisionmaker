# Exp08: Flat Dataset — 30 cases

## Setup
- **Model**: DeepSeek-chat
- **Dataset**: `mw_dataset_flat.json` (best 60-70%, rest 40-60%)
- **Run config**: 30 cases, 100 steps each
- **Prompts**: 4 v2 +hint prompts (weather/online × note/nonote)

## Results

| Prompt | Regret | ±Std |
|--------|--------|------|
| weather +note | 7.4 | 5.2 |
| online +note | 7.9 | 5.3 |
| online -note | 8.2 | 5.1 |
| weather -note | 8.5 | 5.0 |

### Baselines
| Baseline | Regret | ±Std |
|----------|--------|------|
| MW | 3.3 | 2.1 |
| FTL | 3.5 | 2.4 |
| MajVote | 8.4 | 4.7 |
| FPW | 8.6 | 4.6 |
| Random | 15.2 | 5.6 |

## Key Observations
- Note vs nonote gap nearly vanishes (~1 regret point)
- All LLM prompts cluster around MajVote/FPW level
- Weak expert signal → note provides little useful information

## Plot
See `regret_note_vs_nonote.png`
