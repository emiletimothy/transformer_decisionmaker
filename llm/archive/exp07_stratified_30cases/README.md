# Exp07: Stratified Dataset — 30 cases

## Setup
- **Model**: DeepSeek-chat
- **Dataset**: `mw_dataset.json` (best 90-100%, 2nd 65-80%, 3rd 55-70%, worst 45-60%)
- **Run config**: 30 cases, 100 steps each
- **Prompts**: 4 v2 +hint prompts (weather/online × note/nonote)

## Results

| Prompt | Regret | ±Std |
|--------|--------|------|
| weather +note | 8.1 | 6.3 |
| online +note | 9.3 | 8.0 |
| weather -note | 12.7 | 5.9 |
| online -note | 13.9 | 8.2 |

### Baselines
| Baseline | Regret | ±Std |
|----------|--------|------|
| FTL | 1.3 | 1.0 |
| MW | 1.6 | 1.8 |
| FPW | 12.5 | 4.4 |
| MajVote | 14.5 | 4.7 |
| Random | 44.5 | 5.6 |

## Key Observations
- Note vs nonote gap: ~4.5 regret points (clear benefit of note)
- Weather slightly better than online framing
- LLM +note approaches FPW/MajVote level but far from MW/FTL

## Plot
See `regret_note_vs_nonote.png`
