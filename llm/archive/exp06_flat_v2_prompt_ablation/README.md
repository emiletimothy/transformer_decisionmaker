# Exp06: Flat Dataset — v2 Prompt Ablation

## Setup
- **Model**: DeepSeek-chat (deepseek-chat)
- **Dataset**: `mw_dataset_flat.json` (flat expert accuracy distribution)
  - 30 cases, 200 steps, 4 experts
  - Best expert: 60-70%, rest: 40-60%
- **Run config**: 20 cases (indices 0-19), 100 steps each
- **Prompts**: All 8 v2 prompts (2×2×2: weather/online × hint/nohint × note/nonote)
  - v2 adds "Use past outcomes to figure out which experts are most trustworthy and improve your future predictions."

## Results

| Prompt | Acc% | Regret | ±Std |
|--------|------|--------|------|
| online_nohint_v2 | 60.8 | 5.2 | 4.0 |
| weather_no_hint_v2 | 60.2 | 5.8 | 3.9 |
| weather_v2 | 59.8 | 6.2 | 4.5 |
| online_v2 | 58.5 | 7.5 | 4.6 |
| online_nonote_v2 | 58.5 | 7.6 | 5.5 |
| weather_nonote_v2 | 58.2 | 7.8 | 4.9 |
| weather_nonote_nohint_v2 | 57.7 | 8.3 | 4.4 |
| online_nonote_nohint_v2 | 57.4 | 8.7 | 5.2 |

### Baselines (100 steps, same 20 cases)
| Baseline | Regret | ±Std |
|----------|--------|------|
| MW | 3.4 | 2.2 |
| FTL | 3.5 | 2.5 |
| MajVote | 7.1 | 4.2 |
| FPW | 9.1 | 5.9 |

Best Expert avg accuracy: 66.0%

## Key Observations
1. **Tight accuracy range**: All prompts cluster between 57.4-60.8% (vs 65-89% spread on stratified)
2. **Note prompts still better**: Note variants (5.2-7.5 regret) outperform nonote (7.6-8.7)
3. **Nohint slightly better than hint**: Surprising — online_nohint_v2 is best (5.2 regret)
4. **LLM beats MajVote**: Best prompts (5.2-5.8) beat Majority Vote (7.1) and FPW (9.1)
5. **LLM doesn't match MW/FTL**: Gap of ~2 regret points to optimal algorithms (3.4-3.5)
6. **Flat dataset is harder**: Smaller expert accuracy gaps make it harder for LLMs to identify the best expert

## Plot
See `flat_v2_regret_plot.png`
