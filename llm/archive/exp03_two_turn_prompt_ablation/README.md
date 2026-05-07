# Experiment 03: Two-Turn Stateless Protocol — Prompt Ablation

## Best result: DS V3 weather (hint) = 88.8% accuracy (MW optimal = 93.0%)

## Dataset
- mw_dataset.json (stratified, 30 cases, 200 steps, 4 experts)
- Evaluated first 100 steps, 20 cases (idx 0-19)
- Best expert avg: 94.6%, MW optimal avg: 93.0%

## Protocol
- Two-turn stateless: prediction turn + update turn
- Note-passing: model maintains state via explicit short note (max 500 words)
- No conversation history accumulation
- Named experts (Expert A-D)

## Model
- DeepSeek V3 (deepseek-chat, 671B MoE), temperature=0

## Prompts Tested

| Prompt | Goal Framing | Reliability Hint | Mean Acc | Final Regret |
|--------|-------------|-----------------|----------|-------------|
| weather (hint) | weather metaphor | "more reliable" | **88.8%** | **5.8** |
| weather (no hint) | weather metaphor | none | 86.2% | 8.3 |
| online (hint) | accuracy | "more accurate" | 86.3% | 8.2 |
| online (no hint) | accuracy | none | 83.7% | 10.9 |

## Baselines

| Method | Final Regret |
|--------|-------------|
| Follow the Leader | 1.4 |
| MW Optimal | 1.6 |
| weather (hint) | 5.8 |
| online (hint) | 8.2 |
| weather (no hint) | 8.3 |
| online (no hint) | 10.9 |
| Follow Previous Winners | 11.2 |
| Majority Vote (uniform) | 13.0 |
| Random Guessing | ~45 |

## Key Findings

1. **Weather framing +2-3% over online** — metaphor helps LLM understand expert-following
2. **Reliability hint +2-3%** — "some may be more reliable" triggers expert tracking behavior
3. **DS spontaneously develops cumulative counters** in notes (e.g., "A:43/94, B:94/94") in 8/10 weather cases
4. **When counters maintained, DS matches MW** — regret 0-4 in those cases
5. **Gap driven by note degeneration** — 2/10 cases never establish counters, causing regret 10-14
6. **Follow the Leader beats MW** on this dataset (1.4 vs 1.6) due to clear best expert

## Also includes (from earlier runs)
- Qwen3-14B nothink: online (71.5%) and weather (63.5%)
- Qwen3-14B think1024: online (82.7%) and weather (87.1%)
- Thinking boosts +12-24% in two-turn mode

## Cost
- DS V3: ~$2.50 total for all prompt variants
- Qwen3-14B: $0 (local GPU)
