# Experiment 04: 2×2×2 Prompt Ablation

## Dataset
- mw_dataset.json (stratified, best 90-100%, 2nd 65-80%, 3rd 55-70%, worst 45-60%)
- 20 cases, 100 steps

## Model
- DeepSeek V3 (deepseek-chat, 671B MoE), temperature=0

## Design: 2×2×2
- Framing: weather (sunny/rainy) vs online (0/1)
- Hint: "more reliable/accurate" vs no hint
- Note: two-turn note-passing vs multi-turn no-note (feedback only)

## Prompts
Goal text: "Your job is to make your own prediction based on the experts' predictions. Your goal is to be as accurate as possible over time."
