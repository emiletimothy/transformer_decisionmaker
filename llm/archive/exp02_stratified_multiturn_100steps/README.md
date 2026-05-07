# Experiment 02: Stratified Dataset, Multi-turn, 100 steps

## Dataset
- **File**: mw_dataset_stratified.json (now renamed to mw_dataset.json in root)
- **30 cases**, 200 steps each, 4 experts (evaluated first 100)
- **Stratified expert accuracy**:
  - Best: 90-100% (one per case, randomly assigned)
  - Second: 65-80%
  - Third: 55-70%
  - Worst: 45-60%
- Best expert avg: 94.6%, MW optimal avg: 93.0%

## Setup
- **Mode**: Multi-turn interactive (full conversation history accumulated)
- **Steps**: 100 (first 100 of 200-step dataset)
- **Cases**: 20 (idx 0-19)

## Models

| Model | Key | Type | Runs | Cost |
|-------|-----|------|------|------|
| DeepSeek V3 (672B MoE, API) | ds | standard, temp=0 | 80 (20×4 prompts) | ~$7.44 |
| Qwen3-14B (local, vLLM) | qwen3_14b_nothink | no thinking, temp=0.7 | 80 (20×4 prompts) | $0 (local) |
| Qwen3-14B (local, vLLM) | qwen3_14b_think1024 | thinking budget=1024, temp=0.6 | 20 (online only) | $0 (local) |

### vLLM Server Config
```
vllm serve Qwen3-14B \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 16384 \
  --reasoning-parser qwen3 \
  --reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}'
```

## Prompts
- **online**: bare prediction, no algorithm hint
- **weather**: weather metaphor (sunny/rainy)
- **mw_name**: told "use multiplicative weights" (outputs weights + prediction)
- **explicit_update**: full MW algorithm spelled out (outputs weights + prediction)

## Key Results — Mean Accuracy (20 cases)

| Prompt | Best Expert | MW | DeepSeek V3 | Qwen3 nothink | Qwen3 think1024 |
|--------|-----------|------|------------|---------------|-----------------|
| online | 94.6% | 93.0% | 78.2% | 73.0% | 70.8% |
| weather | 94.6% | 93.0% | 80.9% | 77.0% | — |
| mw_name | 94.6% | 93.0% | 77.8% | 57.2% | — |
| explicit_update | 94.6% | 93.0% | 81.3% | 76.3% | — |

## Key Results — Mean Regret (20 cases, online prompt)

| | Mean Loss | Mean Regret |
|---|----------|-------------|
| Best Expert | 5.5 | — |
| MW | 7.0 | 1.6 |
| DeepSeek V3 | 21.8 | 16.4 |
| Qwen3 nothink | 27.1 | 21.6 |
| Qwen3 think1024 | 29.7 | 24.2 |

## Key Findings

1. **DeepSeek V3 (671B) consistently outperforms Qwen3-14B** across all prompts (4-21% gap)
2. **Thinking (1024 budget) does NOT help on Qwen3-14B** — slightly worse than nothink
   - Think wastes budget recapping history instead of analyzing expert performance
   - Think defaults to majority vote, fails to identify best expert
   - Nothink uses implicit pattern matching, sometimes more effective
3. **mw_name prompt is catastrophic for Qwen3-14B** — 57% accuracy (near random)
   - Weights computation is too hard for 14B model without thinking
   - Even with thinking, weights are wrong (stuck near uniform)
4. **All LLMs far below MW optimal** — gap of 12-36% accuracy, 15-24x regret
5. **LLMs do learn** — significantly better than random (50%) and uniform weights (~73%)
6. **Weights output is decorative** — models can't do MW math but still predict via pattern matching

## Cost
- DeepSeek V3: ~$7.44 (API)
- Qwen3-14B: $0 (local GPU, 2× RTX A5000)
