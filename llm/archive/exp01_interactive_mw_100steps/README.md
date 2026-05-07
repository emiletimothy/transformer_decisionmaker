# Experiment 01: Interactive MW (100 steps)

## Setup
- **Task**: Multiplicative Weights Update, interactive (multi-turn) evaluation
- **Dataset**: mw_dataset_uniform.json (archived in old_datasets/)
  - 30 cases, 4 experts, 200 steps, evaluated first 100
  - Expert accuracy: Uniform(0,1), best expert avg 74%, some cases all experts <50%
- **Cases**: 20 cases (idx 0-19)

## Models
| Model | Key | Runs | Cost |
|-------|-----|------|------|
| DeepSeek V3 | ds | 80 (20×4 prompts) | ~$5.56 |
| GPT-4o | gpt4o | 80 (20×4 prompts) | ~$79.84 |
| GPT-5.4 Mini (silent) | gpt54_mini | 80 (20×4 prompts) | ~$19.47 |
| DeepSeek R1 (thinking) | ds_think | 1 (online only) | ~$2.80 |
| Gemini 3 Flash | gemini3_flash | 1 (mw_name only) | ~$0.52 |

## Prompts
- **online**: bare prediction, no algorithm hint
- **weather**: weather metaphor (sunny/rainy)
- **mw_name**: told "use multiplicative weights"
- **explicit_update**: full MW algorithm spelled out

## Key Results (mean accuracy, 20 cases)
| Prompt | Best Expert | MW | ds | gpt4o | gpt54_mini |
|--------|-----------|------|------|-------|------------|
| online | 68.2% | 63.9% | 41.8% | 47.1% | 43.2% |
| weather | 68.2% | 63.9% | 53.0% | 49.5% | 43.5% |
| mw_name | 68.2% | 63.9% | 44.9% | 48.1% | 44.9% |
| explicit_update | 68.2% | 63.9% | 50.9% | 50.6% | 42.4% |

ds_think on case_000 online: 66% accuracy (vs ds 5%, gpt4o 52%, MW 77%)

## Total cost: ~$108
