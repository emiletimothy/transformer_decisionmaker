# Archive

## Experiments

### exp01_interactive_mw_100steps
- **Dataset**: `old_datasets/mw_dataset_uniform.json` — expert acc Uniform(0,1), best expert avg 74%
- **Models**: DeepSeek V3 (20×4), GPT-4o (20×4), GPT-5.4 Mini silent (20×4), DeepSeek R1 (1×1), Gemini 3 Flash (1×1)
- **Mode**: Multi-turn, 100 steps
- **Cost**: ~$108
- **Finding**: All LLMs 42-53% accuracy, far below MW 64%. Dataset too noisy.

### exp02_stratified_multiturn_100steps
- **Dataset**: `mw_dataset_stratified.json` (same as root `mw_dataset.json`) — tiered expert acc (best 90-100%)
- **Models**: DeepSeek V3 (20×4), Qwen3-14B nothink (20×4), Qwen3-14B think1024 (20×1 online only)
- **Mode**: Multi-turn, 100 steps
- **Cost**: ~$7.44 (ds) + $0 (local qwen)
- **Finding**: Better dataset → LLMs reach 73-81%. Thinking doesn't help 14B model. LLMs use pattern matching, not actual MW computation.

## Old Datasets
| File | Description |
|------|-------------|
| `mw_dataset_uniform.json` | Original, expert acc ~ Uniform(0,1). Used in exp01. |
| `mw_dataset_test.json` | 3-case test set, 100 steps |
| `mw_dataset_tiny.json` | 3-case tiny set, 5 steps (for debugging) |
| `mw_dataset_perfect_expert.json` | One perfect expert per case. Not used in final experiments. |
| `mw_dataset_stratified_README.md` | Documentation for stratified dataset |

## Old Scripts
| File | Description |
|------|-------------|
| `api_run.py` | Batch (non-interactive) API runner. Superseded by interactive_api_run.py |
| `run.py` | Old batch mode generate/evaluate CLI |
| `generate_dataset.py` | Dataset generation script |
| `recompute_results.py` | Batch recompute analysis + regenerate plots |
| `auto_stop_think.sh` | Auto-kill think experiments after weather completes |
| `run_qwen3_all.sh` | Sequence script for nothink → think experiments |
| `run_qwen3_sequence.sh` | Earlier version of sequence script |

## Other
- `results_old/` — Very early experiment results (manual, api, interactive_api)
- `old_notebooks/` — Jupyter viewer notebooks
- `train_087/` — Early single-case test results
