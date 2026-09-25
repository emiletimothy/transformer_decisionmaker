# Online Learning with Expert Advice via LLM Prompting

Code and data for evaluating LLMs on the prediction-with-expert-advice problem.

## Contents

```
core/
├── interactive_api_run.py     # Main experiment runner (DS API + local vLLM)
├── mw_lib.py                  # MW algorithm, analysis, plotting utilities
├── generate_dataset.py        # Generate expert-prediction datasets
├── plot_regret.py             # Plot regret curves with baselines
├── prompts/                   # Prompt templates (4 conditions)
│   ├── interactive_weather.txt          # weather framing + note
│   ├── interactive_weather_nonote.txt   # weather framing - note
│   ├── interactive_online.txt           # online framing + note
│   └── interactive_online_nonote.txt    # online framing - note
├── mw_dataset.json            # Stratified regime (30 cases × 200 steps)
├── mw_dataset_flat.json       # Flat regime
└── mw_dataset_antisignal.json # Anti-signal regime
```

## Setup

```bash
pip install openai numpy matplotlib openpyxl
```

For DeepSeek V3:
```bash
export DEEPSEEK_API_KEY=your_key
```

For Qwen3-14B (local, requires GPU):
```bash
pip install vllm
vllm serve /path/to/Qwen3-14B \
  --served-model-name Qwen3-14B \
  --max-model-len 40960 \
  --tensor-parallel-size 2 \
  --enforce-eager
```

## Generate Datasets

```bash
# Generate all three regimes (or use the provided JSON files)
python generate_dataset.py --all
```

## Run Experiments

```bash
# DeepSeek V3, note protocol, stratified dataset, cases 0-29
python interactive_api_run.py cases \
  --idx 0 1 2 ... 29 \
  --model ds \
  --prompt interactive_weather_notehist \
  --dataset mw_dataset.json \
  --steps 100

# DeepSeek V3, no-note protocol
python interactive_api_run.py cases \
  --idx 0 1 2 ... 29 \
  --model ds \
  --prompt interactive_weather_nonote \
  --dataset mw_dataset.json \
  --steps 100

# Qwen3-14B (requires vLLM running on localhost:8000)
python interactive_api_run.py cases \
  --idx 0 1 2 ... 29 \
  --model qwen3_14b_nothink \
  --prompt interactive_weather_notehist \
  --dataset mw_dataset.json \
  --steps 100
```

### Prompt names

| Prompt name | Framing | Note | Protocol |
|---|---|---|---|
| `interactive_weather_notehist` | weather | +note | multi-turn + note |
| `interactive_weather_nonote` | weather | -note | multi-turn |
| `interactive_online_notehist` | online | +note | multi-turn + note |
| `interactive_online_nonote` | online | -note | multi-turn |

### Output directory

Use `RESULTS_DIR` env var to control output location (default: `exp_results/`):
```bash
RESULTS_DIR=my_results python interactive_api_run.py ...
```

Results are saved to `{RESULTS_DIR}/{prompt_short}/{model_key}/cases_{idx}/run001/`.

## Plot Results

```bash
python plot_regret.py \
  --exp_dir exp_results \
  --dataset mw_dataset.json \
  --model ds \
  --save plots/stratified_regret.png
```

## Expert Regimes

| Regime | Best expert | Other experts | Key challenge |
|---|---|---|---|
| Stratified | 90-100% | 45-80% (tiered) | Identify dominant expert |
| Flat | 60-70% | 40-60% | Weak signal |
| Anti-signal | 60-70% | 0-10% (adversarial) + 40-60% | Detect and avoid trap |

## Models

| Model | Key | Decoding | Provider |
|---|---|---|---|
| DeepSeek V3 | `ds` | temperature=0 | API (api.deepseek.com) |
| Qwen3-14B | `qwen3_14b_nothink` | temperature=0.7, top_p=0.8 | Local (vLLM) |
