# Experiment 05: Stratified Dataset — v1 vs v2 Prompt Comparison

## Dataset
- mw_dataset.json (stratified, best 90-100%, 2nd 65-80%, 3rd 55-70%, worst 45-60%)
- 20 cases, 100 steps

## Model
- DeepSeek V3, temperature=0

## v1: 8 prompts without "trustworthy" hint in goal
## v2: 8 prompts with "Use past outcomes to figure out which experts are most trustworthy"

## Key Result
v2 note prompts improved by +3-5.5 regret over v1.
v2 nonote prompts showed no improvement.
Weather vs online: p=0.47, NOT significant (10:10 wins).

## Prompts
Archived in prompts/ subdirectory
