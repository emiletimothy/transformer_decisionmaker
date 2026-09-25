# Multiplicative weights (MWU / weighted majority)

Online prediction with expert advice: 4 experts predict a binary outcome each round, the model
predicts, then sees the outcome. The recurrent models read one round at a time and carry a
latent memory token `M` between rounds (see `train.py` for the round layout).

## Scripts (`scripts/`)

| script | what it does | default output |
|---|---|---|
| `train.py` | trains a recurrent model (model + curriculum training) | `checkpoints/<model>_seed<seed>/` |
| `eval_matched_memory.py` | regret / accuracy of continuous vs discrete vs raw history vs MW, Bayes, majority | `figures/comparison/matched_memory/` |
| `eval_mechanism.py` | state probes, retention rho, behavioural fit, causal steering | `figures/comparison/mechanism/` |
| `eval_theorem_matched.py` | theorem-matched models vs weighted majority, adversarial tests | `figures/continuous_residual/theorem_matched/` |
| `figures_per_model.py` | the submitted paper's figure layouts, redrawn for the final models | `figures/continuous_residual/` |
| `handwired_mwu.py` | the handwired construction (flag-free, standard components) | – |
| `handwired_figure.py` | construction vs exact MW weights | `figures/handwired/` |
| `full_history_model.py`, `train_full_history.py` | the submitted paper's full-history model (the "raw history" baseline) | `checkpoints/full_history/` |
| `multiplicative_weights.py` | reference MW implementation | – |
| `full_history/` | analyses of the full-history model: attention (`eval_attention*.py`), long sequences, robustness, its data generator | `figures/full_history/` |

Jobs (`scripts/jobs/`, submit from this folder, e.g. `sbatch scripts/jobs/train.sh discrete 43`):
`train.sh <model> <seed>`, `eval_matched_memory.sh`, `eval_mechanism.sh`,
`eval_theorem_matched.sh`, `figures_per_model.sh`. Logs go to `logs/`.

## Models (`checkpoints/`, see `paths.py`)

| folder | model |
|---|---|
| `continuous_residual_seed{42,43}` | latent `M_t = M_{t-1} + W h[UPD]` (headline) |
| `continuous_overwrite_seed{42,43}` | latent `M_t = h[UPD]` |
| `discrete_seed{42,43}` | latent snapped to a vocabulary token |
| `theorem_matched_seed{42,43}` | residual latent on the noisy-true-expert data, where weighted majority (eta = ln 4) is Bayes-optimal |
| `full_history/` | the full-history model (re-reads the raw 1024-token history) |
| `earlier_runs/` | superseded recurrent models (v1–v3, forecast-input theorem runs); the leak panel of the paper still plots them |

## Figures (`figures/`)

```
continuous_residual/
  evaluation/{long_sequences,robustness,scenarios}/   regret over long horizons, robustness, OOD scenarios
  attention/                                          attention maps, weight trajectories, latent PCA
  overview/                                           training curves, regret trajectories, attention summary
  theorem_matched/                                    distance to weighted majority, adversarial regret
continuous_overwrite/evaluation/                      matched-memory evaluation of the overwrite model
comparison/
  matched_memory/        final table data: residual vs discrete vs raw history vs MW / Bayes / majority
  mechanism/             probes, retention rho, behavioural fit, steering (json)
  earlier_runs/          the same evaluations for superseded models
handwired/               construction vs exact MW weights
full_history/            the full-history (raw-history) model: attention/ (expert focus, token maps), robustness/
```

`data/earlier/` holds the full-history model's training data (the recurrent models draw fresh
data every epoch and need no dataset file).
