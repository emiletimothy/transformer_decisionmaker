# Online tabular Q-learning

Each step of a random MDP is one token sequence
`<BOS> c_1..c_|A| <Qcurr> s_t a_t r_t <Qnext> (s_{t+1} a_i)_i <Select> a* <Update>`; the model
predicts the teacher's greedy action at `<Select>` and writes its `<Update>` output into the
context slot `c_{a_t}` of the executed action (see `3_train.py` / `2_model.py`).

## Scripts (`scripts/`)

| script | what it does | default output |
|---|---|---|
| `1_generate_data.py` | episodes + tabular Q-learning teacher labels | `data/` |
| `2_model.py`, `3_train.py` | model and training (teacher-forced, BPTT) | `checkpoints/<model>/` |
| `4_evaluate.py --model M` | the standard figure set (agreement, probes, attention, reward, ...) | `figures/<model>/evaluation/` |
| `eval_closed_loop.py --model M` | 1000-step closed loop with the model's own a* (+ eps-greedy via `--astar_modes self_eps0.2`) | `figures/<model>/closed_loop/` |
| `eval_clean_steps.py` | agreement with Q-learning vs simpler rules on steps where copying the last a* fails | `figures/comparison/clean_steps/` |
| `eval_alpha_gamma.py`, `eval_memory_fit.py` | best-fitting behavioural (alpha, gamma) and memory length | `figures/comparison/behavioural_fit/` |
| `eval_q_probe.py` | linear probe from the context slots to the Q-table | `figures/comparison/q_probe/` |
| `eval_drift.py`, `eval_write_decomposition.py` | slot-norm growth in closed loop and what each write adds | `figures/comparison/drift/` |
| `data_shortcuts.py` | how often simpler rules give the teacher's label in a dataset | printed |
| `handwired_q_learning.py` | the handwired construction (flag-free, standard components); running it verifies it | `figures/handwired/verification*.csv` |
| `handwired_figure.py` | construction vs exact tabular Q-learning | `figures/handwired/` |
| `handwired_q_learning_v1.py`, `tabular_q_learning.py` | the submitted paper's construction (used by `relaxation/`) and the tabular reference | – |

Jobs (`scripts/jobs/`, submit from this folder, e.g. `sbatch scripts/jobs/evaluate.sh`):
`train.sh data | <model>`, `evaluate.sh` (every evaluation of the three final models),
`eval_exploration.sh` (eps-greedy closed loop). Logs go to `logs/`.

## Models and data (see `paths.py`)

| path | content |
|---|---|
| `checkpoints/continuous_residual/` | slot write `c <- c + W h[UPDATE]` (headline) |
| `checkpoints/continuous_overwrite/` | slot write `c <- h[UPDATE]` |
| `checkpoints/discrete/` | slot snapped to a vocabulary token |
| `checkpoints/earlier_runs/` | the submitted paper's models and the bootstrap-data models, with their result notes |
| `data/qlv3_dataset.pt` | final data: alpha 0.2, reward noise 0.3, 100–200-step episodes |
| `data/earlier/` | the submitted paper's dataset (`coconut_dataset.pt`) and the bootstrap data |

## Figures (`figures/`)

```
continuous_residual/  continuous_overwrite/  discrete/      (same three folders per model)
  evaluation/                 4_evaluate.py figures (combined_row, probes, attention, reward, size sweep, ...)
  closed_loop/                1000-step closed loop, nonstationary, reward intervention, size sweep (csv/npz)
  closed_loop_exploration/    the same long-horizon runs with eps-greedy exploration (eps 0 ... 0.3)
comparison/
  clean_steps/  behavioural_fit/  q_probe/  drift/
  earlier_runs/               clean-step and probe results of the submitted paper's / bootstrap models
handwired/                    construction vs tabular Q-learning, verification tables
```
