# Transformer Decision Maker

Code, trained models and results for *transformers with a continuous latent memory can run
online learning algorithms*: the multiplicative-weights / weighted-majority algorithm (MWU)
and online tabular Q-learning. For each algorithm there is

* a **handwired construction**: a small transformer (causal softmax heads, ReLU MLPs, residual
  stream) whose weights are set by hand so that it runs the algorithm exactly, and
* **trained models** that differ only in how they carry memory from one step to the next:

  | memory channel | what is carried | folder name |
  |---|---|---|
  | continuous latent, residual write | `c <- c + W h[UPDATE]` (the headline model) | `continuous_residual` |
  | continuous latent, overwrite | `c <- h[UPDATE]` | `continuous_overwrite` |
  | discrete token | `h[UPDATE]` snapped to a vocabulary token | `discrete` |

## Layout

```
multiplicative_weights/     MWU (experts)            ┐ the two projects mirror each other:
tabular_q_learning/         online tabular Q-learning ┘ README.md, paths.py, scripts/, scripts/jobs/,
                                                       checkpoints/, data/, figures/, logs/
relaxation/                 both constructions run with their idealised assumptions relaxed
llm/                        prompted LLMs as online learners on the experts problem (see llm/core/README.md)
paper/                      make_main_figures.py (main-text figures + matched-memory table);
                            the paper source, notes and rendered figures/tables are git-ignored
```

Inside each project, `figures/` has one folder per memory channel, a `comparison/` folder for
anything that compares them, and `handwired/` for the construction:

```
figures/
  continuous_residual/   results of the headline model
  continuous_overwrite/  results of the overwrite ablation
  discrete/              results of the discrete control (Q-learning only; the MWU discrete
                         model appears in comparison/)
  comparison/            matched-memory comparisons, probes, mechanism tests;
                         comparison/earlier_runs/ keeps results of superseded models
  handwired/             the construction vs the exact algorithm
```

## Running things

* Every file location is in the project's `paths.py`; scripts take their defaults from it, so
  they run from any directory and the evaluation scripts default to the final models.
* Slurm jobs live in `scripts/jobs/` and are submitted **from the project folder**, e.g.

  ```bash
  cd tabular_q_learning && sbatch scripts/jobs/evaluate.sh
  cd multiplicative_weights && sbatch scripts/jobs/train.sh continuous_residual 42
  ```

  Logs go to the project's `logs/`.
* Main-text figures and the table: `python3 paper/make_main_figures.py`
  (writes `paper/figures/` and `paper/tables/table_matched.tex`).

See `multiplicative_weights/README.md` and `tabular_q_learning/README.md` for each project's
scripts, jobs, checkpoints and a guide to its figures.

## Installation

```bash
pip install -r requirements.txt
```

Checkpoints (`*.pt`) and the large datasets are git-ignored; they live only on the cluster copy.
