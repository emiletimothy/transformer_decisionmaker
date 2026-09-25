# Reviewer v2 — per-MDP re-evaluation + construction-assumption probes (Q-learning)

Inference only, same two checkpoints as `RESULTS_discrete_vs_continuous.md`
(continuous `coconut-v4-mixed-20260502-1646`, discrete `discrete-v1-20260723-2146`, 3.17M params each).

- Scripts: `scripts/reviewer_v2_eval.py`, `scripts/reviewer_v2_mechanistic.py`, `scripts/reviewer_v2_report.py`, sbatch `scripts/run_reviewer_v2.sh [eval|mech]`
- Jobs: 3608962 (eval, 8m22s, RTX A4000), 3608963 (mechanistic, 26s)
- Outputs: `figures/reviewer_v2/{continuous,discrete}/*_{summary,per_mdp}.csv`, `*_rewards.npz`; `figures/reviewer_v2/mechanistic/`; all tables: `figures/reviewer_v2/report_tables.md`

**n = 50 MDPs per condition** (seeds 9999–10048; previously 10). `±` = SD (SEM) or `± SEM` where marked.
**Parity:** on the first 10 seeds the legacy closed-loop rollout reproduces the old numbers exactly
(contrast 1000-step: continuous 418.70, discrete 274.01; eval: 549.86 / 548.63).

New metrics: `opt_frac` = fraction of steps taking a Q\*-optimal action (value iteration, γ=0.9);
`gap_regret` = Σ_t V\*(s_t) − Q\*(s_t,a_t).

## ⚠ Closed-loop a\* protocol
The closed-loop runners in `4_evaluate.py` feed `a_star = 0` into the step tokens: the UPDATE token is always told that the greedy next action is a₀. In training, a\* is the teacher's argmax. `self` mode instead feeds the model's own SELECT argmax. SELECT comes before a\* causally, so the action choice is unaffected and only the context write changes. **Self mode is *worse* for the continuous model** (contrast 1000-step 39.2 → 32.1 %opt; reward-inversion sensitivity +16.6 → +5.9). So a\*=0 does not explain the flat curves. Either way the closed-loop evaluation has a train/test mismatch in the a\* slot, and the paper should disclose it. Numbers below are **legacy** (the paper's protocol) unless marked.

## 1. Long horizon, 1000 steps (%opt, mean ± SEM, n=50)

| agent | contrast %opt | contrast opt_frac | eval %opt |
|---|---|---|---|
| optimal | 100 | 1.000 | 100 |
| ε-greedy Q | 54.5 ± 1.4 | 0.521 | 73.6 ± 1.1 |
| **continuous** | **39.2 ± 2.2** | **0.360** | 70.7 ± 1.2 |
| discrete | 32.2 ± 1.5 | 0.286 | 68.5 ± 1.4 |
| greedy Q | 30.3 ± 2.0 | 0.266 | 69.8 ± 1.4 |
| random | 28.7 ± 0.2 | 0.249 | 67.7 ± 0.6 |

- The old 10-MDP headline (41.9 vs 27.4, "discrete below random") **does not replicate at n=50**. Continuous is 39.2 ± 2.2 and discrete 32.2 ± 1.5, so discrete is slightly *above* random (+3.5, about 2.3 SEM). Continuous still clearly beats discrete (+7.0), random (+10.5) and greedy Q, but trails ε-greedy Q by 15 pts.
- The eval (Beta(2,2)) family is still saturated: every agent lands at 67–74%.

## 2. Reward interventions, contrast family (%opt ± SEM, 200 steps, n=50)

| condition | continuous | discrete | ε-greedy Q | greedy Q |
|---|---|---|---|---|
| intact | 35.0 ± 2.2 | 32.2 ± 1.7 | 45.9 ± 1.7 | 29.9 ± 2.0 |
| zeroed | 26.1 ± 1.5 | 31.0 ± 1.9 | 28.9 ± 0.4 | 28.9 ± 0.4 |
| constant | 29.5 ± 2.1 | 30.1 ± 1.9 | 27.8 ± 1.7 | 29.9 ± 2.0 |
| decorrelated | 27.4 ± 1.8 | 30.6 ± 1.7 | 28.1 ± 1.7 | 29.9 ± 2.0 |
| inverted | 18.4 ± 1.3 | 29.0 ± 1.7 | 15.0 ± 0.8 | 18.1 ± 1.8 |
| **intact − inverted (paired)** | **+16.6 ± 2.1** | +3.2 ± 1.7 | +30.9 ± 1.8 | +11.8 ± 2.1 |
| **intact − mean(ablated) (paired)** | **+7.3 ± 1.5** | +1.7 ± 0.7 | +17.7 ± 1.4 | +0.3 ± 0.7 |

This is still the cleanest behavioural result. Inverting the reward drives the continuous model below random (18.4%), the same signature as the Q-learners (about 8 paired SEMs). The discrete model reacts only weakly (+3.2 ± 1.7, not "exactly zero" as the n=10 run suggested). On the eval family everything is within ±4 pts.

## 3. Nonstationary (contrast_permute; window rate as % of the adapting optimum, ± SEM)

| agent | pre-switch | just after | end | final cumreward |
|---|---|---|---|---|
| continuous | 33.5 ± 3.1 | 23.8 ± 2.3 | 36.7 ± 3.3 | 47.4 ± 2.6 |
| discrete | 31.6 ± 2.7 | 31.8 ± 2.4 | 31.8 ± 2.8 | 43.9 ± 1.6 |
| ε-greedy Q | 45.5 ± 2.8 | 31.2 ± 2.9 | 38.8 ± 2.4 | 54.5 ± 1.9 |
| random | 28.7 ± 1.8 | 30.3 ± 1.8 | 29.1 ± 1.6 | 43.8 ± 0.7 |
| optimal (frozen) | 100 | 24.4 ± 3.1 | 23.2 ± 3.2 | 74.0 ± 3.0 |

The continuous model shows the Q-learner's dip-and-recover shape: the switch hurts it, which means it had committed to the pre-switch actions, and it then recovers. The discrete model is flat, as random is. Per-window error bars are wide, so treat this as corroborating evidence. The four eval-family variants are uninformative: all agents, including random, land at 65–75%.

## 4. Size sweep (21 cells, |S| 2–8 × |A| 2–4, n=50/cell)

| | continuous | discrete |
|---|---|---|
| agreement with tabular teacher | **0.936 ± 0.038** | 0.624 ± 0.134 |
| model prediction = TRUE optimal action | 0.398 ± 0.116 | 0.368 ± 0.110 |
| teacher target = TRUE optimal action | 0.401 ± 0.118 | 0.401 ± 0.118 |
| contrast return %opt | 48.7 ± 14.6 | 38.6 ± 11.0 (random 39.3) |
| cells beating random (paired, one-sided p<0.05) | **15 / 21** | 1 / 21 |

(± = across-cell SD.) The continuous model imitates the teacher faithfully, *including its suboptimality*: its true-optimal rate (0.398) matches the teacher's (0.401). The teacher is an under-trained 50-step Q-learner. Present agreement as "implements the Q-learning update", not "acts optimally".

## 5. Does the trained model satisfy the construction's assumptions?
Teacher-forced, 50 in-distribution 8×4 MDPs × 50 steps. Figure: `figures/reviewer_v2/mechanistic/mechanistic.png`.

| assumption | metric | continuous | discrete | reference |
|---|---|---|---|---|
| orthogonal token embeddings | mean / max off-diagonal \|cos\| | 0.085 / 0.27 | 0.108 / 0.54 | random Gaussian (d=256): 0.050 |
| near-hard attention | best head's mean max weight at R / QNEXT / UPDATE / QCURR / SELECT | 0.98 / 0.97 / 0.96 / 0.91 / 0.77 | 0.89 / 0.88 / 0.89 / 0.75 / 0.91 | hard = 1 |
| | share of (layer, head, query) pairs with mean max weight > 0.9 | 6.9% | 0.6% | median over all heads: 0.39 |
| fixed-offset routing | share of pairs whose top key is at the same position in > 90% of samples | 6.2% | 1.9% | — |
| Q-row stored in c_a | per-slot context dimensionality (PCA, 90% energy) | **8 per slot (= \|S\|)** | 2–3 | construction: \|S\| |
| | participation ratio of all contexts | 10.7 | 1.3 | — |
| | context energy in span(state embeddings) | 4.0% | 4.9% | random subspace of the same dim: 2.9% |
| | mean principal angle between slot subspaces | 20.6° (mostly shared, a few directions ~85°) | 3.0° | — |
| | distinct carried tokens (discrete only) | — | **4** (UPDATE 50%, a3 27%, a2 22%, NULL <1%) | vocab 19 |

How to read this:
- **Embeddings** are approximately but not exactly orthogonal: about 1.7× the random-vector baseline, and all \|cos\| < 0.27 for continuous.
- **Attention.** Each operand the construction routes (reward, bootstrapped value, UPDATE write) has at least one near-hard head (≥ 0.96) in the continuous model. Most other heads are diffuse. So the trained model is a *sparse near-hard circuit embedded in soft attention*, not uniformly hard. SELECT is the exception (0.77), consistent with the argmax being computed softly.
- **Context state.** Each continuous context slot lives in exactly an |S| = 8-dimensional subspace: the dimensionality of a Q-table row, as the construction predicts. The basis is learned, though, not the state-embedding basis the construction uses (4% energy vs 3% for a random subspace). Together with the linear Q-probe (R² 0.66), this supports "Q-row in a linearly-related basis" rather than the literal superposition of feature embeddings.
- **The discrete model collapses to 4 tokens**, at most 2 bits per slot. This mechanistically explains why it cannot carry a Q-row and why its reward sensitivity is near zero.

## Caveats
- The size-sweep return and the teacher-forced numbers use the same seeds as before, extended to 50. Each cell's per-MDP values are in `size_sweep_per_mdp.csv`.
- The mechanistic probes are single-checkpoint and have no seed replicates. Attention statistics pool 2,500 step-samples per head.
- Minor: axis-label font sizes in `mechanistic.png` are uneven because the rcParams are inherited from 4_evaluate. Cosmetic; re-plot before the camera-ready.
