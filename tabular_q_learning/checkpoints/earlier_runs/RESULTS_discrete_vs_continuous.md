# Discrete vs. Continuous Context: Checkpoint Comparison

All numbers are read off saved runs — no re-evaluation was performed. Sources:

- `coconut_discrete_experiment3352214.out` (Jul 24) — agreement, OOD families, probes, α/γ recovery
- `coconut_reviewer_evals3378331.out` (Jul 27) — reward interventions, size sweep, 1000-step long horizon
- `figures/{discrete,continuous}/{nonstationary,size_sweep,reward_intervention}_summary.csv`

**Error-bar convention.** `±` is **1 SD across the 10 eval MDPs** (seeds 9999–10008), not SEM — divide by √10 ≈ 3.16 for SEM. SDs are quoted only where the run actually saved them:

| Block | SD available? | Source column / field |
|---|---|---|
| ID + OOD agreement (§2) | yes | printed `+/-` in the Jul 24 log |
| Reward interventions (§3) | yes | `final_return_std` |
| Size sweep agreement (§6) | yes | `agreement_std` |
| Per-reward-dist. returns (§4) | yes | printed `±` in the Jul 24 log |
| Size sweep **return** (§6) | **no** | only the cell mean was written |
| Long horizon (§4) | **no** | only final cumreward was printed |
| Nonstationary (§5) | **no** | CSV has no `_std` columns |
| Probes, α/γ, val metrics (§1) | **no** | single fit / single scalar |

Where a block has no SD, the cell is left bare rather than filled with a recomputed number — regenerating them would require re-running the evals.

Checkpoints:

| | continuous | discrete |
|---|---|---|
| checkpoint | `coconut_transformer_coconut-v4-mixed-20260502-1646.pt` | `coconut_transformer_discrete-v1-20260723-2146.pt` |
| epoch / step | 40 / 18,876 | 34 / 14,000 |
| params | 3,171,076 | 3,171,076 |

---

## 1. Headline metrics

| Metric | continuous | discrete | Reference |
|---|---|---|---|
| Val next-action accuracy | **89.1%** | 58.9% | — |
| Val cross-entropy | **0.2090** | 0.7953 | — |
| Mean ID action agreement (8s×4a, 10 MDPs) | **85.8% ± 2.1** | 42.8% ± 9.9 | chance 25% |
| Size-sweep mean agreement (21 cells) | **93.6% ± 3.9** | 62.1% ± 13.8 | chance 25–50% |
| ↳ mean *within*-cell SD (10 MDPs) | **± 2.0** | ± 10.0 | — |
| Contrast intact return, % of optimal | **38.4 ± 17.3** | 28.3 ± 8.6 | random 29.7 |
| Context-probe Q R² | **0.660** | 0.019 | single fit, no SD |
| Context-probe Q R² (no bias) | **0.700** | 0.020 | single fit, no SD |
| Context-probe Frobenius error | **0.123** | 0.212 | lower = better |
| Reward-probe R² (from context delta) | **0.378 – 0.400** | −0.007 | single fit, no SD |
| Reward-probe MAE | **0.138–0.141** | 0.187 | — |
| Effective α, median (mean) | 0.109 (0.107) | 0.012 (0.013) | Q-learner α |
| Effective γ, median (mean) | 0.234 (0.342) | **−5.60 (−3.89)** degenerate | Q-learner γ |
| α/γ fit R², median (mean) | **0.190 (0.179)** | 0.005 (0.025) | 100/100 valid fits both |

For the α/γ block the run saved median **and** mean over 100 fits rather than an SD; the median-vs-mean gap is the available dispersion signal. Continuous is tight (0.109 vs 0.107 for α); discrete's γ swings from −5.60 to −3.89, i.e. the fit is not merely biased but unstable.

The probe / α-γ block is the cleanest separation: the continuous context carries a linearly decodable Q-table and reward signal; the discrete context does not (R² ≈ 0, negative γ fit).

---

## 2. Behavioral cloning agreement — ID and OOD families

Fraction of steps matching the ε-greedy Q-learner's action, 10 MDPs each, 8 states × 4 actions (chance = 25%).

| Family | continuous | discrete | Δ (cont − disc) |
|---|---|---|---|
| ID (Beta(2,2), Dir(1)) | 85.8% ± 2.1 | 42.8% ± 9.9 | +43.0 |
| OOD deterministic transitions | 88.6% ± 5.9 | 47.8% ± 20.5 | +40.8 |
| OOD sparse reward Beta(0.1, 2) | 84.4% ± 4.0 | 44.2% ± 11.0 | +40.2 |
| OOD dense reward Beta(10, 10) | 86.6% ± 4.4 | 37.8% ± 12.6 | +48.8 |
| OOD adversarial (det. + Uniform r) | 89.4% ± 5.6 | 64.2% ± 22.8 | +25.2 |
| OOD high-variance Beta(0.1, 0.1) | 83.0% ± 4.7 | 41.8% ± 6.2 | +41.2 |

Continuous holds 83–89% across every shift (max ID gap 3.6 pts). Discrete is 38–64% with 3–5× the variance across MDPs — its OOD numbers are noise around a near-chance policy, not transfer.

---

## 3. High-contrast rewards (OOD) — reward interventions

**Family `contrast`**: one good action per state, so cumulative reward has real dynamic range. Optimal = 200.0, random = 59.5 (29.7% of optimal). 10 MDPs × 200 steps, values are % of optimal.

| Condition | continuous | discrete | ε-greedy Q | greedy Q | random |
|---|---|---|---|---|---|
| intact (control) | **38.4 ± 17.3** | 28.3 ± 8.6 | 44.5 ± 8.4 | 34.9 ± 14.6 | 29.7 |
| zeroed (r ≡ 0) | 27.1 ± 8.8 | 28.1 ± 11.5 | 28.5 ± 3.3 | 28.5 ± 3.3 | 29.7 |
| constant (r ≡ 0.5) | 28.8 ± 14.4 | 27.9 ± 15.7 | 25.5 ± 7.5 | 34.9 ± 14.6 | 29.7 |
| decorrelated (marginal kept, link broken) | 27.0 ± 13.7 | 26.6 ± 8.9 | 23.7 ± 8.8 | 34.9 ± 14.6 | 29.7 |
| inverted (1 − r) | 18.3 ± 8.8 | 28.2 ± 11.9 | 14.4 ± 4.4 | 13.8 ± 8.1 | 29.7 |
| **Sensitivity: intact − ablated mean** | **+13.1 pts** | **+0.6 pts** | +21.5 pts | +6.9 pts | — |
| **Sensitivity: intact − inverted** | **+20.1 pts** | **+0.1 pts** | +30.1 pts | +21.1 pts | — |

Same numbers as SEM (÷√10), since the SDs are large enough that this matters for reading the table:

| Condition | continuous | discrete | ε-greedy Q |
|---|---|---|---|
| intact | **38.4 ± 5.5** | 28.3 ± 2.7 | 44.5 ± 2.7 |
| zeroed | 27.1 ± 2.8 | 28.1 ± 3.6 | 28.5 ± 1.0 |
| constant | 28.8 ± 4.6 | 27.9 ± 5.0 | 25.5 ± 2.4 |
| decorrelated | 27.0 ± 4.3 | 26.6 ± 2.8 | 23.7 ± 2.8 |
| inverted | 18.3 ± 2.8 | 28.2 ± 3.8 | 14.4 ± 1.4 |

This is the sharpest OOD result. Continuous is the only transformer whose contrast-family return depends on the reward channel: destroying reward costs it ~11 pts and inverting it drives it to 18% (below random), the same qualitative signature as the Q-learners. Discrete sits at 27–28% under **every** condition including inversion — it is ignoring reward entirely and playing a near-random policy that the intervention cannot perturb.

Caveat the error bars force: with n = 10 the per-condition SDs are wide (8–31 in raw-return units), so no *single* continuous-vs-discrete cell is individually significant. The continuous intact − inverted contrast is the one comparison that clears its own noise (38.4 ± 5.5 vs 18.3 ± 2.8 SEM, a ~3σ separation on paired MDPs). The discrete claim rests on the *flatness across all five conditions* — five independent estimates all landing in 26.6–28.3 with SEMs of 2.7–5.0 — not on any one cell.

Raw returns (mean ± SD over 10 MDPs), contrast family:

| Condition | continuous | discrete |
|---|---|---|
| intact | 76.88 ± 34.56 | 56.64 ± 17.25 |
| zeroed | 54.27 ± 17.60 | 56.26 ± 22.93 |
| constant | 57.69 ± 28.80 | 55.89 ± 31.32 |
| decorrelated | 54.08 ± 27.43 | 53.22 ± 17.75 |
| inverted | 36.69 ± 17.62 | 56.36 ± 23.74 |

### Contrast with the saturated `eval` family

**Family `eval`** (Beta(2,2), the paper default): optimal = 152.5, random = 105.4 = **69.1% of optimal**. There is only ~31 pts of headroom, so everything collapses together.

Values are % of optimal, ± 1 SD over 10 MDPs.

| Condition | continuous | discrete | ε-greedy Q | greedy Q |
|---|---|---|---|---|
| intact | 71.6 ± 13.4 | 72.5 ± 8.7 | 77.6 ± 7.8 | 74.7 ± 8.8 |
| zeroed | 67.6 ± 11.2 | 71.3 ± 8.4 | 68.6 ± 5.7 | 68.6 ± 5.7 |
| constant | 70.8 ± 10.9 | 71.3 ± 10.6 | 72.7 ± 9.3 | 74.7 ± 8.8 |
| decorrelated | 71.0 ± 13.1 | 71.9 ± 10.0 | 73.5 ± 9.4 | 74.7 ± 8.8 |
| inverted | 67.7 ± 9.9 | 72.1 ± 11.3 | 71.9 ± 9.4 | 74.7 ± 8.8 |
| **intact − inverted** | +3.9 pts | +0.4 pts | +5.8 pts | 0.0 pts |

The error bars make the point better than the means do: every cell in this table is 67–78% with SDs of 6–13, i.e. **the entire `eval` family is one undifferentiated blob**. Discrete (72%) looks *equal or better* than continuous (72%), but neither is separable from the other, from the Q-learners, or from random (69%) at this noise level. Only the contrast family separates them.

---

## 4. Long-horizon (1000 steps)

Final cumulative reward after 1000 autonomous steps. **No SDs available** — the Jul 27 run printed only the mean curve endpoint per agent, so these rows are bare by necessity.

| Agent | `eval` family | % of optimal | `contrast` family | % of optimal |
|---|---|---|---|---|
| optimal | 768.45 | 100% | 1000.00 | 100% |
| ε-greedy Q | 594.43 | 77.4% | 526.04 | 52.6% |
| greedy Q | 573.10 | 74.6% | 357.13 | 35.7% |
| **continuous** | 549.86 | **71.6%** | **418.69** | **41.9%** |
| **discrete** | 548.63 | **71.4%** | **274.01** | **27.4%** |
| random | 517.19 | 67.3% | 287.79 | 28.8% |

Under the saturated `eval` family the two models are indistinguishable (549.9 vs 548.6, both ~2% above random). Under contrast, continuous clears random by +13 pts and beats greedy Q; discrete lands **below random** (27.4% vs 28.8%). Both transformer curves are essentially flat past t≈50 — neither continues improving over the horizon.

Treat the `eval` gap (549.9 vs 548.6, 0.2%) as zero regardless: the 200-step interventions on the same family carry ±13 SD in % terms, so a 0.2-pt separation is far inside the noise even without this run's own SDs. The contrast gap (41.9 vs 27.4) is ~14 pts and survives any plausible error bar, but note it is a **single** 10-MDP average, not a replicated result.

Shorter-horizon reference (200 steps, `eval` family, from the Jul 24 run): continuous 27.71, discrete 28.12, ε-greedy Q 28.64, optimal 38.19 — again indistinguishable on the saturated metric.

Per-reward-distribution 200-step returns (mean ± 1 SD, 10 MDPs) — the one long-horizon-adjacent block that *does* have error bars:

| Agent | return | ± SD | ± SEM |
|---|---|---|---|
| optimal | 47.00 | ± 4.02 | ± 1.27 |
| greedy Q | 33.50 | ± 6.90 | ± 2.18 |
| ε-greedy Q | 32.10 | ± 8.14 | ± 2.57 |
| **discrete** | 24.20 | ± 10.77 | ± 3.41 |
| **continuous** | 21.50 | ± 11.82 | ± 3.74 |

Both transformers trail both Q baselines here, and the two transformers overlap heavily with each other (21.50 ± 3.74 vs 24.20 ± 3.41 SEM) — this block does *not* separate them.

---

## 5. Nonstationary MDPs

MDP is perturbed mid-episode; rates are per-step reward, `%opt` normalizes to the adapting optimal policy. `steps_to_recover` = steps until post-shift rate returns to within tolerance.

**No SDs available in this block** — `nonstationary_summary.csv` has no `_std` columns; the run wrote per-variant means only. Read every number below as a point estimate over 10 MDPs with an unquantified error bar. Given that the intervention block on comparable data shows ±8–17 in % units, differences under ~10 pts here should be treated as noise.

| Variant | Model | pre %opt | post %opt | end %opt | final cumrew | recover |
|---|---|---|---|---|---|---|
| **action_permute** | continuous | 73.7 | 67.3 | 73.9 | 80.69 | 2 |
| | discrete | 74.2 | 72.5 | 71.5 | 81.91 | 1 |
| | ε-greedy Q | 75.9 | 65.5 | 65.3 | 78.92 | — |
| | optimal (frozen) | 100 | 64.8 | 63.3 | 89.27 | — |
| **reward_resample** | continuous | 73.7 | 67.9 | 76.6 | 78.55 | 0 |
| | discrete | 74.2 | 63.7 | 65.7 | 77.89 | 0 |
| | ε-greedy Q | 75.9 | 65.5 | 68.1 | 77.91 | — |
| | optimal (frozen) | 100 | 67.3 | 70.7 | 87.03 | — |
| **transition_resample** | continuous | 73.7 | 71.5 | 75.9 | 82.82 | 1 |
| | discrete | 74.2 | 69.8 | 73.9 | 81.20 | 1 |
| | ε-greedy Q | 75.9 | 77.3 | 79.3 | 86.80 | — |
| | optimal (frozen) | 100 | 99.8 | 100.5 | 113.93 | — |
| **full_resample** | continuous | 73.7 | 68.5 | 60.9 | 79.43 | 63 |
| | discrete | 74.2 | 71.9 | 67.2 | 78.45 | 0 |
| | ε-greedy Q | 75.9 | 67.9 | 64.4 | 78.70 | — |
| | optimal (frozen) | 100 | 65.0 | 61.9 | 83.37 | — |
| **contrast_permute** (OOD) | **continuous** | **36.3** | 23.0 | **47.7** | **56.23** | 24 |
| | **discrete** | **25.9** | 26.8 | 37.3 | 41.98 | 3 |
| | ε-greedy Q | 35.4 | 33.5 | 35.4 | 51.77 | 0 |
| | greedy Q | 29.7 | 26.8 | 38.2 | 47.59 | 14 |
| | random | 25.9 | 30.6 | 26.8 | 42.84 | 0 |
| | optimal (adapts) | 100 | 100 | 100 | 150.00 | 0 |

The first four variants are on the saturated reward family, where every agent — including *random* (70–73% of optimal) and *frozen optimal* — lands in the same 60–80% band. Those rows carry no signal about competence.

`contrast_permute` is the only discriminative row: continuous starts at 36.3% (vs random 25.9%) and ends at 47.7% with final cumreward 56.23; discrete starts at exactly the random rate (25.9%) and ends at 41.98 cumreward, indistinguishable from random's 42.84. Neither transformer approaches the adapting optimum (150.0). Without SDs this row is suggestive rather than established — but it agrees with the intervention and long-horizon blocks, which do carry error bars, so it is best read as corroboration rather than independent evidence.

---

## 6. Size sweep (|S| ∈ 2..8 × |A| ∈ 2..4, 10 MDPs/cell, 200 steps)

Two distinct dispersions here: **across-cell** SD (how much the grid position matters) and **within-cell** SD (spread over the 10 MDPs in one cell, from `agreement_std`).

| Aggregate over 21 cells | continuous | discrete |
|---|---|---|
| Mean agreement (± across-cell SD) | **93.6% ± 3.9** | 62.1% ± 13.8 |
| Mean *within*-cell SD (10 MDPs) | **± 2.0** | ± 10.0 |
| Mean agreement above chance | **+57.5 pts** | +26.0 pts |
| Worst-cell agreement | **85.8% ± 2.1** (8×4) | 41.2% ± 12.0 (7×4) |
| Best-cell agreement | 99.2% ± 1.0 (2×2) | 90.2% ± 6.5 (2×2) |
| Mean return, % of optimal (± across-cell SD) | **47.2% ± 16.7** | 35.6% ± 13.4 |
| Mean random baseline | 39.5% | 39.5% |
| Cells beating random return | **16 / 21** | 7 / 21 |

Per-cell return has **no within-cell SD** — only the cell mean was written to the CSV, so the ±16.7 / ±13.4 above is across-cell scatter, not eval noise.

Per-cell agreement, mean ± 1 SD over 10 MDPs (continuous / discrete):

| \|S\| \ \|A\| | 2 | 3 | 4 |
|---|---|---|---|
| **2** | 99.2 ± 1.0 / 90.2 ± 6.5 | 98.6 ± 0.9 / 85.0 ± 11.9 | 98.2 ± 0.6 / 80.4 ± 12.2 |
| **3** | 97.0 ± 1.3 / 81.2 ± 11.0 | 96.0 ± 1.3 / 70.2 ± 9.1 | 96.6 ± 0.9 / 58.6 ± 13.7 |
| **4** | 96.4 ± 2.5 / 68.6 ± 10.5 | 96.0 ± 1.5 / 69.2 ± 10.8 | 96.4 ± 1.7 / 58.0 ± 10.7 |
| **5** | 95.6 ± 2.3 / 65.6 ± 10.2 | 93.0 ± 3.3 / 52.2 ± 8.3 | 93.2 ± 1.8 / 52.0 ± 14.9 |
| **6** | 94.6 ± 3.0 / 65.6 ± 7.3 | 92.2 ± 3.2 / 53.0 ± 8.7 | 90.4 ± 2.0 / 46.6 ± 8.2 |
| **7** | 92.2 ± 1.9 / 60.4 ± 10.8 | 89.6 ± 2.3 / 50.8 ± 8.9 | 89.6 ± 2.3 / 41.2 ± 12.0 |
| **8** | 88.6 ± 4.8 / 61.2 ± 5.0 | 87.4 ± 2.0 / 50.6 ± 10.3 | 85.8 ± 2.1 / 42.8 ± 9.9 |

Continuous degrades gracefully (99.2 → 85.8 over the whole grid). Discrete falls from 90.2 to 42.8 — at |S|≥5, |A|=4 it is within ~17 pts of the 25% chance floor and its return never beats random.

**This is the block where the error bars carry real weight**, because unlike the intervention table the separation is enormous relative to the noise: every one of the 21 cells has continuous above discrete by 9–38 pts against within-cell SDs of 0.6–4.8 (continuous) and 5.0–14.9 (discrete). Not one cell overlaps. The 5× gap in within-cell SD (2.0 vs 10.0) is itself a finding: continuous gives the *same* answer on every MDP in a cell, while discrete's policy quality swings wildly seed to seed — consistent with it having learned no transferable rule.

---

## 7. Summary

1. **Continuous context dominates on every discriminative metric**: agreement (85.8 ± 2.1 vs 42.8 ± 9.9 ID; 93.6 ± 3.9 vs 62.1 ± 13.8 swept), probes (Q R² 0.66 vs 0.02; reward R² 0.38 vs −0.01), and contrast-family return (41.9% vs 27.4% at 1000 steps).
2. **The discrete model is at or below random wherever the metric has dynamic range**: contrast long-horizon 27.4% vs random 28.8%, contrast interventions flat at 26.6–28.3% across all five conditions, `contrast_permute` cumreward 41.98 vs random 42.84.
3. **Reward-channel causality is the sharpest test.** Inverting reward costs continuous 20.1 pts on contrast and moves discrete by +0.1 pts. Discrete's policy does not read reward.
4. **Beware the `eval` family.** Random scores 69.1% of optimal there, so discrete "matches" continuous (72% vs 72% intact; 548.6 vs 549.9 at 1000 steps). With ±8–13 SD on every cell the whole family is one blob — that equality is a benchmark artifact, not parity.
5. **Neither model is competitive with the Q-learners it imitates** on the discriminative family: ε-greedy Q hits 52.6% of optimal at 1000 contrast steps vs continuous 41.9%, and both transformers are flat past t≈50.
6. **Where the evidence is strongest, statistically.** Ranked by separation-to-noise: the size sweep is decisive (21/21 cells separated, zero overlap); ID/OOD agreement is decisive (85.8 ± 2.1 vs 42.8 ± 9.9 is a >4σ gap); the contrast interventions are suggestive per-cell but convincing as a pattern (discrete flat across five conditions); long-horizon and nonstationary have **no saved SDs** and should be cited as corroborating, not load-bearing. If any block is worth re-running with per-seed logging, it is the 1000-step long horizon — it produces the headline 41.9-vs-27.4 number and is currently the least defensible.
