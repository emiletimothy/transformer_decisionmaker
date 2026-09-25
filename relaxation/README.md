# Relaxing the idealized assumptions of the handwired constructions

Answers the reviewer point that the constructions rely on orthonormal embeddings, disjoint
buffer subspaces, fixed-offset routing and hard / limiting-softmax attention.

| File | |
|---|---|
| `relax_common.py` | Gram-matrix framework: weights written through embeddings `W' = Phi W Phi^T`, track `y = Phi^T x'`; `G = Phi^T Phi = I` is the exact construction |
| `relax_q.py` | runs the **existing** Q construction (`tabular_q_learning/scripts/handwired_q_learning_v1.py`, matrices unchanged) with relaxations; recurrent context carried as a raw vector |
| `relax_mwu.py` | **new** matrix-form Weighted-Majority construction (the original `transformer_handwired_multiplicative_weights.py`, since removed, was Python loops, not attention, so it can't be relaxed) |
| `run_sweeps.py`, `run_sweeps.sbatch` | all sweeps; 20 instances each, T = 500 (also reported at T = 100) |
| `plot_relaxation.py` → `figures/relaxation_sweeps.{pdf,png}` | 2x4 figure (its panel (d) shows tied read-out only) |
| `plot_relaxation_split.py` → `figures/relaxation_{a,b,c,d}.{pdf,png}` | one figure per sweep; (d) adds the dual read-out |
| `results/{mwu,q}_relaxation.csv` / `_raw.csv` | mean / SEM per setting, and per-instance values |

Setup: MWU with n = 4 experts, qualities U[0.3, 0.9], eta = 0.1. Q-learning on random
6-state x 3-action MDPs (Dir(1) transitions, Beta(2,2) rewards), alpha = 0.1, gamma = 0.9,
eps-greedy trajectories. Every head is causal.

## Findings (T = 500, mean over 20 instances)

- **Sanity check.** Exact execution matches the algorithm: MWU error 0, Q error 2.3e-7.
- **(a) Hardness.** Near-hard attention is not needed. β ≥ 20 is exact to float precision
  for both routing (MWU 1e-6, Q 8e-6) and the MWU mask. The Q max head (soft-max backup)
  has error 5.6e-3 at β = 50 and 5.9e-6 at β = 10³. β ≤ 5 fails.
- **(b) Noise.** Error grows smoothly (linearly) with σ; there is no cliff.
  - Q: 2e-3 at σ = 1e-5, 2e-2 at 1e-4, 0.2 at 1e-3, and greedy agreement stays ≥ 0.999
    up to σ = 1e-3.
  - MWU: moderate β is best. At β = 50 the error is 9e-4, 9e-3 and 0.09 for σ = 1e-5,
    1e-4 and 1e-3. Very hard heads amplify noise: at β = 10⁶ the subtractive mask turns
    σ = 1e-5 into logit noise of about 10, and decision agreement drops to 0.70. So the
    idealized limit is the *least* robust setting; a band of moderate β is exact and robust.
- **(c) Non-orthogonal embeddings.**
  - Tied read-out (Φᵀ, i.e. reading with the same non-orthogonal embeddings) fails at
    every d up to 16384 (max |cos| ≈ 0.02). Each recurrent step multiplies the carried
    state by the Gram matrix, so the error compounds. MWU error is about 0.8–1.0; Q
    diverges (inf).
  - Dual-basis read-out (Φ⁺ = (ΦᵀΦ)⁻¹Φᵀ) is exact for any linearly independent embeddings
    (d ≥ vocabulary size). It then behaves like the noise sweep, with only mild
    amplification near d ≈ V: Q error 2.2e-3 at σ = 1e-5 for d ≥ 256, and 7.5e-3 at d = 16.
  - Takeaway: orthonormality is without loss of generality given linearly independent
    embeddings. The theory does not cover superposition (V > d, near-orthogonal tied
    embeddings), which is where it breaks.
- **(d) Buffer overlap.** Buffer subspaces tilted toward a shared subspace, max cross-block
  |cos| = ρ. Two read-outs, as in (c):
  - Tied read-out fails (ρ = 0.01: MWU error 0.44, Q error 18 with half the runs diverging;
    ρ ≥ 0.1 diverges). This is the same failure as tied read-out in (c), not a separate
    buffer effect. The per-step error is only O(ρ), but the carried state is multiplied by
    the Gram matrix every step, so it compounds: at ρ = 0.01 the Q error grows from 1.8 at
    T = 100 to 17.8 at T = 500.
  - Dual read-out is exact for every ρ ≤ 0.5 (MWU 0, Q 6e-6 at σ = 0). With noise, the
    error barely moves: at σ = 1e-4, Q goes from 0.022 at ρ = 0 to 0.029 at ρ = 0.5.
  - Takeaway: disjoint buffers are not load-bearing. Overlapping but linearly independent
    buffers are equivalent up to an invertible change of basis. What breaks both (c) and (d)
    is reading the state back through the same non-orthogonal directions it was written in.

## Caveats
- **Idealized gating (Q only).** Which positions a Q head writes to is taken from an exact
  pass (the FO trigger and the `q_active` gate in the original code). This is equivalent to
  an attention sink. MWU has no gating.
- **Q fixed-offset heads** are relaxed as softmax over positions with logit β at the target
  offset, with no positional-embedding noise. MWU uses real positional attention.
- **MWU layout differs from the paper text, to make it causal.** Head 1.1 fetches the
  *previous* token (the paper fetches the next one, which a causal mask forbids). The latent
  superposition sits in its own block, with a marker id on z. Labels use separate tokens.
- **Tied vs dual read-out in (c) and (d)** are the two natural ways to write the weights;
  the gap between them is the finding.

## Residual-write constructions (2026-09-25)

`--residual` (run_sweeps.py, plot_relaxation_split.py) runs the residual-write versions:
Q-learning drops Head 3.3 and writes c_{a_t} <- c_{a_t} + P_buf1 h[Update]; MWU restricts
Head 2.1 to <p?> (other rows attend to an attention sink, <BOS>) and writes
lat(z_{t+1}) = lat(z_t) + buf2(<w?>). Outputs: `results/{mwu,q}_relaxation_residual{,_raw}.csv`,
`figures/relaxation_residual_{a,b,c,d}.{pdf,png}` (the old matrix-form MWU weight figures were removed;
the current one is multiplicative_weights/figures/handwired/).

- Exact execution: identical to overwrite (MWU error 0; Q 2.3e-7 at T = 500).
- (c) Tied read-out still degrades but no longer compounds as badly: MWU error falls with d
  (0.44 at d = 1024, 0.16 at 16384; overwrite 0.8-1.0 at every d); Q still diverges in some
  instances up to d = 8192 and reaches 1.4 without divergence at 16384 (overwrite diverges at
  every d). The per-step reads through Phi^T still inject errors into the increment, and the
  Q increment depends on the stored values (-alpha Q, max Q), so errors feed back.
- (d) Overlap with tied read-out still breaks both (MWU 0.33 at rho = 0.01, diverges from
  rho = 0.1; Q diverges from rho = 0.01). Dual read-out is unaffected by the write rule.
- Caveat: these sweeps use the v1-style matrices (relax_q.py reuses
  handwired_q_learning_v1.py; relax_mwu.py has a linear-attention Head 3.2). The
  standard-components constructions are tabular_q_learning/scripts/handwired_q_learning.py
  and multiplicative_weights/scripts/handwired_mwu.py (not yet swept).
