# Relaxing the idealized assumptions of the handwired constructions

Answers the reviewer point that the constructions rely on orthonormal embeddings, disjoint
buffer subspaces, fixed-offset routing and hard / limiting-softmax attention.

| File | |
|---|---|
| `relax_common.py` | Gram-matrix framework: weights written through embeddings `W' = Phi W Phi^T`, track `y = Phi^T x'`; `G = Phi^T Phi = I` is the exact construction |
| `relax_v2.py` | runs the **current** constructions (`tabular_q_learning/scripts/handwired_q_learning.py`, two buffers; `multiplicative_weights/scripts/handwired_mwu.py`, one buffer) with every relaxation applied generically: they are plain weight matrices with no gating outside the network |
| `relax_q.py`, `relax_mwu.py` | the earlier sweeps of the v1-style matrices (`run_sweeps.py --v1`) |
| `run_sweeps.py`, `run_sweeps.sbatch` | all sweeps; 20 instances each, T = 500 (also reported at T = 100) |
| `plot_relaxation.py` → `figures/relaxation_sweeps.{pdf,png}` | 2x4 figure (its panel (d) shows tied read-out only) |
| `plot_relaxation_split.py` → `figures/relaxation_{a,b,c,d}.{pdf,png}` | one figure per sweep; (d) adds the dual read-out |
| `results/{mwu,q}_relaxation.csv` / `_raw.csv` | mean / SEM per setting, and per-instance values |
| `results/earlier_runs/`, `figures/earlier_runs/` | the v1 sweeps (`_v1` overwrite write, `_v1_residual` residual write) |

Setup: MWU with n = 4 experts, qualities U[0.3, 0.9], eta = 0.1. Q-learning on random
6-state x 3-action MDPs (Dir(1) transitions, Beta(2,2) rewards), alpha = 0.1, gamma = 0.9,
eps-greedy trajectories; the Q construction uses the theorem's gate constant
C = 2 R_max / (1 - gamma) = 20. Relaxations act on the identity block and every buffer
(positions and the sink coordinate are not relaxed); noise is added after every sublayer.

## Findings (current constructions, T = 500, mean over 20 instances)

- **Exact execution.** MWU prediction error 8e-14; Q error 2.5e-11.
- **(a) Hardness.** Near-hard attention is not needed. MWU is exact from beta = 20 (1e-7) and
  to float precision from beta = 50. Q routing is exact from beta = 50 (8e-9); the Q max head
  (soft-max backup) has error 1.8e-3 at beta_max = 50, 4e-6 at 1e3 and 3e-11 at 1e4.
  beta <= 10 fails for Q routing (error 3.1).
- **(b) Noise.** Error grows linearly with sigma; there is no cliff.
  - Q: 3.6e-3, 3.6e-2 and 0.28 at sigma = 1e-5, 1e-4, 1e-3 (beta = 50 or 1e3); greedy agreement
    stays >= 0.99 up to sigma = 1e-3. The gate constant sets the slope (noise on a 0/1 selector is
    multiplied by C): with the loose C = 100 of the verification script the errors are ~5x larger (a separate
    check at beta = 1e3, 4 instances; not in the CSVs).
  - MWU: at beta = 50, 3.4e-3, 3.4e-2 and 0.31. Very hard heads amplify noise: at beta = 1e6 the
    error is already ~1 at sigma = 1e-5, so the idealized limit is the least robust setting.
- **(c) Non-orthogonal embeddings.**
  - Tied read-out (Phi^T) fails: MWU error falls with d (44 at d = 32, 0.6 at d = 16384 at beta = 50, decision
    agreement 0.90) but is never exact; Q's error exceeds 1e7 for d <= 2048 (non-finite in some
    instances at d = 32) and stays at ~4-7 beyond.
  - Dual read-out (Phi^+) is exact for any linearly independent embeddings, then behaves like the
    noise sweep with mild amplification near d ~ V (Q: 2.0e-2 at d = 16, 3.6-3.7e-3 for d >= 256 at
    sigma = 1e-5; MWU: 5.7e-3 at d = 16, 3.4-3.5e-3 for d >= 512).
- **(d) Buffer overlap** (max cross-block |cos| = rho).
  - Tied read-out fails (Q error 2.7 at rho = 0.01, diverging from rho = 0.1; MWU diverges at
    every rho > 0, the latent and the prediction sharing one buffer).
  - Dual read-out is exact for every rho <= 0.5, and noise barely moves it (Q at sigma = 1e-4:
    0.036 at rho = 0, 0.043 at rho = 0.5; MWU 0.034 and 0.038).
- **Takeaway.** Orthonormal embeddings and disjoint buffers are without loss of generality given
  linearly independent embeddings (dual read-out); moderate softmax hardness suffices; errors grow
  linearly with residual noise. What breaks is superposition-style read-out through the same
  non-orthogonal directions the state was written in (tied read-out), which compounds through the
  recurrent state.

The v1 sweeps (earlier_runs/) gave the same picture; their Q runs took each head's write positions
from an exact pass, which the current constructions do not need.
