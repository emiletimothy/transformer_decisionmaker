#!/usr/bin/env python3
"""
eval_write_decomposition.py - why the continuous slots' norm grows in closed loop.

Hooks the learned write layer (context_delta) during the 1000-step closed-loop rollout
(model's own a*, contrast MDPs) and splits every write W LN(h[UPDATE]) + b into the part
along the writes' common direction and the rest. Constant-size, partly aligned writes
give the linear slot-norm growth seen in eval_drift.py.

  python3 eval_write_decomposition.py [--model continuous_residual] [--T 1000] [--n_mdps 8]
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import paths  # noqa: E402
_spec = importlib.util.spec_from_file_location('rv2', os.path.join(_HERE, 'eval_closed_loop.py'))
RV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RV)
E = RV.E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='continuous_residual', choices=('continuous_residual',))
    ap.add_argument('--T', type=int, default=1000)
    ap.add_argument('--n_mdps', type=int, default=8)
    args = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, config = RV.load_model(str(paths.checkpoint(args.model)), dev)
    vocab = E.build_vocab(config.max_states, config.max_actions)
    ns, na, T = config.max_states, config.max_actions, args.T
    seeds = list(range(9999, 9999 + args.n_mdps))
    phases = [[(*E.generate_contrast_mdp_seeded(ns, na, seed=sd), T)] for sd in seeds]
    rec = []
    model.context_delta.register_forward_hook(lambda mod, inp, out: rec.append(out.detach().clone()))
    RV.run_tf_batched(model, config, vocab, phases, ns, na, dev, seeds, astar_mode='self')
    D = torch.stack(rec)                     # [calls, B, d]
    if D.shape[0] >= 2 * T:
        D = D[1::2]                          # 'self' mode runs two passes per step; keep the write pass
    D = D[:T]
    n = D.norm(dim=-1)
    mean_vec = D.mean(dim=(0, 1))
    u = mean_vec / mean_vec.norm()
    proj = D @ u
    resid = D - proj[..., None] * u
    print(f'mean |write| = {n.mean():.2f}   |mean write| = {mean_vec.norm():.2f}   '
          f'|bias| = {model.context_delta.bias.norm():.3f}')
    print(f'along the common direction: mean {proj.mean():.2f} (sd {proj.std():.2f});  '
          f'orthogonal remainder: mean norm {resid.norm(dim=-1).mean():.2f}')
    for a, b in [(0, 100), (100, 200), (200, 500), (500, T)]:
        print(f'  steps {a + 1}-{b}: |write| {n[a:b].mean():.2f}, along common direction {proj[a:b].mean():.2f}')
    k = T / na
    print(f'expected slot norm after {k:.0f} writes: coherent {k * proj.mean():.0f}, '
          f'random walk {np.sqrt(k) * resid.norm(dim=-1).mean():.0f}')


if __name__ == '__main__':
    main()
