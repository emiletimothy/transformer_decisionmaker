"""
Relaxed execution of the current handwired constructions (standard components only):
tabular_q_learning/scripts/handwired_q_learning.py (HandwiredQv2, two buffers) and
multiplicative_weights/scripts/handwired_mwu.py (HandwiredMWUv2, one buffer).

Both constructions are plain weight matrices (causal softmax heads, ReLU MLPs, residual
additions, a linear latent write), so every relaxation is applied generically in the
framework of relax_common.py: weights written through the embeddings, W' = Phi W Phi^T,
state tracked as y = Phi^T x'. Then

    token input      y  = G x_canon                   (the carried slots / latent are already y)
    sublayer write   y += G out  (+ noise ~ N(0, sigma^2 Gnoise) after every sublayer)
    latent write     c += G W_ctx y[UPDATE]

with G = Phi^T Phi over the token-direction blocks (the identity block and each buffer);
positions and the constant sink coordinate are not relaxed. G = I is the exact
construction; dual read-out (Phi^+) is G = I with noise covariance G^{-1}. Unlike the
earlier sweeps of the v1 Q construction, no head's write positions are taken from an exact
pass: the constructions need no gating outside the network.
"""
import os
import sys

import numpy as np

from relax_common import psd_sqrt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tabular_q_learning", "scripts"))
sys.path.insert(0, os.path.join(HERE, "..", "multiplicative_weights", "scripts"))
from handwired_q_learning import HandwiredQv2, random_mdp_trajectory  # noqa: E402,F401
from handwired_mwu import HandwiredMWUv2  # noqa: E402


def full_gram(d, token_blocks, Gb):
    """Embed a Gram matrix Gb over len(token_blocks) blocks of equal size into R^{d x d};
    coordinates outside the blocks (positions, the sink coordinate) keep G = I."""
    G = np.eye(d)
    k = token_blocks[0].stop - token_blocks[0].start
    for i, bi in enumerate(token_blocks):
        for j, bj in enumerate(token_blocks):
            G[bi, bj] = Gb[i * k:(i + 1) * k, j * k:(j + 1) * k]
    return G


def relaxed_forward(model, Y, G, sigma, rng, Gsqrt):
    def add(Y, out):
        Y = Y + out @ G.T
        if sigma > 0:
            Y = Y + sigma * rng.standard_normal(Y.shape) @ Gsqrt.T
        return Y
    for heads, mlp in model.blocks:
        Y = add(Y, sum(h(Y) for h in heads))
        if mlp is not None:
            Y = add(Y, mlp(Y))
    return Y


class RelaxedQv2:
    def __init__(self, n_states=6, n_actions=3, alpha=0.1, gamma=0.9):
        self.nS, self.nA, self.alpha, self.gamma = n_states, n_actions, alpha, gamma
        probe = HandwiredQv2(alpha, gamma, max_states=n_states, max_actions=n_actions)
        lay = probe.lay
        self.dTE, self.d = lay.V, lay.d
        self.token_blocks = [lay.id, lay.B1, lay.B2]

    def run(self, traj, G, beta=1e3, beta_max=1e4, sigma=0.0, seed=0, report_T=(100, 500),
            Gnoise=None):
        # rewards are Beta(2,2) in [0, 1]: the gate constant of the theorem, 2 R_max / (1 - gamma)
        m = HandwiredQv2(self.alpha, self.gamma, beta=beta, beta_max=beta_max,
                         max_states=self.nS, max_actions=self.nA, C_gate=2.0 / (1 - self.gamma))
        lay, nS, nA = m.lay, self.nS, self.nA
        P = lay.positions(nA)
        rng = np.random.default_rng(seed)
        Gsqrt = psd_sqrt(G if Gnoise is None else Gnoise) if sigma > 0 else None
        ctx = np.zeros((nA, lay.d))                    # slots, tracked coordinates
        Qtab = np.zeros((nS, nA))
        max_err, agree, n_agree, sel_agree, n_sel = 0.0, 0, 0, 0, 0
        out = {}
        for t, (s, a, r, s2) in enumerate(traj, start=1):
            Qhat = ctx[:, lay.SLOT][:, :nS].T
            a_star = int(np.argmax(Qhat[s2])) if np.all(np.isfinite(Qhat[s2])) else 0
            X = m.embed(np.zeros((nA, lay.d)), s, a, r, s2, a_star)
            Y = X @ G.T
            Y[1:1 + nA] += ctx
            Yo = relaxed_forward(m, Y, G, sigma, rng, Gsqrt)
            srt = np.sort(Qtab[s2])
            if srt[-1] - srt[-2] > 1e-6:
                sel_agree += int(np.argmax(Yo[P['sel'], lay.ASEL][:nA]) == np.argmax(Qtab[s2]))
                n_sel += 1
            Qtab[s, a] += self.alpha * (r + self.gamma * Qtab[s2].max() - Qtab[s, a])
            ctx[a] = ctx[a] + G @ (m.W_ctx @ Yo[P['upd']])
            Qhat = ctx[:, lay.SLOT][:, :nS].T
            err = float(np.abs(Qhat - Qtab).max())
            if not np.isfinite(err):
                err = np.inf
            max_err = max(max_err, err)
            srt = np.sort(Qtab, axis=1)
            uniq = (srt[:, -1] - srt[:, -2]) > 1e-6
            agree += int((np.argmax(Qhat, 1) == np.argmax(Qtab, 1))[uniq].sum())
            n_agree += int(uniq.sum())
            if t in report_T:
                out[t] = dict(max_err=max_err, final_err=err, greedy_agree=agree / max(n_agree, 1),
                              select_agree=sel_agree / max(n_sel, 1))
            if not np.isfinite(err):                   # diverged: report and stop
                for tt in report_T:
                    if tt >= t and tt not in out:
                        out[tt] = dict(max_err=np.inf, final_err=np.inf, greedy_agree=np.nan,
                                       select_agree=np.nan)
                break
        return out


class RelaxedMWUv2:
    def __init__(self, n_experts=4, eta=0.1):
        self.n, self.eta = n_experts, eta
        probe = HandwiredMWUv2(n_experts, eta)
        self.dTE = probe.buf.stop - probe.buf.start
        self.d = probe.d
        self.token_blocks = [probe.id, probe.buf]

    def run(self, preds_seq, labels, G, beta=1e3, sigma=0.0, seed=0, report_T=(100, 500),
            Gnoise=None):
        m = HandwiredMWUv2(self.n, self.eta, beta)
        rng = np.random.default_rng(seed)
        Gsqrt = psd_sqrt(G if Gnoise is None else Gnoise) if sigma > 0 else None
        M = np.zeros(m.d)
        lam = np.zeros(self.n)
        res = {}
        max_perr, max_linf, n_agree, n_valid, mist_tf, mist_ex = 0.0, 0.0, 0, 0, 0, 0
        for t in range(1, len(labels) + 1):
            p, yl = preds_seq[t - 1], labels[t - 1]
            Y = m.embed(np.zeros(m.d), p, yl) @ G.T
            Y[0] += M
            Yo = relaxed_forward(m, Y, G, sigma, rng, Gsqrt)
            w = np.exp(lam - lam.max()); w /= w.sum()
            p_ex = float(w[np.array(p) == 1].sum())
            p_tf = float(Yo[m.p_sep, m.PRED])
            dec_tf, dec_ex = int(p_tf > 0.5), int(p_ex > 0.5)
            if abs(p_ex - 0.5) > 1e-9:
                n_valid += 1
                n_agree += int(dec_tf == dec_ex)
            mist_tf += int(dec_tf != yl)
            mist_ex += int(dec_ex != yl)
            max_perr = max(max_perr, abs(p_tf - p_ex)) if np.isfinite(p_tf) else np.inf
            lh = M[m.LAT]
            wh = np.exp(lh - lh.max()); wh /= wh.sum()
            linf = float(np.abs(wh - w).max())
            max_linf = max(max_linf, linf) if np.isfinite(linf) else np.inf
            lam = lam + self.eta * (np.array(p) == yl)
            M = M + G @ (m.W_ctx @ Yo[m.p_upd])
            if t in report_T:
                lh = M[m.LAT]
                wh = np.exp(lh - lh.max()); wh /= wh.sum()
                w2 = np.exp(lam - lam.max()); w2 /= w2.sum()
                res[t] = dict(max_pred_err=max_perr, w_linf=float(np.abs(wh - w2).max()),
                              w_l1=float(np.abs(wh - w2).sum()),
                              max_w_linf=max(max_linf, float(np.abs(wh - w2).max())),
                              decision_agree=n_agree / max(n_valid, 1),
                              regret_diff=mist_tf - mist_ex)
        return res
