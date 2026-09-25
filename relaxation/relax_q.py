"""
Relaxed execution of the handwired tabular Q-learning construction
(tabular_q_learning/scripts/handwired_q_learning_v1.py).

The weight matrices are taken verbatim from ContinuousThoughtQTransformer; only the
execution is changed so that the following idealizations can be relaxed:

  * orthonormal token embeddings       -> Gram matrix G of random unit vectors in R^d
  * disjoint id / buf1 / buf2 subspaces -> overlapping blocks (cross-block cos <= rho)
  * hard routing (HARD_TEMP=1e6, exact fixed-offset shift) -> softmax at inverse temp beta
  * limiting softmax in the max head (beta=1e3)            -> finite beta
  * exact arithmetic                    -> Gaussian noise on the residual stream per layer

The recurrent context is carried as a raw continuous vector (not decoded and re-encoded),
so errors accumulate across steps exactly as they would in the autoregressive model.

residual=True runs the residual-write construction (Head 3.3 dropped; see
handwired_q_learning_v1.py): c_{a_t} <- c_{a_t} + P_buf1 h[Update], so the carried
slot never passes through the network and only the increment is written through the
embeddings.

Caveat (kept idealized): *which* positions a head writes to (the FO trigger and the
q_active gate in the original code) is taken from an exact pass on the same step.  In a
standard transformer this corresponds to an attention sink (e.g. BOS with zero value)
absorbing attention from non-query positions.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tabular_q_learning", "scripts"))
import handwired_q_learning_v1 as H  # noqa: E402

from relax_common import softmax_rows  # noqa: E402


def _np(t):
    return t.detach().double().numpy()


class RelaxedQ:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, residual=False):
        cfg = H.QLearningConfig(n_states=n_states, n_actions=n_actions, alpha=alpha, gamma=gamma,
                                residual=residual)
        self.residual = residual
        tf = H.ContinuousThoughtQTransformer(cfg)
        self.tf = tf
        self.cfg = cfg
        self.dTE, self.d = tf.d_TE, tf.d_model
        self.layers = []
        for layer in [tf.l1, tf.l2, tf.l3, tf.l4]:
            heads = []
            for h in layer:
                if isinstance(h, H.FixedOffsetChooser):
                    heads.append(dict(kind="fo", T=list(h.T), off=h.offset,
                                      WV=_np(h.W_V), WO=_np(h.W_O)))
                elif isinstance(h, H.SoftmaxHead):
                    heads.append(dict(kind="soft", WQ=_np(h.WQ), WK=_np(h.WK), WV=_np(h.WV),
                                      WO=_np(h.WO), temp=h.temp, causal=h.causal,
                                      routing=(h.temp == H.HARD_TEMP)))
                else:
                    heads.append(dict(kind="lin", WQ=_np(h.WQ), WK=_np(h.WK), WV=_np(h.WV),
                                      WO=_np(h.WO), per_token=h.per_token))
            self.layers.append(heads)

    # ------------------------------------------------------------------ forward
    def forward(self, y, G, beta_route=None, beta_max=None, sigma=0.0, rng=None,
                masks=None, force_causal=False):
        """
        y: [L, d] input in Phi^T coordinates.  beta_route=None -> exact routing
        (HARD_TEMP softmax and exact fixed-offset shift).  beta_max=None -> original 1e3.
        masks: list of per-head gate vectors from an exact pass (or None to compute them
        here, which is what the exact pass does).  Returns (y_out, masks).
        """
        L = y.shape[0]
        idx = np.arange(L)
        causal = idx[None, :] <= idx[:, None]
        rec = [] if masks is None else None
        k = 0
        for heads in self.layers:
            out = np.zeros_like(y)
            for h in heads:
                if h["kind"] == "fo":
                    if masks is None:
                        trig = np.zeros(L)
                        for t in h["T"]:
                            trig += (y[:, t] > 0).astype(float)
                        rec.append(trig)
                    else:
                        trig = masks[k]
                    V = y @ h["WV"].T
                    off = h["off"]
                    if beta_route is None:
                        A = np.zeros((L, L))
                        A[idx[off:], idx[off:] - off] = 1.0
                    else:
                        S = np.zeros((L, L))
                        S[idx[off:], idx[off:] - off] = beta_route
                        A = softmax_rows(S, causal)
                        A[:off] = 0.0  # no target exists for the first `off` rows
                    out += (trig[:, None] * (A @ V)) @ h["WO"].T
                elif h["kind"] == "soft":
                    Q = y @ h["WQ"].T
                    K = y @ h["WK"].T
                    V = y @ h["WV"].T
                    if h["routing"]:
                        temp = h["temp"] if beta_route is None else beta_route
                    else:
                        temp = 1.0 if beta_max is None else beta_max / H.BETA
                    S = (Q @ K.T) * temp
                    m = causal if (h["causal"] or force_causal) else None
                    A = softmax_rows(S, m)
                    if masks is None:
                        gate = (np.abs(Q).sum(-1) > 1e-6).astype(float)
                        rec.append(gate)
                    else:
                        gate = masks[k]
                    out += (gate[:, None] * (A @ V)) @ h["WO"].T
                else:
                    Q = y @ h["WQ"].T
                    K = y @ h["WK"].T
                    V = y @ h["WV"].T
                    if h["per_token"]:
                        o = (Q * K).sum(-1, keepdims=True) * V
                    else:
                        o = (Q @ K.T) @ V
                    out += o @ h["WO"].T
                    if masks is None:
                        rec.append(None)
                k += 1
            y = y + out @ G.T
            if sigma > 0:
                y = y + sigma * rng.standard_normal(y.shape) @ self._Gsqrt.T
        return y, (rec if masks is None else masks)

    # ------------------------------------------------------------------ helpers
    def _onehot(self, i):
        v = np.zeros(self.d)
        v[i] = 1.0
        return v

    def ctx_canon(self, a):
        tf = self.tf
        return self._onehot(tf.A[a]) + self._onehot(tf.UPDATE)

    def step_tokens_canon(self, s, a, r, s_next, a_star):
        tf, nA = self.tf, self.cfg.n_actions
        toks = [self._onehot(tf.QCURR), self._onehot(tf.S[s]), self._onehot(tf.A[a])]
        rt = self._onehot(tf.R)
        rt[self.dTE + tf.R] = r
        toks += [rt, self._onehot(tf.QNEXT)]
        for b in range(nA):
            toks += [self._onehot(tf.S[s_next]), self._onehot(tf.A[b])]
        toks += [self._onehot(tf.SELECT), self._onehot(tf.A[a_star]), self._onehot(tf.UPDATE)]
        return np.stack(toks)

    def decode_Q(self, ctx):
        tf = self.tf
        return np.array([[ctx[a][self.dTE + tf.S[s]] for a in range(self.cfg.n_actions)]
                         for s in range(self.cfg.n_states)])

    # ------------------------------------------------------------------ rollout
    def run(self, traj, G, beta_route=None, beta_max=None, sigma=0.0, seed=0,
            force_causal=False, report_T=(100, 500), Gnoise=None):
        """Gnoise: covariance (/sigma^2) of the per-layer noise in the tracked coordinates;
        defaults to G (tied read-out Phi^T).  Dual read-out Phi^+ uses G=I, Gnoise=G^{-1}."""
        nS, nA = self.cfg.n_states, self.cfg.n_actions
        alpha, gamma = self.cfg.alpha, self.cfg.gamma
        rng = np.random.default_rng(seed)
        self._Gsqrt = None
        if sigma > 0:
            from relax_common import psd_sqrt
            self._Gsqrt = psd_sqrt(G if Gnoise is None else Gnoise)
        I = np.eye(self.d)
        P_buf1 = np.zeros((self.d, self.d))
        P_buf1[self.dTE:2 * self.dTE, self.dTE:2 * self.dTE] = np.eye(self.dTE)

        ctx = [G @ self.ctx_canon(a) for a in range(nA)]        # relaxed context
        ctx_ex = [self.ctx_canon(a) for a in range(nA)]         # exact context (for gates)
        Qtab = np.zeros((nS, nA))

        max_err, agree, n_agree, sel_agree, n_sel = 0.0, 0, 0, 0, 0
        out = {}
        for t, (s, a, r, s_next) in enumerate(traj, start=1):
            Qhat = self.decode_Q(ctx)
            a_star = int(np.argmax(Qhat[s_next]))
            # exact pass on exact state -> structural gates
            a_star_ex = int(np.argmax(Qtab[s_next]))
            x_ex = np.vstack([np.stack(ctx_ex), self.step_tokens_canon(s, a, r, s_next, a_star_ex)])
            y_ex, masks = self.forward(x_ex, I, force_causal=force_causal)

            x = self.step_tokens_canon(s, a, r, s_next, a_star) @ G.T
            y_in = np.vstack([np.stack(ctx), x])
            y_out, _ = self.forward(y_in, G, beta_route, beta_max, sigma, rng, masks,
                                    force_causal=force_causal)

            # model's selected action at SELECT (id block, action dims)
            sel_pos = nA + 5 + 2 * nA
            sel_scores = np.array([y_out[sel_pos, self.tf.A[b]] for b in range(nA)])
            gap_next = np.sort(Qtab[s_next])[-1] - np.sort(Qtab[s_next])[-2] if nA > 1 else 1
            if gap_next > 1e-6:
                sel_agree += int(np.argmax(sel_scores) == np.argmax(Qtab[s_next]))
                n_sel += 1

            # tabular update and recurrence
            Qtab[s, a] += alpha * (r + gamma * Qtab[s_next].max() - Qtab[s, a])
            upd = y_out[-1]
            upd_ex = y_ex[-1]
            if self.residual:
                # residual write: only the increment goes through the embeddings
                ctx[a] = ctx[a] + G @ (P_buf1 @ upd)
                ctx_ex[a] = ctx_ex[a] + P_buf1 @ upd_ex
            else:
                ctx[a] = G @ (self.ctx_canon(a) + P_buf1 @ upd)
                ctx_ex[a] = self.ctx_canon(a) + P_buf1 @ upd_ex

            Qhat = self.decode_Q(ctx)
            err = np.abs(Qhat - Qtab).max()
            if not np.isfinite(err):
                err = np.inf
            max_err = max(max_err, err)
            srt = np.sort(Qtab, axis=1)
            uniq = (srt[:, -1] - srt[:, -2]) > 1e-6
            agree += int((np.argmax(Qhat, 1) == np.argmax(Qtab, 1))[uniq].sum())
            n_agree += int(uniq.sum())
            if t in report_T:
                out[t] = dict(max_err=max_err, final_err=err,
                              greedy_agree=agree / max(n_agree, 1),
                              select_agree=sel_agree / max(n_sel, 1))
        return out


# ---------------------------------------------------------------------- MDPs
def random_mdp_trajectory(n_states, n_actions, T, seed, alpha=0.1, gamma=0.9, eps=0.3):
    """Random MDP (Dir(1) transitions, Beta(2,2) rewards) + eps-greedy behavior trajectory."""
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(n_states), size=(n_states, n_actions))
    R = rng.beta(2, 2, size=(n_states, n_actions))
    Qp = np.zeros((n_states, n_actions))
    s = int(rng.integers(n_states))
    traj = []
    for _ in range(T):
        a = int(rng.integers(n_actions)) if rng.random() < eps else int(np.argmax(Qp[s]))
        s2 = int(rng.choice(n_states, p=P[s, a]))
        r = float(R[s, a])
        traj.append((s, a, r, s2))
        Qp[s, a] += alpha * (r + gamma * Qp[s2].max() - Qp[s, a])
        s = s2
    return traj


if __name__ == "__main__":
    # Sanity: exact execution reproduces tabular Q-learning; also test forced causality.
    m = RelaxedQ(6, 3)
    traj = random_mdp_trajectory(6, 3, 500, seed=0)
    I = np.eye(m.d)
    print("exact:", m.run(traj, I))
    print("exact, all heads causal:", m.run(traj, I, force_causal=True))
    mr = RelaxedQ(6, 3, residual=True)
    print("residual, exact, all heads causal:", mr.run(traj, np.eye(mr.d), force_causal=True))
