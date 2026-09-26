"""
Matrix-form implementation of the Weighted-Majority / exponential-weights construction
(paper Sec. 3), executed with relaxable idealizations.

The original construction (transformer_handwired_multiplicative_weights.py, since removed)
computes MWU with Python loops rather than attention matrices, so it cannot be relaxed.
This file re-implements the construction as actual causal attention heads acting on a
block-structured residual stream  [id | buf1 | buf2 | buf3 | lat | pos | one].

Per round the input is (causal version of the paper's token stream):

    <BOS> <w> <z_t> <e_1><p_1> ... <e_n><p_n> <p?> <y_t> <w?>

where z_t is the continuous latent token carried from the previous round: id(z_t) = u_Z
(a marker) and lat(z_t) = sum_i lambda_i u_{e_i} (the superposition of expert embeddings,
lambda_i = log w_i).  p_i in {q0, q1} and y_t in {y0, y1} (separate label tokens).

Differences from the paper's text (needed for a *causal* decoder):
  * Head 1.1 copies the PREVIOUS token's id into buf1 (the paper fetches the next token,
    which a causal mask forbids).  So buf1(p_i) = u_{e_i} and buf1(w?) = u_{y_t}.
  * The superposition lives in its own 'lat' block of the latent token rather than in its
    id block, so the fixed-offset copy in Head 1.1 cannot leak it into other tokens' keys.
  * Head 3.1 masks non-slot keys with a -beta penalty (hard mask on id in {q0,q1}) so
    that only the n expert-prediction slots enter the exponential-weights softmax.

Layers
  L1  H1.1  fixed-offset (-1) positional attention, id_{i-1} -> buf1_i         [hard, beta]
  L2  H2.1  <p?>,<w?> attend to the token with id u_Z (z_t), copy lat -> buf2     [hard, beta]
  L3  H3.1  <p?>: softmax_j( <buf2_i, buf1_j>_E  - beta [j not an expert slot] ),
            value id_j restricted to {q0,q1} -> buf3.  Scores are exactly lambda_i:
            this is the exponential-weights softmax, temperature 1 (not relaxed).
      H3.2  <w?>: linear attention, score eta*<y_t, p_j>, value buf1_j = u_{e_j} -> buf2
            so buf2(w?) = sum_i (lambda_i + eta 1{p_i=y_t}) u_{e_i} = lat(z_{t+1}).
  Readout: p_hat = buf3(<p?>)[q1];  next latent: lat(z_{t+1}) = buf2(<w?>).

Relaxations (see relax_common): Gram matrix G of token embeddings (non-orthogonal /
overlapping buffers), finite beta for the hard heads, residual noise sigma per layer.

residual=True: residual state write, lat(z_{t+1}) = lat(z_t) + buf2(<w?>). Head 2.1 then
copies z_t to <p?> only (the prediction still needs it); every other row attends to an
attention sink (<BOS>, zero latent value), so buf2(<w?>) holds just the increment
sum_i eta 1{p_i = y_t} u_{e_i}. The carried state never passes through the network, and
only the increment is written through the embeddings.
"""
import numpy as np

from relax_common import softmax_rows, psd_sqrt


class RelaxedMWU:
    def __init__(self, n_experts=4, eta=0.1, residual=False):
        self.n, self.eta = n_experts, eta
        self.residual = residual
        n = n_experts
        # vocabulary (identity directions)
        self.BOS, self.W = 0, 1
        self.E = list(range(2, 2 + n))
        self.Q0, self.Q1 = 2 + n, 3 + n
        self.Y0, self.Y1 = 4 + n, 5 + n
        self.PQ, self.WQ = 6 + n, 7 + n
        self.Z = 8 + n
        self.dTE = 9 + n
        self.L = 2 * n + 6                      # tokens per round
        self.n_blocks = 5                       # id, buf1, buf2, buf3, lat
        self.pos0 = self.n_blocks * self.dTE
        self.one = self.pos0 + self.L
        self.d = self.one + 1

    def blk(self, b):
        return slice(b * self.dTE, (b + 1) * self.dTE)

    def col(self, b, v):
        return b * self.dTE + v

    # --------------------------------------------------------------- tokens
    def round_tokens(self, preds, label):
        """Canonical embeddings; the latent content of z (row 2) is filled in by run()."""
        n, X = self.n, np.zeros((self.L, self.d))
        ids = [self.BOS, self.W, self.Z]
        for i in range(n):
            ids += [self.E[i], self.Q1 if preds[i] == 1 else self.Q0]
        ids += [self.PQ, self.Y1 if label == 1 else self.Y0, self.WQ]
        for pos, v in enumerate(ids):
            X[pos, self.col(0, v)] = 1.0
            X[pos, self.pos0 + pos] = 1.0
            X[pos, self.one] = 1.0
        return X

    # --------------------------------------------------------------- forward
    def forward(self, y, G, beta, sigma, rng, Gsqrt):
        L, dTE = self.L, self.dTE
        idx = np.arange(L)
        causal = idx[None, :] <= idx[:, None]
        E = np.array(self.E)
        id_, b1, b2, b3, lat = (self.blk(k) for k in range(5))
        pos = slice(self.pos0, self.pos0 + L)

        def add(y, out):
            y = y + out @ G.T
            if sigma > 0:
                y = y + sigma * rng.standard_normal(y.shape) @ Gsqrt.T
            return y

        # ---- Layer 1, Head 1.1: fixed offset -1 via positional attention
        shift = np.zeros((L, L))               # (P pos_i) = pos_{i-1}
        shift[idx[1:] - 1, idx[1:]] = 1.0
        Qp = y[:, pos] @ shift.T               # query: shifted position
        Kp = y[:, pos]
        A = softmax_rows(beta * Qp @ Kp.T, causal)
        out = np.zeros_like(y)
        out[:, b1] = A @ y[:, id_]
        y = add(y, out)

        # ---- Layer 2, Head 2.1: <p?>,<w?> fetch the latent superposition from z_t
        if self.residual:
            # only <p?> fetches z_t; every other row attends to an attention sink (<BOS>,
            # zero latent value) instead of averaging over earlier tokens
            q = y[:, self.col(0, self.PQ)]
            sink_q = y[:, self.one] - q
            S = beta * (np.outer(q, y[:, self.col(0, self.Z)])
                        + np.outer(sink_q, y[:, self.col(0, self.BOS)]))
            A = softmax_rows(S, causal)
        else:
            q = y[:, self.col(0, self.PQ)] + y[:, self.col(0, self.WQ)]
            k = y[:, self.col(0, self.Z)]
            A = softmax_rows(beta * np.outer(q, k), causal)
        out = np.zeros_like(y)
        out[:, b2] = A @ y[:, lat]
        y = add(y, out)

        # ---- Layer 3
        out = np.zeros_like(y)
        # Head 3.1: exponential-weights softmax over expert slots (masked by -beta)
        qv = np.concatenate([y[:, b2][:, E], y[:, [self.col(0, self.PQ)]]], axis=1)
        kv = np.concatenate([y[:, b1][:, E],
                             beta * (y[:, [self.col(0, self.Q0)]] + y[:, [self.col(0, self.Q1)]]
                                     - y[:, [self.one]])],
                            axis=1)
        A = softmax_rows(qv @ kv.T, causal)
        vals = y[:, [self.col(0, self.Q0), self.col(0, self.Q1)]]
        o = A @ vals
        out[:, self.col(3, self.Q0)] = o[:, 0]
        out[:, self.col(3, self.Q1)] = o[:, 1]
        # Head 3.2: linear-attention multiplicative update
        qv = self.eta * y[:, [self.col(1, self.Y0), self.col(1, self.Y1)]]
        kv = y[:, [self.col(0, self.Q0), self.col(0, self.Q1)]]
        S = (qv @ kv.T) * causal
        upd = S @ y[:, b1][:, E]
        out[:, self.col(2, 0) + E] += upd
        y = add(y, out)
        return y

    # --------------------------------------------------------------- rollout
    def run(self, preds_seq, labels, G, beta=1e6, sigma=0.0, seed=0, report_T=(100, 500),
            Gnoise=None):
        """Gnoise: noise covariance (/sigma^2) in tracked coordinates (default G, tied
        read-out Phi^T).  Dual read-out Phi^+ corresponds to G=I, Gnoise=G^{-1}."""
        rng = np.random.default_rng(seed)
        Gsqrt = psd_sqrt(G if Gnoise is None else Gnoise) if sigma > 0 else None
        n, E = self.n, np.array(self.E)
        z_lat = np.zeros(self.dTE)                      # lat(z) in Phi^T coordinates
        lam = np.zeros(n)                               # exact log-weights
        pq_pos, wq_pos = 2 * n + 3, 2 * n + 5
        res = {}
        max_perr, max_linf, n_agree, n_valid = 0.0, 0.0, 0, 0
        mist_tf, mist_ex = 0, 0
        for t in range(1, len(labels) + 1):
            p, yl = preds_seq[t - 1], labels[t - 1]
            X = self.round_tokens(p, yl)
            Y = X @ G.T
            Y[2, self.blk(4)] = z_lat                   # continuous latent (already Phi^T coords)
            Yout = self.forward(Y, G, beta, sigma, rng, Gsqrt)

            # exact exponential weights
            w = np.exp(lam - lam.max())
            w /= w.sum()
            p_ex = float(w[np.array(p) == 1].sum())
            p_tf = float(Yout[pq_pos, self.col(3, self.Q1)])
            s0 = float(Yout[pq_pos, self.col(3, self.Q0)])
            dec_tf = int(p_tf > s0)
            dec_ex = int(p_ex > 0.5)
            if abs(p_ex - 0.5) > 1e-9:
                n_valid += 1
                n_agree += int(dec_tf == dec_ex)
            mist_tf += int(dec_tf != yl)
            mist_ex += int(dec_ex != yl)
            max_perr = max(max_perr, abs(p_tf - p_ex)) if np.isfinite(p_tf) else np.inf

            # weights implied by the latent vs exact
            lam_hat = z_lat[E]
            wh = np.exp(lam_hat - lam_hat.max())
            wh /= wh.sum()
            linf = float(np.abs(wh - w).max())
            l1 = float(np.abs(wh - w).sum())
            max_linf = max(max_linf, linf) if np.isfinite(linf) else np.inf

            # updates
            lam = lam + self.eta * (np.array(p) == yl)
            # next latent: lat(z_{t+1}) <- buf2(<w?>)  (linear map, then re-embedded: G S y)
            S_y = np.zeros(self.d)
            S_y[self.blk(4)] = Yout[wq_pos, self.blk(2)]
            if self.residual:
                z_lat = z_lat + (G @ S_y)[self.blk(4)]
            else:
                z_lat = (G @ S_y)[self.blk(4)]
            if t in report_T:
                lam_hat = z_lat[E]
                wh = np.exp(lam_hat - lam_hat.max()); wh /= wh.sum()
                w2 = np.exp(lam - lam.max()); w2 /= w2.sum()
                res[t] = dict(max_pred_err=max_perr,
                              w_linf=float(np.abs(wh - w2).max()),
                              w_l1=float(np.abs(wh - w2).sum()),
                              max_w_linf=max(max_linf, float(np.abs(wh - w2).max())),
                              decision_agree=n_agree / max(n_valid, 1),
                              regret_diff=mist_tf - mist_ex)
        return res


def expert_sequence(n, T, seed):
    """Paper's data model: expert qualities U[0.3, 0.9], random binary labels."""
    rng = np.random.default_rng(seed)
    qual = rng.uniform(0.3, 0.9, size=n)
    labels = rng.integers(0, 2, size=T)
    correct = rng.random((T, n)) < qual[None, :]
    preds = np.where(correct, labels[:, None], 1 - labels[:, None])
    return preds.tolist(), labels.tolist()


if __name__ == "__main__":
    m = RelaxedMWU(4, eta=0.1)
    preds, labels = expert_sequence(4, 500, seed=0)
    print("exact:", m.run(preds, labels, np.eye(m.d)))
    mr = RelaxedMWU(4, eta=0.1, residual=True)
    print("residual, exact:", mr.run(preds, labels, np.eye(mr.d)))
