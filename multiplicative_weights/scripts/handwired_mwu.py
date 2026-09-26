"""
Handwired exponential-weights (MWU) transformer, v2 (flag-free): standard components only.

Causal softmax attention, ReLU MLPs and residual-stream addition -- no linear attention,
no Python-side routing or gating inside the forward pass, and no role information in the
position embeddings. It uses the TRAINED round layout and vocabulary roles of
train.py (the label y uses the same PRED_0 / PRED_1 tokens as expert predictions)
and the residual latent write of the trained residual models.

Idealisations (the only ones): hard attention as a softmax with large beta; orthonormal token
and absolute position embeddings (one-hot in this code, w.l.o.g.); no LayerNorm. (Queries that need a constant use the direction
sum_v u_v: inner product 1 with every token's identity, 0 with the carried latent M, which is not a
token. One-hot is just the basis used here; any orthonormal embeddings give the same computation.)

Round layout (n = 4 experts, 20 positions):
  0 M (carried latent) | 1..8: E_1 p_1 ... E_n p_n | 9 SEP | 10 y |
  11..18: E_1 l_1 ... E_n l_n | 19 UPD
p_i, y in {PRED_0, PRED_1}; l_i in {LOSS_0, LOSS_1}. Expert i's tokens sit at fixed positions,
so a position identifies the expert (as a position identifies a slot in the Q construction).
M carries lambda = sum_i lambda_i u_{e_i} (lambda_i = eta * number of correct predictions of
expert i, the paper's gamma = e^eta; the weights equal MWU on 0/1 losses, softmax being
shift-invariant).

Residual stream: id (V) | pos (20) | buf (V), d = 2 V + L. ONE buffer, holding, by position,
  M      lambda (expert directions)
  SEP    lambda, copied from M (expert directions); PRED = exponential-weights P(y = 1) (u_{PRED_1})
  l_i    G = (1 - l_i) u_{e_i}: expert i was correct
  UPD    INC = eta sum_i (1 - l_i) u_{e_i}

Block 1
  H1    SEP reads the latent: every query attends to its own position (value: its buffer, empty
        at the input except at M), SEP scores higher on position 0 and copies lambda.
  MLP1  G at l_i = ReLU(pos[l_i] + id[LOSS_0] - 1) u_{e_i}, exact for 0/1 inputs.
Block 2
  H2.1  exponential weights at SEP: scores <buf_query, u_{e_i}> on the key at p_i's position
        (a fixed map from pos(p_i) to u_{e_i}) + a large bonus on keys with identity PRED_0 or
        PRED_1, i.e. softmax_i(lambda_i) over the n prediction tokens (lambda >= 0, so the bonus
        is independent of T; y, also a PRED token, comes after SEP); value id[PRED_1] -> PRED.
  H2.2  UPD attends uniformly to keys with identity LOSS_0 or LOSS_1; value G, scaled by
        eta * n -> INC.
Layer-2 outputs at other positions are never read. Decision: predict 1 iff PRED(SEP) > 1/2.
Recurrence (residual): M_{t+1} = M_t + W_ctx h[UPD], W_ctx = projection onto the buffer's
expert directions.
"""
import numpy as np



class Head:
    def __init__(self, WQ, WK, WV, WO):
        self.WQ, self.WK, self.WV, self.WO = WQ, WK, WV, WO

    def __call__(self, x):
        L = x.shape[0]
        S = (x @ self.WQ.T) @ (x @ self.WK.T).T
        S = np.where(np.tril(np.ones((L, L), dtype=bool)), S, -np.inf)
        S = S - S.max(1, keepdims=True)
        A = np.exp(S)
        A /= A.sum(1, keepdims=True)
        return (A @ (x @ self.WV.T)) @ self.WO.T


class MLP:
    def __init__(self, W1, b1, W2):
        self.W1, self.b1, self.W2 = W1, b1, W2

    def __call__(self, x):
        return np.maximum(x @ self.W1.T + self.b1, 0.0) @ self.W2.T


class HandwiredMWUv2:
    def __init__(self, n_experts=4, eta=0.1, beta=1e3):
        n = self.n = n_experts
        self.eta = eta
        # vocabulary: E_1..E_n, PRED_0, PRED_1, SEP, LOSS_0, LOSS_1, UPD
        self.E = list(range(n))
        self.P0, self.P1, self.SEP, self.L0, self.L1, self.UPD = range(n, n + 6)
        V = n + 6
        self.L = 4 * n + 4
        self.p_pred = [2 + 2 * i for i in range(n)]
        self.p_sep, self.p_y = 2 * n + 1, 2 * n + 2
        self.p_loss = [2 * n + 4 + 2 * i for i in range(n)]
        self.p_upd = 4 * n + 3
        off = 0
        def f(k):
            nonlocal off
            s = slice(off, off + k)
            off += k
            return s
        self.id, self.pos = f(V), f(self.L)
        self.buf = f(V)
        self.LAT = slice(self.buf.start + self.E[0], self.buf.start + self.E[0] + n)   # expert directions
        self.PRED = self.buf.start + self.P1
        d = self.d = off
        Z = lambda r: np.zeros((r, d))
        ex = lambda k: self.buf.start + self.E[k]

        # H1: every query attends to its own position; SEP scores higher on M (position 0)
        WQ, WK, WV, WO = Z(self.L + 1), Z(self.L + 1), Z(n), Z(n).T.copy()
        for p in range(self.L):
            WQ[p, self.pos.start + p] = beta
            WK[p, self.pos.start + p] = 1.0
        WQ[self.L, self.id.start + self.SEP] = 2 * beta
        WK[self.L, self.pos.start + 0] = 1.0
        for k in range(n):
            WV[k, ex(k)] = 1.0
            WO[ex(k), k] = 1.0
        h1 = Head(WQ, WK, WV, WO)
        # MLP1: G = (1 - l_i) u_{e_i} at l_i's position
        W1, b1, W2 = np.zeros((n, d)), -np.ones(n), np.zeros((d, n))
        for k in range(n):
            W1[k, self.pos.start + self.p_loss[k]] = 1.0
            W1[k, self.id.start + self.L0] = 1.0
            W2[ex(k), k] = 1.0
        mlp1 = MLP(W1, b1, W2)
        # H2.1 exponential weights over the prediction tokens
        big = 4 * beta
        WQ, WK, WV, WO = Z(n + 1), Z(n + 1), Z(1), Z(1).T.copy()
        WQ[0, self.id] = big                           # membership (<sum_v u_v, u_tok> = 1 on every token)
        WK[0, self.id.start + self.P0] = WK[0, self.id.start + self.P1] = 1.0
        for k in range(n):                             # lambda_k on the key at p_k's position
            WQ[1 + k, ex(k)] = 1.0
            WK[1 + k, self.pos.start + self.p_pred[k]] = 1.0
        WV[0, self.id.start + self.P1] = 1.0
        WO[self.PRED, 0] = 1.0
        h21 = Head(WQ, WK, WV, WO)
        # H2.2 UPD averages the gated loss vectors of the n loss tokens
        WQ, WK, WV, WO = Z(1), Z(1), Z(n), Z(n).T.copy()
        WQ[0, self.id.start + self.UPD] = big          # UPD -> LOSS_0 / LOSS_1 tokens
        WK[0, self.id.start + self.L0] = WK[0, self.id.start + self.L1] = 1.0
        for k in range(n):
            WV[k, ex(k)] = 1.0
            WO[ex(k), k] = eta * n
        h22 = Head(WQ, WK, WV, WO)
        self.blocks = [([h1], mlp1), ([h21, h22], None)]
        self.W_ctx = np.zeros((d, d))
        for k in range(n):
            self.W_ctx[ex(k), ex(k)] = 1.0
        self.pos_emb = np.zeros((self.L, d))            # one-hot absolute positions only
        for p in range(self.L):
            self.pos_emb[p, self.pos.start + p] = 1.0

    def embed(self, M, preds, label):
        X = self.pos_emb.copy()
        X[0] += M                                      # carried latent (not a token)
        toks = {}
        for i in range(self.n):
            toks[1 + 2 * i] = self.E[i]
            toks[2 + 2 * i] = self.P1 if preds[i] == 1 else self.P0
            toks[2 * self.n + 3 + 2 * i] = self.E[i]
            toks[2 * self.n + 4 + 2 * i] = self.L0 if preds[i] == label else self.L1
        toks[self.p_sep] = self.SEP
        toks[self.p_y] = self.P1 if label == 1 else self.P0
        toks[self.p_upd] = self.UPD
        for p, v in toks.items():                      # token embedding: one-hot identity
            X[p, self.id.start + v] = 1.0
        return X

    def forward(self, X):
        for heads, mlp in self.blocks:
            X = X + sum(h(X) for h in heads)
            if mlp is not None:
                X = X + mlp(X)
        return X

    def run(self, preds_seq, labels):
        M = np.zeros(self.d)
        lam = np.zeros(self.n)
        out = []
        for p, y in zip(preds_seq, labels):
            H = self.forward(self.embed(M, p, y))
            w = np.exp(lam - lam.max()); w /= w.sum()
            p_ex = float(w[np.array(p) == 1].sum())
            lat = M[self.LAT]
            wh = np.exp(lat - lat.max()); wh /= wh.sum()
            out.append((float(H[self.p_sep, self.PRED]), p_ex, wh, w))
            M = M + H[self.p_upd] @ self.W_ctx.T               # residual write
            lam = lam + self.eta * (np.array(p) == y)
        return out


def verify(betas=(20, 50, 100, 1e3, 1e4), n_seq=10, T=500, eta=0.1):
    rows = []
    for beta in betas:
        perr, werr, agree = [], [], []
        for seed in range(n_seq):
            rng = np.random.default_rng(seed)
            q = rng.uniform(0.3, 0.9, 4)
            y = rng.integers(0, 2, T)
            c = rng.random((T, 4)) < q
            preds = np.where(c, y[:, None], 1 - y[:, None])
            out = HandwiredMWUv2(4, eta, beta).run(preds.tolist(), y.tolist())
            perr.append(max(abs(a - b) for a, b, _, _ in out))
            werr.append(max(np.abs(c1 - c2).max() for _, _, c1, c2 in out))
            agree.append(np.mean([(a > .5) == (b > .5) for a, b, _, _ in out if abs(b - .5) > 1e-9]))
        rows.append((beta, max(perr), max(werr), float(np.mean(agree))))
        print(f"beta={beta:<7g} max|p_hat - p_MW| {rows[-1][1]:.2e}  max|w_hat - w_MW| "
              f"{rows[-1][2]:.2e}  decision agreement {rows[-1][3]:.4f}", flush=True)
    return rows


if __name__ == "__main__":
    verify()
