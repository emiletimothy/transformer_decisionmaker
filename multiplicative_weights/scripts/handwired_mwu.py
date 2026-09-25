"""
Handwired exponential-weights (MWU) transformer, v2 (flag-free): standard components only.

Causal softmax attention, ReLU MLPs and residual-stream addition -- no linear attention,
no Python-side routing or gating inside the forward pass, and no role information in the
position embeddings. It uses the TRAINED round layout and vocabulary roles of
train.py (the label y uses the same PRED_0 / PRED_1 tokens as expert predictions)
and the residual latent write of the trained residual models.

Idealisations (the only ones): hard attention as a softmax with large beta; one-hot token
and one-hot absolute position embeddings; a constant direction ONE shared by every token
embedding (the carried latent M is not a token and does not carry it); no LayerNorm.
(The previous version with role flags in the position embedding is
handwired_mwu (flags variant, removed).py.)

Round layout (n = 4 experts, 20 positions):
  0 M (carried latent) | 1..8: E_1 p_1 ... E_n p_n | 9 SEP | 10 y |
  11..18: E_1 l_1 ... E_n l_n | 19 UPD
p_i, y in {PRED_0, PRED_1}; l_i in {LOSS_0, LOSS_1}. M carries lambda = sum_i lambda_i u_{e_i}
(lambda_i = eta * number of correct predictions of expert i, the paper's gamma = e^eta;
the weights equal MWU on 0/1 losses, softmax being shift-invariant) in its LAT field.
Position 0 (M) is the attention sink: its value is 0 in every field a head reads except LAT.

Residual stream fields: id (V) | pos (20) | ONE |
  LAT (n) latent log-weights | EXP (n) expert id of the previous token |
  LATQ (n) copy of M's LAT | PRED (1) exponential-weights P(y = 1) |
  G (n) gated expert id ((1 - l_i) u_{e_i}) | INC (n) increment eta sum_i (1 - l_i) u_{e_i}

Block 1
  H1.1  previous-token head (query pos(i) -> key pos(i-1)); value expert id -> EXP.
  H1.2  every token (query ONE) attends to position 0 (M); value LAT -> LATQ.
  MLP1  G = AND(EXP, id[LOSS_0]) = ReLU(EXP[k] + id[LOSS_0] - 1), exact for 0/1 inputs.
Block 2
  H2.1  exponential-weights head: scores <LATQ_query, EXP_key> + large bonus on keys with
        identity PRED_0 or PRED_1, i.e. softmax_i(lambda_i) over the n prediction tokens
        (the label token, also a PRED token, comes after SEP and is masked); value
        id[PRED_1] -> PRED. At SEP this is sum_i w_i 1{p_i = 1}, w = softmax(lambda).
  H2.2  queries with identity UPD attend uniformly to keys with identity LOSS_0 or LOSS_1;
        value G, output scaled by eta * n -> INC = eta sum_i (1 - l_i) u_{e_i}.
Every query also carries ONE, which scores 0.5 beta on the sink (position 0).
Decision: predict 1 iff PRED(SEP) > 1/2. Recurrence (residual):
M_{t+1} = M_t + W_ctx h[UPD], W_ctx: INC -> LAT.
"""
import numpy as np

SINK = 0.5


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
        self.one = f(1).start
        self.LAT, self.EXP, self.LATQ = f(n), f(n), f(n)
        self.PRED = f(1).start
        self.G, self.INC = f(n), f(n)
        d = self.d = off
        Z = lambda r: np.zeros((r, d))

        # H1.1 previous token -> EXP
        WQ, WK, WV, WO = Z(self.L), Z(self.L), Z(n), Z(n).T.copy()
        for i in range(1, self.L):
            WQ[i - 1, self.pos.start + i] = beta
        for j in range(self.L):
            WK[j, self.pos.start + j] = 1.0
        for k in range(n):
            WV[k, self.id.start + self.E[k]] = 1.0
            WO[self.EXP.start + k, k] = 1.0
        h11 = Head(WQ, WK, WV, WO)
        # H1.2 every token attends to M (position 0) -> LATQ
        WQ, WK, WV, WO = Z(1), Z(1), Z(n), Z(n).T.copy()
        WQ[0, self.one] = beta
        WK[0, self.pos.start + 0] = 1.0
        for k in range(n):
            WV[k, self.LAT.start + k] = 1.0
            WO[self.LATQ.start + k, k] = 1.0
        h12 = Head(WQ, WK, WV, WO)
        # MLP1: G = AND(EXP, LOSS_0)  (expert k was correct this round)
        W1, b1, W2 = np.zeros((n, d)), -np.ones(n), np.zeros((d, n))
        for k in range(n):
            W1[k, self.EXP.start + k] = 1.0
            W1[k, self.id.start + self.L0] = 1.0
            W2[self.G.start + k, k] = 1.0
        mlp1 = MLP(W1, b1, W2)
        # H2.1 exponential weights over the prediction tokens
        big = 4 * beta
        WQ, WK, WV, WO = Z(n + 2), Z(n + 2), Z(1), Z(1).T.copy()
        WQ[0, self.one] = beta * SINK                  # sink baseline on M (position 0)
        WK[0, self.pos.start + 0] = 1.0
        WQ[1, self.one] = big                          # membership: PRED_0 / PRED_1 tokens
        WK[1, self.id.start + self.P0] = WK[1, self.id.start + self.P1] = 1.0
        for k in range(n):                             # lambda_k on expert k's p token
            WQ[2 + k, self.LATQ.start + k] = 1.0
            WK[2 + k, self.EXP.start + k] = 1.0
        WV[0, self.id.start + self.P1] = 1.0
        WO[self.PRED, 0] = 1.0
        h21 = Head(WQ, WK, WV, WO)
        # H2.2 UPD averages the gated loss vectors of the n loss tokens
        WQ, WK, WV, WO = Z(2), Z(2), Z(n), Z(n).T.copy()
        WQ[0, self.one] = beta * SINK
        WK[0, self.pos.start + 0] = 1.0
        WQ[1, self.id.start + self.UPD] = big         # UPD -> LOSS_0 / LOSS_1 tokens
        WK[1, self.id.start + self.L0] = WK[1, self.id.start + self.L1] = 1.0
        for k in range(n):
            WV[k, self.G.start + k] = 1.0
            WO[self.INC.start + k, k] = eta * n
        h22 = Head(WQ, WK, WV, WO)
        self.blocks = [([h11, h12], mlp1), ([h21, h22], None)]
        self.W_ctx = np.zeros((d, d))
        for k in range(n):
            self.W_ctx[self.LAT.start + k, self.INC.start + k] = 1.0
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
        for p, v in toks.items():                      # token embedding: identity + ONE
            X[p, self.id.start + v] = 1.0
            X[p, self.one] = 1.0
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
