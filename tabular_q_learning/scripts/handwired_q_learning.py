"""
Handwired tabular Q-learning transformer, v2 (flag-free): standard components only.

Causal softmax attention, ReLU MLPs and residual-stream addition only -- no Python-side
routing, gating, shifting or indexing inside the forward pass, and no role information in
the position embeddings. One instance (max_states = 8, max_actions = 4, as the trained
model) serves every MDP with |S| <= 8 and |A| <= 4. It uses the TRAINED token layout
(3_train.build_step_tokens plus the |A| context slots of 2_model.forward_step) and the
residual context write of the trained residual models.

Idealisations (the only ones):
  * hard attention is a softmax with a large inverse temperature (beta for routing heads,
    beta_max for the max head); error vs beta is reported by verify();
  * one-hot token embeddings and one-hot absolute position embeddings (no role flags);
  * a constant direction ONE shared by every token embedding (the attention-sink baseline
    and the MLP bias path); context slots are not tokens and do not carry it;
  * no LayerNorm.
(The previous version with role flags in the position embedding is
handwired_q_learning (flags variant, removed).py.)

Per step, positions (n = |A| of this MDP, o = n + 1):
  0 BOS | 1..n context slots c_1..c_n | o QCURR, o+1 s_t, o+2 a_t, o+3 R(r_t), o+4 QNEXT,
  o+5+2i s_{t+1}, o+6+2i a_i (i = 0..n-1) | o+5+2n SELECT, o+6+2n a*, o+7+2n UPDATE.
Slot j is identified only by its POSITION 1 + j; its content is what the model wrote there
(zeros at t = 0) plus its position embedding. The reward is a scalar feature RVAL of the R
token (as reward_proj in the trained model).

Residual stream: id (V) | pos (L_max) | ONE | buf1 (V) | buf2 (V), d = 3 V + L_max + 1.
  buf1  token identities and scalars: PID + P2ID = identities of the tokens 1 and 2 back (action
        tokens; they never collide), ST = u_{s_t} (UPDATE); scalars along special-token
        directions: RVAL / RU (reward at R / UPDATE), QV (action tokens), IS_AT, IS_PAIR (roles of
        action tokens), MAXQ, QCUR; ASEL (selected action) along the action directions
  buf2  Q columns: SLOT = sum_s Q_t(s, a) u_s (slots), COL = column of this token's action
        (action tokens), INC = TD increment (UPDATE)

Block 1 (attention on the input, then MLP1)
  H1.1  FO(A, -1): an action token at pos(i) attends to pos(i-1), value identity -> PID;
        every other token scores higher on the BOS sink (query 2 beta (ONE - is_action)),
        whose value is 0, and reads nothing.
  H1.2  FO(A, -2): the same with pos(i-2), value identity -> P2ID.
        (Context slots are not tokens: they carry no ONE, and nothing reads their PID/P2ID.)
  H1.3  slot fetch: a token with identity A_j queries pos(1 + j), value SLOT -> COL;
        other tokens fall to the BOS sink (value 0).
  MLP1  (exact ReLU arithmetic on 0/1 features, |values| <= C)
        QV       = sum_s PID[S_s] * COL[s]   via  e*c = ReLU(c - C(1-e)) - ReLU(-c - C(1-e))
        IS_AT    = ReLU(is_action + P2ID[QCURR] - 1)
        IS_PAIR  = ReLU(is_action - P2ID[QCURR] - PID[SELECT])
        with is_action = sum_j id[A_j].
Block 2
  H2.1  max head: queries with identity SELECT or UPDATE; keys IS_PAIR (large bonus) plus
        beta_max * QV; values action identity -> ASEL, QV -> MAXQ.
  H2.2  UPDATE -> key IS_AT; values QV -> QCUR, PID[S] (= u_{s_t}) -> ST.
  H2.3  UPDATE -> key id[R]; value RVAL -> RU.
  MLP2  INC[s] = gate(ST[s], alpha (RU + gamma MAXQ - QCUR)).
Sink: every attention query carries ONE, which scores 0.5 beta on the BOS key; BOS has
value 0 in every field any head reads, so a query without a target reads nothing.

The a* token also fetches a slot (H1.3) but its QV is 0 (its previous token is SELECT,
not a state) and IS_PAIR = IS_AT = 0, so no block-2 head reads it; the computation never
uses the a* token, so the model's own a* and the teacher's a* give identical results.
Output: a* = argmax ASEL at SELECT. Recurrence: c_{a_t} <- c_{a_t} + W_ctx h[UPDATE],
W_ctx: INC -> SLOT (the write target a_t is chosen by the recurrence protocol, as in
training).
"""
import numpy as np

SINK = 0.5
C_GATE = 100.0


class Layout:
    def __init__(self, max_states=8, max_actions=4):
        self.nS, self.nA = max_states, max_actions
        self.S = list(range(max_states))
        self.A = list(range(max_states, max_states + max_actions))
        base = max_states + max_actions
        self.BOS, self.R, self.SELECT, self.UPDATE, self.QCURR, self.QNEXT = range(base, base + 6)
        self.V = base + 6
        self.Lmax = 3 * max_actions + 9
        off = 0

        def f(k):
            nonlocal off
            s = slice(off, off + k)
            off += k
            return s
        self.id, self.pos = f(self.V), f(self.Lmax)
        self.ONE = f(1).start
        # two buffers of d_TE = V dimensions; a buffer holds several quantities along
        # orthogonal token directions (states / actions / special tokens) or at different positions
        self.B1, self.B2 = f(self.V), f(self.V)
        b1, b2 = self.B1.start, self.B2.start
        states = lambda b: slice(b + self.S[0], b + self.S[0] + max_states)
        # buf1: token identities and scalars.
        #   identities of the tokens 1 and 2 back (action tokens: 1 back is a state or SELECT,
        #   2 back is QCURR, QNEXT or an action, so they never collide); u_{s_t} at UPDATE;
        #   scalars along the special-token directions that are free where they are written
        #   (at action tokens BOS, R, UPDATE; the role indicators avoid R, which carries the
        #   reward at the R token and would otherwise enter heads 2.1 / 2.2 as a key);
        #   the selected action along the action directions (SELECT / UPDATE)
        self.PID = self.P2ID = self.B1
        self.ST = states(b1)
        self.RVAL = self.RU = b1 + self.R       # reward at R; copied to UPDATE
        self.QV = b1 + self.R                   # Q value at action tokens
        self.IS_AT = b1 + self.BOS              # role indicators at action tokens
        self.IS_PAIR = b1 + self.UPDATE
        self.MAXQ = b1 + self.SELECT            # max_a Q_t(s_{t+1}, a) at SELECT / UPDATE
        self.QCUR = b1 + self.QCURR             # Q_t(s_t, a_t) at UPDATE
        self.ASEL = slice(b1 + self.A[0], b1 + self.A[0] + max_actions)
        # buf2: Q columns: slot content (slots), fetched column (action tokens), TD increment (UPDATE)
        self.SLOT = self.COL = self.INC = states(b2)
        self.d = off

    def positions(self, n):
        o = n + 1
        return dict(at=o + 2, r=o + 3, pair=[o + 6 + 2 * i for i in range(n)],
                    sel=o + 5 + 2 * n, astar=o + 6 + 2 * n, upd=o + 7 + 2 * n, L=3 * n + 9)


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


class HandwiredQv2:
    def __init__(self, alpha=0.1, gamma=0.9, beta=1e3, beta_max=1e4,
                 max_states=8, max_actions=4):
        self.lay = Lay = Layout(max_states, max_actions)
        self.alpha, self.gamma = alpha, gamma
        d, nS, nA, V = Lay.d, max_states, max_actions, Lay.V
        Z = lambda r: np.zeros((r, d))
        idc = lambda v: Lay.id.start + v

        def sink(WQ, WK):
            WQ[0, Lay.ONE] = beta * SINK
            WK[0, idc(Lay.BOS)] = 1.0

        def offset_head(k, field):
            # FO(A, -k): action tokens attend k positions back; every other token (ONE, not
            # an action) scores 2 beta on the BOS key and reads BOS, whose value is 0
            WQ, WK, WV, WO = Z(Lay.Lmax + 1), Z(Lay.Lmax + 1), Z(V), Z(V).T.copy()
            for i in range(k, Lay.Lmax):
                WQ[1 + i - k, Lay.pos.start + i] = beta
            for j in range(Lay.Lmax):
                WK[1 + j, Lay.pos.start + j] = 1.0
            WQ[0, Lay.ONE] = 2 * beta
            for a in Lay.A:
                WQ[0, idc(a)] = -2 * beta
            WK[0, idc(Lay.BOS)] = 1.0
            for v in range(V):
                if v != Lay.BOS:
                    WV[v, idc(v)] = 1.0
                    WO[field.start + v, v] = 1.0
            return Head(WQ, WK, WV, WO)

        h11, h12 = offset_head(1, Lay.PID), offset_head(2, Lay.P2ID)

        # H1.3 slot fetch
        WQ, WK, WV, WO = Z(Lay.Lmax + 1), Z(Lay.Lmax + 1), Z(nS), Z(nS).T.copy()
        sink(WQ, WK)
        for j in range(nA):
            WQ[1 + (1 + j), idc(Lay.A[j])] = beta
        for p in range(Lay.Lmax):
            WK[1 + p, Lay.pos.start + p] = 1.0
        for s in range(nS):
            WV[s, Lay.SLOT.start + s] = 1.0
            WO[Lay.COL.start + s, s] = 1.0
        h13 = Head(WQ, WK, WV, WO)

        # MLP1: QV (gated products) and the roles of action tokens
        n_u = 2 * nS + 2
        W1, b1, W2 = np.zeros((n_u, d)), np.zeros(n_u), np.zeros((d, n_u))
        for s in range(nS):
            for sign, row in ((1.0, 2 * s), (-1.0, 2 * s + 1)):
                W1[row, Lay.COL.start + s] = sign
                W1[row, Lay.PID.start + Lay.S[s]] = C_GATE
                b1[row] = -C_GATE
                W2[Lay.QV, row] = sign
        r_at, r_pr = 2 * nS, 2 * nS + 1
        for j in range(nA):
            for r in (r_at, r_pr):
                W1[r, idc(Lay.A[j])] = 1.0
        W1[r_at, Lay.P2ID.start + Lay.QCURR] = 1.0
        b1[r_at] = -1.0
        W1[r_pr, Lay.P2ID.start + Lay.QCURR] = -1.0
        W1[r_pr, Lay.PID.start + Lay.SELECT] = -1.0
        W2[Lay.IS_AT, r_at] = W2[Lay.IS_PAIR, r_pr] = 1.0
        mlp1 = MLP(W1, b1, W2)

        # H2.1 max head
        big = 4 * beta + 2 * beta_max * C_GATE
        WQ, WK, WV, WO = Z(3), Z(3), Z(nA + 1), Z(nA + 1).T.copy()
        sink(WQ, WK)
        WQ[1, idc(Lay.SELECT)] = WQ[1, idc(Lay.UPDATE)] = big
        WK[1, Lay.IS_PAIR] = 1.0
        WQ[2, idc(Lay.SELECT)] = WQ[2, idc(Lay.UPDATE)] = beta_max
        WK[2, Lay.QV] = 1.0
        for a in range(nA):
            WV[a, idc(Lay.A[a])] = 1.0
            WO[Lay.ASEL.start + a, a] = 1.0
        WV[nA, Lay.QV] = 1.0
        WO[Lay.MAXQ, nA] = 1.0
        h21 = Head(WQ, WK, WV, WO)

        # H2.2 UPDATE <- a_t
        WQ, WK, WV, WO = Z(2), Z(2), Z(nS + 1), Z(nS + 1).T.copy()
        sink(WQ, WK)
        WQ[1, idc(Lay.UPDATE)] = beta
        WK[1, Lay.IS_AT] = 1.0
        WV[0, Lay.QV] = 1.0
        WO[Lay.QCUR, 0] = 1.0
        for s in range(nS):
            WV[1 + s, Lay.PID.start + Lay.S[s]] = 1.0
            WO[Lay.ST.start + s, 1 + s] = 1.0
        h22 = Head(WQ, WK, WV, WO)

        # H2.3 UPDATE <- R
        WQ, WK, WV, WO = Z(2), Z(2), Z(1), Z(1).T.copy()
        sink(WQ, WK)
        WQ[1, idc(Lay.UPDATE)] = beta
        WK[1, idc(Lay.R)] = 1.0
        WV[0, Lay.RVAL] = 1.0
        WO[Lay.RU, 0] = 1.0
        h23 = Head(WQ, WK, WV, WO)

        # MLP2: INC[s] = gate(ST[s], alpha (RU + gamma MAXQ - QCUR))
        W1, b1, W2 = np.zeros((2 * nS, d)), np.zeros(2 * nS), np.zeros((d, 2 * nS))
        for s in range(nS):
            for sign, row in ((1.0, 2 * s), (-1.0, 2 * s + 1)):
                W1[row, Lay.RU] = sign * alpha
                W1[row, Lay.MAXQ] = sign * alpha * gamma
                W1[row, Lay.QCUR] = -sign * alpha
                W1[row, Lay.ST.start + s] = C_GATE
                b1[row] = -C_GATE
                W2[Lay.INC.start + s, row] = sign
        mlp2 = MLP(W1, b1, W2)

        self.blocks = [([h11, h12, h13], mlp1), ([h21, h22, h23], mlp2)]
        self.W_ctx = np.zeros((d, d))
        for s in range(nS):
            self.W_ctx[Lay.SLOT.start + s, Lay.INC.start + s] = 1.0

    # ---------------------------------------------------------------- model
    def embed(self, ctx, s, a, r, s_next, a_star):
        Lay, n = self.lay, ctx.shape[0]
        toks = ([Lay.BOS] + [None] * n + [Lay.QCURR, Lay.S[s], Lay.A[a], Lay.R, Lay.QNEXT])
        for i in range(n):
            toks += [Lay.S[s_next], Lay.A[i]]
        toks += [Lay.SELECT, Lay.A[a_star], Lay.UPDATE]
        X = np.zeros((len(toks), Lay.d))
        for p, v in enumerate(toks):
            X[p, Lay.pos.start + p] = 1.0              # one-hot absolute position
            if v is not None:                          # token embedding: identity + ONE
                X[p, Lay.id.start + v] = 1.0
                X[p, Lay.ONE] = 1.0
        X[1:1 + n] += ctx                              # carried context slots
        X[self.lay.positions(n)['r'], Lay.RVAL] = r    # reward scalar on the R token
        return X

    def forward(self, X):
        for heads, mlp in self.blocks:
            X = X + sum(h(X) for h in heads)
            X = X + mlp(X)
        return X

    def run(self, traj, n_states, n_actions, astar_mode='self'):
        """traj: list of (s, a, r, s_next). astar_mode 'self': the model's own SELECT argmax
        is fed back as the a* token; 'teacher': the tabular argmax Q_{t+1}(s_next)."""
        Lay = self.lay
        P = Lay.positions(n_actions)
        ctx = np.zeros((n_actions, Lay.d))
        Qtab = np.zeros((n_states, n_actions))
        Qs, sel = [], []
        for s, a, r, s2 in traj:
            if astar_mode == 'self':
                H0 = self.forward(self.embed(ctx, s, a, r, s2, 0))
                a_star = int(np.argmax(H0[P['sel'], Lay.ASEL][:n_actions]))
            else:
                q2 = Qtab.copy()
                q2[s, a] += self.alpha * (r + self.gamma * Qtab[s2].max() - Qtab[s, a])
                a_star = int(np.argmax(q2[s2]))
            H = self.forward(self.embed(ctx, s, a, r, s2, a_star))
            sel.append((int(np.argmax(H[P['sel'], Lay.ASEL][:n_actions])), Qtab[s2].copy()))
            ctx[a] = ctx[a] + H[P['upd']] @ self.W_ctx.T         # residual write
            Qtab[s, a] += self.alpha * (r + self.gamma * Qtab[s2].max() - Qtab[s, a])
            Qs.append((ctx[:, Lay.SLOT][:, :n_states].T.copy(), Qtab.copy()))
        return Qs, sel


def random_mdp_trajectory(n_states, n_actions, T, seed, alpha=0.1, gamma=0.9, eps=0.3,
                          shift=0.0, noise=0.0):
    """Random MDP (Dir(1) transitions, Beta(2,2) mean rewards, minus `shift`, plus N(0, noise^2)
    per observation) and an eps-greedy behaviour trajectory."""
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(n_states), size=(n_states, n_actions))
    R = rng.beta(2, 2, size=(n_states, n_actions)) - shift
    Q = np.zeros((n_states, n_actions))
    s = int(rng.integers(n_states))
    traj = []
    for _ in range(T):
        a = int(rng.integers(n_actions)) if rng.random() < eps else int(np.argmax(Q[s]))
        s2 = int(rng.choice(n_states, p=P[s, a]))
        r = float(R[s, a] + noise * rng.standard_normal())
        traj.append((s, a, r, s2))
        Q[s, a] += alpha * (r + gamma * Q[s2].max() - Q[s, a])
        s = s2
    return traj


def _one(args):
    beta, bm, nA, alpha, gamma, rew, seed, T = args
    rng = np.random.default_rng(1000 * seed + 7)
    nS = int(rng.integers(2, 9))
    shift, noise = {'clean': (0.0, 0.0), 'noisy_negative': (0.5, 0.3)}[rew]
    traj = random_mdp_trajectory(nS, nA, T, seed, alpha, gamma, shift=shift, noise=noise)
    m = HandwiredQv2(alpha, gamma, beta=beta, beta_max=bm)
    Qs, sel = m.run(traj, nS, nA, 'self')
    Qt, _ = m.run(traj, nS, nA, 'teacher')
    err = max(np.abs(qh - q).max() for qh, q in Qs)
    ok = []
    for qh, q in Qs:
        srt = np.sort(q, 1)
        u = (srt[:, -1] - srt[:, -2]) > 1e-6
        ok += list((qh.argmax(1) == q.argmax(1))[u])
    diff = max(np.abs(a[0] - b[0]).max() for a, b in zip(Qs, Qt))
    return dict(beta=beta, beta_max=bm, nA=nA, nS=nS, alpha=alpha, gamma=gamma, rewards=rew,
                seed=seed, max_err=float(err), greedy_agree=float(np.mean(ok)),
                self_vs_teacher=float(diff))


def verify(betas=(50, 1e3, 1e4), beta_maxes=(1e3, 1e4, 1e5), n_seeds=2, T=500, procs=16):
    from itertools import product
    from multiprocessing import Pool
    tasks = [(b, bm, nA, al, g, rw, sd, T) for b, bm, nA, al, g, rw, sd in product(
        betas, beta_maxes, (2, 3, 4), (0.1, 0.2, 0.5), (0.9, 0.95),
        ('clean', 'noisy_negative'), range(n_seeds))]
    with Pool(procs) as pool:
        rows = pool.map(_one, tasks)
    import pandas as pd
    df = pd.DataFrame(rows)
    summ = df.groupby(['beta', 'beta_max']).agg(
        max_err=('max_err', 'max'), mean_err=('max_err', 'mean'),
        greedy_agree=('greedy_agree', 'mean'), min_greedy_agree=('greedy_agree', 'min'),
        self_vs_teacher=('self_vs_teacher', 'max'), n=('max_err', 'size')).reset_index()
    print(summ.to_string(index=False))
    print('\nby |A| (beta=1e3, beta_max=1e4):')
    print(df[(df.beta == 1e3) & (df.beta_max == 1e4)].groupby('nA').agg(
        max_err=('max_err', 'max'), greedy_agree=('greedy_agree', 'mean')).to_string())
    return df, summ


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import paths
    df, summ = verify()
    out = paths.FIGURES / 'handwired'
    out.mkdir(parents=True, exist_ok=True)
    summ.to_csv(out / 'verification.csv', index=False)
    df.to_csv(out / 'verification_raw.csv', index=False)
    print('wrote', out / 'verification.csv')
