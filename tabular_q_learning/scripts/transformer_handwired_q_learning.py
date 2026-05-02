"""
Continuous Thought Q-Learning Transformer
Fully aligned with updated 4-layer theoretical construction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# Inverse temperature for the behavior policy (softmax over Q_next). Set by user.
BETA = 1e3

# Inverse temperature used solely to harden routing-only softmax heads into
# argmax operations. Independent of the policy temperature BETA.
HARD_TEMP = 1e6


# =========================
# Config
# =========================

@dataclass
class QLearningConfig:
    n_states: int = 4
    n_actions: int = 2
    alpha: float = 0.1
    gamma: float = 0.9
    beta: float = BETA


# =========================
# Vocabulary
# =========================

def build_vocab(cfg):
    idx = 0
    S = list(range(idx, idx + cfg.n_states)); idx += cfg.n_states
    A = list(range(idx, idx + cfg.n_actions)); idx += cfg.n_actions

    BOS = idx; idx += 1
    R = idx; idx += 1

    SELECT = idx; idx += 1
    UPDATE = idx; idx += 1

    QCURR = idx; idx += 1
    QNEXT = idx; idx += 1

    return idx, S, A, BOS, R, SELECT, UPDATE, QCURR, QNEXT


# =========================
# Geometry primitives
# =========================

def P_block(start, end, d):
    P = torch.zeros(d, d)
    P[start:end, start:end] = torch.eye(end - start)
    return P


def S_swap(src, dst, size, d):
    S = torch.zeros(d, d)
    S[dst:dst+size, src:src+size] = torch.eye(size)
    return S


def Pi(indices, start, d):
    P = torch.zeros(d, d)
    for i in (indices if isinstance(indices, list) else [indices]):
        P[start + i, start + i] = 1
    return P


def u(i, start, d):
    v = torch.zeros(d)
    v[start + i] = 1
    return v


# =========================
# Attention heads
# =========================

class FixedOffsetChooser(nn.Module):
    """
    FO(T, -ℓ): routes any token in T to ℓ steps back.
    """
    def __init__(self, T, offset, W_V, W_O):
        super().__init__()
        self.T = T
        self.offset = offset
        self.register_buffer("W_V", W_V)
        self.register_buffer("W_O", W_O)

    def forward(self, x):
        B, L, D = x.shape

        trigger = torch.zeros(B, L, 1, device=x.device)
        for t in self.T:
            trigger += (x[:, :, t:t+1] > 0).float()

        V = x @ self.W_V.T

        shifted = torch.zeros_like(V)
        if L > self.offset:
            shifted[:, self.offset:] = V[:, :-self.offset]

        return (trigger * shifted) @ self.W_O.T


class SoftmaxHead(nn.Module):
    def __init__(self, WQ, WK, WV, WO, temp=1.0, causal=False):
        super().__init__()
        self.register_buffer("WQ", WQ)
        self.register_buffer("WK", WK)
        self.register_buffer("WV", WV)
        self.register_buffer("WO", WO)
        self.temp = temp
        self.causal = causal

    def forward(self, x):
        Q = x @ self.WQ.T
        K = x @ self.WK.T
        V = x @ self.WV.T

        attn = (Q @ K.transpose(-2, -1)) * self.temp
        if self.causal:
            L = attn.shape[-1]
            mask = torch.triu(torch.ones(L, L, device=attn.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = attn @ V

        q_active = (Q.abs().sum(dim=-1, keepdim=True) > 1e-6).to(out.dtype)
        out = out * q_active

        return out @ self.WO.T


class LinearHead(nn.Module):
    def __init__(self, WQ, WK, WV, WO, per_token=False):
        super().__init__()
        self.register_buffer("WQ", WQ)
        self.register_buffer("WK", WK)
        self.register_buffer("WV", WV)
        self.register_buffer("WO", WO)
        self.per_token = per_token

    def forward(self, x):
        Q = x @ self.WQ.T
        K = x @ self.WK.T
        V = x @ self.WV.T

        if self.per_token:
            scalar = (Q * K).sum(dim=-1, keepdim=True)
            out = scalar * V
        else:
            scores = Q @ K.transpose(-2, -1)
            out = scores @ V
        return out @ self.WO.T


# =========================
# Transformer
# =========================

class ContinuousThoughtQTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        d_vocab, S, A, BOS, R, SELECT, UPDATE, QCURR, QNEXT = build_vocab(cfg)
        self.d_TE = d_vocab
        self.d_model = 3 * d_vocab

        self.S, self.A = S, A
        self.BOS, self.R = BOS, R
        self.SELECT, self.UPDATE = SELECT, UPDATE
        self.QCURR, self.QNEXT = QCURR, QNEXT

        d = self.d_model
        dTE = self.d_TE

        # blocks
        P_id = P_block(0, dTE, d)
        P_buf1 = P_block(dTE, 2*dTE, d)
        P_buf2 = P_block(2*dTE, 3*dTE, d)

        Pi_S = Pi(S, 0, d)
        Pi_A = Pi(A, 0, d)

        Pi_Qc = Pi([QCURR], 0, d)
        Pi_Qn = Pi([QNEXT], 0, d)

        Pi_S_b1 = Pi(S, dTE, d)
        Pi_R_b1 = Pi([R], dTE, d)
        Pi_Qc_b2 = Pi([QCURR], 2*dTE, d)
        Pi_Qn_b2 = Pi([QNEXT], 2*dTE, d)
        Pi_sel_b2 = Pi([SELECT], 2*dTE, d)

        S_b2_to_id = S_swap(2*dTE, 0, dTE, d)
        S_b1_to_id = S_swap(dTE, 0, dTE, d)
        S_id_to_b1 = S_swap(0, dTE, dTE, d)
        S_id_to_b2 = S_swap(0, 2*dTE, dTE, d)

        I = torch.eye(d)

        beta = cfg.beta
        sum_uA = sum(u(act, 0, d) for act in A)

        # =========================
        # Layer 1: routing
        # =========================

        self.l1 = nn.ModuleList([
            FixedOffsetChooser(A, 1, P_id @ Pi_S, I),
            FixedOffsetChooser([SELECT], 2, P_id @ Pi_S, I),
            FixedOffsetChooser([R], 2, P_id @ Pi_S, I),

            FixedOffsetChooser(A, 2, P_id @ Pi_Qc, I),

            SoftmaxHead(
                torch.outer(u(QNEXT,0,d), sum(u(a,0,d) for a in A)) @ P_id,
                Pi_Qn @ P_id,
                P_id @ Pi_Qn,
                I,
                temp=HARD_TEMP,
                causal=True,
            ),

            SoftmaxHead(
                (Pi_A + torch.outer(u(UPDATE,0,d), sum(u(a,0,d) for a in A))) @ P_id,
                (Pi_A + Pi([UPDATE], 0, d)) @ P_id,
                P_buf1,
                I,
                temp=HARD_TEMP,
            ),
        ])

        # =========================
        # Layer 2: Q evaluation
        # =========================

        self.l2 = nn.ModuleList([
            SoftmaxHead(
                torch.outer(u(QCURR,0,d) + sum_uA, u(SELECT,0,d)) @ P_id,
                (Pi_Qc + Pi_A) @ P_id,
                Pi_S @ P_id,
                S_id_to_b1,
                temp=HARD_TEMP,
            ),

            LinearHead(Pi_S, S_b1_to_id, Pi_Qn - Pi_Qc, S_id_to_b2, per_token=True),
        ])

        # =========================
        # Layer 3: selection + max + Update inheritance
        # =========================

        self.l3 = nn.ModuleList([
            SoftmaxHead(
                torch.outer(u(QNEXT,0,d), u(SELECT,0,d)) @ P_id,
                beta * S_b2_to_id @ Pi_Qn_b2,
                P_id @ Pi_A,
                I
            ),

            SoftmaxHead(
                torch.outer(u(QNEXT,0,d), u(SELECT,0,d)) @ P_id,
                beta * S_b2_to_id @ Pi_Qn_b2,
                torch.outer(u(SELECT,0,d), u(QNEXT, 2*dTE, d)) @ P_buf2,
                S_id_to_b2
            ),

            SoftmaxHead(
                torch.outer(u(QCURR,0,d) + sum_uA, u(UPDATE,0,d)) @ P_id,
                (Pi_Qc + Pi_A) @ P_id,
                Pi_A + P_buf1,
                I,
                temp=HARD_TEMP
            ),
        ])

        # =========================
        # Layer 4: TD assembly
        # =========================

        a, g = cfg.alpha, cfg.gamma

        self.l4 = nn.ModuleList([
            LinearHead(
                torch.outer(u(QCURR,0,d), u(UPDATE,0,d)) @ P_id,
                S_b2_to_id @ Pi_Qc_b2,
                a * S_id_to_b1 @ Pi_S,
                I
            ),

            LinearHead(
                torch.outer(u(R,0,d), u(UPDATE,0,d)) @ P_id,
                S_b1_to_id @ Pi_R_b1,
                a * S_id_to_b1 @ Pi_S,
                I
            ),

            LinearHead(
                torch.outer(u(SELECT,0,d), u(UPDATE,0,d)) @ P_id,
                S_b2_to_id @ Pi_sel_b2,
                a * g * P_buf1 @ Pi_S_b1,
                I
            ),
        ])

    def _token_label(self, seq_len):
        cfg = self.cfg
        nA = cfg.n_actions
        labels = []
        for a in range(nA):
            labels.append(f"c_a{a}")
        labels += ["QCURR", "S_t", "A_t", "R", "QNEXT"]
        for a in range(nA):
            labels += [f"S_next({a})", f"A({a})"]
        labels += ["SELECT", "A_star", "UPDATE"]
        if len(labels) != seq_len:
            return [f"tok{i}" for i in range(seq_len)]
        return labels

    def _summarize_token(self, tok):
        dTE = self.d_TE
        id_part = tok[:dTE]
        b1 = tok[dTE:2*dTE]
        b2 = tok[2*dTE:3*dTE]

        def nz(v):
            idxs = (v.abs() > 1e-6).nonzero(as_tuple=True)[0].tolist()
            return {self._name(i): float(v[i]) for i in idxs}
        return {"id": nz(id_part), "buf1": nz(b1), "buf2": nz(b2)}

    def _name(self, i):
        if i in self.S:
            return f"S{self.S.index(i)}"
        if i in self.A:
            return f"A{self.A.index(i)}"
        return {self.BOS: "BOS", self.R: "R", self.SELECT: "SEL",
                self.UPDATE: "UPD", self.QCURR: "QCURR",
                self.QNEXT: "QNEXT"}.get(i, f"v{i}")

    def forward(self, x, log=False, log_tokens=None):
        if log:
            L = x.shape[1]
            labels = self._token_label(L)
            print("\n===== INPUT =====")
            picks = log_tokens if log_tokens is not None else range(L)
            for i in picks:
                print(f"  [{i}] {labels[i]}: {self._summarize_token(x[0, i])}")

        for li, layer in enumerate([self.l1, self.l2, self.l3, self.l4], start=1):
            res = x
            out = 0
            for h in layer:
                out = out + h(res)
            x = res + out
            if log:
                print(f"\n===== AFTER LAYER {li} =====")
                picks = log_tokens if log_tokens is not None else range(x.shape[1])
                for i in picks:
                    print(f"  [{i}] {labels[i]}: {self._summarize_token(x[0, i])}")
        return x