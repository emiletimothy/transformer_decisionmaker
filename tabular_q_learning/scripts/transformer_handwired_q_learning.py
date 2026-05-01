"""
True PyTorch Handwired Q-Learning Transformer (Coconut Construction).
Strictly follows the mathematical notation and proof from the LaTeX paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class QLearningConfig:
    n_states: int = 4
    n_actions: int = 2
    alpha: float = 0.1
    gamma: float = 0.9

def _compute_vocab(cfg: QLearningConfig):
    idx = 0
    S = list(range(idx, idx + cfg.n_states)); idx += cfg.n_states
    A = list(range(idx, idx + cfg.n_actions)); idx += cfg.n_actions
    BOS = idx; idx += 1
    R = idx; idx += 1
    EVAL = idx; idx += 1
    SELECT = idx; idx += 1
    QCURR = idx; idx += 1
    QNEXT = idx; idx += 1
    UPDATE = idx; idx += 1
    return idx, S, A, BOS, R, EVAL, SELECT, QCURR, QNEXT, UPDATE

# --- Mathematical Subspace Projectors (from the paper) ---
def P_B(start, end, d_model):
    """Orthogonal projector onto block B."""
    P = torch.zeros(d_model, d_model)
    for i in range(start, end):
        P[i, i] = 1.0
    return P

def S_A_to_B(src_start, dst_start, length, d_model):
    """Rigid swap that copies block A into block B."""
    S = torch.zeros(d_model, d_model)
    for i in range(length):
        S[dst_start + i, src_start + i] = 1.0
    return S

def Pi_tau(indices, start, d_model):
    """Rank-1 projector onto orthonormal axis u_tau within block B."""
    Pi = torch.zeros(d_model, d_model)
    if isinstance(indices, int):
        indices = [indices]
    for idx in indices:
        Pi[start + idx, start + idx] = 1.0
    return Pi

def u_vec(idx, start, d_model):
    """Helper to get a specific embedding axis u_v."""
    u = torch.zeros(d_model)
    u[start + idx] = 1.0
    return u

# --- Attention Head Architectures ---

class FixedOffsetChooser(nn.Module):
    """Generalized Attention Chooser FO(T, -ell)."""
    def __init__(self, target_tokens, offset, W_V, W_O, d_model):
        super().__init__()
        self.target_tokens = target_tokens
        self.offset = offset
        self.register_buffer('W_V', W_V)
        self.register_buffer('W_O', W_O)

    def forward(self, x):
        B, seq_len, D = x.shape
        
        # Check presence in the identity subspace exactly
        trigger = torch.zeros(B, seq_len, 1, device=x.device)
        for tok_idx in self.target_tokens:
            trigger += (x[:, :, tok_idx:tok_idx+1] > 0.5).float()
        trigger = (trigger > 0).float()
        
        V = x @ self.W_V.T
        
        V_shifted = torch.zeros_like(V)
        if seq_len > self.offset:
            V_shifted[:, self.offset:, :] = V[:, :-self.offset, :]
        
        return (trigger * V_shifted) @ self.W_O.T
    
class SoftmaxAttentionHead(nn.Module):
    """Standard Softmax Content Matching Head (with indicator scaling)."""
    def __init__(self, W_Q, W_K, W_V, W_O, temp=1000.0):
        super().__init__()
        self.register_buffer('W_Q', W_Q)
        self.register_buffer('W_K', W_K)
        self.register_buffer('W_V', W_V)
        self.register_buffer('W_O', W_O)
        self.temp = temp # Temp → inf perfectly recovers orthonormal indicator matches

    def forward(self, x):
        Q = x @ self.W_Q.T
        K = x @ self.W_K.T
        V = x @ self.W_V.T
        scores = Q @ K.transpose(-2, -1) * self.temp
        seq_len = x.shape[1]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        return (weights @ V) @ self.W_O.T

class LinearAttentionHead(nn.Module):
    """Linear Attention Head (no softmax) for exact scalar dot-product assembly."""
    def __init__(self, W_Q, W_K, W_V, W_O, diagonal_only=False):
        super().__init__()
        self.register_buffer('W_Q', W_Q)
        self.register_buffer('W_K', W_K)
        self.register_buffer('W_V', W_V)
        self.register_buffer('W_O', W_O)
        self.diagonal_only = diagonal_only

    def forward(self, x):
        Q = x @ self.W_Q.T
        K = x @ self.W_K.T
        V = x @ self.W_V.T
        scores = Q @ K.transpose(-2, -1) # Scalar dot product
        seq_len = x.shape[1]
        if self.diagonal_only:
            mask = torch.eye(seq_len, device=x.device)
        else:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        scores = scores * mask # Nullify future (or non-diagonal) entirely
        return (scores @ V) @ self.W_O.T


# --- Main Transformer Module ---
class ContinuousThoughtQTransformer(nn.Module):
    def __init__(self, cfg: QLearningConfig):
        super().__init__()
        self.cfg = cfg
        v_size, self.S, self.A, self.BOS, self.R, self.EVAL, self.SELECT, self.QCURR, self.QNEXT, self.UPDATE = _compute_vocab(cfg)
        self.d_TE = v_size
        self.d_model = 3 * self.d_TE 
        d_m, d_TE = self.d_model, self.d_TE
        
        # Build strict mathematical projectors
        P_id = P_B(0, d_TE, d_m)
        P_buf1 = P_B(d_TE, 2*d_TE, d_m)
        P_buf2 = P_B(2*d_TE, 3*d_TE, d_m)
        
        Pi_S_id = Pi_tau(self.S, 0, d_m)
        Pi_A_id = Pi_tau(self.A, 0, d_m)
        Pi_BOS_id = Pi_tau([self.BOS], 0, d_m)
        Pi_Qcurr_id = Pi_tau(self.QCURR, 0, d_m)
        Pi_Qnext_id = Pi_tau(self.QNEXT, 0, d_m)
        
        Pi_S_buf1 = Pi_tau(self.S, d_TE, d_m)
        Pi_Qcurr_buf2 = Pi_tau(self.QCURR, 2*d_TE, d_m)
        Pi_Qnext_buf2 = Pi_tau(self.QNEXT, 2*d_TE, d_m)
        Pi_Select_buf2 = Pi_tau(self.SELECT, 2*d_TE, d_m)
        Pi_r_buf1 = Pi_tau(self.R, d_TE, d_m)
        
        S_id_to_id = S_A_to_B(0, 0, d_TE, d_m)
        S_id_to_buf1 = S_A_to_B(0, d_TE, d_TE, d_m)
        S_id_to_buf2 = S_A_to_B(0, 2*d_TE, d_TE, d_m)
        S_buf1_to_id = S_A_to_B(d_TE, 0, d_TE, d_m)
        S_buf2_to_id = S_A_to_B(2*d_TE, 0, d_TE, d_m)
        I = torch.eye(d_m)
        
        # --- Layer 1: Workspace routing and context fetching ---
        W_V_1 = P_id @ Pi_S_id
        
        # 1.4 Current-phase tag for a_t
        W_V_1_4 = P_id @ Pi_Qcurr_id
        
        # 1.5 Next-phase tag for a_i
        sum_u_A = sum(u_vec(a, 0, d_m) for a in self.A)
        W_Q_1_5 = torch.outer(u_vec(self.QNEXT, 0, d_m), sum_u_A) @ P_id
        W_K_1_5 = Pi_Qnext_id @ P_id
        W_V_1_5 = P_id @ Pi_Qnext_id
        
        # 1.6 Context fetch — boost c_a via u_BOS marker so c_a wins over Phase-1 a-tokens
        bos_boost = sum(torch.outer(u_vec(self.BOS, 0, d_m), u_vec(a, 0, d_m)) for a in self.A)
        W_Q_1_6 = Pi_A_id @ P_id + bos_boost
        W_K_1_6 = (Pi_A_id + Pi_BOS_id) @ P_id
        
        # 1.7 Select fetches s_t
        W_Q_1_7 = torch.outer(u_vec(self.QCURR, 0, d_m), u_vec(self.SELECT, 0, d_m)) @ P_id
        W_K_1_7 = Pi_Qcurr_id @ P_id
        W_V_1_7 = Pi_S_id @ P_id
        
        # 1.8 Route a_t (Phase 1) into final UPDATE's id so UPDATE knows which column to inherit
        # UPDATE is at offset 2*n_actions+5 after a_t (Phase 1); UPDATE is the only token with u_UPDATE
        # in id (context tokens now carry u_BOS instead), so target=[UPDATE] fires uniquely.
        offset_update_to_at = 2 * cfg.n_actions + 5
        W_V_1_8 = P_id @ Pi_A_id

        # 1.9 Route s_t into SELECT's buf1 so head 4.3's αγ·max term lands at u_{s_t}
        offset_select_to_st = 2 * cfg.n_actions + 4
        W_V_1_9 = P_id @ Pi_S_id

        # --- Layer 2: Self-attention state evaluation ---
        # 2.1 Compute Q values (Identity-kernel isolated to self)
        W_Q_2_1 = Pi_S_id
        W_K_2_1 = S_buf1_to_id
        W_V_2_1 = Pi_Qnext_id - Pi_Qcurr_id

        # 2.2 UPDATE inherits Q[:, a_t] from c_{a_t}. After head 1.8, UPDATE has u_{a_t} in id;
        # c_{a_t} is uniquely boosted by u_BOS so softmax concentrates on it. Layer-1 head 1.6
        # leaves c_a's residual buf1 = 2·Q[:, a] (input + self-attended copy), so we scale by 1/2.
        W_Q_2_2 = Pi_A_id @ P_id + bos_boost
        W_K_2_2 = (Pi_A_id + Pi_BOS_id) @ P_id
        W_V_2_2 = 0.5 * P_buf1
        
        # --- Layer 3: Q-value maximization and action selection ---
        # 3.1 Argmax and max value
        W_Q_3_1 = torch.outer(u_vec(self.QNEXT, 0, d_m), u_vec(self.SELECT, 0, d_m)) @ P_id
        W_K_3_1 = 1000.0 * S_buf2_to_id @ Pi_Qnext_buf2 # Pre-scaled beta -> inf
        W_V_3_1 = P_id @ Pi_A_id + P_buf2 @ torch.outer(u_vec(self.SELECT, 2*d_TE, d_m), u_vec(self.QNEXT, 2*d_TE, d_m)) @ P_buf2
        
        # (Layer 3.2 "update inheritance" removed — that head was broken: it pulled from QCURR
        # whose buf1 is 0, AND was pulled toward a_t Phase 1 with equal score, halving Q[:, a_t].
        # Inheritance is now handled correctly by Layer 2 head 2.2 via c_{a_t}.)

        # --- Layer 4: TD error assembly (Parallel Linear Heads) ---
        alpha, gamma = cfg.alpha, cfg.gamma
        
        # 4.1 Subtract Qcurr
        W_Q_4_1 = torch.outer(u_vec(self.QCURR, 0, d_m), u_vec(self.UPDATE, 0, d_m)) @ P_id
        W_K_4_1 = S_buf2_to_id @ Pi_Qcurr_buf2
        W_V_4_1 = alpha * S_id_to_buf1 @ Pi_S_id
        
        # 4.2 Add reward
        W_Q_4_2 = torch.outer(u_vec(self.R, 0, d_m), u_vec(self.UPDATE, 0, d_m)) @ P_id
        W_K_4_2 = S_buf1_to_id @ Pi_r_buf1
        W_V_4_2 = alpha * S_id_to_buf1 @ Pi_S_id
        
        # 4.3 Add discounted max next Q
        W_Q_4_3 = torch.outer(u_vec(self.SELECT, 0, d_m), u_vec(self.UPDATE, 0, d_m)) @ P_id
        W_K_4_3 = S_buf2_to_id @ Pi_Select_buf2
        W_V_4_3 = alpha * gamma * P_buf1 @ Pi_S_buf1
        
        self.layer1 = nn.ModuleList([
            FixedOffsetChooser(self.A, 1, W_V_1, S_id_to_id, d_m),
            FixedOffsetChooser([self.SELECT], 2, W_V_1, S_id_to_id, d_m),
            FixedOffsetChooser([self.R], 2, W_V_1, S_id_to_id, d_m),
            FixedOffsetChooser(self.A, 2, W_V_1_4, I, d_m),
            SoftmaxAttentionHead(W_Q_1_5, W_K_1_5, W_V_1_5, I),
            SoftmaxAttentionHead(W_Q_1_6, W_K_1_6, P_buf1, I),
            SoftmaxAttentionHead(W_Q_1_7, W_K_1_7, W_V_1_7, S_id_to_buf1),
            FixedOffsetChooser([self.UPDATE], offset_update_to_at, W_V_1_8, S_id_to_id, d_m),
            FixedOffsetChooser([self.SELECT], offset_select_to_st, W_V_1_9, S_id_to_buf1, d_m),
        ])
        self.layer2 = nn.ModuleList([
            LinearAttentionHead(W_Q_2_1, W_K_2_1, W_V_2_1, S_id_to_buf2, diagonal_only=True),
            SoftmaxAttentionHead(W_Q_2_2, W_K_2_2, W_V_2_2, I),
        ])
        self.layer3 = nn.ModuleList([
            SoftmaxAttentionHead(W_Q_3_1, W_K_3_1, W_V_3_1, I, temp=1.0), # beta=1000 baked into W_K
        ])
        self.layer4 = nn.ModuleList([
            LinearAttentionHead(W_Q_4_1, W_K_4_1, W_V_4_1, I),
            LinearAttentionHead(W_Q_4_2, W_K_4_2, W_V_4_2, I),
            LinearAttentionHead(W_Q_4_3, W_K_4_3, W_V_4_3, I)
        ])

    def forward(self, x):
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            residual = x
            out = torch.zeros_like(x)
            for head in layer:
                out += head(residual)
            x = residual + out
        return x