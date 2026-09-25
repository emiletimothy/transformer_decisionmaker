"""
Shared utilities for relaxing the idealized assumptions of the handwired constructions.

Framework
---------
Both constructions are written in *canonical* coordinates: every token has a one-hot
identity embedding and the residual stream is split into disjoint blocks
[id | buf_1 | ... | buf_k | (pos)].  A physical transformer instead lives in R^D with
token embeddings Phi (columns = embedding directions).  Writing every weight matrix of
the construction as  W' = Phi W Phi^T  (i.e. all weights are expressed through the token
embeddings, exactly as in the paper's matrices) and tracking  y = Phi^T x'  gives:

    embed:     y_0      = G x_canon
    head read: uses y  (Phi^T x')
    head write: y      += G * out_canon
    noise:     y       += Phi^T eps,   eps ~ N(0, sigma^2 I_D)   =>   N(0, sigma^2 G)

where G = Phi^T Phi is the Gram matrix of the embedding directions.  G = I recovers the
exact construction; every relaxation below is a particular G (plus finite attention
inverse temperature beta and residual noise sigma).
"""
import numpy as np


def random_unit_gram(n_tokens, d, rng):
    """Gram matrix of n_tokens random unit vectors in R^d (non-orthogonal embeddings)."""
    U = rng.standard_normal((d, n_tokens))
    U /= np.linalg.norm(U, axis=0, keepdims=True)
    G = U.T @ U
    off = G - np.eye(n_tokens)
    return G, float(np.abs(off).max())


def overlap_blocks_gram(dTE, n_blocks, rho, rng):
    """
    Block Gram for n_blocks subspaces of dimension dTE that are each orthonormal
    internally but overlap each other:  Phi_j = sqrt(1-rho) E_j + sqrt(rho) F R_j,
    with F a shared dTE-dim subspace and R_j random orthogonal.  Cross-block Gram
    is rho * R_j^T R_k, so rho = 0 is the paper's disjoint buffers and
    max cross-block |cos| <= rho.
    """
    Rs = []
    for _ in range(n_blocks):
        A = rng.standard_normal((dTE, dTE))
        Qm, _ = np.linalg.qr(A)
        Rs.append(Qm)
    G = np.zeros((n_blocks * dTE, n_blocks * dTE))
    for j in range(n_blocks):
        for k in range(n_blocks):
            sl_j = slice(j * dTE, (j + 1) * dTE)
            sl_k = slice(k * dTE, (k + 1) * dTE)
            G[sl_j, sl_k] = np.eye(dTE) if j == k else rho * Rs[j].T @ Rs[k]
    off = G - np.eye(n_blocks * dTE)
    return G, float(np.abs(off).max())


def block_diag(*mats):
    n = sum(m.shape[0] for m in mats)
    out = np.zeros((n, n))
    i = 0
    for m in mats:
        k = m.shape[0]
        out[i:i + k, i:i + k] = m
        i += k
    return out


def psd_sqrt(G):
    w, V = np.linalg.eigh(G)
    w = np.clip(w, 0, None)
    return (V * np.sqrt(w)) @ V.T


def softmax_rows(S, mask=None):
    """Row-wise softmax; mask (bool, True = allowed)."""
    S = S.copy()
    if mask is not None:
        S = np.where(mask, S, -np.inf)
    m = S.max(axis=-1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    E = np.exp(S - m)
    if mask is not None:
        E = np.where(mask, E, 0.0)
    Z = E.sum(axis=-1, keepdims=True)
    Z = np.where(Z > 0, Z, 1.0)
    return E / Z


def mean_sem(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
