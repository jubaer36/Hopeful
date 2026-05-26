"""Block C: Multi-Frequency Module (M³Net §3.3, FAGCN-style self-gating).

Global heterogeneous graph (fully-connected within modality + cross-modal
same-utterance edges).  Per-edge self-gating balances low-pass and high-pass
signals without a separate ε hyperparameter.

Update rule (K layers):
  F^{k+1}_i = F^k_i + Σ_j r_ij * Â_ij * F^k_j
  r_ij = tanh(W3 [F^k_i ∥ F^k_j])    (r^l - r^h = r, so r ∈ [-1,1])
  Â = D^{-1/2} A D^{-1/2}

Node layout: [T_0..T_{L-1}, A_0..A_{L-1}, V_0..V_{L-1}] — same as IGM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_global_graph(L: int, device: torch.device):
    """
    Fully-connected within each modality + cross-modal same-utterance edges.
    Returns adjacency A (3L, 3L) with unit weights (will be normalised).
    """
    N = 3 * L
    A = torch.zeros(N, N, device=device)

    # Intra-modality: fully-connected within each block (excluding self)
    for off in (0, L, 2 * L):
        for i in range(L):
            for j in range(L):
                if i != j:
                    A[off + i, off + j] = 1.0

    # Cross-modal same utterance: t_i ↔ a_i ↔ v_i
    for i in range(L):
        ti, ai, vi = i, L + i, 2 * L + i
        for u, v in [(ti, ai), (ti, vi), (ai, vi)]:
            A[u, v] = 1.0
            A[v, u] = 1.0

    return A


def _normalise_adj(A: torch.Tensor):
    """Symmetric normalisation: Â = D^{-1/2} A D^{-1/2}."""
    deg = A.sum(dim=1).clamp(min=1e-6)
    d_inv_sqrt = deg.pow(-0.5)
    return d_inv_sqrt.unsqueeze(-1) * A * d_inv_sqrt.unsqueeze(0)


class MultiFrequencyModule(nn.Module):
    """
    Block C of HyFIN-Net.

    Same 3L node layout as IGM/HM.
    One shared W3 gate (2*d_h → 1) per layer, applied per-edge.
    """

    def __init__(self, d_h: int, k_freq: int = 4, dropout: float = 0.1):
        super().__init__()
        self.k_freq = k_freq

        # Gate weight matrices (one per layer for capacity)
        self.W3 = nn.ModuleList([
            nn.Linear(d_h * 2, 1, bias=True) for _ in range(k_freq)
        ])
        self.ln = nn.ModuleList([
            nn.LayerNorm(d_h) for _ in range(k_freq)
        ])
        self.drop = nn.Dropout(dropout)

    def forward(self, T: torch.Tensor, A: torch.Tensor, V: torch.Tensor):
        """
        T, A, V: (L, d_h)
        Returns: F_T, F_A, F_V each (L, d_h)
        """
        L = T.size(0)
        F = torch.cat([T, A, V], dim=0)          # (3L, d_h)
        N = F.size(0)
        device = F.device

        A_raw = _build_global_graph(L, device)   # (3L, 3L)
        A_hat = _normalise_adj(A_raw)            # (3L, 3L)
        edge_mask = (A_raw > 0)                  # (3L, 3L) bool

        for k in range(self.k_freq):
            # Compute self-gating coefficients r_ij for all edge pairs
            F_i = F.unsqueeze(1).expand(N, N, -1)   # (N, N, d_h)
            F_j = F.unsqueeze(0).expand(N, N, -1)   # (N, N, d_h)
            pairs = torch.cat([F_i, F_j], dim=-1)   # (N, N, 2*d_h)

            # Only compute gates where edges exist (mask others to 0)
            r = torch.zeros(N, N, device=device)
            if edge_mask.any():
                # Vectorised over all edges
                pairs_flat = pairs[edge_mask]            # (E, 2*d_h)
                r_flat = torch.tanh(self.W3[k](pairs_flat).squeeze(-1))  # (E,)
                r[edge_mask] = r_flat

            # Gated propagation: F ← F + (r ⊙ Â) @ F
            gated_A = r * A_hat                          # (N, N)
            F = F + self.drop(self.ln[k](gated_A @ F))  # (N, d_h)

        F_T = F[0:L]
        F_A = F[L:2*L]
        F_V = F[2*L:3*L]
        return F_T, F_A, F_V
