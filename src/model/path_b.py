"""Stage 3 Path B: M3Net-style multi-frequency Chebyshev spectral filtering."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SpectralFilter(nn.Module):
    """
    Per-modality spectral filter on an undirected window graph.
    Chebyshev polynomial basis (degree k_poly), K_f learnable filters,
    adaptive per-utterance frequency integration.
    """

    def __init__(self, d: int, window_size: int = 6, k_poly: int = 2, k_freq: int = 4):
        super().__init__()
        self.window_size = window_size
        self.k_poly      = k_poly
        self.k_freq      = k_freq

        # Dedicated edge projection — separate from filtered features to avoid circularity
        self.edge_proj = nn.Linear(d, 64, bias=False)

        # θ[k, j]: scalar coefficient for filter k, polynomial degree j
        self.theta = nn.Parameter(torch.randn(k_freq, k_poly + 1) * 0.02)

        # Adaptive per-utterance frequency integration
        self.freq_proj = nn.Linear(d, k_freq)

    def _laplacian(self, nodes: torch.Tensor) -> torch.Tensor:
        """Build normalized graph Laplacian L_norm = 2L/λ_max − I."""
        N    = nodes.size(0)
        efts = self.edge_proj(nodes)                           # (N, 64)
        norm = F.normalize(efts, p=2, dim=-1)
        cos  = norm @ norm.t()                                 # (N, N)

        idx  = torch.arange(N, device=nodes.device)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        mask = (dist <= self.window_size) & (dist > 0)

        W    = cos * mask.float()
        W    = (W + W.t()) / 2                                 # symmetrise
        D    = W.sum(dim=1)
        L    = torch.diag(D) - W

        lam_max = torch.linalg.eigvalsh(L).max().clamp(min=1e-6)
        I       = torch.eye(N, device=nodes.device)
        return 2.0 * L / lam_max - I                          # (N, N)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        """nodes: (N, d) — stop-grad applied upstream → (N, d)"""
        L_norm = self._laplacian(nodes)

        # Chebyshev recurrence: T0, T1, T2
        T0 = nodes
        T1 = L_norm @ nodes
        T2 = 2 * (L_norm @ T1) - T0
        basis = [T0, T1, T2]  # k_poly=2 → degrees 0,1,2

        # K_f filters, each a linear combination of basis vectors
        filtered = [
            sum(self.theta[k, j] * basis[j] for j in range(self.k_poly + 1))
            for k in range(self.k_freq)
        ]                                                      # list of (N, d)

        # Adaptive integration: each utterance learns its filter weighting
        freq_w  = torch.softmax(self.freq_proj(nodes), dim=-1)  # (N, K_f)
        X_freq  = sum(
            freq_w[:, k:k+1] * filtered[k] for k in range(self.k_freq)
        )                                                      # (N, d)
        return X_freq


class PathB(nn.Module):
    """
    Path B: three independent spectral filters, one per modality.
    Input nodes are stop-gradients from Stage 2 — Path B cannot
    influence Stage 2 parameter updates.
    """

    def __init__(self, d: int, window_size: int = 6, k_poly: int = 2, k_freq: int = 4):
        super().__init__()
        self.filter_T = _SpectralFilter(d, window_size, k_poly, k_freq)
        self.filter_A = _SpectralFilter(d, window_size, k_poly, k_freq)
        self.filter_V = _SpectralFilter(d, window_size, k_poly, k_freq)

    def forward(self, T, A, V):
        """T, A, V: (N, d) from Stage 2 → T_freq, A_freq, V_freq each (N, d)"""
        T_sg = T.detach()
        A_sg = A.detach()
        V_sg = V.detach()
        return self.filter_T(T_sg), self.filter_A(A_sg), self.filter_V(V_sg)
