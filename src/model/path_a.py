"""Stage 3 Path A: per-modality dynamic hypergraph + cross-modal gated attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F

_K_C_MAX = 32  # maximum contextual hyperedges; K_c per conversation ≤ this


def _top_tau_sparse(S: torch.Tensor, tau: int) -> torch.Tensor:
    """
    S: (N, K)  — raw attention scores
    Returns H: (N, K) — sparse incidence matrix.
    For each hyperedge column: keep top-τ nodes, softmax-normalize those τ entries.
    """
    N, K = S.shape
    tau = max(1, min(tau, N))
    topk_idx = torch.topk(S, k=tau, dim=0).indices             # (tau, K)
    mask = torch.zeros_like(S).scatter_(0, topk_idx, 1.0)
    # Softmax over unmasked positions per column
    S_masked = S.masked_fill(mask == 0, float("-inf"))
    return torch.softmax(S_masked, dim=0) * mask               # (N, K)


def _adaptive_kc_tau(N: int) -> tuple:
    K_c = max(2, N // 4)
    K_c = min(K_c, _K_C_MAX)
    tau = max(2, N // (K_c + 1))
    tau = min(tau, max(1, int(0.4 * N)))
    tau = min(tau, N)
    tau = max(tau, 1)
    return K_c, tau


class _HypergraphConv(nn.Module):
    """
    2-layer hypergraph convolution with residual connections.
    Implements: D_v^{-1} H W_e D_e^{-1} H^T X per layer.
    """

    def __init__(self, d: int, num_speakers: int):
        super().__init__()
        self.d            = d
        self.num_speakers = num_speakers
        self.W_q          = nn.Linear(d, d, bias=False)
        self.W_k          = nn.Linear(d, d, bias=False)
        self.prototypes   = nn.Parameter(torch.randn(_K_C_MAX, d) * 0.02)

        # Learnable hyperedge weights (log-parameterised for positivity via softplus)
        self.w_ctx = nn.Parameter(torch.zeros(_K_C_MAX))
        self.w_spk = nn.Parameter(torch.zeros(num_speakers))

        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)

    def _propagate(
        self,
        H:   torch.Tensor,  # (N, E)
        w_e: torch.Tensor,  # (E,)  positive weights
        X:   torch.Tensor,  # (N, d)
    ) -> torch.Tensor:
        D_v     = H.sum(dim=1).clamp(min=1e-6)    # (N,)
        D_e     = H.sum(dim=0).clamp(min=1e-6)    # (E,)
        we_de   = w_e / D_e                        # (E,)
        mid     = H * we_de.unsqueeze(0)           # (N, E)
        out     = mid @ (H.t() @ X)               # (N, d)
        return out / D_v.unsqueeze(-1)

    def forward(self, nodes: torch.Tensor, speaker_ids: torch.Tensor) -> torch.Tensor:
        """nodes: (N, d), speaker_ids: (N,) global int"""
        N   = nodes.size(0)
        K_c, tau = _adaptive_kc_tau(N)

        # Contextual hyperedges via attention over learned prototypes
        Q   = self.W_q(nodes)                                  # (N, d)
        K   = self.W_k(self.prototypes[:K_c])                  # (K_c, d)
        S   = Q @ K.t() / (self.d ** 0.5)                     # (N, K_c)
        H_ctx = _top_tau_sparse(S, tau)                        # (N, K_c)

        # Speaker hyperedges (fixed binary)
        H_spk = torch.zeros(N, self.num_speakers, device=nodes.device)
        H_spk.scatter_(1, speaker_ids.unsqueeze(1), 1.0)      # (N, num_speakers)

        H   = torch.cat([H_ctx, H_spk], dim=1)                # (N, K_c + num_speakers)
        w_e = torch.cat([
            F.softplus(self.w_ctx[:K_c]),
            F.softplus(self.w_spk),
        ], dim=0)                                              # (K_c + num_speakers,)

        # 2-layer HGNN with residuals
        prop1 = self._propagate(H, w_e, nodes)
        out1  = self.ln1(prop1 + nodes)
        prop2 = self._propagate(H, w_e, out1)
        out2  = self.ln2(prop2 + out1)
        return out2


class _CrossModalAttention(nn.Module):
    """
    Per-utterance gated cross-modal attention (5 directions).
    Gates start near-zero via bias_init=-3.0 and learn from data.
    """

    def __init__(self, d: int):
        super().__init__()
        self.scale = d ** -0.5
        # 5 gate linears: VT, VA, AT, TA, TV
        gates = {
            "gate_VT": nn.Linear(d * 2, d),
            "gate_VA": nn.Linear(d * 2, d),
            "gate_AT": nn.Linear(d * 2, d),
            "gate_TA": nn.Linear(d * 2, d),
            "gate_TV": nn.Linear(d * 2, d),
        }
        for name, layer in gates.items():
            nn.init.constant_(layer.bias, -3.0)
            setattr(self, name, layer)

    def _cross_attn(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Single-head cross-attention. Q,K,V: (N, d) → (N, d)."""
        attn = torch.softmax(Q @ K.t() * self.scale, dim=-1)  # (N, N)
        return attn @ V

    def forward(self, T, A, V):
        # Visual → Text (unconditional elevation)
        g_vt = torch.sigmoid(self.gate_VT(torch.cat([V, T], dim=-1)))
        V    = V + g_vt * self._cross_attn(V, T, T)

        # Visual → Audio
        g_va = torch.sigmoid(self.gate_VA(torch.cat([V, A], dim=-1)))
        V    = V + g_va * self._cross_attn(V, A, A)

        # Audio → Text
        g_at = torch.sigmoid(self.gate_AT(torch.cat([A, T], dim=-1)))
        A    = A + g_at * self._cross_attn(A, T, T)

        # Text → Audio (near-zero start, learns if data supports it)
        g_ta = torch.sigmoid(self.gate_TA(torch.cat([T, A], dim=-1)))
        T    = T + g_ta * self._cross_attn(T, A, A)

        # Text → Visual
        g_tv = torch.sigmoid(self.gate_TV(torch.cat([T, V], dim=-1)))
        T    = T + g_tv * self._cross_attn(T, V, V)

        return T, A, V


class PathA(nn.Module):
    """
    Path A: per-modality dynamic hypergraph followed by cross-modal gated attention.
    Three independent hypergraphs (one per modality) then 5-direction cross-modal mixing.
    """

    def __init__(self, d: int, num_speakers: int):
        super().__init__()
        self.hyper_T    = _HypergraphConv(d, num_speakers)
        self.hyper_A    = _HypergraphConv(d, num_speakers)
        self.hyper_V    = _HypergraphConv(d, num_speakers)
        self.cross_attn = _CrossModalAttention(d)

    def forward(self, T, A, V, speaker_ids):
        """T, A, V: (N, d); speaker_ids: (N,) → T', A', V' each (N, d)"""
        T = self.hyper_T(T, speaker_ids)
        A = self.hyper_A(A, speaker_ids)
        V = self.hyper_V(V, speaker_ids)
        T, A, V = self.cross_attn(T, A, V)
        return T, A, V
