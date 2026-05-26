"""Block B: Inception Hyper-Graph Module (IGM + HM in parallel).

Node layout (3L nodes stacked):
  indices 0   … L-1   → text
  indices L   … 2L-1  → audio
  indices 2L  … 3L-1  → visual

IGM: n branches × (implicit-edge injection + k-GNN) → mean-pool → P^t, P^a, P^v
HM : M³Net-style hypergraph with edge-dependent γ_e(v) weights → Q^t, Q^a, Q^v
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: graph construction
# ─────────────────────────────────────────────────────────────────────────────

def _angular_weight(hi: torch.Tensor, hj: torch.Tensor) -> torch.Tensor:
    """Scalar angular similarity: 1 - arccos(cos(hi,hj)) / π  ∈ [0,1]."""
    cos = F.cosine_similarity(hi, hj, dim=-1).clamp(-1 + 1e-7, 1 - 1e-7)
    return 1.0 - torch.acos(cos) / 3.14159265358979


def _build_graph(h: torch.Tensor, past: int, future: int):
    """
    Build heterogeneous graph over 3L nodes (vectorised — no Python loops).
    Returns (inter, intra_past, intra_future) each as (src, tgt, weight).
    h: (3L, d_h)
    """
    L = h.size(0) // 3
    device = h.device

    idx = torch.arange(L, device=device)

    # ── R_inter: cross-modal same utterance ──────────────────────────────────
    # 3 pairs × L utterances × 2 directions = 6L edges
    t_idx = idx
    a_idx = idx + L
    v_idx = idx + 2 * L

    inter_pairs = torch.stack([
        torch.stack([t_idx, a_idx]), torch.stack([a_idx, t_idx]),
        torch.stack([t_idx, v_idx]), torch.stack([v_idx, t_idx]),
        torch.stack([a_idx, v_idx]), torch.stack([v_idx, a_idx]),
    ], dim=0)  # (6, 2, L)
    src_inter = inter_pairs[:, 0, :].reshape(-1)   # (6L,)
    tgt_inter = inter_pairs[:, 1, :].reshape(-1)
    w_inter   = _angular_weight(h[src_inter], h[tgt_inter])

    # ── R_intra^past / ^future: sliding window per modality ──────────────────
    src_p_list, tgt_p_list = [], []
    src_f_list, tgt_f_list = [], []

    for off in (0, L, 2 * L):
        i_all = idx  # (L,)
        for delta in range(1, max(past, future) + 1):
            j = i_all - delta                       # past: i - delta
            valid_p = (j >= 0) & (delta <= past)
            if valid_p.any():
                ii = i_all[valid_p] + off
                jj = j[valid_p]    + off
                src_p_list.append(ii); tgt_p_list.append(jj)
                src_p_list.append(jj); tgt_p_list.append(ii)

            j = i_all + delta                       # future: i + delta
            valid_f = (j < L) & (delta <= future)
            if valid_f.any():
                ii = i_all[valid_f] + off
                jj = j[valid_f]     + off
                src_f_list.append(ii); tgt_f_list.append(jj)
                src_f_list.append(jj); tgt_f_list.append(ii)

    def _mk_intra(srcs, tgts):
        if not srcs:
            return (torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, device=device))
        s = torch.cat(srcs); t = torch.cat(tgts)
        return s, t, _angular_weight(h[s], h[t])

    return (
        (src_inter, tgt_inter, w_inter),
        _mk_intra(src_p_list, tgt_p_list),
        _mk_intra(src_f_list, tgt_f_list),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Implicit Edge Detector (HRG-SSA)
# ─────────────────────────────────────────────────────────────────────────────

class ImplicitEdgeDetector(nn.Module):
    """
    Per-modality causal implicit edge detector.
    For modality τ and utterance pair (i, j) with j < i:
      s_ij = LeakyReLU(W_τ [h_i^τ ∥ h_j^τ])
      α = softmax_j(s_ij)
      edge exists if α_ij > 1/L
    Returns extra (src, tgt, weight) edges for ALL modalities combined.
    """

    def __init__(self, d_h: int):
        super().__init__()
        self.W_t = nn.Linear(d_h * 2, 1, bias=False)
        self.W_a = nn.Linear(d_h * 2, 1, bias=False)
        self.W_v = nn.Linear(d_h * 2, 1, bias=False)

    def forward(self, h: torch.Tensor):
        """h: (3L, d_h) — returns (src, tgt, weight) tensors."""
        L = h.size(0) // 3
        device = h.device
        if L < 2:
            return (torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, device=device))

        threshold = 1.0 / L
        all_src, all_tgt, all_w = [], [], []

        modality_blocks = [
            (0,       self.W_t),
            (L,       self.W_a),
            (2 * L,   self.W_v),
        ]

        # Causal mask: lower-triangular (j < i only)
        causal = torch.tril(torch.ones(L, L, device=device), diagonal=-1).bool()  # (L,L)

        for off, W in modality_blocks:
            h_mod = h[off: off + L]           # (L, d_h)

            # Vectorised pairwise: (L, L, 2*d_h) → (L, L, 1) → (L, L)
            hi = h_mod.unsqueeze(1).expand(L, L, -1)   # (L, L, d_h)
            hj = h_mod.unsqueeze(0).expand(L, L, -1)   # (L, L, d_h)
            pairs = torch.cat([hi, hj], dim=-1)         # (L, L, 2*d_h)
            scores = F.leaky_relu(W(pairs).squeeze(-1)) # (L, L)
            scores = scores.masked_fill(~causal, float("-inf"))

            # Softmax row-wise; rows with all -inf → uniform (masked out by causal)
            alpha = torch.softmax(scores, dim=1)        # (L, L)
            alpha = alpha * causal.float()              # zero out non-causal

            keep = (alpha > threshold) & causal
            rows, cols = keep.nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue

            w_vals = alpha[rows, cols]
            all_src.append(off + rows);  all_src.append(off + cols)
            all_tgt.append(off + cols);  all_tgt.append(off + rows)
            all_w.append(w_vals);        all_w.append(w_vals)

        if not all_src:
            return (torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, device=device))

        return (
            torch.cat(all_src),
            torch.cat(all_tgt),
            torch.cat(all_w),
        )


# ─────────────────────────────────────────────────────────────────────────────
# k-GNN layer (heterogeneous message passing)
# ─────────────────────────────────────────────────────────────────────────────

class kGNNLayer(nn.Module):
    """
    One layer of heterogeneous GNN with 4 edge-type-specific weights.
    h_v ← LN( h_v + ReLU( W_self h_v + Σ_type W_type · scatter_sum(w·h_u) ) )
    """

    def __init__(self, d_h: int, dropout: float = 0.1):
        super().__init__()
        self.W_self    = nn.Linear(d_h, d_h)
        self.W_inter   = nn.Linear(d_h, d_h)
        self.W_past    = nn.Linear(d_h, d_h)
        self.W_future  = nn.Linear(d_h, d_h)
        self.W_implicit= nn.Linear(d_h, d_h)
        self.ln        = nn.LayerNorm(d_h)
        self.drop      = nn.Dropout(dropout)

    def _agg(self, h, src, tgt, w, W, N):
        """Aggregate weighted neighbor messages and project."""
        if src.numel() == 0:
            return torch.zeros(N, h.size(1), device=h.device)
        msgs = w.unsqueeze(-1) * h[src]               # (E, d_h)
        agg  = torch.zeros(N, h.size(1), device=h.device)
        agg.scatter_add_(0, tgt.unsqueeze(-1).expand_as(msgs), msgs)
        return W(agg)

    def forward(self, h, inter, intra_past, intra_future, implicit):
        """
        h: (N, d_h)
        Each edge tuple: (src, tgt, weight) tensors
        """
        N = h.size(0)
        out = (self.W_self(h)
               + self._agg(h, *inter,         self.W_inter,    N)
               + self._agg(h, *intra_past,     self.W_past,     N)
               + self._agg(h, *intra_future,   self.W_future,   N)
               + self._agg(h, *implicit,       self.W_implicit, N))
        return self.ln(h + self.drop(F.relu(out)))


# ─────────────────────────────────────────────────────────────────────────────
# Hypergraph Module (M³Net-style with edge-dependent γ_e(v))
# ─────────────────────────────────────────────────────────────────────────────

class HyperGraphModule(nn.Module):
    """
    M³Net hypergraph propagation on 3L nodes with (3 + L) hyperedges.

    Hyperedges:
      e_T (idx 0): all L text nodes
      e_A (idx 1): all L audio nodes
      e_V (idx 2): all L visual nodes
      e_i (idx 3+i): {i, L+i, 2L+i} for each utterance i

    Edge-dependent node weights γ_e(v):
      • modality edges: shared γ per modality block (γ_T, γ_A, γ_V)
      • utterance edges: per-modality γ (γ_t, γ_a, γ_v) shared across all L

    Propagation (M³Net Eq. 5):
      V^{l+1} = σ( D_H^{-1} Ĥ W_e B^{-1} Ĥ^T V^l )  + V^l
    """

    def __init__(self, d_h: int, n_hyp: int = 2, use_edge_weights: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        self.n_hyp = n_hyp
        self.use_edge_weights = use_edge_weights

        # Edge weights ω(e) per hyperedge type (log-parameterised → softplus)
        # 3 modality edges + 1 shared utterance edge weight
        self.log_w_modal = nn.Parameter(torch.zeros(3))    # ω_T, ω_A, ω_V
        self.log_w_utt   = nn.Parameter(torch.zeros(1))    # ω_utt shared

        if use_edge_weights:
            # γ_e(v): node weights
            # For modality edges: one γ per modality-block
            self.log_gamma_modal = nn.Parameter(torch.zeros(3))  # γ_T, γ_A, γ_V
            # For utterance edges: per-modality node contribution
            self.log_gamma_utt   = nn.Parameter(torch.zeros(3))  # γ_t, γ_a, γ_v

        # Per-layer transform
        self.W_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(d_h, d_h), nn.LayerNorm(d_h))
            for _ in range(n_hyp)
        ])
        self.drop = nn.Dropout(dropout)

    def _build_incidence(self, L: int, device: torch.device):
        """
        Build Ĥ ∈ ℝ^{3L × (3+L)} with γ_e(v) weights.
        """
        N = 3 * L
        E = 3 + L
        H = torch.zeros(N, E, device=device)

        if self.use_edge_weights:
            gamma_m = F.softplus(self.log_gamma_modal)  # (3,)
            gamma_u = F.softplus(self.log_gamma_utt)    # (3,)
        else:
            gamma_m = torch.ones(3, device=device)
            gamma_u = torch.ones(3, device=device)

        # Modality edges
        H[0:L,       0] = gamma_m[0]   # e_T: text nodes
        H[L:2*L,     1] = gamma_m[1]   # e_A: audio nodes
        H[2*L:3*L,   2] = gamma_m[2]   # e_V: visual nodes

        # Utterance edges
        for i in range(L):
            H[i,         3 + i] = gamma_u[0]   # text node
            H[L + i,     3 + i] = gamma_u[1]   # audio node
            H[2 * L + i, 3 + i] = gamma_u[2]   # visual node

        return H  # (3L, 3+L)

    def _propagate(self, V: torch.Tensor, H: torch.Tensor, L: int):
        """
        V: (3L, d_h)
        H: (3L, 3+L)  incidence matrix with γ weights
        """
        E = H.size(1)

        # Edge weights ω(e)
        w_e = torch.cat([
            F.softplus(self.log_w_modal),
            F.softplus(self.log_w_utt).expand(L),
        ])  # (3+L,)

        # B = diag of Ĥ^T 1 (weighted hyperedge sizes)
        B = H.sum(dim=0).clamp(min=1e-6)        # (E,)

        # Node → hyperedge aggregation: x = B^{-1} W_e Ĥ^T V
        #   Ĥ^T: (E, 3L) @ V: (3L, d) → (E, d)
        x = (H.t() @ V) / B.unsqueeze(-1)       # (E, d_h)
        x = w_e.unsqueeze(-1) * x               # (E, d_h)

        # Hyperedge → node aggregation: out = Ĥ x
        out = H @ x                             # (3L, d_h)

        # Node degree normalisation
        D = (H * w_e.unsqueeze(0) / B.unsqueeze(0)).sum(dim=1).clamp(min=1e-6)
        out = out / D.unsqueeze(-1)

        return out

    def forward(self, h: torch.Tensor, feature_dropout: float = 0.0):
        """
        h: (3L, d_h)
        feature_dropout: applied to h before propagation (for DualCL views)
        Returns: (3L, d_h)
        """
        L = h.size(0) // 3
        device = h.device
        H = self._build_incidence(L, device)    # (3L, 3+L)

        if feature_dropout > 0 and self.training:
            h = F.dropout(h, p=feature_dropout, training=True)

        V = h
        for W_layer in self.W_layers:
            prop = self._propagate(V, H, L)
            V = V + self.drop(F.relu(W_layer(prop)))

        return V


# ─────────────────────────────────────────────────────────────────────────────
# Inception Hyper-Graph Module (IGM branches + HM in parallel)
# ─────────────────────────────────────────────────────────────────────────────

class InceptionHyperGraphModule(nn.Module):
    """
    Block B of HyFIN-Net.

    IGM: n branches with window pairs (p_b, f_b).  Each branch:
      1. Build heterogeneous graph G_b
      2. Inject implicit edges (ImplicitEdgeDetector)
      3. Stack N_inc k-GNN layers
      Mean-aggregate branches → P^t, P^a, P^v.

    HM: M³Net hypergraph on the same 3L nodes → Q^t, Q^a, Q^v.

    Both branches receive the same h (output of UnimodalEncoder).
    """

    def __init__(self, cfg):
        super().__init__()
        d_h = cfg.d_h
        n_inc = cfg.n_inc
        drop  = cfg.dropout

        # IGM branches
        self.use_igm = cfg.use_igm
        self.igm_windows = cfg.igm_windows

        if self.use_igm:
            # Shared k-GNN layers across ALL branches (parameter efficiency)
            # Each branch still uses the same weights; multi-scale comes from graph topology
            self.kgnn_layers = nn.ModuleList([
                kGNNLayer(d_h, dropout=drop) for _ in range(n_inc)
            ])
        else:
            # Single global-window branch (ablation)
            self.kgnn_layers = nn.ModuleList([
                kGNNLayer(d_h, dropout=drop) for _ in range(n_inc)
            ])

        # Implicit edge detector (shared across branches)
        self.use_implicit = cfg.use_implicit_edge
        if self.use_implicit:
            self.impl_det = ImplicitEdgeDetector(d_h)

        # Hypergraph module
        self.use_hm = cfg.use_hm
        if self.use_hm:
            self.hm = HyperGraphModule(
                d_h, cfg.n_hyp, cfg.use_edge_weights, drop
            )

        self.d_h = d_h

    def _run_branch(self, h: torch.Tensor, past: int, future: int):
        """Run one IGM branch; returns (3L, d_h)."""
        inter, intra_past, intra_fut = _build_graph(h, past, future)

        if self.use_implicit:
            impl = self.impl_det(h)
        else:
            dev = h.device
            impl = (torch.empty(0, dtype=torch.long, device=dev),
                    torch.empty(0, dtype=torch.long, device=dev),
                    torch.empty(0, device=dev))

        for layer in self.kgnn_layers:
            h = layer(h, inter, intra_past, intra_fut, impl)
        return h

    def forward(self, T: torch.Tensor, A: torch.Tensor, V: torch.Tensor,
                dual_cl_drop: float = 0.0):
        """
        T, A, V: (L, d_h)
        dual_cl_drop: feature dropout for DualCL second view on HM
        Returns:
          P_t, P_a, P_v : (L, d_h)  — IGM output
          Q_t, Q_a, Q_v : (L, d_h)  — HM output
          q2             : (3L, d_h) or None — HM second view (DualCL)
        """
        L = T.size(0)
        h = torch.cat([T, A, V], dim=0)          # (3L, d_h)

        # ── IGM ────────────────────────────────────────────────────────────
        if self.use_igm:
            windows = self.igm_windows
        else:
            # Ablation: single branch with full window
            windows = [(L - 1, L - 1)]

        branch_outs = []
        h_branch = h.detach() if not self.use_igm else h
        for (past, future) in windows:
            out = self._run_branch(h_branch.clone() if len(windows) > 1 else h_branch,
                                   past, future)
            branch_outs.append(out)

        if branch_outs:
            igm_out = torch.stack(branch_outs, dim=0).mean(0)  # (3L, d_h)
        else:
            igm_out = h

        P_t = igm_out[0:L]
        P_a = igm_out[L:2*L]
        P_v = igm_out[2*L:3*L]

        # ── HM ─────────────────────────────────────────────────────────────
        if self.use_hm:
            q1 = self.hm(h, feature_dropout=0.0)
            Q_t = q1[0:L];   Q_a = q1[L:2*L];   Q_v = q1[2*L:3*L]

            if dual_cl_drop > 0 and self.training:
                q2 = self.hm(h, feature_dropout=dual_cl_drop)
            else:
                q2 = None
        else:
            zeros = torch.zeros(L, self.d_h, device=T.device)
            Q_t = Q_a = Q_v = zeros
            q2 = None

        return P_t, P_a, P_v, Q_t, Q_a, Q_v, q2
