"""HyFIN-Net: Hyper-Frequency Inception Network.

Full forward pass: A → B → C → D → E → F
  A : UnimodalEncoder
  B : InceptionHyperGraphModule (IGM + HM)
  C : MultiFrequencyModule
  D : Multi-view concatenation [p^τ ∥ q^τ ∥ f̄^τ]
  E : Cross-modal attention fusion (text-anchored)
  F : MLP classifier

model.forward(batch, training=False) returns:
  logits   (sum_N, C)
  labels   (sum_N,)
  z        (sum_N, d_h)  — utterance representations (for CBFC)
  dl_loss  scalar or None  — per-batch mean DualCL loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import UnimodalEncoder
from .igm     import InceptionHyperGraphModule
from .mfm     import MultiFrequencyModule


# ─────────────────────────────────────────────────────────────────────────────
# Block E: Cross-Modal Attention Fusion
# ─────────────────────────────────────────────────────────────────────────────

class CrossModalFusion(nn.Module):
    """
    Text-anchored cross-modal attention.
    Input:  m^t, m^a, m^v  each (L, 3*d_h)  [p∥q∥f̄ per modality]
    Output: z (L, d_h)

    Step 1: compress each [p∥q∥f̄] ∈ ℝ^{3*d_h} → ℝ^{d_h}  (3 separate projections)
    Step 2: cross-attend in d_h space  (W_Q/K/V all d_h × d_h)
    Step 3: W_z: 3*d_h → d_h

    This replaces the original formulation where W_V and W_z operated on 3*d_h (=1536
    for d_h=512), which caused 12M+ params in this block alone.
    """

    def __init__(self, d_in: int, d_h: int, dropout: float = 0.1):
        """d_in = 3 * d_h."""
        super().__init__()

        # Step 1: per-modality multi-view compression
        self.proj_t = nn.Sequential(nn.Linear(d_in, d_h), nn.LayerNorm(d_h), nn.ReLU())
        self.proj_a = nn.Sequential(nn.Linear(d_in, d_h), nn.LayerNorm(d_h), nn.ReLU())
        self.proj_v = nn.Sequential(nn.Linear(d_in, d_h), nn.LayerNorm(d_h), nn.ReLU())

        # Step 2: cross-attention in d_h space
        self.W_Q_v = nn.Linear(d_h, d_h, bias=False)
        self.W_K_v = nn.Linear(d_h, d_h, bias=False)
        self.W_V_v = nn.Linear(d_h, d_h, bias=False)

        self.W_Q_a = nn.Linear(d_h, d_h, bias=False)
        self.W_K_a = nn.Linear(d_h, d_h, bias=False)
        self.W_V_a = nn.Linear(d_h, d_h, bias=False)

        self.scale = d_h ** -0.5

        # Step 3: fuse [f̂^t ∥ e_a ∥ e_v] → d_h
        self.W_z  = nn.Linear(d_h * 3, d_h)
        self.ln   = nn.LayerNorm(d_h)
        self.drop = nn.Dropout(dropout)

    def _cross_attn(self, Q_proj, K_proj, V_proj, q_src, k_src):
        Q    = Q_proj(q_src)                                       # (L, d_h)
        K    = K_proj(k_src)                                       # (L, d_h)
        V    = V_proj(k_src)                                       # (L, d_h)
        attn = torch.softmax(Q @ K.t() * self.scale, dim=-1)      # (L, L)
        return attn @ V                                            # (L, d_h)

    def forward(self, m_t, m_a, m_v):
        """m_t, m_a, m_v: (L, 3*d_h) → z: (L, d_h)"""
        e_t = self.proj_t(m_t)     # (L, d_h)
        e_a = self.proj_a(m_a)
        e_v = self.proj_v(m_v)

        ca_v    = self._cross_attn(self.W_Q_v, self.W_K_v, self.W_V_v, e_t, e_v)
        ca_a    = self._cross_attn(self.W_Q_a, self.W_K_a, self.W_V_a, e_t, e_a)
        f_hat_t = e_t + ca_v + ca_a                               # (L, d_h)

        fused = torch.cat([f_hat_t, e_a, e_v], dim=-1)            # (L, 3*d_h)
        z     = F.relu(self.W_z(self.drop(fused)))
        return self.ln(z)


# ─────────────────────────────────────────────────────────────────────────────
# Fallback fusion (ablation: no cross-modal attention)
# ─────────────────────────────────────────────────────────────────────────────

class PlainConcatFusion(nn.Module):
    def __init__(self, d_in: int, d_h: int, dropout: float = 0.1):
        super().__init__()
        self.proj_t = nn.Linear(d_in, d_h)
        self.proj_a = nn.Linear(d_in, d_h)
        self.proj_v = nn.Linear(d_in, d_h)
        self.fc   = nn.Linear(d_h * 3, d_h)
        self.ln   = nn.LayerNorm(d_h)
        self.drop = nn.Dropout(dropout)

    def forward(self, m_t, m_a, m_v):
        x = torch.cat([self.proj_t(m_t), self.proj_a(m_a), self.proj_v(m_v)], dim=-1)
        return self.ln(F.relu(self.fc(self.drop(x))))


# ─────────────────────────────────────────────────────────────────────────────
# DualCL loss helper (InfoNCE on HM node embeddings)
# ─────────────────────────────────────────────────────────────────────────────

def _dual_cl_loss(q1: torch.Tensor, q2: torch.Tensor, tau: float) -> torch.Tensor:
    """
    q1, q2: (N, d) — two dropout views of HM node embeddings.
    Returns scalar InfoNCE loss.
    """
    q1 = F.normalize(q1, dim=-1)
    q2 = F.normalize(q2, dim=-1)
    # Positive: (q1_i, q2_i); negatives: all other j in q2
    sim = q1 @ q2.t() / tau          # (N, N)
    labels = torch.arange(sim.size(0), device=sim.device)
    return F.cross_entropy(sim, labels)


# ─────────────────────────────────────────────────────────────────────────────
# HyFIN-Net
# ─────────────────────────────────────────────────────────────────────────────

class HyFIN(nn.Module):
    def __init__(self, cfg, num_speakers: int):
        super().__init__()
        d_h = cfg.d_h

        # A: unimodal encoder
        self.encoder = UnimodalEncoder(cfg, num_speakers)

        # B: inception hyper-graph module
        self.ighm = InceptionHyperGraphModule(cfg)

        # C: multi-frequency module
        if cfg.use_mfm:
            self.mfm = MultiFrequencyModule(d_h, cfg.k_freq, cfg.dropout)
        self.use_mfm = cfg.use_mfm

        # D→E: fusion
        d_in = d_h * 3   # [p^τ ∥ q^τ ∥ f̄^τ]
        if cfg.use_cross_modal_attn:
            self.fusion = CrossModalFusion(d_in, d_h, cfg.dropout)
        else:
            self.fusion = PlainConcatFusion(d_in, d_h, cfg.dropout)

        # F: classifier
        self.drop_cls  = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(d_h, cfg.num_classes)

        # DualCL config
        self.use_dual_cl = cfg.use_dual_cl
        self.dual_cl_drop = cfg.dual_cl_drop
        self.tau_cl       = cfg.tau_cl

    def _forward_conv(self, conv: dict, training: bool):
        T, A, V = self.encoder(conv)               # (L, d_h) × 3
        L = T.size(0)
        d_h = T.size(1)

        # B: IGM + HM
        dual_drop = self.dual_cl_drop if (training and self.use_dual_cl) else 0.0
        P_t, P_a, P_v, Q_t, Q_a, Q_v, q2 = self.ighm(T, A, V, dual_cl_drop=dual_drop)

        # C: MFM
        if self.use_mfm:
            F_t, F_a, F_v = self.mfm(T.detach(), A.detach(), V.detach())
        else:
            zeros = torch.zeros(L, d_h, device=T.device)
            F_t = F_a = F_v = zeros

        # D: multi-view concatenation
        m_t = torch.cat([P_t, Q_t, F_t], dim=-1)   # (L, 3*d_h)
        m_a = torch.cat([P_a, Q_a, F_a], dim=-1)
        m_v = torch.cat([P_v, Q_v, F_v], dim=-1)

        # E: fusion
        z = self.fusion(m_t, m_a, m_v)              # (L, d_h)

        # F: classify
        logits = self.classifier(self.drop_cls(z))  # (L, num_classes)

        # DualCL loss
        dl = None
        if training and self.use_dual_cl and q2 is not None:
            q1_nodes = torch.cat([Q_t, Q_a, Q_v], dim=0)   # (3L, d_h)
            q2_nodes = q2                                    # (3L, d_h)
            dl = _dual_cl_loss(q1_nodes, q2_nodes, self.tau_cl)

        return logits, conv["labels"], z, dl

    def forward(self, batch: list, training: bool = False):
        """
        Returns:
          logits   (sum_L, C)
          labels   (sum_L,)
          z        (sum_L, d_h)
          dl_loss  scalar or None
        """
        logits_l, labels_l, z_l, dl_l = [], [], [], []
        for conv in batch:
            lg, lb, zz, dl = self._forward_conv(conv, training)
            logits_l.append(lg)
            labels_l.append(lb)
            z_l.append(zz)
            if dl is not None:
                dl_l.append(dl)

        logits = torch.cat(logits_l, dim=0)
        labels = torch.cat(labels_l, dim=0)
        z      = torch.cat(z_l, dim=0)
        dl_loss = (sum(dl_l) / len(dl_l)) if dl_l else None
        return logits, labels, z, dl_loss
