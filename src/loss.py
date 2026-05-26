"""
Loss functions for HyFIN-Net.

ℒ = ℒ_CBCE  +  μ · ℒ_CBFC  +  λ · ℒ_DualCL

CBCE : Class-Balanced Cross-Entropy (effective-number reweighting, β=0.999)
CBFC : Class-Balanced Focal Contrastive loss (ConxGNN Eq. 18)
DualCL: computed inside the model forward and passed in as a pre-scalar.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Effective-number class weights (CBN weighting)
# ─────────────────────────────────────────────────────────────────────────────

def effective_num_weights(
    counts: torch.Tensor,
    beta: float = 0.999,
    use_class_balanced: bool = True,
) -> torch.Tensor:
    """
    counts: (C,) integer class counts in the training set.
    Returns w_c = (1-β) / (1-β^{n_c}), normalised to sum to C.
    """
    if not use_class_balanced:
        return torch.ones_like(counts.float())
    counts = counts.float().clamp(min=1)
    w = (1.0 - beta) / (1.0 - beta ** counts)
    w = w / w.sum() * len(counts)   # normalise so mean = 1
    return w


# ─────────────────────────────────────────────────────────────────────────────
# Class-Balanced Cross-Entropy
# ─────────────────────────────────────────────────────────────────────────────

class CBCELoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor):
        super().__init__()
        self.register_buffer("w", class_weights)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels, weight=self.w.to(logits.device))


# ─────────────────────────────────────────────────────────────────────────────
# Class-Balanced Focal Contrastive Loss (CBFC — ConxGNN Eq. 18)
# ─────────────────────────────────────────────────────────────────────────────

class CBFCLoss(nn.Module):
    """
    For each anchor utterance i:
      ℒ_i = -w_{c_i} · (1/|P(i)|) · Σ_{j∈P(i)} (1-sim(i,j))^γ
            · log( exp(sim(i,j)/τ) / Σ_{k≠i} exp(sim(i,k)/τ) )

    where:
      P(i) = {j : label_j = label_i, j ≠ i}
      sim   = cosine similarity on z features
      w_c   = effective-number class weight
    """

    def __init__(self, class_weights: torch.Tensor, gamma: float = 1.5,
                 tau: float = 0.5):
        super().__init__()
        self.gamma = gamma
        self.tau   = tau
        self.register_buffer("w", class_weights)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """z: (N, d), labels: (N,) → scalar loss."""
        N = z.size(0)
        if N < 2:
            return z.sum() * 0.0

        z_n = F.normalize(z, dim=-1)
        sim = z_n @ z_n.t()                     # (N, N) cosine sim

        # Exclude self-similarity from denominator
        mask_self = torch.eye(N, dtype=torch.bool, device=z.device)
        sim_no_self = sim.masked_fill(mask_self, float("-inf"))

        log_denom = torch.logsumexp(sim_no_self / self.tau, dim=1)  # (N,)

        # Per-anchor loss
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)       # (N, N)
        pos_mask  = labels_eq & ~mask_self                            # (N, N)

        w_anchor = self.w.to(z.device)[labels]                       # (N,)
        loss = torch.zeros(1, device=z.device)
        n_valid = 0

        for i in range(N):
            pos_idx = pos_mask[i].nonzero(as_tuple=False).squeeze(-1)
            if pos_idx.numel() == 0:
                continue
            n_pos = pos_idx.numel()

            sim_pos   = sim[i, pos_idx]                                # (n_pos,)
            focal_w   = (1.0 - sim_pos.detach()).clamp(min=0) ** self.gamma

            log_num   = sim_pos / self.tau
            log_prob  = log_num - log_denom[i]                        # (n_pos,)

            term = (focal_w * log_prob).sum() / n_pos
            loss = loss - w_anchor[i] * term
            n_valid += 1

        return loss / max(n_valid, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Combined loss
# ─────────────────────────────────────────────────────────────────────────────

class HyFINLoss(nn.Module):
    def __init__(self, cfg, class_counts: torch.Tensor):
        super().__init__()
        w = effective_num_weights(class_counts, cfg.beta, cfg.use_class_balanced)

        self.cbce = CBCELoss(w)
        self.use_cbfc    = cfg.use_cbfc
        self.use_dual_cl = cfg.use_dual_cl
        self.mu       = cfg.mu  if cfg.use_cbfc    else 0.0
        self.lam      = cfg.lam if cfg.use_dual_cl else 0.0
        self.lam_live = 0.0  # set externally each epoch for warmup

        if cfg.use_cbfc:
            self.cbfc = CBFCLoss(w, gamma=cfg.gamma_cbfc, tau=cfg.tau_cl)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        z:      torch.Tensor,
        dl_loss: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict]:
        l_ce = self.cbce(logits, labels)
        l_fc = self.cbfc(z, labels) if self.use_cbfc else logits.new_zeros(1).squeeze()
        l_cl = dl_loss if (self.use_dual_cl and dl_loss is not None) else logits.new_zeros(1).squeeze()

        total = l_ce + self.mu * l_fc + self.lam_live * l_cl
        return total, {"ce": l_ce.item(), "fc": l_fc.item(), "cl": l_cl.item()}
