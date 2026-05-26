"""Annealed class-balanced focal cross-entropy loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnnealedFocalLoss(nn.Module):
    """
    Gradually introduces focal weighting and class-balancing:
      epochs 0 .. focal_start-1 : standard CE, uniform weights
      epochs focal_start .. focal_end-1 : linear ramp
      epochs focal_end+ : full focal CE with inverse-frequency class weights
    """

    def __init__(
        self,
        class_weights: torch.Tensor,   # inverse-frequency weights, (C,)
        focal_start:   int   = 5,
        focal_end:     int   = 10,
        gamma_max:     float = 2.0,
    ):
        super().__init__()
        self.focal_start = focal_start
        self.focal_end   = focal_end
        self.gamma_max   = gamma_max
        self.register_buffer("w_target", class_weights)

    def _schedule(self, epoch: int):
        if epoch < self.focal_start:
            return 0.0, None                     # standard CE, no class weights
        if epoch >= self.focal_end:
            return self.gamma_max, self.w_target

        t     = (epoch - self.focal_start) / max(self.focal_end - self.focal_start, 1)
        gamma = t * self.gamma_max
        alpha = 1.0 + t * (self.w_target - 1.0)
        return gamma, alpha

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, epoch: int) -> torch.Tensor:
        gamma, alpha = self._schedule(epoch)

        log_p   = F.log_softmax(logits, dim=-1)           # (N, C)
        p_true  = log_p.exp().gather(1, labels.unsqueeze(1)).squeeze(1)  # (N,)
        focal_w = (1.0 - p_true).pow(gamma)               # (N,)
        nll     = -log_p.gather(1, labels.unsqueeze(1)).squeeze(1)       # (N,)

        if alpha is None:
            class_w = torch.ones_like(nll)
        else:
            class_w = alpha.to(logits.device)[labels]     # (N,)

        return (focal_w * class_w * nll).mean()
