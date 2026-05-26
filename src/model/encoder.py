"""Stage 1: unimodal encoders for text, audio, and visual modalities."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    def __init__(self, text_dim: int = 1024, d: int = 128):
        super().__init__()
        self.ln   = nn.LayerNorm(text_dim)
        self.proj = nn.Linear(text_dim, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.ln(x))


class AudioEncoder(nn.Module):
    def __init__(self, audio_dim: int = 1024, d: int = 128):
        super().__init__()
        self.ln   = nn.LayerNorm(audio_dim)
        self.proj = nn.Linear(audio_dim, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.ln(x))


class VisualEncoder(nn.Module):
    """
    Dual-stream encoder: SigLIP2 holistic + OpenFace AU FACS groups.
    Input:
        siglip2:  (N, 3, siglip2_dim)
        openface: (N, 3, au_dim)
    Output: (N, d)
    """

    def __init__(self, siglip2_dim: int = 1152, au_dim: int = 8, d: int = 128):
        super().__init__()
        # Learnable mask tokens (zero-init so missing face starts at zero)
        self.mask_tok_sig = nn.Parameter(torch.zeros(siglip2_dim))
        self.mask_tok_au  = nn.Parameter(torch.zeros(au_dim))

        # SigLIP2 stream
        self.ln_sig      = nn.LayerNorm(siglip2_dim)
        self.pos_emb_sig = nn.Parameter(torch.zeros(3, siglip2_dim))
        self.proj_sig    = nn.Linear(siglip2_dim, d)

        # AU stream — FACS anatomical groups
        self.ln_au      = nn.LayerNorm(au_dim)
        self.pos_emb_au = nn.Parameter(torch.zeros(3, au_dim))
        # Brow: AU1(0), AU2(1), AU4(2)
        self.brow_proj  = nn.Linear(3, 32)
        # Cheek/Nose: AU6(3), AU9(4)
        self.cheek_proj = nn.Linear(2, 16)
        # Mouth: AU12(5), AU25(6), AU26(7)
        self.mouth_proj = nn.Linear(3, 32)
        self.au_ln      = nn.LayerNorm(80)
        self.au_proj    = nn.Linear(80, d)

        # Per-segment gated fusion
        self.gate = nn.Linear(d * 2, d)

        # Temporal aggregation (3 learnable scalars → softmax)
        self.w_temp = nn.Parameter(torch.ones(3))

    def _encode_segment(
        self,
        sig_s: torch.Tensor,   # (N, siglip2_dim)
        au_s:  torch.Tensor,   # (N, au_dim)
        s:     int,
    ) -> torch.Tensor:
        N = sig_s.size(0)

        # Replace zero segments with mask tokens
        sig_missing = sig_s.abs().sum(dim=-1, keepdim=True) == 0
        au_missing  = au_s.abs().sum(dim=-1, keepdim=True) == 0
        sig_s = torch.where(sig_missing, self.mask_tok_sig.unsqueeze(0).expand(N, -1), sig_s)
        au_s  = torch.where(au_missing,  self.mask_tok_au.unsqueeze(0).expand(N, -1),  au_s)

        # SigLIP2 stream: LN → add positional emb → project
        x_sig = self.ln_sig(sig_s) + self.pos_emb_sig[s]
        v_sig = self.proj_sig(x_sig)                           # (N, d)

        # AU stream: LN → add positional emb → FACS group projections
        x_au    = self.ln_au(au_s) + self.pos_emb_au[s]
        brow    = self.brow_proj(x_au[:, [0, 1, 2]])           # (N, 32)
        cheek   = self.cheek_proj(x_au[:, [3, 4]])             # (N, 16)
        mouth   = self.mouth_proj(x_au[:, [5, 6, 7]])          # (N, 32)
        au_grp  = self.au_ln(torch.cat([brow, cheek, mouth], dim=-1))  # (N, 80)
        v_au    = self.au_proj(au_grp)                         # (N, d)

        # Per-segment gated fusion: gate learns relative AU vs holistic contribution
        gate_s  = torch.sigmoid(self.gate(torch.cat([v_sig, v_au], dim=-1)))
        return gate_s * v_au + (1 - gate_s) * v_sig            # (N, d)

    def forward(
        self,
        siglip2:  torch.Tensor,  # (N, 3, siglip2_dim)
        openface: torch.Tensor,  # (N, 3, au_dim)
    ) -> torch.Tensor:
        segments = [
            self._encode_segment(siglip2[:, s, :], openface[:, s, :], s)
            for s in range(3)
        ]
        w = torch.softmax(self.w_temp, dim=0)                  # (3,)
        return sum(w[s] * segments[s] for s in range(3))       # (N, d)
