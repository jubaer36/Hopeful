"""Block A: unimodal encoders + speaker embedding injection.

Text   : 1-layer Transformer (self-attention over L utterances) → d_h
Audio  : LayerNorm + Linear + ReLU → d_h
Visual : SigLIP2 (3-segment) + AU FACS dual-stream → d_h
Speaker: Embedding(speaker_id) added to each modality output.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    """
    Project RoBERTa-Large CLS tokens (L, 1024) → (L, d_h) via 1-layer
    Transformer self-attention with sinusoidal positional encoding.
    """

    def __init__(self, text_dim: int = 1024, d_h: int = 512, n_heads: int = 8,
                 max_len: int = 300, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(text_dim, d_h)
        self.ln_in = nn.LayerNorm(d_h)

        # Sinusoidal PE
        pe = torch.zeros(max_len, d_h)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_h, 2).float() * (-math.log(10000.0) / d_h))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

        # 1-layer Transformer
        self.attn = nn.MultiheadAttention(d_h, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_h, d_h * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_h * 2, d_h),
        )
        self.ln1 = nn.LayerNorm(d_h)
        self.ln2 = nn.LayerNorm(d_h)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (L, text_dim) → (L, d_h)"""
        # x is (L, text_dim) — add batch dim for MHA
        L = x.size(0)
        h = self.ln_in(self.proj(x))          # (L, d_h)
        h = h + self.pe[:L]                   # add sinusoidal PE
        h = h.unsqueeze(0)                    # (1, L, d_h)
        a, _ = self.attn(h, h, h)             # (1, L, d_h)
        h = self.ln1(h + self.drop(a))
        h = self.ln2(h + self.drop(self.ff(h)))
        return h.squeeze(0)                   # (L, d_h)


class AudioEncoder(nn.Module):
    def __init__(self, audio_dim: int = 1024, d_h: int = 512):
        super().__init__()
        self.ln = nn.LayerNorm(audio_dim)
        self.proj = nn.Linear(audio_dim, d_h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (L, audio_dim) → (L, d_h)"""
        return F.relu(self.proj(self.ln(x)))


class VisualEncoder(nn.Module):
    """
    Dual-stream: SigLIP2 holistic + OpenFace AU FACS groups.
    Temporal (3 segments) weighted-mean pool → (L, d_h).
    """

    def __init__(self, siglip2_dim: int = 1152, au_dim: int = 8, d_h: int = 512):
        super().__init__()
        self.mask_tok_sig = nn.Parameter(torch.zeros(siglip2_dim))
        self.mask_tok_au  = nn.Parameter(torch.zeros(au_dim))

        self.ln_sig      = nn.LayerNorm(siglip2_dim)
        self.pos_emb_sig = nn.Parameter(torch.zeros(3, siglip2_dim))
        self.proj_sig    = nn.Linear(siglip2_dim, d_h)

        self.ln_au       = nn.LayerNorm(au_dim)
        self.pos_emb_au  = nn.Parameter(torch.zeros(3, au_dim))
        self.brow_proj   = nn.Linear(3, 32)
        self.cheek_proj  = nn.Linear(2, 16)
        self.mouth_proj  = nn.Linear(3, 32)
        self.au_ln       = nn.LayerNorm(80)
        self.au_proj     = nn.Linear(80, d_h)

        self.gate        = nn.Linear(d_h * 2, d_h)
        self.w_temp      = nn.Parameter(torch.ones(3))

    def _encode_segment(self, sig_s, au_s, s):
        N = sig_s.size(0)
        sig_missing = sig_s.abs().sum(dim=-1, keepdim=True) == 0
        au_missing  = au_s.abs().sum(dim=-1, keepdim=True) == 0
        sig_s = torch.where(sig_missing, self.mask_tok_sig.unsqueeze(0).expand(N, -1), sig_s)
        au_s  = torch.where(au_missing,  self.mask_tok_au.unsqueeze(0).expand(N, -1),  au_s)

        x_sig = self.ln_sig(sig_s) + self.pos_emb_sig[s]
        v_sig = self.proj_sig(x_sig)

        x_au    = self.ln_au(au_s) + self.pos_emb_au[s]
        brow    = self.brow_proj(x_au[:, [0, 1, 2]])
        cheek   = self.cheek_proj(x_au[:, [3, 4]])
        mouth   = self.mouth_proj(x_au[:, [5, 6, 7]])
        v_au    = self.au_proj(self.au_ln(torch.cat([brow, cheek, mouth], dim=-1)))

        gate_s  = torch.sigmoid(self.gate(torch.cat([v_sig, v_au], dim=-1)))
        return gate_s * v_au + (1 - gate_s) * v_sig

    def forward(self, siglip2, openface):
        """siglip2: (L, 3, siglip2_dim), openface: (L, 3, au_dim) → (L, d_h)"""
        segments = [
            self._encode_segment(siglip2[:, s, :], openface[:, s, :], s)
            for s in range(3)
        ]
        w = torch.softmax(self.w_temp, dim=0)
        return sum(w[s] * segments[s] for s in range(3))


class UnimodalEncoder(nn.Module):
    """Encode all 3 modalities and inject speaker embeddings (additive)."""

    def __init__(self, cfg, num_speakers: int):
        super().__init__()
        d_h = cfg.d_h
        self.text_enc   = TextEncoder(cfg.text_dim, d_h, dropout=cfg.dropout)
        self.audio_enc  = AudioEncoder(cfg.audio_dim, d_h)
        self.visual_enc = VisualEncoder(cfg.siglip2_dim, cfg.au_dim, d_h)
        # Speaker embedding same dim as d_h — added directly
        self.speaker_emb = nn.Embedding(num_speakers, d_h)
        nn.init.normal_(self.speaker_emb.weight, std=0.02)

    def forward(self, conv: dict):
        """
        conv keys: text (L, 1024), audio (L, 1024), siglip2 (L, 3, 1152),
                   openface (L, 3, 8), speaker_ids (L,)
        Returns: T, A, V each (L, d_h)
        """
        T = self.text_enc(conv["text"])
        A = self.audio_enc(conv["audio"])
        V = self.visual_enc(conv["siglip2"], conv["openface"])

        spk = self.speaker_emb(conv["speaker_ids"])  # (L, d_h)
        T = T + spk
        A = A + spk
        V = V + spk
        return T, A, V
