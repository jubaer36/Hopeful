"""
Full MERC model: Stages 2–6 plus the top-level forward pass.

Stage 2:  Speaker embedding injection + sinusoidal positional encoding
Stage 3:  Path A (dynamic hypergraph) ‖ Path B (spectral filtering)
Stage 4:  Modality-specific gated path fusion
Stage 5:  Utterance-level modality fusion MLP
Stage 6:  Classifier
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import TextEncoder, AudioEncoder, VisualEncoder
from .path_a  import PathA
from .path_b  import PathB


# ---------------------------------------------------------------------------
# Stage 2: Speaker embedding injection + sinusoidal positional encoding
# ---------------------------------------------------------------------------

class _SpeakerPositionInjector(nn.Module):
    def __init__(
        self,
        num_speakers: int,
        d: int = 64,
        d_spk: int = 32,
        max_len: int = 300,
        use_speaker: bool = True,
    ):
        super().__init__()
        self.use_speaker = use_speaker

        if use_speaker:
            self.speaker_emb = nn.Embedding(num_speakers, d_spk)
            self.proj_T = nn.Linear(d + d_spk, d)
            self.proj_A = nn.Linear(d + d_spk, d)
            self.proj_V = nn.Linear(d + d_spk, d)

        pe  = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, T, A, V, speaker_ids):
        """T, A, V: (N, d); speaker_ids: (N,) → T, A, V each (N, d)"""
        N   = T.size(0)
        pos = self.pe[:N]

        if self.use_speaker:
            spk = self.speaker_emb(speaker_ids)
            T = self.proj_T(torch.cat([T, spk], dim=-1)) + pos
            A = self.proj_A(torch.cat([A, spk], dim=-1)) + pos
            V = self.proj_V(torch.cat([V, spk], dim=-1)) + pos
        else:
            T = T + pos
            A = A + pos
            V = V + pos
        return T, A, V


# ---------------------------------------------------------------------------
# Stage 4: Modality-specific gated path fusion
# ---------------------------------------------------------------------------

class _PathFusion(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.gate_T = nn.Linear(d * 2, d)
        self.gate_A = nn.Linear(d * 2, d)
        self.gate_V = nn.Linear(d * 2, d)
        self.ln_T   = nn.LayerNorm(d)
        self.ln_A   = nn.LayerNorm(d)
        self.ln_V   = nn.LayerNorm(d)

    def forward(self, hA_T, hB_T, hA_A, hB_A, hA_V, hB_V):
        g_T = torch.sigmoid(self.gate_T(torch.cat([hA_T, hB_T], dim=-1)))
        g_A = torch.sigmoid(self.gate_A(torch.cat([hA_A, hB_A], dim=-1)))
        g_V = torch.sigmoid(self.gate_V(torch.cat([hA_V, hB_V], dim=-1)))
        T   = self.ln_T(g_T * hA_T + (1 - g_T) * hB_T)
        A   = self.ln_A(g_A * hA_A + (1 - g_A) * hB_A)
        V   = self.ln_V(g_V * hA_V + (1 - g_V) * hB_V)
        return T, A, V


# ---------------------------------------------------------------------------
# Stage 5: Utterance-level modality fusion MLP
# ---------------------------------------------------------------------------

class _ModalityFusionMLP(nn.Module):
    def __init__(self, d: int, dropout: float = 0.3):
        super().__init__()
        self.ln   = nn.LayerNorm(d * 3)
        self.fc1  = nn.Linear(d * 3, d)
        self.fc2  = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, T, A, V):
        x = torch.cat([T, A, V], dim=-1)              # (N, 3d)
        x = F.gelu(self.fc1(self.ln(x)))              # (N, d)
        x = self.fc2(self.drop(x))                    # (N, d)
        return x


# ---------------------------------------------------------------------------
# Full MERC model
# ---------------------------------------------------------------------------

class MERC(nn.Module):
    def __init__(self, cfg, num_speakers: int):
        super().__init__()
        d = cfg.d
        self._use_path_a = cfg.use_path_a
        self._use_path_b = cfg.use_path_b

        # Stage 1
        self.text_enc   = TextEncoder(cfg.text_dim,    d)
        self.audio_enc  = AudioEncoder(cfg.audio_dim,  d)
        self.visual_enc = VisualEncoder(cfg.siglip2_dim, cfg.au_dim, d)

        # Stage 2
        self.spk_pos = _SpeakerPositionInjector(
            num_speakers, d, cfg.d_spk, use_speaker=cfg.use_speaker_emb
        )

        # Stage 3
        if cfg.use_path_a:
            self.path_a = PathA(d, num_speakers, use_cross_modal=cfg.use_cross_modal_attn)
        if cfg.use_path_b:
            self.path_b = PathB(d, cfg.window_size, cfg.k_poly, cfg.k_freq)

        # Stage 4 — only needed when both paths active
        if cfg.use_path_a and cfg.use_path_b:
            self.path_fusion = _PathFusion(d)

        # Stage 5
        self.utt_fusion = _ModalityFusionMLP(d, cfg.dropout)

        # Stage 6
        self.ln_cls   = nn.LayerNorm(d)
        self.drop_cls = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(d, cfg.num_classes)

    def _forward_conv(self, conv: dict) -> torch.Tensor:
        """Process one conversation; returns logits (N, num_classes)."""
        text       = conv["text"]        # (N, text_dim)
        audio      = conv["audio"]       # (N, audio_dim)
        siglip2    = conv["siglip2"]     # (N, 3, siglip2_dim)
        openface   = conv["openface"]    # (N, 3, au_dim)
        speaker_ids = conv["speaker_ids"]  # (N,)

        # Stage 1
        T = self.text_enc(text)
        A = self.audio_enc(audio)
        V = self.visual_enc(siglip2, openface)

        # Stage 2
        T, A, V = self.spk_pos(T, A, V, speaker_ids)

        # Stage 3 — conditional path routing
        if self._use_path_a and self._use_path_b:
            hA_T, hA_A, hA_V = self.path_a(T, A, V, speaker_ids)
            hB_T, hB_A, hB_V = self.path_b(T, A, V)
            fT, fA, fV = self.path_fusion(hA_T, hB_T, hA_A, hB_A, hA_V, hB_V)
        elif self._use_path_a:
            fT, fA, fV = self.path_a(T, A, V, speaker_ids)
        elif self._use_path_b:
            fT, fA, fV = self.path_b(T, A, V)
        else:
            fT, fA, fV = T, A, V

        # Stage 5
        u = self.utt_fusion(fT, fA, fV)

        # Stage 6
        return self.classifier(self.drop_cls(self.ln_cls(u)))

    def forward(self, batch: list):
        """
        batch: list of conversation dicts (all tensors already on device).
        Returns:
            logits: (sum_N, num_classes)
            labels: (sum_N,)
        """
        logits_list = []
        labels_list = []
        for conv in batch:
            logits_list.append(self._forward_conv(conv))
            labels_list.append(conv["labels"])
        return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)
