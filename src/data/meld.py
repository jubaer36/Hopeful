import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


MELD_EMOTIONS = {
    "neutral": 0, "joy": 1, "surprise": 2, "anger": 3,
    "sadness": 4, "disgust": 5, "fear": 6,
}

_MELD_MISSING = {"dia125_utt3", "dia110_utt7"}


def _build_speaker_map(train_df: pd.DataFrame, rare_threshold: int):
    counts = train_df["speaker"].value_counts()
    rare = set(counts[counts < rare_threshold].index)
    regular = sorted(counts[counts >= rare_threshold].index.tolist())
    spk_map = {s: i for i, s in enumerate(regular)}
    background_id = len(regular)
    for s in rare:
        spk_map[s] = background_id
    num_speakers = background_id + 1
    return spk_map, num_speakers


class MELDDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        rare_threshold: int = 5,
        speaker_map: dict = None,
        num_speakers: int = None,
    ):
        feat_root = os.path.join(root, "MELD", "features")
        text_feats  = torch.load(f"{feat_root}/text_roberta_large_{split}.pt",
                                 weights_only=False)
        audio_feats = torch.load(f"{feat_root}/audio_microsoft_wavlm_large_{split}.pt",
                                 weights_only=False)
        sig_feats   = torch.load(f"{feat_root}/video_siglip2_temporal_{split}.pt",
                                 weights_only=False)
        au_feats    = torch.load(f"{feat_root}/video_openface_au_{split}.pt",
                                 weights_only=False)

        df_all = pd.read_csv(os.path.join(root, "MELD", "labels.csv"))
        train_df = df_all[df_all["split"] == "train"].copy()

        if speaker_map is None:
            speaker_map, num_speakers = _build_speaker_map(train_df, rare_threshold)

        self.speaker_map  = speaker_map
        self.num_speakers = num_speakers
        self._background  = num_speakers - 1

        df = df_all[df_all["split"] == split].copy()
        df = df[~df["clip_name"].isin(_MELD_MISSING)]

        self.conversations = []

        for dia_id, grp in df.groupby("dia_id", sort=False):
            grp = grp.sort_values("utt_id")
            clips   = grp["clip_name"].tolist()
            labels  = [MELD_EMOTIONS[e] for e in grp["emotion"].tolist()]
            spk_ids = [
                speaker_map.get(s, self._background)
                for s in grp["speaker"].tolist()
            ]

            # Keep only utterances whose text features exist
            valid = [c for c in clips if c in text_feats]
            if len(valid) < 2:
                continue

            mask    = [c in text_feats for c in clips]
            clips   = [c for c, v in zip(clips, mask)   if v]
            labels  = [l for l, v in zip(labels, mask)  if v]
            spk_ids = [s for s, v in zip(spk_ids, mask) if v]

            text_t  = torch.from_numpy(np.stack([text_feats[c]  for c in clips])).float()
            audio_t = torch.from_numpy(np.stack([
                audio_feats.get(c, np.zeros(1024, dtype=np.float32)) for c in clips
            ])).float()
            sig_t   = torch.from_numpy(np.stack([
                sig_feats.get(c, np.zeros((3, 1152), dtype=np.float32)) for c in clips
            ])).float()
            au_t    = torch.from_numpy(np.stack([
                au_feats.get(c, np.zeros((3, 8), dtype=np.float32)) for c in clips
            ])).float()

            self.conversations.append({
                "dialog_id":   f"dia{dia_id}",
                "text":        text_t,
                "audio":       audio_t,
                "siglip2":     sig_t,
                "openface":    au_t,
                "labels":      torch.tensor(labels,  dtype=torch.long),
                "speaker_ids": torch.tensor(spk_ids, dtype=torch.long),
            })

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        return self.conversations[idx]

    def get_class_weights(self) -> torch.Tensor:
        all_labels = []
        for c in self.conversations:
            all_labels.extend(c["labels"].tolist())
        counts = np.bincount(all_labels, minlength=len(MELD_EMOTIONS)).astype(float)
        counts = np.maximum(counts, 1)
        w = 1.0 / counts
        w /= w.sum()
        return torch.from_numpy(w).float()
