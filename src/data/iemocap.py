import os
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


IEMOCAP_EMOTIONS = {"hap": 0, "sad": 1, "neu": 2, "ang": 3, "exc": 4, "fru": 5}

_IEMOCAP_SPEAKERS = [
    f"Session{s}_{g}"
    for s in range(1, 6)
    for g in ("F", "M")
]
IEMOCAP_SPEAKER_MAP = {spk: i for i, spk in enumerate(_IEMOCAP_SPEAKERS)}
IEMOCAP_NUM_SPEAKERS = len(_IEMOCAP_SPEAKERS)  # 10


class IEMOCAPDataset(Dataset):
    """
    Loads IEMOCAP conversations from a specified list of sessions.
    Use sessions=trainval_sessions for the full trainval pool,
    sessions=test_sessions for the held-out test set.
    K-fold splitting is done externally in the run script.
    """

    def __init__(self, root: str, sessions: List[str]):
        feat_root   = os.path.join(root, "IEMOCAP", "features")
        text_feats  = torch.load(os.path.join(feat_root, "text_roberta_large.pt"),
                                 weights_only=False)
        audio_feats = torch.load(os.path.join(feat_root, "audio_microsoft_wavlm_large.pt"),
                                 weights_only=False)
        sig_feats   = torch.load(os.path.join(feat_root, "video_siglip2_temporal.pt"),
                                 weights_only=False)
        au_feats    = torch.load(os.path.join(feat_root, "video_openface_au.pt"),
                                 weights_only=False)

        df = pd.read_csv(os.path.join(root, "IEMOCAP", "labels.csv"))
        df = df[df["emotion"].isin(IEMOCAP_EMOTIONS)].copy()
        df = df[df["session"].isin(sessions)].copy()

        self.num_speakers  = IEMOCAP_NUM_SPEAKERS
        self.conversations = []

        for dialog_id, grp in df.groupby("dialog", sort=False):
            grp     = grp.sort_values("start_time")
            utt_ids = grp["utt_id"].tolist()
            labels  = [IEMOCAP_EMOTIONS[e] for e in grp["emotion"].tolist()]
            spk_ids = [
                IEMOCAP_SPEAKER_MAP[f"{row.session}_{row.speaker}"]
                for _, row in grp.iterrows()
            ]

            if len(utt_ids) < 2:
                continue

            text_t  = torch.from_numpy(np.stack([text_feats[u]  for u in utt_ids])).float()
            audio_t = torch.from_numpy(np.stack([audio_feats[u] for u in utt_ids])).float()
            sig_t   = torch.from_numpy(np.stack([
                sig_feats.get(u, np.zeros((3, 1152), dtype=np.float32)) for u in utt_ids
            ])).float()
            au_t    = torch.from_numpy(np.stack([
                au_feats.get(u, np.zeros((3, 8), dtype=np.float32)) for u in utt_ids
            ])).float()

            self.conversations.append({
                "dialog_id":   dialog_id,
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


class ConvSubset(Dataset):
    """Thin wrapper around a plain list of conversation dicts."""

    def __init__(self, conversations: list):
        self.conversations = conversations

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        return self.conversations[idx]

    def get_class_weights(self) -> torch.Tensor:
        labels = [l for c in self.conversations for l in c["labels"].tolist()]
        counts = np.bincount(labels, minlength=len(IEMOCAP_EMOTIONS)).astype(float)
        counts = np.maximum(counts, 1)
        w = 1.0 / counts
        w /= w.sum()
        return torch.from_numpy(w).float()

    def get_speaker_text_means(self) -> dict:
        spk_feats = defaultdict(list)
        for conv in self.conversations:
            for i, sid in enumerate(conv["speaker_ids"].tolist()):
                spk_feats[sid].append(conv["text"][i])
        return {sid: torch.stack(fs).mean(0) for sid, fs in spk_feats.items()}


def make_kfold_splits(conversations: list, k: int, seed: int = 42):
    """
    Shuffle conversations and split into k folds.
    Yields (train_convs, dev_convs) for each fold.
    """
    rng   = np.random.default_rng(seed)
    idx   = rng.permutation(len(conversations)).tolist()
    folds = [idx[i::k] for i in range(k)]          # k nearly-equal index lists

    for fold_i in range(k):
        dev_idx   = folds[fold_i]
        train_idx = [j for fi, fold in enumerate(folds) if fi != fold_i for j in fold]
        yield (
            [conversations[j] for j in train_idx],
            [conversations[j] for j in dev_idx],
        )
