"""
Evaluate a saved MERC checkpoint on the test set.

IEMOCAP (k-fold — evaluates all saved fold checkpoints):
    python -m src.test --dataset iemocap --data_root Dataset/Processed

IEMOCAP (single fold):
    python -m src.test --dataset iemocap --ckpt checkpoints/iemocap_fold2.pt

MELD:
    python -m src.test --dataset meld --data_root Dataset/Processed

MELD (explicit checkpoint):
    python -m src.test --dataset meld --ckpt checkpoints/meld_best.pt
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config       import MERCConfig
from .data.iemocap import IEMOCAPDataset, IEMOCAP_NUM_SPEAKERS
from .data.meld    import MELDDataset, _build_speaker_map
from .loss         import AnnealedFocalLoss
from .model.merc   import MERC
from .train        import list_collate, evaluate
from .visualize    import plot_kfold_summary, plot_meld_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(ckpt_path: str, device: torch.device):
    """Load model weights + metadata. Returns (state_dict, meta_dict)."""
    meta_path = ckpt_path.replace(".pt", "_meta.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)

    meta = None
    if os.path.isfile(meta_path):
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
    return state_dict, meta


def _build_model(cfg: MERCConfig, num_speakers: int, state_dict, device: torch.device) -> MERC:
    model = MERC(cfg, num_speakers).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# IEMOCAP evaluation
# ---------------------------------------------------------------------------

def test_iemocap(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Discover fold checkpoints
    if args.ckpt:
        ckpt_paths = [args.ckpt]
    else:
        ckpt_paths = sorted([
            os.path.join("checkpoints", f)
            for f in os.listdir("checkpoints")
            if f.startswith("iemocap_fold") and f.endswith(".pt")
            and "_meta" not in f
        ])
        if not ckpt_paths:
            raise FileNotFoundError("No iemocap_fold*.pt checkpoints found in checkpoints/")

    print(f"Found {len(ckpt_paths)} IEMOCAP checkpoint(s): {[os.path.basename(p) for p in ckpt_paths]}")

    # Load config from first checkpoint's meta (or use defaults)
    _, meta0 = _load_checkpoint(ckpt_paths[0], device)
    if meta0 is not None:
        cfg           = meta0["cfg"]
        num_speakers  = meta0["num_speakers"]
        test_sessions = meta0.get("test_sessions", cfg.iemocap_test_sessions)
    else:
        print("  [warn] no _meta.pt found — using default config")
        cfg           = MERCConfig(dataset="iemocap", data_root=args.data_root)
        num_speakers  = IEMOCAP_NUM_SPEAKERS
        test_sessions = cfg.iemocap_test_sessions

    cfg.data_root = args.data_root  # allow override

    print(f"  test sessions: {test_sessions}")
    test_ds = IEMOCAPDataset(cfg.data_root, sessions=test_sessions)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             collate_fn=list_collate, num_workers=0)
    print(f"  test conversations: {len(test_ds)}")

    class_weights = torch.ones(cfg.num_classes).to(device)
    loss_fn       = AnnealedFocalLoss(class_weights, cfg.focal_start,
                                      cfg.focal_end, cfg.focal_gamma)

    fold_results = []
    for ckpt_path in ckpt_paths:
        state_dict, meta = _load_checkpoint(ckpt_path, device)
        model = _build_model(cfg, num_speakers, state_dict, device)
        te = evaluate(model, test_loader, loss_fn, cfg.focal_end, device, cfg.num_classes)

        fold_name = os.path.basename(ckpt_path)
        best_ep   = meta["best_epoch"] if meta else "?"
        print(f"  {fold_name} (best ep {best_ep}) → "
              f"WA {te['wa']:.4f}  WF1 {te['wf1']:.4f}  UF1 {te['uf1']:.4f}")
        fold_results.append(te)

    if len(fold_results) > 1:
        print(f"\n  {'Metric':<6}  {'Mean':>7}  {'Std':>7}  {'Min':>7}  {'Max':>7}")
        print("  " + "-" * 38)
        for metric in ("wa", "wf1", "uf1"):
            vals = [r[metric] for r in fold_results]
            print(f"  {metric.upper():<6}  {np.mean(vals):7.4f}  {np.std(vals):7.4f}  "
                  f"{np.min(vals):7.4f}  {np.max(vals):7.4f}")

        plot_kfold_summary(fold_results, cfg.iemocap_emotions,
                           save_dir=args.plot_dir, tag="iemocap_test")
    else:
        te = fold_results[0]
        for metric in ("wa", "wf1", "uf1"):
            print(f"  {metric.upper()}: {te[metric]:.4f}")


# ---------------------------------------------------------------------------
# MELD evaluation
# ---------------------------------------------------------------------------

def test_meld(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.ckpt or "checkpoints/meld_best.pt"
    state_dict, meta = _load_checkpoint(ckpt_path, device)

    if meta is not None:
        cfg          = meta["cfg"]
        num_speakers = meta["num_speakers"]
    else:
        print("  [warn] no _meta.pt found — rebuilding speaker map")
        cfg          = MERCConfig(dataset="meld", data_root=args.data_root)
        df_all       = pd.read_csv(os.path.join(args.data_root, "MELD", "labels.csv"))
        train_df     = df_all[df_all["split"] == "train"]
        _, num_speakers = _build_speaker_map(train_df, cfg.meld_rare_threshold)

    cfg.data_root = args.data_root

    # Rebuild speaker map for dataset construction
    df_all   = pd.read_csv(os.path.join(cfg.data_root, "MELD", "labels.csv"))
    train_df = df_all[df_all["split"] == "train"]
    speaker_map, _ = _build_speaker_map(train_df, cfg.meld_rare_threshold)

    test_ds = MELDDataset(cfg.data_root, "test", cfg.meld_rare_threshold,
                          speaker_map, num_speakers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             collate_fn=list_collate, num_workers=0)
    print(f"  test conversations: {len(test_ds)}")

    model = _build_model(cfg, num_speakers, state_dict, device)

    class_weights = torch.ones(cfg.num_classes).to(device)
    loss_fn       = AnnealedFocalLoss(class_weights, cfg.focal_start,
                                      cfg.focal_end, cfg.focal_gamma)

    best_ep = meta["best_epoch"] if meta else "?"
    te = evaluate(model, test_loader, loss_fn, cfg.focal_end, device, cfg.num_classes)

    print(f"\n  Checkpoint: {os.path.basename(ckpt_path)} (best ep {best_ep})")
    print(f"  {'Metric':<6}  {'Score':>7}")
    print("  " + "-" * 16)
    for metric in ("wa", "wf1", "uf1"):
        print(f"  {metric.upper():<6}  {te[metric]:7.4f}")
    print()
    emotion_names = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]
    for name, f1 in zip(emotion_names, te["f1_per_class"]):
        print(f"  {name:<10}  F1 {f1:.4f}")

    plot_meld_summary(te, emotion_names, save_dir=args.plot_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate saved MERC checkpoint")
    parser.add_argument("--dataset",   required=True, choices=["iemocap", "meld"])
    parser.add_argument("--data_root", default="Dataset/Processed",
                        help="Path to Dataset/Processed/")
    parser.add_argument("--ckpt",      default=None,
                        help="Explicit checkpoint path. Omit to auto-discover.")
    parser.add_argument("--plot_dir",  default="plots",
                        help="Directory for output plots")
    args = parser.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)

    if args.dataset == "iemocap":
        test_iemocap(args)
    else:
        test_meld(args)


if __name__ == "__main__":
    main()
