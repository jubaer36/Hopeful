"""Train and evaluate MERC on MELD (train/dev/test splits)."""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config     import MERCConfig
from .data.meld  import MELDDataset, _build_speaker_map
from .logger     import ResultLogger
from .loss       import AnnealedFocalLoss
from .model.merc import MERC
from .train      import list_collate, train_epoch, evaluate
from .visualize  import plot_meld_curves, plot_meld_summary

_MELD_EMOTIONS = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]


def main(cfg: MERCConfig = None):
    if cfg is None:
        cfg = MERCConfig(dataset="meld")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)

    logger = ResultLogger("meld", cfg)
    logger.log("Building MELD speaker map from train split…")

    df_all      = pd.read_csv(os.path.join(cfg.data_root, "MELD", "labels.csv"))
    train_df    = df_all[df_all["split"] == "train"]
    speaker_map, num_speakers = _build_speaker_map(train_df, cfg.meld_rare_threshold)
    logger.log(f"  speakers: {num_speakers} ({num_speakers-1} regular + 1 background)")

    logger.log("Loading MELD datasets…")
    train_ds = MELDDataset(cfg.data_root, "train", cfg.meld_rare_threshold,
                           speaker_map, num_speakers)
    dev_ds   = MELDDataset(cfg.data_root, "dev",   cfg.meld_rare_threshold,
                           speaker_map, num_speakers)
    test_ds  = MELDDataset(cfg.data_root, "test",  cfg.meld_rare_threshold,
                           speaker_map, num_speakers)
    logger.log(f"  train: {len(train_ds)}  dev: {len(dev_ds)}  test: {len(test_ds)} convs")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=list_collate, num_workers=0)
    dev_loader   = DataLoader(dev_ds,   batch_size=cfg.batch_size, shuffle=False,
                              collate_fn=list_collate, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False,
                              collate_fn=list_collate, num_workers=0)

    model = MERC(cfg, num_speakers=num_speakers).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"  trainable parameters: {total_params:,}")

    class_weights = train_ds.get_class_weights().to(device)
    loss_fn       = AnnealedFocalLoss(class_weights, cfg.focal_start,
                                      cfg.focal_end, cfg.focal_gamma)
    optimizer     = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    best_wf1   = 0.0
    best_epoch = 0
    best_ckpt  = "checkpoints/meld_best.pt"

    history = {
        "train_loss": [], "dev_loss":  [],
        "dev_wf1":    [], "dev_uf1":   [],
        "test_wf1":   [],
    }

    logger.log(f"\n{'Ep':>4}  {'TrLoss':>7}  {'DevLoss':>8}  {'DevWF1':>7}  "
               f"{'DevUF1':>7}  {'TeWF1':>7}")
    logger.log("-" * 56)

    for epoch in range(cfg.epochs):
        tr  = train_epoch(model, train_loader, optimizer, loss_fn,
                          epoch, device, cfg.grad_clip, cfg.num_classes)
        dev = evaluate(model, dev_loader,  loss_fn, epoch, device, cfg.num_classes)
        te  = evaluate(model, test_loader, loss_fn, epoch, device, cfg.num_classes)

        history["train_loss"].append(tr["loss"])
        history["dev_loss"].append(dev["loss"])
        history["dev_wf1"].append(dev["wf1"])
        history["dev_uf1"].append(dev["uf1"])
        history["test_wf1"].append(te["wf1"])

        logger.log_epoch("history", epoch, {
            "train_loss": tr["loss"],  "train_wf1": tr["wf1"],
            "dev_loss":   dev["loss"], "dev_wf1":   dev["wf1"], "dev_uf1": dev["uf1"],
            "test_loss":  te["loss"],  "test_wf1":  te["wf1"],  "test_uf1": te["uf1"],
        })

        logger.log(f"{epoch:4d}  {tr['loss']:7.4f}  {dev['loss']:8.4f}  "
                   f"{dev['wf1']:7.4f}  {dev['uf1']:7.4f}  {te['wf1']:7.4f}")

        if dev["wf1"] > best_wf1:
            best_wf1   = dev["wf1"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_ckpt)
            torch.save({
                "dataset":      "meld",
                "num_speakers": num_speakers,
                "num_classes":  cfg.num_classes,
                "cfg":          cfg,
                "best_epoch":   best_epoch,
            }, best_ckpt.replace(".pt", "_meta.pt"))

    # Final evaluation with best-dev checkpoint
    model.load_state_dict(torch.load(best_ckpt, weights_only=True))
    te_final = evaluate(model, test_loader, loss_fn, cfg.epochs - 1,
                        device, cfg.num_classes)

    logger.log(f"\nBest ep {best_epoch} | dev WF1 {best_wf1:.4f}")
    logger.log_fold_result(0, te_final, best_epoch)   # reuse fold mechanism for single-run summary
    logger.log_final_summary(_MELD_EMOTIONS)

    plot_dir = os.path.join(logger.run_dir, "plots")
    plot_meld_curves(history, best_epoch, save_dir=plot_dir)
    plot_meld_summary(te_final, _MELD_EMOTIONS, save_dir=plot_dir)
    logger.close()
    return {"te_final": te_final, "best_wf1": best_wf1, "run_dir": logger.run_dir}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  default="Dataset/Processed")
    parser.add_argument("--epochs",     type=int,   default=60)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    cfg = MERCConfig(
        dataset    = "meld",
        data_root  = args.data_root,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
        seed       = args.seed,
    )
    main(cfg)
