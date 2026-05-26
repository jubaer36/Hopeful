"""Train and evaluate HyFIN-Net on MELD (train/dev/test splits)."""

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config      import HyFINConfig
from .data.meld   import MELDDataset, _build_speaker_map
from .logger      import ResultLogger
from .loss        import HyFINLoss
from .model.hyfin import HyFIN
from .train       import list_collate, train_epoch, evaluate
from .visualize   import plot_meld_curves, plot_meld_summary

MELD_EMOTIONS = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]


def _class_counts_from_dataset(ds: MELDDataset, num_classes: int) -> torch.Tensor:
    labels = [l for c in ds.conversations for l in c["labels"].tolist()]
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    return torch.from_numpy(np.maximum(counts, 1)).float()


def main(cfg: HyFINConfig = None):
    if cfg is None:
        cfg = HyFINConfig(dataset="meld")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)

    logger = ResultLogger("meld", cfg)
    logger.log("Loading MELD features…")

    # Build speaker map from train split
    import pandas as pd
    df_train = pd.read_csv(os.path.join(cfg.data_root, "MELD", "labels.csv"))
    df_train = df_train[df_train["split"] == "train"]
    speaker_map, num_speakers = _build_speaker_map(df_train, cfg.meld_rare_threshold)

    train_ds = MELDDataset(cfg.data_root, "train", cfg.meld_rare_threshold,
                           speaker_map, num_speakers)
    dev_ds   = MELDDataset(cfg.data_root, "dev",   cfg.meld_rare_threshold,
                           speaker_map, num_speakers)
    test_ds  = MELDDataset(cfg.data_root, "test",  cfg.meld_rare_threshold,
                           speaker_map, num_speakers)

    logger.log(f"  train: {len(train_ds)} convs  dev: {len(dev_ds)}  test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=list_collate, num_workers=0)
    dev_loader   = DataLoader(dev_ds,   batch_size=cfg.batch_size, shuffle=False,
                              collate_fn=list_collate, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False,
                              collate_fn=list_collate, num_workers=0)

    model = HyFIN(cfg, num_speakers=num_speakers).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"  trainable parameters: {total_params:,}")

    counts  = _class_counts_from_dataset(train_ds, cfg.num_classes).to(device)
    loss_fn = HyFINLoss(cfg, counts)
    opt     = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                               weight_decay=cfg.weight_decay)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    ckpt_path    = "checkpoints/meld_best.pt"
    best_dev_wf1 = 0.0
    best_epoch   = 0

    history = {
        "train_loss": [], "dev_loss": [], "test_loss": [],
        "dev_wf1": [], "dev_uf1": [], "test_wf1": [],
    }

    logger.log(f"  {'Ep':>4}  {'TrLoss':>7}  {'DevWF1':>7}  {'TeWF1':>7}")

    for epoch in range(cfg.epochs):
        tr  = train_epoch(model, train_loader, opt, loss_fn, device,
                          cfg.grad_clip, cfg.num_classes)
        dev = evaluate(model, dev_loader,  loss_fn, device, cfg.num_classes)
        te  = evaluate(model, test_loader, loss_fn, device, cfg.num_classes)
        sched.step()

        history["train_loss"].append(tr["loss"])
        history["dev_loss"].append(dev["loss"])
        history["test_loss"].append(te["loss"])
        history["dev_wf1"].append(dev["wf1"])
        history["dev_uf1"].append(dev["uf1"])
        history["test_wf1"].append(te["wf1"])

        logger.log_epoch("history", epoch, {
            "train_loss": tr["loss"], "train_wf1": tr["wf1"],
            "dev_loss":   dev["loss"], "dev_wf1": dev["wf1"], "dev_uf1": dev["uf1"],
            "test_loss":  te["loss"], "test_wf1": te["wf1"],
        })

        if dev["wf1"] > best_dev_wf1:
            best_dev_wf1 = dev["wf1"]
            best_epoch   = epoch
            torch.save(model.state_dict(), ckpt_path)
            torch.save({"cfg": cfg, "num_speakers": num_speakers,
                        "best_epoch": best_epoch}, ckpt_path.replace(".pt", "_meta.pt"))

        if epoch % 5 == 0 or epoch == cfg.epochs - 1:
            logger.log(f"  {epoch:4d}  {tr['loss']:7.4f}  "
                       f"{dev['wf1']:7.4f}  {te['wf1']:7.4f}")

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    te_best = evaluate(model, test_loader, loss_fn, device, cfg.num_classes)
    logger.log(f"\n  Best epoch {best_epoch} | Test WF1 {te_best['wf1']:.4f}")

    plot_meld_curves(history, best_epoch,
                     save_dir=os.path.join(logger.run_dir, "plots"))
    plot_meld_summary(te_best, MELD_EMOTIONS,
                      save_dir=os.path.join(logger.run_dir, "plots"))
    logger.close()
    return {"test": te_best, "best_epoch": best_epoch, "run_dir": logger.run_dir}
