"""
Train and evaluate MERC on IEMOCAP.

Split protocol:
  Test    : Session 5 (fixed, never touched during training)
  Trainval: Sessions 1–4, split into k folds
  Each fold: train on (k-1) folds, select best checkpoint by dev WF1,
             evaluate on session-5 test set.
  Final report: mean ± std of test metrics across k folds.
"""

import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config       import MERCConfig
from .data.iemocap import (
    IEMOCAPDataset, ConvSubset, make_kfold_splits, IEMOCAP_NUM_SPEAKERS,
)
from .logger       import ResultLogger
from .loss         import AnnealedFocalLoss
from .model.merc   import MERC
from .train        import list_collate, train_epoch, evaluate
from .visualize    import plot_fold_curves, plot_kfold_summary


# ---------------------------------------------------------------------------
# Speaker embedding initialisation
# ---------------------------------------------------------------------------

def _init_speaker_embeddings(model: MERC, train_subset: ConvSubset, device: torch.device):
    spk_means = train_subset.get_speaker_text_means()
    emb_dim   = model.spk_pos.speaker_emb.embedding_dim
    text_dim  = next(iter(spk_means.values())).shape[0]

    torch.manual_seed(0)
    proj = torch.randn(text_dim, emb_dim, device=device) / (text_dim ** 0.5)

    with torch.no_grad():
        for spk_id, mean_feat in spk_means.items():
            model.spk_pos.speaker_emb.weight.data[spk_id] = mean_feat.to(device) @ proj


# ---------------------------------------------------------------------------
# Single fold training
# ---------------------------------------------------------------------------

def _train_fold(
    fold_i:      int,
    train_convs: list,
    dev_convs:   list,
    test_ds:     IEMOCAPDataset,
    cfg:         MERCConfig,
    device:      torch.device,
    logger:      ResultLogger,
) -> dict:
    train_sub = ConvSubset(train_convs)
    dev_sub   = ConvSubset(dev_convs)

    train_loader = DataLoader(train_sub, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=list_collate, num_workers=0)
    dev_loader   = DataLoader(dev_sub,   batch_size=cfg.batch_size, shuffle=False,
                              collate_fn=list_collate, num_workers=0)
    test_loader  = DataLoader(test_ds,   batch_size=cfg.batch_size, shuffle=False,
                              collate_fn=list_collate, num_workers=0)

    model = MERC(cfg, num_speakers=IEMOCAP_NUM_SPEAKERS).to(device)
    _init_speaker_embeddings(model, train_sub, device)

    class_weights = train_sub.get_class_weights().to(device)
    loss_fn       = AnnealedFocalLoss(class_weights, cfg.focal_start,
                                      cfg.focal_end, cfg.focal_gamma)
    optimizer     = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    ckpt_path    = f"checkpoints/iemocap_fold{fold_i}.pt"
    best_dev_wf1 = 0.0
    best_epoch   = 0
    csv_name     = f"fold{fold_i}_history"

    history = {
        "train_loss": [], "dev_loss": [], "test_loss": [],
        "dev_wf1":    [], "dev_uf1":  [], "test_wf1":  [],
    }

    logger.log(f"\n  Fold {fold_i}  train={len(train_convs)} dev={len(dev_convs)} convs")
    logger.log(f"  {'Ep':>4}  {'TrLoss':>7}  {'DevLoss':>8}  {'DevWF1':>7}  "
               f"{'DevUF1':>7}  {'TeWF1':>7}")

    for epoch in range(cfg.epochs):
        tr  = train_epoch(model, train_loader, optimizer, loss_fn,
                          epoch, device, cfg.grad_clip, cfg.num_classes)
        dev = evaluate(model, dev_loader,  loss_fn, epoch, device, cfg.num_classes)
        te  = evaluate(model, test_loader, loss_fn, epoch, device, cfg.num_classes)

        history["train_loss"].append(tr["loss"])
        history["dev_loss"].append(dev["loss"])
        history["test_loss"].append(te["loss"])
        history["dev_wf1"].append(dev["wf1"])
        history["dev_uf1"].append(dev["uf1"])
        history["test_wf1"].append(te["wf1"])

        # Log every epoch to CSV
        logger.log_epoch(csv_name, epoch, {
            "train_loss": tr["loss"],  "train_wf1": tr["wf1"],
            "dev_loss":   dev["loss"], "dev_wf1":   dev["wf1"], "dev_uf1": dev["uf1"],
            "test_loss":  te["loss"],  "test_wf1":  te["wf1"],  "test_uf1": te["uf1"],
        })

        if dev["wf1"] > best_dev_wf1:
            best_dev_wf1 = dev["wf1"]
            best_epoch   = epoch
            torch.save(model.state_dict(), ckpt_path)
            torch.save({
                "dataset":       "iemocap",
                "num_speakers":  IEMOCAP_NUM_SPEAKERS,
                "num_classes":   cfg.num_classes,
                "cfg":           cfg,
                "fold":          fold_i,
                "best_epoch":    best_epoch,
                "test_sessions": cfg.iemocap_test_sessions,
            }, ckpt_path.replace(".pt", "_meta.pt"))

        if epoch % 10 == 0 or epoch == cfg.epochs - 1:
            logger.log(f"  {epoch:4d}  {tr['loss']:7.4f}  {dev['loss']:8.4f}  "
                       f"{dev['wf1']:7.4f}  {dev['uf1']:7.4f}  {te['wf1']:7.4f}")

    # Load best-dev checkpoint, final test eval
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    te_best = evaluate(model, test_loader, loss_fn, cfg.epochs - 1, device, cfg.num_classes)

    plot_fold_curves(fold_i, history, best_epoch, save_dir=os.path.join(logger.run_dir, "plots"))
    logger.log_fold_result(fold_i, te_best, best_epoch)
    return te_best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: MERCConfig = None):
    if cfg is None:
        cfg = MERCConfig(dataset="iemocap")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)

    logger = ResultLogger("iemocap", cfg)
    logger.log("Loading IEMOCAP features…")

    trainval_ds = IEMOCAPDataset(cfg.data_root, sessions=cfg.iemocap_trainval_sessions)
    test_ds     = IEMOCAPDataset(cfg.data_root, sessions=cfg.iemocap_test_sessions)
    logger.log(f"  trainval: {len(trainval_ds)} conversations  |  "
               f"test (Session5): {len(test_ds)}")

    model_tmp    = MERC(cfg, IEMOCAP_NUM_SPEAKERS)
    total_params = sum(p.numel() for p in model_tmp.parameters() if p.requires_grad)
    logger.log(f"  trainable parameters: {total_params:,}")
    del model_tmp

    fold_results = []
    for fold_i, (train_convs, dev_convs) in enumerate(
        make_kfold_splits(trainval_ds.conversations, cfg.iemocap_k_folds, cfg.seed)
    ):
        te = _train_fold(fold_i, train_convs, dev_convs, test_ds, cfg, device, logger)
        fold_results.append(te)

    logger.log(f"\n{'='*55}")
    logger.log(f"  {cfg.iemocap_k_folds}-fold CV  —  Session-5 test set")
    logger.log(f"{'='*55}")
    logger.log_final_summary(cfg.iemocap_emotions)

    plot_kfold_summary(fold_results, cfg.iemocap_emotions,
                       save_dir=os.path.join(logger.run_dir, "plots"))
    logger.close()
    return fold_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   default="Dataset/Processed")
    parser.add_argument("--epochs",      type=int,   default=60)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--k_folds",     type=int,   default=5)
    args = parser.parse_args()

    cfg = MERCConfig(
        dataset         = "iemocap",
        data_root       = args.data_root,
        epochs          = args.epochs,
        batch_size      = args.batch_size,
        lr              = args.lr,
        seed            = args.seed,
        iemocap_k_folds = args.k_folds,
    )
    main(cfg)
