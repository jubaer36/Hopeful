"""Train and evaluate HyFIN-Net on IEMOCAP (5-fold CV, Session-5 test set)."""

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config       import HyFINConfig
from .data.iemocap import (
    IEMOCAPDataset, ConvSubset, make_kfold_splits, IEMOCAP_NUM_SPEAKERS,
)
from .logger       import ResultLogger
from .loss         import HyFINLoss, effective_num_weights
from .metrics      import compute_metrics
from .model.hyfin  import HyFIN
from .train        import list_collate, train_epoch, evaluate
from .visualize    import plot_fold_curves, plot_kfold_summary


def _class_counts(subset: ConvSubset, num_classes: int) -> torch.Tensor:
    labels = [l for c in subset.conversations for l in c["labels"].tolist()]
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    return torch.from_numpy(np.maximum(counts, 1)).float()


def _train_fold(fold_i, train_convs, dev_convs, test_ds, cfg, device, logger):
    train_sub = ConvSubset(train_convs)
    dev_sub   = ConvSubset(dev_convs)

    train_loader = DataLoader(train_sub, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=list_collate, num_workers=0)
    dev_loader   = DataLoader(dev_sub,   batch_size=cfg.batch_size, shuffle=False,
                              collate_fn=list_collate, num_workers=0)
    test_loader  = DataLoader(test_ds,   batch_size=cfg.batch_size, shuffle=False,
                              collate_fn=list_collate, num_workers=0)

    model = HyFIN(cfg, num_speakers=IEMOCAP_NUM_SPEAKERS).to(device)
    counts  = _class_counts(train_sub, cfg.num_classes).to(device)
    loss_fn = HyFINLoss(cfg, counts)
    opt     = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                               weight_decay=cfg.weight_decay)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    ckpt_path    = f"checkpoints/iemocap_fold{fold_i}.pt"
    best_dev_wf1 = 0.0
    best_epoch   = 0
    csv_name     = f"fold{fold_i}_history"

    history = {
        "train_loss": [], "dev_loss": [], "test_loss": [],
        "dev_wf1": [],    "dev_uf1": [],  "test_wf1": [],
    }

    logger.log(f"\n  Fold {fold_i}  train={len(train_convs)} dev={len(dev_convs)} convs")
    logger.log(f"  {'Ep':>4}  {'TrLoss':>7}  {'DevWF1':>7}  {'TeWF1':>7}  "
               f"{'CE':>7}  {'FC':>7}  {'CL':>7}")

    for epoch in range(cfg.epochs):
        # DualCL warmup: ramp λ from 0 → lam over dual_cl_warmup epochs
        warmup = cfg.dual_cl_warmup
        loss_fn.lam_live = loss_fn.lam * min(1.0, (epoch + 1) / max(warmup, 1))

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

        logger.log_epoch(csv_name, epoch, {
            "train_loss": tr["loss"],   "train_wf1": tr["wf1"],
            "dev_loss":   dev["loss"],  "dev_wf1":   dev["wf1"],  "dev_uf1": dev["uf1"],
            "test_loss":  te["loss"],   "test_wf1":  te["wf1"],   "test_uf1": te["uf1"],
            "ce": tr["ce"], "fc": tr["fc"], "cl": tr["cl"],
        })

        if dev["wf1"] > best_dev_wf1:
            best_dev_wf1 = dev["wf1"]
            best_epoch   = epoch
            torch.save(model.state_dict(), ckpt_path)
            torch.save({
                "dataset": "iemocap", "num_speakers": IEMOCAP_NUM_SPEAKERS,
                "num_classes": cfg.num_classes, "cfg": cfg,
                "fold": fold_i, "best_epoch": best_epoch,
            }, ckpt_path.replace(".pt", "_meta.pt"))

        if epoch % 5 == 0 or epoch == cfg.epochs - 1:
            logger.log(f"  {epoch:4d}  {tr['loss']:7.4f}  {dev['wf1']:7.4f}  "
                       f"{te['wf1']:7.4f}  {tr['ce']:7.4f}  "
                       f"{tr['fc']:7.4f}  {tr['cl']:7.4f}")

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    te_best = evaluate(model, test_loader, loss_fn, device, cfg.num_classes)
    plot_fold_curves(fold_i, history, best_epoch,
                     save_dir=os.path.join(logger.run_dir, "plots"))
    logger.log_fold_result(fold_i, te_best, best_epoch)
    return te_best


def main(cfg: HyFINConfig = None):
    if cfg is None:
        cfg = HyFINConfig(dataset="iemocap")

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
    logger.log(f"  trainval: {len(trainval_ds)} convs  |  test: {len(test_ds)} convs")

    model_tmp    = HyFIN(cfg, IEMOCAP_NUM_SPEAKERS)
    total_params = sum(p.numel() for p in model_tmp.parameters() if p.requires_grad)
    logger.log(f"  trainable parameters: {total_params:,}")
    del model_tmp

    fold_results = []
    for fold_i, (train_convs, dev_convs) in enumerate(
        make_kfold_splits(trainval_ds.conversations, cfg.iemocap_k_folds, cfg.seed)
    ):
        if fold_i >= cfg.iemocap_max_folds:
            break
        te = _train_fold(fold_i, train_convs, dev_convs, test_ds, cfg, device, logger)
        fold_results.append(te)

    logger.log(f"\n{'='*55}\n  {cfg.iemocap_k_folds}-fold CV — Session-5 test\n{'='*55}")
    logger.log_final_summary(cfg.iemocap_emotions)
    plot_kfold_summary(fold_results, cfg.iemocap_emotions,
                       save_dir=os.path.join(logger.run_dir, "plots"))
    logger.close()
    return {"fold_results": fold_results, "run_dir": logger.run_dir}
