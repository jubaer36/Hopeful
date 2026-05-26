"""Training curve and summary visualisations."""

import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ---------------------------------------------------------------------------
# Per-fold training curve
# ---------------------------------------------------------------------------

def plot_fold_curves(
    fold_i:   int,
    history:  dict,           # keys: train_loss, dev_loss, dev_wf1, te_wf1 (lists over epochs)
    best_epoch: int,
    save_dir: str = "plots",
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    epochs = list(range(len(history["train_loss"])))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"IEMOCAP  Fold {fold_i}", fontsize=13, fontweight="bold")

    # --- Loss curves ---
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train loss",  color="#4C72B0", lw=1.5)
    ax.plot(epochs, history["dev_loss"],   label="Dev loss",    color="#DD8452", lw=1.5)
    if "test_loss" in history:
        ax.plot(epochs, history["test_loss"], label="Test loss", color="#55A868",
                lw=1.0, linestyle="--", alpha=0.7)
    ax.axvline(best_epoch, color="red", linestyle=":", lw=1.2, label=f"Best ep {best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- WF1 curves ---
    ax = axes[1]
    ax.plot(epochs, history["dev_wf1"],  label="Dev WF1",  color="#DD8452", lw=1.5)
    ax.plot(epochs, history["dev_uf1"],  label="Dev UF1",  color="#DD8452", lw=1.0,
            linestyle="--", alpha=0.8)
    if "test_wf1" in history:
        ax.plot(epochs, history["test_wf1"], label="Test WF1", color="#55A868",
                lw=1.5, alpha=0.85)
    ax.axvline(best_epoch, color="red", linestyle=":", lw=1.2, label=f"Best ep {best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.set_title("WF1 / UF1")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"iemocap_fold{fold_i}_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {path}")


# ---------------------------------------------------------------------------
# Cross-fold summary
# ---------------------------------------------------------------------------

def plot_kfold_summary(
    fold_results: List[dict],     # list of {wa, wf1, uf1, f1_per_class, ...}
    emotion_names: List[str],
    save_dir: str = "plots",
    tag: str = "iemocap",
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    k = len(fold_results)

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle(f"IEMOCAP {k}-fold CV — Session 5 test", fontsize=13, fontweight="bold")

    fold_ids = list(range(k))
    colors   = plt.cm.tab10.colors

    # --- WA per fold ---
    ax0 = fig.add_subplot(gs[0, 0])
    wa_vals  = [r["wa"]  for r in fold_results]
    wf1_vals = [r["wf1"] for r in fold_results]
    uf1_vals = [r["uf1"] for r in fold_results]
    x  = np.arange(k)
    w  = 0.25
    ax0.bar(x - w,   wa_vals,  w, label="WA",  color="#4C72B0", alpha=0.85)
    ax0.bar(x,       wf1_vals, w, label="WF1", color="#DD8452", alpha=0.85)
    ax0.bar(x + w,   uf1_vals, w, label="UF1", color="#55A868", alpha=0.85)
    ax0.axhline(np.mean(wa_vals),  color="#4C72B0", lw=0.8, linestyle="--")
    ax0.axhline(np.mean(wf1_vals), color="#DD8452", lw=0.8, linestyle="--")
    ax0.axhline(np.mean(uf1_vals), color="#55A868", lw=0.8, linestyle="--")
    ax0.set_xticks(x); ax0.set_xticklabels([f"F{i}" for i in fold_ids])
    ax0.set_ylim(0, 1); ax0.set_ylabel("Score"); ax0.set_title("Per-fold metrics")
    ax0.legend(fontsize=8); ax0.grid(True, alpha=0.3, axis="y")

    # --- Per-class F1 across folds (box) ---
    ax1 = fig.add_subplot(gs[0, 1])
    n_cls = len(emotion_names)
    cls_f1 = [
        [fold_results[fi]["f1_per_class"][c] for fi in range(k)]
        for c in range(n_cls)
    ]
    bp = ax1.boxplot(cls_f1, patch_artist=True,
                     medianprops=dict(color="black", lw=1.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax1.set_xticklabels(emotion_names, rotation=30, ha="right", fontsize=8)
    ax1.set_ylim(0, 1); ax1.set_ylabel("F1"); ax1.set_title("Per-class F1 across folds")
    ax1.grid(True, alpha=0.3, axis="y")

    # --- Mean ± std bar ---
    ax2 = fig.add_subplot(gs[1, 0])
    metrics = ["WA", "WF1", "UF1"]
    means   = [np.mean(wa_vals), np.mean(wf1_vals), np.mean(uf1_vals)]
    stds    = [np.std(wa_vals),  np.std(wf1_vals),  np.std(uf1_vals)]
    bars = ax2.bar(metrics, means, yerr=stds, capsize=6,
                   color=["#4C72B0", "#DD8452", "#55A868"], alpha=0.85, width=0.4)
    for bar, m, s in zip(bars, means, stds):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 m + s + 0.01, f"{m:.4f}\n±{s:.4f}",
                 ha="center", va="bottom", fontsize=8)
    ax2.set_ylim(0, 1); ax2.set_ylabel("Score"); ax2.set_title("Mean ± std")
    ax2.grid(True, alpha=0.3, axis="y")

    # --- Mean per-class F1 bar ---
    ax3 = fig.add_subplot(gs[1, 1])
    mean_cls = [np.mean([fold_results[fi]["f1_per_class"][c] for fi in range(k)])
                for c in range(n_cls)]
    std_cls  = [np.std( [fold_results[fi]["f1_per_class"][c] for fi in range(k)])
                for c in range(n_cls)]
    ax3.bar(range(n_cls), mean_cls, yerr=std_cls, capsize=5,
            color=colors[:n_cls], alpha=0.8)
    ax3.set_xticks(range(n_cls))
    ax3.set_xticklabels(emotion_names, rotation=30, ha="right", fontsize=8)
    ax3.set_ylim(0, 1); ax3.set_ylabel("F1"); ax3.set_title("Mean per-class F1")
    ax3.grid(True, alpha=0.3, axis="y")

    path = os.path.join(save_dir, f"{tag}_kfold_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {path}")


# ---------------------------------------------------------------------------
# MELD single-run curves
# ---------------------------------------------------------------------------

def plot_meld_curves(
    history:    dict,      # train_loss, dev_loss, dev_wf1, dev_uf1, test_wf1 (optional)
    best_epoch: int,
    save_dir:   str = "plots",
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    epochs = list(range(len(history["train_loss"])))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("MELD training curves", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train", color="#4C72B0", lw=1.5)
    ax.plot(epochs, history["dev_loss"],   label="Dev",   color="#DD8452", lw=1.5)
    ax.axvline(best_epoch, color="red", linestyle=":", lw=1.2, label=f"Best ep {best_epoch}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, history["dev_wf1"],  label="Dev WF1",  color="#DD8452", lw=1.5)
    ax.plot(epochs, history["dev_uf1"],  label="Dev UF1",  color="#DD8452", lw=1.0,
            linestyle="--", alpha=0.8)
    if "test_wf1" in history:
        ax.plot(epochs, history["test_wf1"], label="Test WF1", color="#55A868",
                lw=1.5, alpha=0.85)
    ax.axvline(best_epoch, color="red", linestyle=":", lw=1.2, label=f"Best ep {best_epoch}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1"); ax.set_title("WF1 / UF1")
    ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "meld_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {path}")


def plot_meld_summary(
    test_metrics:  dict,
    emotion_names: List[str],
    save_dir:      str = "plots",
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    n_cls  = len(emotion_names)
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("MELD test results", fontsize=13, fontweight="bold")

    ax = axes[0]
    metrics = ["WA", "WF1", "UF1"]
    vals    = [test_metrics["wa"], test_metrics["wf1"], test_metrics["uf1"]]
    bars = ax.bar(metrics, vals, color=["#4C72B0", "#DD8452", "#55A868"], alpha=0.85, width=0.4)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1); ax.set_title("Overall metrics"); ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    f1s = test_metrics["f1_per_class"]
    ax.bar(range(n_cls), f1s, color=colors[:n_cls], alpha=0.8)
    ax.set_xticks(range(n_cls))
    ax.set_xticklabels(emotion_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1); ax.set_title("Per-class F1"); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(save_dir, "meld_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {path}")
