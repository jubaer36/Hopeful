"""
Structured result logging for MERC training runs.

Creates a timestamped directory under results/<dataset>_<timestamp>/ containing:
  config.json          — full hyperparameter snapshot
  fold{i}_history.csv  — per-epoch metrics for each fold (IEMOCAP)
  history.csv          — per-epoch metrics (MELD single run)
  results.json         — per-fold final metrics + aggregate stats
  run.log              — full console output mirrored to file
"""

import csv
import json
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Any


class ResultLogger:
    def __init__(self, dataset: str, cfg, results_root: str = "results"):
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(results_root, f"{dataset}_{ts}")
        os.makedirs(self.run_dir, exist_ok=True)

        # Mirror stdout to run.log
        log_path = os.path.join(self.run_dir, "run.log")
        self._setup_logging(log_path)

        # Save config
        self._save_config(cfg)

        self._csv_writers: dict[str, tuple] = {}   # name → (file_handle, csv.writer)
        self._fold_results: list[dict]       = []
        self.log(f"Run directory: {self.run_dir}")

    # ------------------------------------------------------------------
    # Logging / stdout mirror
    # ------------------------------------------------------------------

    def _setup_logging(self, log_path: str):
        logger = logging.getLogger("merc")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        fmt = logging.Formatter("%(message)s")

        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        self._logger = logger

    def log(self, msg: str = ""):
        self._logger.info(msg)

    # ------------------------------------------------------------------
    # Config snapshot
    # ------------------------------------------------------------------

    def _save_config(self, cfg):
        try:
            cfg_dict = asdict(cfg)
        except Exception:
            cfg_dict = {k: str(v) for k, v in vars(cfg).items()}
        path = os.path.join(self.run_dir, "config.json")
        with open(path, "w") as f:
            json.dump(cfg_dict, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Per-epoch CSV history
    # ------------------------------------------------------------------

    def _get_csv_writer(self, name: str, fieldnames: list):
        if name not in self._csv_writers:
            path = os.path.join(self.run_dir, f"{name}.csv")
            fh   = open(path, "w", newline="", encoding="utf-8")
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            self._csv_writers[name] = (fh, writer)
        return self._csv_writers[name][1]

    def log_epoch(self, name: str, epoch: int, metrics: dict):
        """Append one row to <name>.csv. metrics dict must be flat scalars."""
        row = {"epoch": epoch, **{k: f"{v:.6f}" if isinstance(v, float) else v
                                  for k, v in metrics.items()}}
        fieldnames = list(row.keys())
        writer = self._get_csv_writer(name, fieldnames)
        writer.writerow(row)
        # Flush so the file is readable mid-run
        self._csv_writers[name][0].flush()

    # ------------------------------------------------------------------
    # Fold / run results
    # ------------------------------------------------------------------

    def log_fold_result(self, fold_i: int, metrics: dict, best_epoch: int):
        record = {
            "fold":        fold_i,
            "best_epoch":  best_epoch,
            "wa":          metrics["wa"],
            "wf1":         metrics["wf1"],
            "uf1":         metrics["uf1"],
            "f1_per_class": metrics.get("f1_per_class", []),
        }
        self._fold_results.append(record)
        self.log(f"  [fold {fold_i}] WA {metrics['wa']:.4f}  "
                 f"WF1 {metrics['wf1']:.4f}  UF1 {metrics['uf1']:.4f}  "
                 f"(best ep {best_epoch})")
        self._flush_results()

    def log_final_summary(self, emotion_names: list[str]):
        import numpy as np
        results = self._fold_results
        if not results:
            return

        summary: dict[str, Any] = {"folds": results}
        for metric in ("wa", "wf1", "uf1"):
            vals = [r[metric] for r in results]
            summary[f"{metric}_mean"] = float(np.mean(vals))
            summary[f"{metric}_std"]  = float(np.std(vals))
            summary[f"{metric}_min"]  = float(np.min(vals))
            summary[f"{metric}_max"]  = float(np.max(vals))

        # Mean per-class F1
        if results[0]["f1_per_class"]:
            n_cls = len(results[0]["f1_per_class"])
            mean_cls = [
                float(np.mean([r["f1_per_class"][c] for r in results]))
                for c in range(n_cls)
            ]
            summary["mean_f1_per_class"] = dict(zip(emotion_names, mean_cls))

        summary_path = os.path.join(self.run_dir, "results.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        self.log(f"\n  Results saved → {summary_path}")
        self.log(f"\n  {'Metric':<6}  {'Mean':>7}  {'Std':>7}  {'Min':>7}  {'Max':>7}")
        self.log("  " + "-" * 38)
        for metric in ("wa", "wf1", "uf1"):
            self.log(f"  {metric.upper():<6}  "
                     f"{summary[metric+'_mean']:7.4f}  "
                     f"{summary[metric+'_std']:7.4f}  "
                     f"{summary[metric+'_min']:7.4f}  "
                     f"{summary[metric+'_max']:7.4f}")

        if "mean_f1_per_class" in summary:
            self.log("\n  Per-class F1 (mean across folds):")
            for name, f1 in summary["mean_f1_per_class"].items():
                self.log(f"    {name:<12}  {f1:.4f}")

    def _flush_results(self):
        path = os.path.join(self.run_dir, "results.json")
        with open(path, "w") as f:
            json.dump({"folds": self._fold_results}, f, indent=2)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        for fh, _ in self._csv_writers.values():
            fh.close()
        self._csv_writers.clear()
        for handler in self._logger.handlers[:]:
            handler.close()
            self._logger.removeHandler(handler)
