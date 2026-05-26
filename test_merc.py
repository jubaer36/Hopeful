"""Entry point: python test_merc.py --dataset iemocap|meld [--ckpt path]"""
import argparse
import os
from src.test import test_iemocap, test_meld

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate saved MERC checkpoint")
    parser.add_argument("--dataset",   required=True, choices=["iemocap", "meld"])
    parser.add_argument("--data_root", default="Dataset/Processed")
    parser.add_argument("--ckpt",      default=None,
                        help="Explicit .pt path. Omit to auto-discover.")
    parser.add_argument("--plot_dir",  default="plots")
    parser.add_argument("--device",    default="cuda")
    args = parser.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)

    if args.dataset == "iemocap":
        test_iemocap(args)
    else:
        test_meld(args)
