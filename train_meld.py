"""Entry point: python train_meld.py [--epochs N] [--batch_size N] ..."""
import argparse
from src.config import MERCConfig
from src.run_meld import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  default="Dataset/Processed")
    parser.add_argument("--epochs",     type=int,   default=60)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()

    cfg = MERCConfig(
        dataset    = "meld",
        data_root  = args.data_root,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
        seed       = args.seed,
        device     = args.device,
    )
    main(cfg)
