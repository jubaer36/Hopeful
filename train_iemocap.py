"""Entry point: python train_iemocap.py [--epochs N] [--batch_size N] ..."""
import argparse
from src.config import MERCConfig
from src.run_iemocap import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  default="Dataset/Processed")
    parser.add_argument("--epochs",     type=int,   default=60)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--k_folds",    type=int,   default=5)
    parser.add_argument("--d",          type=int,   default=64)
    parser.add_argument("--d_spk",      type=int,   default=32)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--device",     default="cuda")
    # Ablation flags
    parser.add_argument("--no_path_a",        action="store_true")
    parser.add_argument("--no_path_b",        action="store_true")
    parser.add_argument("--no_cross_modal",   action="store_true")
    parser.add_argument("--no_speaker_emb",   action="store_true")
    args = parser.parse_args()

    cfg = MERCConfig(
        dataset              = "iemocap",
        data_root            = args.data_root,
        epochs               = args.epochs,
        batch_size           = args.batch_size,
        lr                   = args.lr,
        seed                 = args.seed,
        iemocap_k_folds      = args.k_folds,
        d                    = args.d,
        d_spk                = args.d_spk,
        dropout              = args.dropout,
        device               = args.device,
        use_path_a           = not args.no_path_a,
        use_path_b           = not args.no_path_b,
        use_cross_modal_attn = not args.no_cross_modal,
        use_speaker_emb      = not args.no_speaker_emb,
    )
    main(cfg)
