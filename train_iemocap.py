"""Entry point: python train_iemocap.py [options]"""
import argparse
from src.config    import HyFINConfig
from src.run_iemocap import main


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  default="Dataset/Processed")
    p.add_argument("--epochs",     type=int,   default=40)
    p.add_argument("--batch_size", type=int,   default=12)
    p.add_argument("--lr",         type=float, default=4e-4)
    p.add_argument("--d_h",        type=int,   default=512)
    p.add_argument("--dropout",    type=float, default=0.3)
    p.add_argument("--n_hyp",      type=int,   default=2)
    p.add_argument("--k_freq",     type=int,   default=4)
    p.add_argument("--n_inc",      type=int,   default=3)
    p.add_argument("--mu",         type=float, default=0.8)
    p.add_argument("--lam",        type=float, default=0.1)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--device",     default="cuda")
    # Ablation
    p.add_argument("--no_igm",            action="store_true")
    p.add_argument("--no_hm",             action="store_true")
    p.add_argument("--no_mfm",            action="store_true")
    p.add_argument("--no_implicit_edge",  action="store_true")
    p.add_argument("--no_edge_weights",   action="store_true")
    p.add_argument("--no_cross_modal",    action="store_true")
    p.add_argument("--no_cbfc",           action="store_true")
    p.add_argument("--no_dual_cl",        action="store_true")
    p.add_argument("--no_class_balanced", action="store_true")
    args = p.parse_args()

    cfg = HyFINConfig(
        dataset                = "iemocap",
        data_root              = args.data_root,
        epochs                 = args.epochs,
        batch_size             = args.batch_size,
        lr                     = args.lr,
        d_h                    = args.d_h,
        dropout                = args.dropout,
        n_hyp                  = args.n_hyp,
        k_freq                 = args.k_freq,
        n_inc                  = args.n_inc,
        mu                     = args.mu,
        lam                    = args.lam,
        seed                   = args.seed,
        device                 = args.device,
        # IEMOCAP-best windows from §6
        igm_windows            = [(10, 9), (5, 3), (3, 2)],
        use_igm                = not args.no_igm,
        use_hm                 = not args.no_hm,
        use_mfm                = not args.no_mfm,
        use_implicit_edge      = not args.no_implicit_edge,
        use_edge_weights       = not args.no_edge_weights,
        use_cross_modal_attn   = not args.no_cross_modal,
        use_cbfc               = not args.no_cbfc,
        use_dual_cl            = not args.no_dual_cl,
        use_class_balanced     = not args.no_class_balanced,
    )
    main(cfg)
