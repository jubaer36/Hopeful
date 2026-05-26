from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class HyFINConfig:
    dataset: str = "iemocap"
    data_root: str = "Dataset/Processed"

    # ── Model dims ──────────────────────────────────────────────────────────
    # 256 for IEMOCAP (~5.5K train utterances); 512 for MELD (~10K)
    d_h: int = 256

    # ── IGM ─────────────────────────────────────────────────────────────────
    # List of (past_window, future_window) per branch.  Override per dataset.
    igm_windows: List = field(default_factory=lambda: [(10, 9), (5, 3), (3, 2)])
    n_inc: int = 3           # k-GNN layers per IGM branch

    # ── Hypergraph Module ────────────────────────────────────────────────────
    n_hyp: int = 2           # propagation layers

    # ── Multi-Frequency Module ───────────────────────────────────────────────
    k_freq: int = 4          # frequency layers

    # ── Dropout ─────────────────────────────────────────────────────────────
    dropout: float = 0.3

    # ── Loss weights ─────────────────────────────────────────────────────────
    beta: float = 0.999      # effective-number beta
    mu: float = 0.8          # CBFC weight
    lam: float = 0.1         # DualCL weight
    tau_cl: float = 0.5      # contrastive temperature
    gamma_cbfc: float = 1.5  # focal gamma for CBFC
    dual_cl_drop: float = 0.1
    dual_cl_warmup: int = 5  # ramp λ from 0 → lam over this many epochs

    # ── Optimiser ────────────────────────────────────────────────────────────
    lr: float = 4e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    epochs: int = 40
    batch_size: int = 16
    seed: int = 42

    # ── Feature dims ─────────────────────────────────────────────────────────
    text_dim: int = 1024
    audio_dim: int = 1024
    siglip2_dim: int = 1152
    au_dim: int = 8

    # ── IEMOCAP ──────────────────────────────────────────────────────────────
    iemocap_test_sessions: List[str] = field(
        default_factory=lambda: ["Session5"]
    )
    iemocap_trainval_sessions: List[str] = field(
        default_factory=lambda: ["Session1", "Session2", "Session3", "Session4"]
    )
    iemocap_k_folds: int = 5
    iemocap_emotions: List[str] = field(
        default_factory=lambda: ["hap", "sad", "neu", "ang", "exc", "fru"]
    )

    # ── MELD ─────────────────────────────────────────────────────────────────
    meld_rare_threshold: int = 5

    # ── Ablation flags ────────────────────────────────────────────────────────
    use_igm: bool = True              # False → single k-GNN branch, global window
    use_hm: bool = True               # False → q^τ = 0
    use_mfm: bool = True              # False → f̄^τ = 0
    use_implicit_edge: bool = True    # False → skip HRG-SSA detector
    use_edge_weights: bool = True     # False → uniform Ĥ (no γ_e)
    use_cross_modal_attn: bool = True # False → plain concat-FC
    use_cbfc: bool = True             # False → μ = 0
    use_dual_cl: bool = True          # False → λ = 0
    use_class_balanced: bool = True   # False → uniform class weights

    device: str = "cuda"

    @property
    def num_classes(self) -> int:
        return 6 if self.dataset == "iemocap" else 7

    @property
    def n_igm_branches(self) -> int:
        return len(self.igm_windows)
