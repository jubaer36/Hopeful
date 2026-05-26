from dataclasses import dataclass, field
from typing import List


@dataclass
class MERCConfig:
    dataset: str = "iemocap"
    data_root: str = "Dataset/Processed"

    d: int = 128
    d_spk: int = 64
    window_size: int = 6
    k_poly: int = 2
    k_freq: int = 4
    dropout: float = 0.3

    text_dim: int = 1024
    audio_dim: int = 1024
    siglip2_dim: int = 1152
    au_dim: int = 8

    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    epochs: int = 60
    batch_size: int = 8
    seed: int = 42

    focal_start: int = 5
    focal_end: int = 10
    focal_gamma: float = 2.0

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

    meld_rare_threshold: int = 5

    device: str = "cuda"

    @property
    def num_classes(self) -> int:
        return 6 if self.dataset == "iemocap" else 7
