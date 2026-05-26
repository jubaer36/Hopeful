# Implementation Progress

## Stage 0: Data Collection (Complete)

- IEMOCAP Full Release — 5 sessions, dyadic conversations, 6 emotion classes
- MELD Raw — 3 splits (train/dev/test), 7 emotion classes, Friends TV-show clips
  - IEMOCAP: 10,039/10,039 utterances → `Dataset/Processed/IEMOCAP/labels.csv`
  - MELD: 13,707/13,707 utterances  → `Dataset/Processed/MELD/labels.csv`

---

## Stage 1: Multimodal Preprocessing (Complete)

Output location: `Dataset/Processed/`

### IEMOCAP — `preprocess_iemocap.ipynb`

Extracts per-utterance modality files from raw dataset. Handles split-screen dialog AVIs by
cropping to the speaking side using filename convention (index 5 of dialog name = gender on left).

| Modality | Count | Location | Notes |
|---|---|---|---|
| Audio | 10,039 | `Processed/IEMOCAP/audio/*.wav` | Copied from `sentences/wav/` |
| Video | 10,039 | `Processed/IEMOCAP/video/*.mp4` | Speaker-cropped, audio stripped, re-encoded libx264 |
| Text  | 10,039 | `Processed/IEMOCAP/text/*.txt`  | Parsed from `dialog/transcriptions/` (latin-1 encoding) |
| Labels | 10,039 rows | `Processed/IEMOCAP/labels.csv` | utt_id, session, dialog, speaker, emotion, V/A/D, timestamps |

**Speaker crop logic:**
- `dialog_name[5]` → which gender is on the LEFT of the split-screen frame
- `utt_id.rsplit('_',1)[-1][0]` → which gender is speaking
- Match → crop `x=0` (left); mismatch → crop `x=iw/2` (right)

**Emotion labels:** all 10 original classes kept (neu/hap/exc/sad/ang/fru/sur/dis/fea/xxx). No filtering.

### MELD — `preprocess_meld.ipynb`

| Modality | train | dev | test | Location |
|---|---|---|---|---|
| Audio | 9,988 | 1,108 | 2,610 | `Processed/MELD/audio/{split}/*.wav` (16kHz mono PCM) |
| Video | 9,988 | 1,108 | 2,610 | `Processed/MELD/video/{split}/*.mp4` (audio stripped) |
| Text  | 9,989 | 1,109 | 2,610 | `Processed/MELD/text/{split}/*.txt` |
| Labels | 13,708 rows | — | — | `Processed/MELD/labels.csv` (includes `status` column) |

**Note:** 1 train clip (`dia125_utt3.mp4`) permanently unplayable (missing moov atom) → `SKIP: unplayable`. Text file written, audio/video skipped.

---

## Stage 2: Feature Extraction (Text ✓ · Audio ✓ · Video: IEMOCAP ✓ / MELD pending)

### Text — `extract_text_iemocap.ipynb` / `extract_text_meld.ipynb`

Encoder: **RoBERTa-Large** (frozen) — CLS token from last hidden state  
Feature dim: 1024  

| Dataset | Notebook | Output | Count | Status |
|---|---|---|---|---|
| IEMOCAP | `extract_text_iemocap.ipynb` | `Processed/IEMOCAP/features/text_roberta.pt` | 10,039 | Ready to run |
| MELD train | `extract_text_meld.ipynb` | `Processed/MELD/features/text_roberta_train.pt` | 9,989 | Ready to run |
| MELD dev   | `extract_text_meld.ipynb` | `Processed/MELD/features/text_roberta_dev.pt`   | 1,109  | Ready to run |
| MELD test  | `extract_text_meld.ipynb` | `Processed/MELD/features/text_roberta_test.pt`  | 2,610  | Ready to run |

Output format: `dict {utt_id / clip_name → np.array(1024,)}` saved via `torch.save`  
Tokenization: max_length=512, truncation, padding  
Batch size: 8

### Audio — `extract_audio_iemocap.ipynb` / `extract_audio_meld.ipynb`

Encoder: **WavLM-Large** (default) or **HuBERT-Large** — configurable via `MODEL_NAME`  
Feature: masked mean-pool over last hidden state frames — excludes padding via `_get_feat_extract_output_lengths`  
Feature dim: 1024 (both WavLM-Large and HuBERT-Large)  
Sample rate: 16 kHz mono (auto-resampled if needed)  
Batch size: 8 (variable-length audio, 12 GB VRAM)  

| Dataset | Notebook | Output | Count | Status |
|---|---|---|---|---|
| IEMOCAP | `extract_audio_iemocap.ipynb` | `Processed/IEMOCAP/features/audio_{MODEL_TAG}.pt` | 10,039 | Ready to run |
| MELD train | `extract_audio_meld.ipynb` | `Processed/MELD/features/audio_{MODEL_TAG}_train.pt` | 9,988 | Ready to run |
| MELD dev   | `extract_audio_meld.ipynb` | `Processed/MELD/features/audio_{MODEL_TAG}_dev.pt`   | 1,108  | Ready to run |
| MELD test  | `extract_audio_meld.ipynb` | `Processed/MELD/features/audio_{MODEL_TAG}_test.pt`  | 2,610  | Ready to run |

Output format: `dict {utt_id / clip_name → np.array(1024,)}` saved via `torch.save`  
Missing audio files skipped automatically (logged in `skipped` list); `dia125_utt3` expected skip in MELD train.

### Video — `extract_video_meld.ipynb` / `extract_video_iemocap.ipynb`

Three feature types extracted per utterance via **unified face-crop pipeline** — RetinaFace runs once per frame; the same crop feeds all three models (no redundant detection).

**Pipeline per utterance:**
1. Sample frames at 2 fps, max 60 frames
2. Each frame saved to temp `.jpg` → RetinaFace (`vis_threshold=0.5`) → BGR face crop
3. Valid crops split into temporal segments: **beginning / middle / end** (up to 3 crops each)
4. All three models encode the same crops; mean-pool per segment → stack → `(3, dim)`

**Why temporal `(3, dim)` not flat `(dim,)`?**  
Captures expression arc (onset → apex → offset). Single mean-pool discards temporal ordering. Consistent with audio (WavLM first/mid/last) and text (CLS/token/SEP regions).

**Feature types:**

| Feature | Model | Per-segment dim | Output shape | Notes |
|---|---|---|---|---|
| CLIP | ViT-L/14 | 768 | `(3, 768)` | L2-normalised per crop before mean-pool |
| SigLIP 2 | so400m-patch14-384 | 1152 | `(3, 1152)` | Raw `pooler_output` — no normalisation (sigmoid loss) |
| OpenFace AU | OF3 MTL backbone | 8 | `(3, 8)` | 8 AU intensities per crop, mean-pool per segment |

**OpenFace 3.0 (`openface-test 0.1.26`):**  
Python MTL model (EfficientNet-B0 backbone), `au_numbers=8` hardcoded.  
Outputs 8 AU intensity scores ≈ [0, 1] via GNN head (cosine-similarity output).  
Zero vector stored for utterances with no detected face in any frame.

**Checkpointing:** saves `*_ckpt.pt` every 500 utterances → safe to stop/resume without restarting. Checkpoint files deleted on successful split completion.

| Feature | IEMOCAP output | MELD output (per split) |
|---|---|---|
| CLIP `(3, 768)` | `video_clip_temporal.pt` | `video_clip_temporal_{split}.pt` |
| SigLIP 2 `(3, 1152)` | `video_siglip2_temporal.pt` | `video_siglip2_temporal_{split}.pt` |
| OpenFace AU `(3, 8)` | `video_openface_au.pt` | `video_openface_au_{split}.pt` |

**Status:**

| Dataset | Count | Face coverage | Status |
|---|---|---|---|
| IEMOCAP | 10,039 / 10,039 | 9,960 / 10,039 (99.2%) | ✓ Complete |
| MELD train | — / 9,989 | — | Notebooks ready — extraction pending |
| MELD dev | — / 1,109 | — | Notebooks ready — extraction pending |
| MELD test | — / 2,610 | — | Notebooks ready — extraction pending |

MELD expected face coverage: ~99% (TV-show footage, group shots, profile views).

**Verification notebooks:** `inspect_visual_meld.ipynb` / `inspect_visual_iemocap.ipynb`  
Walk through every step (frame sampling → detection → crops → segmentation → CLIP → SigLIP2 → AU) and verify computed features match saved `.pt` files.

**Install (once):**
```bash
conda run -n hopeful pip install git+https://github.com/openai/CLIP.git
conda run -n hopeful pip install openface-test && conda run -n hopeful openface download
```

---

## Stage 3: Graph Construction + Model (In Progress — arch2 branch)

Two architectures implemented as Python packages under `src/` (git branches).

### Branch `arch` — MERC / arch1 (Complete, has results)

Path A: dynamic hypergraph + cross-modal gated attention  
Path B: Chebyshev spectral filtering  
Hidden dim: d=64 | Loss: AnnealedFocalLoss (anneals focal γ over epochs 5–10)  
Checkpoints: `checkpoints/iemocap_fold{0-4}.pt`, `checkpoints/meld_best.pt`  
Results: `results/`

### Branch `arch2` — HyFIN-Net (Implemented 2026-05-26, training in progress)

Plan: `Possible Plans/arch2-shahi.md`  
Hidden dim: d_h=**256** (IEMOCAP) | **3.8M params**

**Architecture (A→B→C→D→E→F):**

| Block | Module | Key |
|---|---|---|
| A | `src/model/encoder.py` | Text: 1-layer Transformer; Audio: linear+ReLU; Visual: SigLIP2+AU dual-stream; Speaker: add |
| B | `src/model/igm.py` | IGM: n-branch heterogeneous k-GNN (angular-weighted, causal implicit edges); HM: M³Net hypergraph with γ_e(v) edge-dependent node weights |
| C | `src/model/mfm.py` | Global graph, per-edge self-gated low/high-pass (FAGCN-style, K layers) |
| D | concat | `[p^τ ∥ q^τ ∥ f̄^τ]` per modality → 3·d_h |
| E | `src/model/hyfin.py` | Per-modality projection (3d_h→d_h) → text-anchored cross-modal attention → fuse |
| F | `src/model/hyfin.py` | Linear → softmax |

**Loss:** CBCE + μ·CBFC + λ·DualCL with 5-epoch λ warmup

**Hyperparams:**

| | IEMOCAP | MELD |
|---|---|---|
| d_h | 256 | 512 |
| IGM windows | `[(10,9),(5,3),(3,2)]` | `[(11,11),(7,4),(6,4)]` |
| HM layers | 2 | 4 |
| Freq layers K | 4 | 3 |
| Dropout | 0.3 | 0.4 |
| μ (CBFC) | 0.8 | 0.8 |
| λ (DualCL) | 0.1 | 0.1 |
| DualCL warmup | 5 epochs | 5 epochs |

**Entry points:**
```bash
conda run -n hopeful python train_iemocap.py   # 5-fold CV, Session-5 test
conda run -n hopeful python train_meld.py      # train/dev/test splits
```

**Ablation flags:** `--no_igm`, `--no_hm`, `--no_mfm`, `--no_implicit_edge`, `--no_edge_weights`, `--no_cross_modal`, `--no_cbfc`, `--no_dual_cl`, `--no_class_balanced`

**Feature files expected** (from Stage 2):
- IEMOCAP: `text_roberta_large.pt`, `audio_microsoft_wavlm_large.pt`, `video_siglip2_temporal.pt`, `video_openface_au.pt`
- MELD: same pattern with `_{split}.pt` suffix

**Fixes applied 2026-05-26:**
- `CrossModalFusion`: added per-modality projections 3d_h→d_h before cross-attn; removed 7M-param W_V/W_z bottleneck (12.2M → 1.8M in fusion block)
- `ImplicitEdgeDetector`: vectorised O(L²) Python loop → 3 matrix ops; ~100× speedup
- `_build_graph`: vectorised Python loops → tensor indexing
- `d_h` default: 512 → 256 for IEMOCAP (19M → 3.8M total)
- DualCL λ warmup (0 → 0.1 over 5 epochs) to stabilise early CE learning

**Param count history:**

| Config | Params |
|---|---|
| Original d_h=512 | 19.0M |
| d_h=512 + fixed fusion | 13.5M |
| **d_h=256 + fixed fusion** | **3.8M** |

**Expected performance:** IEMOCAP WF1 75.8–77.0, MELD WF1 67.3–68.5 (see plan §9)

---

## Environment

Conda env: `hopeful`
Key deps: `ffmpeg`, `pandas`, `joblib`, `tqdm`, `ipykernel`

## GPU
RTX 3060 12gb VRAM
