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

## Stage 3: Graph Construction + Model (Training Complete — IEMOCAP ✓ / MELD retraining)

Architecture: **Plan A — Temporal Heterogeneous Hypergraph** (see `Possible Plans/stage3_design.md`)

### Notebooks

| Notebook | Dataset | Status |
|---|---|---|
| `stage3_plan_a_iemocap.ipynb` | IEMOCAP | ✓ 5-fold LOSO complete |
| `stage3_plan_a_meld.ipynb` | MELD | Retraining (fixed loss settings) |
| `ablation_iemocap.ipynb` | IEMOCAP | Ablation complete — informs final config |

### Architecture summary

- **5 nodes/utterance**: text (RoBERTa-1024), audio (WavLM-1024), vis_begin/mid/end (AU⊕SigLIP2 = 1160)
- **4 hyperedge types**: Multimodal-utterance (type 0), Visual-arc / expression arc (type 1), Contextual ×5 (type 2), Speaker (type 3)
- **Propagation**: two-level node→edge / edge→node attention, alternating inter/intra-modal schedule
- **Loss**: CB-Focal (Cui et al. CVPR 2019) + λ·BCL (Zhu et al. CVPR 2022)
- **OOM guard**: `HypergraphConvLayer` chunks hedge attention over 64 nodes at a time (~110 MB peak)

### Final confirmed configs (after ablation)

| Setting | IEMOCAP | MELD | Reason |
|---|---|---|---|
| HIDDEN | 256 | 256 | sufficient capacity |
| K_LAYERS | 2 | 2 | K=4 slightly better but negligible for IEMOCAP; K=2 suitable for short MELD dialogs |
| DROPOUT | 0.3 | 0.3 | 0.5 was too aggressive → slow convergence |
| WEIGHT_DECAY | 1e-4 | 5e-4 | MELD needs stronger regularization |
| BETA_CB | 0.9999 | 0.999 | MELD: beta=0.9999 gives 13:1 neutral:fear weight ratio → kills neutral learning; 0.999 → 4:1 ratio |
| LAMBDA_BCL | 0.5 | 0.1 | MELD dialogs (median N≈10) give noisy BCL gradients; reduce influence |
| PATIENCE | 12 | 20 | MELD converges ~3x slower |
| EPOCHS | 60 | 100 | MELD needs more epochs |

### Ablation findings (IEMOCAP, 2-fold, Session1+Session5)

| Config | S1 WF1 | S5 WF1 | Mean | Gap |
|---|---|---|---|---|
| A_baseline (K=2, spk edges, BCL) | 0.5785 | 0.6485 | 0.6135 | 0.1122 |
| B_no_speaker | 0.5667 | 0.6353 | 0.6010 | 0.2066 |
| C_bcl_detach | 0.5730 | 0.6430 | 0.6080 | 0.1199 |
| D_no_spk+BCL | 0.5748 | 0.6309 | 0.6029 | 0.1465 |
| E_all changes | 0.5743 | 0.6377 | 0.6060 | 0.1513 |

Key conclusions: speaker edges help (removing increases gap 0.11→0.21); BCL detach/intra-first neutral; val loss rise is 89% BCL inflation (not classification degradation) → early stop on WF1 is correct.

### IEMOCAP Results (5-fold LOSO, final)

| Session | WF1 | UAF1 | ACC |
|---|---|---|---|
| Session1 | 0.5788 | 0.5871 | 0.5868 |
| Session2 | 0.5921 | 0.5794 | 0.5846 |
| Session3 | 0.6205 | 0.6127 | 0.6275 |
| Session4 | 0.6358 | 0.6072 | 0.6362 |
| Session5 | 0.6556 | 0.6390 | 0.6603 |
| **Mean±std** | **0.6165±0.0281** | **0.6051±0.0210** | **0.6191±0.0293** |

Beats MM-DFN baseline (WF1≈58.18). Saved to `Dataset/Processed/IEMOCAP/stage3_results/`.

### MELD Results (retrained with fixed loss settings)

| Metric | Broken run | Fixed run | MM-DFN baseline |
|---|---|---|---|
| Test WF1 | 0.3610 | **0.4631** | 0.5817 |
| Test UAF1 | 0.2710 | 0.2960 | — |
| Test ACC | 0.3460 | 0.4642 | — |
| Dev best WF1 | 0.3714 | 0.4751 | — |

Per-class F1 (fixed run): neutral=0.632, joy=0.320, surprise=0.327, anger=0.383, sadness=0.257, disgust=0.067, fear=0.087  
Root cause confirmed: neutral F1 jumped 0.40→0.63 after fixing CB-Focal beta (0.9999→0.999).

Still below MM-DFN baseline (-0.12 WF1). Fear/disgust remain near zero (267/269 train samples).  
Possible next step: increase HIDDEN=384, K_LAYERS=4 to push toward baseline.

### Key implementation details

- `build_incidence_matrix(N, speakers)` → `(5N, E)` H matrix + `edge_types (E,)`; pre-built per dialog, cached
- `PlanAModel.forward(text, audio, vis, vis_absent, H_mat, edge_types, return_feats)` — H_mat passed pre-built
- Visual absent: `AU.sum()==0 AND SigLIP2.sum()==0` → replaced with learned `visual_absent_embed (3, d)`
- Missing MELD audio: `dia125_utt3` (train), `dia110_utt7` (dev) → zero-filled 1024-dim vector

### Checkpointing / resume

- **IEMOCAP**: per-epoch `resume_fold{N}.pt` saved each epoch; deleted on clean fold completion. `fold{N}_best_model.pt` skipped if already present → full crash recovery across all 5 folds.
- **MELD**: per-epoch `resume_checkpoint.pt`; skips training entirely if `best_model.pt` exists.
- All checkpoints and results → `Dataset/Processed/{IEMOCAP,MELD}/stage3_results/`

### Evaluation protocol

| Dataset | Protocol | Classes | Utterances |
|---|---|---|---|
| IEMOCAP | 5-fold LOSO (leave-one-session-out) | 6 (ang/exc/fru/hap/neu/sad) | ~7,380 |
| MELD | fixed train/dev/test splits | 7 (neutral/joy/surprise/anger/sadness/disgust/fear) | ~13,708 |

### Targets

| Dataset | Baseline (MM-DFN) | Aim | Current ceiling |
|---|---|---|---|
| IEMOCAP WF1 | ~58.18 | ≥63–68 | ~75.47 (HRG-SSA IJCAI-25) |
| MELD WF1 | ~58.17 | ≥60–64 | ~65+ |

---

## Environment

Conda env: `hopeful`
Key deps: `ffmpeg`, `pandas`, `joblib`, `tqdm`, `ipykernel`

## GPU
RTX 3060 12gb VRAM
