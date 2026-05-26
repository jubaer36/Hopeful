# Extracted Feature Shapes

## IEMOCAP — 10,039 utterances (no splits)

| File | Key | Shape | Dim | Model / Extractor |
|---|---|---|---|---|
| `audio_facebook_hubert_large_ll60k.pt` | `utt_id` | `(1024,)` | 1024 | HuBERT-large-ll60k, masked mean-pool last hidden state |
| `audio_opensmile_IS10.pt` | `utt_id` | `(1582,)` | 1582 | openSMILE IS10 functionals |
| `text_roberta_large.pt` | `utt_id` | `(1024,)` | 1024 | RoBERTa-large, CLS token |
| `video_clip_temporal.pt` | `utt_id` | `(3, 768)` | 3×768 | CLIP ViT-L/14, 3 temporal segments |
| `video_siglip2_temporal.pt` | `utt_id` | `(3, 1152)` | 3×1152 | SigLIP2 so400m-patch14-384, 3 temporal segments |
| `video_openface_au.pt` | `utt_id` | `(3, 8)` | 3×8 | OpenFace 3.0 MTL, AU intensities, 3 temporal segments ⚠️ |

---

## MELD — 13,708 utterances (train 9989 / dev 1109 / test 2610)

| File pattern | Key | Shape | Dim | Model / Extractor |
|---|---|---|---|---|
| `audio_facebook_hubert_large_ll60k_{split}.pt` | `clip_name` | `(1024,)` | 1024 | HuBERT-large-ll60k, masked mean-pool (chunks averaged for clips >30s) |
| `audio_opensmile_IS10_{split}.pt` | `clip_name` | `(1582,)` | 1582 | openSMILE IS10 functionals ⚠️ |
| `text_roberta_large_{split}.pt` | `clip_name` | `(1024,)` | 1024 | RoBERTa-large, CLS token |
| `video_clip_temporal_{split}.pt` | `clip_name` | `(3, 768)` | 3×768 | CLIP ViT-L/14, 3 temporal segments |
| `video_siglip2_temporal_{split}.pt` | `clip_name` | `(3, 1152)` | 3×1152 | SigLIP2 so400m-patch14-384, 3 temporal segments |
| `video_openface_au_{split}.pt` | `clip_name` | `(3, 8)` | 3×8 | OpenFace 3.0 MTL, AU intensities, 3 temporal segments ⚠️ |

⚠️ `audio_opensmile_IS10_train.pt` and `audio_opensmile_IS10_dev.pt` contain **NaN** values (short clips trigger openSMILE "Segment too short" warning). Test split is clean.

⚠️ `video_openface_au_{split}.pt` — the 8 AU indices are **unnamed**. The MTL backbone ([`GuillaumeRochette/openface`](https://github.com/GuillaumeRochette/openface) v0.0.0, EfficientNet-B0 + GNN head) does not document which AUs map to which output index. Identity depends on training dataset labels (likely DISFA: AU1/2/4/6/9/12/25/26 or BP4D subset), but this is unconfirmed. Do **not** assume index correspondence with OpenFace 2.0's named 18-AU outputs.

---

## Temporal Video Segments — How `(3, D)` is Constructed

Frames sampled at 2 fps (max 60). RetinaFace crops face per frame. Valid crops split into 3 segments:

| Segment | Frames used |
|---|---|
| Beginning | first 3 crops |
| Middle | 3 crops centred on midpoint |
| End | last 3 crops |

Each segment → mean-pooled → 1 vector of dim D. Zero vector stored if no face detected in any frame.

---

## Configurable Alternatives

### Audio SSL
| Model | Dim |
|---|---|
| `microsoft/wavlm-large` | 1024 |
| `facebook/hubert-large-ll60k` | 1024 |
| `facebook/wav2vec2-large-960h` | 1024 |
| `facebook/hubert-base-ls960` | 768 |

### Audio openSMILE
| Feature set | Dim |
|---|---|
| `eGeMAPSv02` | 88 |
| `GeMAPSv01b` | 62 |
| `IS09` | 384 |
| `IS10` | 1582 |
| `ComParE_2016` / `IS13` | 6373 |

### Text
| Model | Dim |
|---|---|
| `roberta-large` | 1024 |
| `bert-large-uncased` | 1024 |
| `microsoft/deberta-v3-large` | 1024 |
| `bert-base-uncased` | 768 |

---

## Notes

- All `.pt` files: `torch.load(path)` → `dict[str, np.ndarray]`
- Video features are `(3, D)` — temporal; audio and text are flat `(D,)` — needs flatten or temporal attention for fusion
- openSMILE IS10 features are **unnormalized** (mean≈28, std≈230, L2≈5000); audio SSL mean≈0.001, std≈0.1, L2≈2–4 — normalize before fusion
- MELD permanently missing: `dia125_utt3` (train audio), `dia110_utt7` (dev audio/video)
- OpenFace 2.0 fallback notebooks exist (`extract_video_au_openface2_*.ipynb`) — output shape `(18,)` flat, not yet run; AUs explicitly named: AU01/02/04/05/06/07/09/10/12/14/15/17/20/23/25/26/28/45 (`_r` intensity)
- OpenFace 3.0 AU indices are **unnamed** — do not assume index correspondence with OF2 outputs (see ⚠️ note above)
