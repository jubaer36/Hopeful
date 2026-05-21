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

## Stage 2: Feature Extraction (In Progress)

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
Batch size: 32

### Audio (Not Started)
Planned: **WavLM-Large , Hubert Lg**

### Video (Not Started)
Planned: **SigLIP 2 + OpenFace 3.0 Action Units**

---

## Stage 3: Graph Construction + Model (Not Started)

Target architecture: SC-VAH / PHASE (see `Possible Plans/`)

---

## Environment

Conda env: `hopeful`
Key deps: `ffmpeg`, `pandas`, `joblib`, `tqdm`, `ipykernel`
