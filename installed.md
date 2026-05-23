# Environment: `hopeful` — Install Log

All commands run as `conda run -n hopeful <cmd>` unless noted.

---

## Base Environment

```bash
conda create -n hopeful python=3.10 -y
```

---

## PyTorch (CUDA 12.8)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Installed: `torch 2.11.0+cu128`, `torchvision 0.26.0+cu128`, `torchaudio 2.11.0+cu128`

---

## Jupyter

```bash
pip install jupyter jupyterlab ipywidgets
```

---

## General Data / ML

```bash
pip install numpy pandas scipy scikit-image matplotlib seaborn tqdm
```

---

## Stage 1 — Text Features (RoBERTa-Large)

```bash
pip install transformers safetensors tokenizers
```

- `transformers 5.9.0` — RoBERTa-Large tokenizer + model (`roberta-large`, 1024-dim CLS)
- SigLIP 2 (`google/siglip2-so400m-patch14-384`) also loaded via `transformers` (no extra install)

---

## Stage 2 — Audio Features (WavLM-Large)

No extra install. `torchaudio` (bundled with torch) handles audio I/O and resampling.  
WavLM-Large loaded via `transformers` (already installed).

---

## Stage 3 — Visual Features

### OpenCV (cv2)

```bash
pip install opencv-python-headless
```

Installed: `opencv-python-headless 4.13.0.92`  
Note: `opencv-python 4.11.0.86` and `opencv-contrib-python 4.11.0.86` also present (likely pulled in by another dep). Headless version used in notebooks to avoid display/Qt dependencies.

### CLIP (ViT-L/14)

```bash
pip install git+https://github.com/openai/CLIP.git
```

Installed: `clip 1.0`  
Pulls `ftfy` and `regex` as dependencies (text tokenizer, not used in vision-only path here).

### SigLIP 2

No extra install — loaded via `transformers.AutoModel` / `AutoProcessor`.

### OpenFace 3.0 (primary AU extractor)

```bash
pip install openface-test
```

Installed: `openface-test 0.1.26`  
Weights downloaded automatically from HuggingFace (`nutPace/openface_weights`) to `~/.cache/openface/` on first import.  
Dependencies pulled: `timm 1.0.15` (MTL backbone = `tf_efficientnet_b0_ns`)

**Patch applied to site-packages** (workaround for hardcoded backbone path bug):  
`/mnt/Work/Environments/Ubuntu/Conda/envs/hopeful/lib/python3.10/site-packages/openface/face_detection.py`  
Line added in `_load_retinaface_model` before `RetinaFace(...)` call:
```python
self.cfg = {**self.cfg, 'pretrain': False}  # skip hardcoded ./weights/mobilenetV1X0.25_pretrain.tar
```
After patching, must delete `__pycache__` for the patch to take effect:
```bash
find /mnt/Work/Environments/Ubuntu/Conda/envs/hopeful/lib/python3.10/site-packages/openface -name "*.pyc" -delete
```

### huggingface_hub (compatibility fix)

```bash
pip install --upgrade huggingface_hub
```

Installed: `huggingface_hub 1.16.1`  
Required because `transformers 5.9.0` needs `huggingface_hub >= 1.x`; old `0.21.0` caused `ImportError: cannot import name 'is_offline_mode'`.

### torchcodec (video decoding)

```bash
pip install torchcodec
```

Installed: `torchcodec 0.13.0`  
GPU-accelerated video frame decoding. Used alongside `cv2` for frame sampling.

### ffmpeg (system + Python wrapper)

```bash
pip install ffmpeg
conda install -c conda-forge ffmpeg -y   # system binary (if not already present)
```

Installed Python wrapper: `ffmpeg 1.4`

### OpenFace 2.0 (fallback — binary, NOT pip)

Not installed as a Python package. Compile from source if needed:
```bash
sudo apt-get install cmake libopenblas-dev libopencv-dev libdlib-dev
git clone https://github.com/TadasBaltrusaitis/OpenFace.git
cd OpenFace && mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=Release .. && make -j$(nproc)
# Binary at: OpenFace/build/bin/FeatureExtraction
```
Set `OPENFACE2_BIN` in `extract_video_au_openface2_{meld,iemocap}.ipynb`.

---

## Package Versions (key packages)

| Package | Version | Purpose |
|---|---|---|
| torch | 2.11.0+cu128 | Core DL framework |
| torchaudio | 2.11.0+cu128 | Audio I/O + WavLM |
| torchvision | 0.26.0+cu128 | Image transforms |
| torchcodec | 0.13.0 | Video frame decoding |
| transformers | 5.9.0 | RoBERTa, WavLM, SigLIP2 |
| huggingface_hub | 1.16.1 | Model weight downloads |
| clip | 1.0 | CLIP ViT-L/14 |
| openface-test | 0.1.26 | OpenFace 3.0 AU extraction |
| timm | 1.0.15 | EfficientNet backbone (OpenFace 3.0 MTL) |
| opencv-python-headless | 4.13.0.92 | Frame extraction / image I/O |
| numpy | 1.26.4 | Arrays |
| pandas | 2.2.3 | CSV / label loading |
| tqdm | 4.66.2 | Progress bars |
