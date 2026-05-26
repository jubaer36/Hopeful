# Architecture Plan: Multimodal Emotion Recognition in Conversation (MERC)

---

## 1. Overview

This document describes a complete, fully-specified architecture for Multimodal Emotion Recognition in Conversation (MERC). The architecture is a classification-based system operating on text, audio, and visual modalities, targeting both IEMOCAP and MELD benchmarks as separate models. The design is built around two parallel graph reasoning paths — a per-modality dynamic hypergraph and an M3Net-style multi-frequency spectral filter — fused via modality-specific gating, with a particular emphasis on visual underutilization correction and overfitting control at d=128.

---

## 2. Design Principles

- **Classification, not generation.** The model is a discriminative classifier. No T5 decoder, no text generation, no auxiliary reconstruction losses.
- **Dynamic graph structure.** The hypergraph incidence matrix is learned via attention over node features, not fixed by rule. Sparse by design via top-τ sparsemax.
- **Visual underutilization addressed.** Visual features receive a dedicated dual-stream encoder (SigLIP2 + AU group encoding), modality-separate graph nodes, and asymmetric cross-modal attention that explicitly elevates visual toward text-level representation richness.
- **Overfitting controlled.** d=128, pretrained encoders permanently frozen, single loss term, conservative MLP sizing, aggressive regularisation throughout.
- **Two separate models.** IEMOCAP and MELD are trained and evaluated independently. No shared parameters, no cross-dataset label conflict, no joint batch construction.

---

## 3. Feature Specification

### 3.1 Input Features (Per Utterance)

| Modality | Source | Raw Shape | Notes |
|---|---|---|---|
| Text | RoBERTa-large, CLS token | (1024,) | Frozen throughout training |
| Audio | WavLM-Large, masked mean-pool last hidden state | (1024,) | Frozen throughout training |
| Visual holistic | SigLIP2 so400m-patch14-384, 3 temporal segments | (3, 1152) | Frozen throughout training |
| Visual facial | OpenFace 3.0 MTL, AU intensities, 3 temporal segments | (3, 8) | Frozen throughout training |

### 3.2 Confirmed AU Identity (OpenFace 3.0 MTL — DISFA Subset)

| Index | AU | Description | Emotion Relevance |
|---|---|---|---|
| 0 | AU1 | Inner Brow Raiser | Fear, sadness, surprise |
| 1 | AU2 | Outer Brow Raiser | Surprise, fear |
| 2 | AU4 | Brow Lowerer | Anger, sadness, concentration |
| 3 | AU6 | Cheek Raiser | Genuine happiness (Duchenne marker) |
| 4 | AU9 | Nose Wrinkler | Disgust |
| 5 | AU12 | Lip Corner Puller | Happiness, smiling |
| 6 | AU25 | Lips Part | Various |
| 7 | AU26 | Jaw Drop | Surprise, fear |

**FACS-informed groupings used in encoding:**
- Brow group: AU1, AU2, AU4 (indices 0, 1, 2)
- Cheek/Nose group: AU6, AU9 (indices 3, 4)
- Mouth group: AU12, AU25, AU26 (indices 5, 6, 7)

### 3.3 Temporal Segment Construction

Frames sampled at 2 fps (max 60). RetinaFace crops face per frame. Valid crops split into 3 segments:

| Segment | Frames Used |
|---|---|
| Beginning (s=0) | First 3 crops |
| Middle (s=1) | 3 crops centred on midpoint |
| End (s=2) | Last 3 crops |

Each segment → mean-pooled → 1 vector. Zero vector stored if no face detected.

### 3.4 Dropped Features and Reasons

| Feature | Reason Dropped |
|---|---|
| OpenSMILE IS10 (1582,) | NaN values in MELD train and dev splits; extreme dimensionality |
| eGeMAPS v02 (88,) | Dropped in favour of WavLM-Large only audio; NaN presence unconfirmed |
| CLIP ViT-L/14 (3, 768) | Superseded by SigLIP2 (stronger sigmoid contrastive objective, better fine-grained semantics) |
| HuBERT-Large (1024,) | Superseded by WavLM-Large (stronger on paralinguistic/emotion tasks) |
| OpenFace 2.0 (18,) flat | Superseded by OpenFace 3.0 MTL with temporal segmentation |

### 3.5 Dataset Notes

**IEMOCAP:**
- 10,039 utterances total (no official splits — use standard 5-fold or last-2-session test convention)
- 6 emotion classes: happiness, sadness, neutrality, anger, excitement, frustration
- Dyadic (2 speakers per conversation)

**MELD:**
- 13,708 utterances (train: 9,989 / dev: 1,109 / test: 2,610)
- 7 emotion classes: joy, sadness, neutral, anger, surprise, fear, disgust
- Multi-party (up to 7 speakers per conversation)
- Permanently missing: `dia125_utt3` (train audio), `dia110_utt7` (dev audio/video) — **exclude from training**
- Frequent multi-face ambiguity in visual features (TV show, group shots)

---

## 4. Global Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Hidden dimension d | 128 | Conservative — ~1.33M non-pretrained params vs 7,500 IEMOCAP utterances |
| Speaker embedding dim d_spk | 64 | Sufficient for speaker identity encoding without dominating modality representations |
| Window size w (Path B) | 6 | Covers ~80% of typical IEMOCAP/MELD conversation segment lengths |
| Polynomial degree K_poly | 2 | Second-order Chebyshev; sufficient for low/high frequency separation |
| Number of frequency filters K_f | 4 | Low and high frequency variants per modality |
| Dropout rate | 0.3 | Applied at Stage 5 fusion and before classifier |
| Weight decay | 1e-4 | Applied to all trainable parameters |
| Gradient clip norm | 1.0 | Global max norm |
| Bias initialisation (cross-modal gates) | -3.0 | Gates start near-zero; model learns cross-modal dependence from data |

---

## 5. Architecture

### 5.1 Stage 1 — Unimodal Encoding

#### 5.1.1 Text Encoding

```
Input:  RoBERTa-large CLS token  (1024,)  [frozen]

T_raw = LayerNorm(roberta_cls)
T_i   = Linear(1024 → 128)(T_raw)        → (128,)
```

RoBERTa is frozen throughout. LayerNorm applied before projection to normalise CLS token scale. One vector per utterance.

#### 5.1.2 Audio Encoding

```
Input:  WavLM-Large masked mean-pool  (1024,)  [frozen]

A_raw = LayerNorm(wavlm_output)
A_i   = Linear(1024 → 128)(A_raw)        → (128,)
```

WavLM-Large is frozen throughout. Single-stream encoding — no handcrafted feature fusion. Utterance-level representation already pooled upstream.

#### 5.1.3 Visual Encoding

**Step 1 — Missing face handling (per temporal segment):**

```
For each segment s in {0, 1, 2}:
    if siglip2[s].abs().sum() == 0.0:
        siglip2[s] = mask_token_siglip    # learnable (1152,), init: zeros
    if openface[s].abs().sum() == 0.0:
        openface[s] = mask_token_au       # learnable (8,), init: zeros
```

Strict zero-equality detection per segment independently. Partial detection (face present in some segments, absent in others) is preserved.

**Step 2 — SigLIP2 stream (per segment, shared projection weights across segments):**

```
For each segment s in {0, 1, 2}:
    x_norm    = LayerNorm(siglip2[s])              # (1152,)
    x_pos     = x_norm + pos_emb_sig[s]            # add learned pos emb AFTER LN
    v_sig[s]  = Linear(1152 → 128)(x_pos)          → (128,)
```

`pos_emb_sig`: learnable embeddings of shape (3, 1152), one per temporal position. Added after LayerNorm to avoid normalisation washing out positional signal.

**Step 3 — AU stream — FACS group-aware encoding (per segment, shared weights across segments):**

```
For each segment s in {0, 1, 2}:
    x_norm    = LayerNorm(openface[s])             # (8,)
    x_pos     = x_norm + pos_emb_au[s]            # add learned pos emb AFTER LN

    brow      = Linear(3 → 32)(x_pos[[0,1,2]])    # AU1, AU2, AU4
    cheek     = Linear(2 → 16)(x_pos[[3,4]])       # AU6, AU9
    mouth     = Linear(3 → 32)(x_pos[[5,6,7]])     # AU12, AU25, AU26

    au_grp    = LayerNorm(cat([brow, cheek, mouth]))  → (80,)
    v_au[s]   = Linear(80 → 128)(au_grp)           → (128,)
```

`pos_emb_au`: learnable embeddings of shape (3, 8), separate from SigLIP2 positional embeddings.

Group-aware encoding provides structural prior from FACS anatomy before learning — brow, cheek/nose, and mouth muscles are encoded in their natural anatomical groups before projection to d.

**Step 4 — Per-timestep gated fusion of SigLIP2 and AU:**

```
For each segment s in {0, 1, 2}:
    gate[s]     = sigmoid(Linear(256 → 128)([v_sig[s] || v_au[s]]))
    v_fused[s]  = gate[s] ⊙ v_au[s] + (1 - gate[s]) ⊙ v_sig[s]    → (128,)
```

Gate modulates AU contribution relative to SigLIP2 at each temporal position independently. Learns to suppress AU signal when face detection is unreliable (mask token scenario) and to amplify it when AU activations are strongly discriminative.

**Step 5 — Temporal aggregation across 3 segments:**

```
w_temp = softmax(w_learnable)          # 3 learnable scalars, init: uniform
V_i    = Σ_{s=0}^{2} w_temp[s] * v_fused[s]              → (128,)
```

3 learnable scalars passed through softmax. Mathematically equivalent to attention over 3 tokens but without Q/K/V projection overhead. Model learns which temporal segment is most emotionally informative at the dataset level.

---

### 5.2 Stage 2 — Speaker Embedding Injection and Positional Encoding

#### 5.2.1 Speaker Embeddings

```
spk_emb_i = SpeakerEmbedding(speaker_id_i)    → (64,)
```

**IEMOCAP initialisation**: Mean of raw RoBERTa CLS vectors per speaker, passed through a fixed random projection to (64,). Computed before training begins.

**MELD initialisation**: Zero initialisation for all speakers.

**MELD rare speaker handling**: Speakers with fewer than 5 utterances in the training set are assigned a shared learnable "background speaker" embedding rather than individual embeddings. Prevents undertrained embeddings from injecting noise for rare characters.

#### 5.2.2 Speaker Injection (Concatenation, Not Addition)

```
T_i = Linear(192 → 128)([T_i || spk_emb_i])
A_i = Linear(192 → 128)([A_i || spk_emb_i])
V_i = Linear(192 → 128)([V_i || spk_emb_i])
```

Concatenation before projection — not addition. Avoids subspace alignment assumption. Allows the projection to learn how to integrate speaker identity with modality content rather than assuming additive compatibility.

#### 5.2.3 Utterance Position Encoding

```
pos_i    = sinusoidal_encoding(position=i, d=128)
T_i     += pos_i
A_i     += pos_i
V_i     += pos_i
```

Fixed sinusoidal encoding indexed by utterance position within the conversation (0, 1, ..., N-1). Same encoding applied to all three modalities — positional offset is conversation-structural, not modality-specific. Sinusoidal (not learned) to handle variable conversation lengths without extrapolation risk.

Applied once, after speaker injection, before graph construction. Both Path A and Path B receive position-aware nodes.

**Output of Stage 2**: 3N nodes, each (128,), encoding modality content + speaker identity + utterance position.

---

### 5.3 Stage 3 — Parallel Graph Paths

Two paths operate simultaneously on the 3N node set. They are structurally independent: different graph topologies, different operations, different gradient flows.

**Node set notation:**
- Text nodes: {T₁, ..., Tₙ}, each (128,)
- Audio nodes: {A₁, ..., Aₙ}, each (128,)
- Visual nodes: {V₁, ..., Vₙ}, each (128,)

#### 5.3.1 Batch Processing Strategy

PyTorch Geometric-style graph batching is used throughout. Multiple conversation graphs within a mini-batch are treated as disconnected components of one large graph. Node indices are offset per conversation to prevent cross-conversation edges. Batch vectors track which node belongs to which conversation for aggregation operations.

Padding is not used. Conversations of different lengths are handled natively via block-diagonal graph structure. For hypergraphs, incidence matrices are block-diagonal in the batched representation — conversation i's hyperedges connect only conversation i's nodes.

---

#### Path A — Per-Modality Dynamic Hypergraph

Three independent hypergraphs, one per modality (T, A, V). Each operates only on its N modality-specific nodes.

**A.1 Adaptive Hyperparameters**

```
K_c  = max(2, N//4)                  # contextual hyperedge count
τ    = max(2, N//(K_c + 1))          # max nodes per hyperedge
τ    = min(τ, int(0.4 * N))          # hard cap: no hyperedge contains >40% of nodes
```

Examples:
| N | K_c | τ |
|---|---|---|
| 3 | 2 | 2 |
| 8 | 2 | 2 |
| 20 | 5 | 3 |
| 40 | 10 | 3 |
| 100 | 25 | 3 |

**A.2 Incidence Matrix Construction**

For modality m with N nodes:

```
# Contextual hyperedges (learned, K_c prototypes)
S_m     = (W_q · nodes_m) @ (W_k · prototypes_m).T / sqrt(128)   → (N, K_c)
H_ctx_m = top_τ_sparsemax(S_m)                                    → (N, K_c)

# Speaker hyperedges (fixed, not learned)
H_spk_m[i, s] = 1  if utterance i spoken by speaker s
               = 0  otherwise                                      → (N, num_speakers)

# Combined incidence matrix
H_m = cat([H_ctx_m, H_spk_m], dim=1)                             → (N, K_c + num_speakers)
```

`W_q`, `W_k`: linear projections (128→128), separate per modality.
`prototypes_m`: K_c learnable vectors of shape (128,), one per modality. Global across conversations; conversation-specific variation comes from node features.

Top-τ sparsemax: for each hyperedge column, keep the τ highest-scoring nodes, zero the rest, normalise. Produces genuinely sparse incidence matrix — directly prevents the over-smoothing that dense incidence causes.

**A.3 Hypergraph Convolution (2 Layers with Residual)**

```
# Layer 1
D_v   = diag(H_m @ ones(K_c + num_spk))          # node degree matrix   (N, N)
D_e   = diag(H_m.T @ ones(N))                     # hyperedge degree     (K+S, K+S)
prop  = D_v^{-1} @ H_m @ W_e @ D_e^{-1} @ H_m.T @ nodes_m
out_1 = LayerNorm(prop + nodes_m)                 # residual connection

# Layer 2
prop  = D_v^{-1} @ H_m @ W_e @ D_e^{-1} @ H_m.T @ out_1
out_2 = LayerNorm(prop + out_1)                   # residual connection
```

`W_e`: learnable diagonal hyperedge weight matrix (K_c + num_speakers, K_c + num_speakers), separate per modality. Residual connections after both layers slow over-smoothing. Sparsity in H_m is the primary anti-over-smoothing mechanism; residuals are secondary.

**A.4 Per-Utterance Gated Cross-Modal Attention**

Applied after hypergraph convolution. All cross-modal gates use linear layers with bias initialised to -3.0, so sigmoid outputs start near zero — model starts nearly non-attending and learns cross-modal dependencies from data.

```
# Visual → Text
gate_VT_i  = sigmoid(Linear(256 → 128, bias_init=-3.0)([V_i || T_i]))
V_i'      += gate_VT_i ⊙ CrossAttn(Q=V_i,  K=T_i, V=T_i)

# Visual → Audio
gate_VA_i  = sigmoid(Linear(256 → 128, bias_init=-3.0)([V_i' || A_i]))
V_i''     += gate_VA_i ⊙ CrossAttn(Q=V_i', K=A_i, V=A_i)

# Audio → Text
gate_AT_i  = sigmoid(Linear(256 → 128, bias_init=-3.0)([A_i || T_i]))
A_i'      += gate_AT_i ⊙ CrossAttn(Q=A_i,  K=T_i, V=T_i)

# Text → Audio (starts near-zero, can learn)
gate_TA_i  = sigmoid(Linear(256 → 128, bias_init=-3.0)([T_i || A_i]))
T_i'      += gate_TA_i ⊙ CrossAttn(Q=T_i,  K=A_i, V=A_i)

# Text → Visual (starts near-zero, can learn)
gate_TV_i  = sigmoid(Linear(256 → 128, bias_init=-3.0)([T_i' || V_i]))
T_i''     += gate_TV_i ⊙ CrossAttn(Q=T_i', K=V_i, V=V_i)
```

CrossAttn: single-head cross-attention (d=128). Per-utterance gate is computed from both modality representations — it is utterance-specific, not a global scalar.

Visual attends to text and audio unconditionally. Text attends to audio and visual through gates that start near-zero but can learn bidirectional cross-modal signal if data supports it.

**Path A output**: enriched node representations {T''_1,...,T''_N}, {A'_1,...,A'_N}, {V''_1,...,V''_N}, each (128,).

---

#### Path B — M3Net Multi-Frequency Spectral Filtering

Operates on a separate, independent graph. Input is stop-gradient of Stage 2 outputs — Path B receives no gradients from Path A.

**B.1 Stop-Gradient and Input**

```
nodes_B = stop_gradient(stage2_nodes + sinusoidal_pos_enc)
```

Full stop-gradient on Path B's input. Stage 2 parameters are optimised only via Path A's gradient signal. This keeps the two paths independent.

**B.2 Per-Modality Undirected Window Graph Construction**

For each modality m independently:

```
# Dedicated edge projection (prevents circularity)
edge_feats_m = Linear(128 → 64)(nodes_B_m)

# Sliding window edges with cosine similarity weights
W_ij = cosine_similarity(edge_feats_m[i], edge_feats_m[j])  if |i - j| ≤ 6
     = 0                                                      otherwise

W    = (W + W.T) / 2                         # symmetrise → undirected
D    = diag(W @ ones(N))                     # degree matrix
L    = D - W                                 # graph Laplacian
L_norm = 2 * L / λ_max - I                  # rescale eigenvalues to [-1, 1]
```

`λ_max`: largest eigenvalue of L, computed per graph. Window size w=6. Edge weights computed from a dedicated 128→64 projection, not the same features being filtered — eliminates the circularity where edge weights and filtered features share the same feature space.

**B.3 Chebyshev Polynomial Frequency Filtering (Degree 2)**

```
# Chebyshev basis (degree 2)
T0 = nodes_B_m                              # (N, 128)
T1 = L_norm @ nodes_B_m                     # (N, 128)
T2 = 2 * L_norm @ T1 - T0                   # (N, 128)

# K_f=4 learnable filters
For k in {0, 1, 2, 3}:
    filtered_k = θ[k,0]*T0 + θ[k,1]*T1 + θ[k,2]*T2    → (N, 128)

# Adaptive frequency integration (per utterance)
freq_weights   = softmax(Linear(128 → 4)(nodes_B_m))    → (N, 4)
X_freq_m       = Σ_k freq_weights[:, k:k+1] * filtered_k → (N, 128)
```

`θ[k, j]`: learnable scalar per filter k, per polynomial degree j. 4 filters × 3 degree coefficients = 12 scalars total per modality.

Adaptive integration: each utterance learns its own weighting over the 4 filters, allowing the model to prefer low-frequency (emotional commonality, sustained states) or high-frequency (emotional contrast, abrupt shifts) per utterance rather than applying a fixed spectral profile.

Note: high-frequency signal on this temporal window graph captures **temporal contrast** (utterances that differ from their temporal neighbours) — this is related to but distinct from the multivariate modal discrepancy that M3Net's original setting demonstrated. Both are emotionally meaningful.

Applied independently for T, A, V modality graphs.

**Path B output**: frequency-filtered node representations {X_freq_T}, {X_freq_A}, {X_freq_V}, each (N, 128).

---

### 5.4 Stage 4 — Modality-Specific Gated Path Fusion

Scale information between Path A and Path B is preserved during gate computation. LayerNorm applied after fusion, not before.

```
# Text nodes
gate_T_i  = sigmoid(W_g_T · [h_A_T_i || h_B_T_i])           → (128,)
fused_T_i = LayerNorm(gate_T_i ⊙ h_A_T_i + (1 - gate_T_i) ⊙ h_B_T_i)

# Audio nodes
gate_A_i  = sigmoid(W_g_A · [h_A_A_i || h_B_A_i])           → (128,)
fused_A_i = LayerNorm(gate_A_i ⊙ h_A_A_i + (1 - gate_A_i) ⊙ h_B_A_i)

# Visual nodes
gate_V_i  = sigmoid(W_g_V · [h_A_V_i || h_B_V_i])           → (128,)
fused_V_i = LayerNorm(gate_V_i ⊙ h_A_V_i + (1 - gate_V_i) ⊙ h_B_V_i)
```

`W_g_T`, `W_g_A`, `W_g_V`: separate (256→128) linear layers per modality. Modality-specific gates allow the model to learn that visual should weight Path A (cross-modal hypergraph grounding) differently from how audio weights Path B (prosodic frequency dynamics).

---

### 5.5 Stage 5 — Utterance-Level Modality Fusion

```
concat_i = cat([fused_T_i, fused_A_i, fused_V_i])    → (384,)
h_i      = GELU(Linear(384 → 128)(LayerNorm(concat_i)))
u_i      = Linear(128 → 128)(Dropout(h_i, p=0.3))    → (128,)
```

Two-layer MLP with intermediate dimension d (not 2d — corrected from earlier version). Captures non-linear cross-modal interactions that a single linear projection cannot model. Dropout before the second projection at the highest-risk overfitting point.

---

### 5.6 Stage 6 — Classifier

```
logits_i = Linear(128 → num_classes)(Dropout(LayerNorm(u_i), p=0.3))

# IEMOCAP: num_classes = 6
# MELD:    num_classes = 7
```

---

## 6. Loss Function — Annealed Class-Balanced Focal Cross-Entropy

### 6.1 Formulation

```
L = -(1/N) Σ_i Σ_c y_{ic} · (1 - p_{ic})^γ · w_c · log(p_{ic})
```

- `w_c`: inverse-frequency class weight for class c, computed from training set per dataset independently
- `γ`: focal modulation parameter
- `y_{ic}`: one-hot ground truth
- `p_{ic}`: predicted probability

### 6.2 Curriculum Annealing Schedule

Hard switching between loss regimes causes gradient instability due to Adam momentum buffer mismatch. Both γ and w_c are annealed gradually:

| Epoch Range | γ | w_c |
|---|---|---|
| 0–4 | 0 (standard CE) | 1.0 (uniform) |
| 5–9 | Linear: 0 → 2 | Linear: 1.0 → w_target |
| 10+ | 2 (full focal) | w_target (inverse-frequency) |

`w_target` is computed once from training set class frequencies before training begins. Annealing gives the optimizer time to adapt to the changing gradient scale as focal weighting and class balancing are introduced.

### 6.3 Per-Dataset Class Weights

Computed independently for IEMOCAP and MELD. IEMOCAP's less severe imbalance results in less aggressive weighting than MELD's highly skewed distribution (fear and disgust appear rarely in MELD).

---

## 7. Regularisation — Complete Specification

| Mechanism | Location | Value / Detail |
|---|---|---|
| Pretrained encoder freeze | RoBERTa, WavLM-Large, SigLIP2 | Permanent — never unfrozen |
| Dropout | After MLP h_i (Stage 5) and before classifier (Stage 6) | p=0.3 |
| LayerNorm | After every linear projection; after every gated fusion | Throughout |
| Weight decay | All trainable parameters | 1e-4 |
| Gradient clipping | Global | max norm 1.0 |
| Sparse incidence matrix | Path A top-τ sparsemax | τ = max(2, min(int(0.4N), N//(K_c+1))) |
| Residual connections | Hypergraph conv layers 1 and 2 | Both layers |
| Stop-gradient | Path B input | Full stop — paths are gradient-independent |
| Chebyshev approximation | Path B polynomial filters | Degree 2 recurrence — no explicit L^j |
| Background speaker embedding | MELD rare speakers (<5 utterances) | Shared learnable embedding |
| Sinusoidal position encoding | After Stage 2, before graph construction | Fixed, not learned — handles variable N |
| Gate bias initialisation | All cross-modal attention gates (Stage 3A) | -3.0 — starts near-zero attending |
| Loss curriculum | γ and w_c annealing | Epochs 5–10 linear ramp |
| AU group encoding | Stage 1 visual AU stream | FACS-informed group projection |
| Modality-specific gate matrices | Stage 4 gated path fusion | W_g_T, W_g_A, W_g_V separate |

---

## 8. Parameter Budget

All estimates at d=128.

| Component | Parameters |
|---|---|
| Text projection (1024→128) | 131,072 |
| Audio projection (1024→128) | 131,072 |
| SigLIP2 projection (1152→128, shared across segments) | 147,456 |
| SigLIP2 positional embeddings (3×1152) | 3,456 |
| AU group projections (3→32, 2→16, 3→32) | 208 |
| AU combined projection (80→128, shared across segments) | 10,240 |
| AU positional embeddings (3×8) | 24 |
| Visual gate per timestep (256→128) × 3 | 98,304 |
| Temporal weighted sum | 3 |
| Speaker embeddings — IEMOCAP (10 × 64) | 640 |
| Speaker embeddings — MELD (~260 × 64) | 16,640 |
| Speaker projections ×3 (192→128) | 73,728 |
| Path A: prototype vectors ×3 modalities | ~K_c × 128 × 3 ≈ 12,288 (K_c≈32 avg) |
| Path A: W_q, W_k per modality ×3 | 98,304 |
| Path A: hyperedge weight W_e ×3 | ~12,288 |
| Path A: cross-modal gate linears ×5 | 163,840 |
| Path B: edge projection ×3 (128→64) | 24,576 |
| Path B: Chebyshev filter coefficients ×3 (4 filters × 3 coeffs) | 36 |
| Path B: adaptive integration ×3 (128→4) | 1,536 |
| Stage 4: modality-specific gate matrices ×3 (256→128) | 98,304 |
| Stage 5: MLP (384→128, 128→128) | 65,536 |
| Stage 6: classifier (128→6 or 128→7) | 768 / 896 |
| **Total (IEMOCAP)** | **~1.07M** |
| **Total (MELD)** | **~1.09M** |

~1.07–1.09M non-pretrained parameters. Ratio to training utterances: ~142 params/utterance (IEMOCAP), ~109 params/utterance (MELD). Within manageable overfitting regime with the specified regularisation.

---

## 9. Architecture Diagram

```
INPUTS (per utterance i in conversation of N utterances):
  Text:    RoBERTa-large CLS      (1024,)   [frozen]
  Audio:   WavLM-Large pool       (1024,)   [frozen]
  Visual:  SigLIP2 temporal       (3,1152)  [frozen]  ← mask_tok_sig if zero
           OpenFace3 AU           (3,8)     [frozen]  ← mask_tok_au  if zero
                     │
                     ▼
    ┌────────────────────────────────────────────────┐
    │           STAGE 1: UNIMODAL ENCODING           │
    │                                                │
    │  Text:                                         │
    │    LN → Linear(1024→128)              → (128,) │
    │                                                │
    │  Audio:                                        │
    │    LN → Linear(1024→128)              → (128,) │
    │                                                │
    │  Visual (per segment s in {0,1,2}):            │
    │    SigLIP2: LN → +pos_emb → Linear   → (128,) │
    │    AU:      LN → +pos_emb →           → (128,) │
    │             [group_enc(brow,cheek,mouth)]       │
    │    Gate fusion: sigmoid([sig||au])             │
    │    → v_fused[s]                       → (128,) │
    │                                                │
    │  Temporal aggregation:                         │
    │    softmax(w_learnable) weighted sum  → (128,) │
    └────────────────────────────────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────────────┐
    │       STAGE 2: SPEAKER + POSITION              │
    │                                                │
    │  spk_emb = SpeakerEmb(speaker_id)    → (64,)  │
    │  [modality || spk_emb] → Linear(192→128)       │
    │  + sinusoidal(position=i, d=128)               │
    │                                                │
    │  → T_i, A_i, V_i each (128,)                  │
    │  3N nodes total                                │
    └────────────────────────────────────────────────┘
                     │
           ┌─────────┴──────────┐
           ▼                    ▼
    ┌─────────────┐    ┌──────────────────────────┐
    │   PATH A    │    │         PATH B            │
    │             │    │  stop_grad(Stage 2 nodes) │
    │ Per-Modality│    │                           │
    │ Dynamic     │    │  Per-modality undirected  │
    │ Hypergraph  │    │  window graph (w=6)        │
    │             │    │  Edge proj (128→64)        │
    │ K_c=max(2,  │    │  Cosine sim weighting      │
    │   N//4)     │    │  Graph Laplacian           │
    │ τ=adaptive  │    │  Chebyshev filters (K=2)   │
    │             │    │  K_f=4 frequency filters   │
    │ Contextual  │    │  Adaptive integration       │
    │ hyperedges  │    │                           │
    │ (top-τ      │    │  X_freq_T (N,128)         │
    │ sparsemax)  │    │  X_freq_A (N,128)         │
    │             │    │  X_freq_V (N,128)         │
    │ Speaker     │    └──────────────────────────┘
    │ hyperedges  │               │
    │ (fixed)     │               │
    │             │               │
    │ 2-layer     │               │
    │ HGNN conv   │               │
    │ + residuals │               │
    │             │               │
    │ Per-utterance│              │
    │ gated cross-│               │
    │ modal attn  │               │
    │ (5 directions│              │
    │ bias=-3.0)  │               │
    └─────────────┘               │
           │                      │
           └──────────┬───────────┘
                      ▼
    ┌────────────────────────────────────────────────┐
    │      STAGE 4: MODALITY-SPECIFIC GATED FUSION   │
    │                                                │
    │  gate_m = sigmoid(W_g_m · [h_A_m || h_B_m])  │
    │  fused_m = LN(gate_m⊙h_A_m+(1-gate_m)⊙h_B_m)│
    │                                                │
    │  Separate W_g_T, W_g_A, W_g_V (256→128)      │
    └────────────────────────────────────────────────┘
                      │
                      ▼
    ┌────────────────────────────────────────────────┐
    │      STAGE 5: UTTERANCE MODALITY FUSION        │
    │                                                │
    │  [fused_T || fused_A || fused_V]    → (384,)  │
    │  LN → Linear(384→128) → GELU        → (128,)  │
    │  Dropout(0.3) → Linear(128→128)     → (128,)  │
    └────────────────────────────────────────────────┘
                      │
                      ▼
    ┌────────────────────────────────────────────────┐
    │           STAGE 6: CLASSIFIER                  │
    │                                                │
    │  LN → Dropout(0.3) → Linear(128→C)            │
    │  C = 6 (IEMOCAP) / C = 7 (MELD)              │
    │                                                │
    │  Loss: Annealed CB Focal Cross-Entropy         │
    │  γ: 0 → 2  (epochs 5–10, linear)             │
    │  w_c: uniform → inverse-freq (epochs 5–10)    │
    └────────────────────────────────────────────────┘
```

---

## 10. Design Decisions and Rationale

### 10.1 Why Two Parallel Paths?

Path A (hypergraph) and Path B (frequency filtering) capture genuinely different properties of the same conversation:

| Path | Graph Type | What It Captures |
|---|---|---|
| A | Dynamic sparse hypergraph | Higher-order joint dependencies across modalities and utterances; speaker structure |
| B | Undirected window graph | Spectral domain: emotional commonality (low-freq) and temporal contrast (high-freq) |

These are orthogonal views. The gated fusion learns per-modality which path is more informative.

### 10.2 Why Not VHGAE?

VHGAE (HAUCL, ACM MM 2024) introduces an ELBO (reconstruction + KL divergence) as a training loss alongside the classification loss. On IEMOCAP (7,500 utterances) and MELD (13,700 utterances), two competing loss terms with different gradient scales cause training instability. The ELBO pushes toward graph structure reconstruction; cross-entropy pushes toward emotion discrimination. These conflict in early training.

The attention-based incidence matrix adopted here gives dynamic structure learning (hyperedge membership changes per conversation because node features differ) without any reconstruction loss. The trade-off: less conversation-specific than VHGAE's posterior, but significantly more stable to train.

### 10.3 Why Not IGM (ConxGNN's Inception Graph Module)?

IGM and the hypergraph both answer the same question: which utterances should exchange information, and how strongly? Running both simultaneously creates two competing topology-learning components on the same node set, with redundant computation and competing gradient signals. The hypergraph (Path A) is the stronger choice for multivariate relationship modelling; IGM is dropped.

### 10.4 Why Permanently Freeze Pretrained Encoders?

At d=128 with ~1.07M non-pretrained parameters, the architecture is already at the boundary of the safe overfitting regime for IEMOCAP. Fine-tuning RoBERTa-large (355M), WavLM-Large (317M), and SigLIP2 (400M+) on 7,500 utterances would be catastrophic. The pretrained representations are strong enough without task-specific fine-tuning for emotion classification when the downstream graph and fusion modules are well-designed.

### 10.5 Why AU Group Encoding?

The 8 AUs (AU1, AU2, AU4, AU6, AU9, AU12, AU25, AU26) are now confirmed to be known. Encoding them as a flat 8-vector and projecting to d treats them as anonymous. Encoding them in FACS-informed anatomical groups (brow, cheek/nose, mouth) provides the model with a structural prior that reflects how these muscles actually co-activate during emotional expression. This is a low-cost inductive bias (80 parameters in group projections) with meaningful semantic grounding.

### 10.6 Why Sinusoidal (Not Learned) Positional Encoding?

IEMOCAP conversations vary from ~5 to ~100+ utterances. MELD clips are typically shorter (2–15 utterances). A learned positional embedding indexed from 0 to max_N requires a fixed maximum conversation length and cannot extrapolate beyond it. Sinusoidal encoding is position-continuous and handles any conversation length without modification.

---

## 11. Key Differences From Prior Work

| Problem in Prior Work | Solution in This Architecture |
|---|---|
| Fixed graph topology (all methods) | Attention-learned sparse incidence matrix — dynamic per conversation |
| Visual treated as a single flat vector | Dual-stream AU + SigLIP2 with FACS group encoding and temporal gating |
| Uniform graph topology across modalities | Per-modality independent hypergraphs — text, audio, visual have separate graph structures |
| Standard GNNs assume pairwise relationships | Hyperedges encode higher-order (multivariate) dependencies naturally |
| High-frequency emotional contrast lost in GNN smoothing | M3Net Chebyshev filters on dedicated undirected graph preserve spectral signal |
| Cross-modal attention suppresses visual/audio (text dominance) | Per-utterance gated attention; text attending to others starts near-zero |
| MELD class imbalance (fear, disgust underrepresented) | Class-Balanced Focal Cross-Entropy with curriculum annealing |
| Training instability from multiple loss terms | Single classification loss — no ELBO, no contrastive, no reconstruction |
| Speaker identity injected as additive noise | Concatenation before projection; rare speakers tied to background embedding |
| Short conversations degenerate hypergraph structure | Adaptive K_c and τ as functions of N; hard 40% density cap |

---

## 12. Open Limitations

The following limitations are acknowledged and not addressed by this architecture:

- **Missing modality robustness at inference**: If audio or visual is entirely absent at test time, the architecture uses zero vectors (audio) or mask tokens (visual) but has no dedicated missing-modality imputation or uncertainty mechanism. Modality dropout training was considered and rejected to keep training simple — this is a known gap.
- **MELD multi-face ambiguity**: Multiple characters on screen simultaneously means OpenFace 3 may extract AUs from the wrong speaker. The visual gate can learn to suppress unreliable visual features but has no explicit confidence signal from the face detector.
- **Prototype vectors are global, not conversation-specific**: Contextual hyperedge prototypes are the same for all conversations. Conversation-specific prototypes would require variational machinery (VHGAE) reintroducing the ELBO. Accepted as a design trade-off.
- **Stop-gradient prevents Stage 2 from receiving Path B gradient signal**: Unimodal encoders are optimised only for Path A's representational needs. Path B exploits Stage 2 outputs but cannot influence them.
- **No LLM-based contextual grounding**: LLM-derived utterance representations (from dialogue-fine-tuned models) have shown consistent gains in recent ERC work. Not included to keep the architecture self-contained and avoid additional dependencies.
- **Emotion shift detection**: Neither path explicitly models transitions between emotional states across utterances. Both treat each utterance's emotion as independently conditioned on context.

---

## 13. References and Intellectual Lineage

| Component | Source Paper |
|---|---|
| Per-modality dynamic hypergraph | Adapted from HAUCL (Yi et al., ACM MM 2024) and Hypergraph-based Multimodal Adaptive Fusion (Supercomputing 2025) |
| Multi-frequency spectral filtering | M3Net (Chen et al., CVPR 2023) |
| Hyperedge multivariate relationship modelling | M3Net (Chen et al., CVPR 2023) |
| Modality-separate graph nodes | HRG-SSA (Ji et al., IJCAI 2025) |
| Speaker hyperedges | HAUCL (Yi et al., ACM MM 2024) |
| Class-Balanced Focal Cross-Entropy | ConxGNN (Tran Van et al., ICASSP 2025) |
| Cross-task sentiment-emotion edges (not adopted, informed design) | M3GAT (Zhang et al., ACM TOIS 2024) |
| Interaction pattern distinction (informed design) | DIB-HGCN (Chen & Shi, AAAI 2025) |
| SigLIP2 visual features | SigLIP2 (Google, 2024) |
| WavLM-Large audio features | WavLM (Microsoft, 2022) |
| OpenFace 3.0 AU features | Rochette et al., GuillaumeRochette/openface |
