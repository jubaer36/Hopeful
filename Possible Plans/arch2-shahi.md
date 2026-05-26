# Novel Architecture for Multimodal Emotion Recognition in Conversation (MERC)

## Proposed Name: **HyFIN-Net** — Hyper-Frequency Inception Network

---

## 1. Source Paper Recap (one-line summaries)

| Paper | Core Contribution | What we borrow |
|-------|------------------|----------------|
| **ConxGNN** (arXiv 2412.16444) | Inception Graph Module (IGM) with multi-window k-GNN branches + a parallel Hypergraph Module + cross-modal attention fusion + Class-Balanced Focal Contrastive (CBFC) loss | IGM topology, multi-window scheme, CBFC + CBCE losses, cross-modal attention fusion |
| **M³Net** (CVPR 2023) | Hypergraph for multivariate propagation + Multi-frequency propagation (low-pass + high-pass via FAGCN-style self-gating) + edge-dependent node weights | Multi-frequency propagation block, edge-dependent hyperedge weights, hypergraph construction with multimodal + contextual hyperedges |
| **HAUCL** (ACM MM 2024) | Variational Hypergraph Autoencoder (VHGAE) + dual-view hypergraph contrastive learning | We do **NOT** include VHGAE itself (see §2). We borrow only the *dual-view contrastive* idea, lightly. |
| **HRG-SSA** (IJCAI 2025) | Hybrid Relational Graph (explicit adjacency + implicit-connection detector) + Sentiment-laden Semantic Alignment + T5 generative decoder | We borrow only the **implicit-connection detector** (lightweight feed-forward + masked softmax) to enrich the multi-modal graph. We do **NOT** use the T5 decoder or sentiment-text generation pipeline. |

---

## 2. Verification — What Works vs. What Will Not Work

### ✅ Components that WILL Work (kept)

| Component | Origin | Reason it works in this combination |
|-----------|--------|-------------------------------------|
| **Inception Graph Module (IGM)** with N parallel k-GNN branches with different `[past, future]` window sizes | ConxGNN | Multi-scale context windows are orthogonal to hypergraph and frequency modules. Ablation in ConxGNN shows IGM removal drops Acc by 27.8% on IEMOCAP — the single most important component. |
| **Hypergraph Module (HM)** with multimodal + contextual hyperedges and edge-dependent node weights γ_e(v) | ConxGNN + M³Net | Hyperedges natively model higher-arity dependencies across modalities and utterances. Edge-dependent weights (from M³Net) refine fine-grained contributions. Compatible with the IGM because both operate on the same 3L node set. |
| **Multi-frequency propagation** with adaptive low-pass + high-pass self-gating (FAGCN-style) | M³Net | High-frequency signals capture emotion *discrepancy* (sarcasm, modality conflict) while low-frequency captures *commonality*. M³Net ablation showed removing it drops Acc by ~2.4%. Operates on a standard graph in parallel to the hypergraph — no conflict. |
| **Cross-modal attention** between text and {audio, visual} during fusion | ConxGNN | Text carries the strongest emotional signal; aligning a/v to t via CA improves grounding. |
| **Class-Balanced Focal Contrastive (CBFC) loss** + Class-Balanced Cross-Entropy (CBCE) | ConxGNN | IEMOCAP and MELD are class-imbalanced (MELD heavily neutral-dominated). Re-weighting is essential. |
| **Implicit-connection detector** (per-modality FFN + masked softmax) — *lightweight, used to inject extra edges into a single IGM branch* | HRG-SSA | Adds modality-adaptive implicit edges without changing graph topology aggressively. Cheap (a single FFN + softmax). Drops ~1.7% Acc when removed in HRG-SSA. |
| **Dual-view contrastive auxiliary loss** on hypergraph node embeddings produced under *two different perturbations* (feature-dropout views, not VHGAE-reconstructed) | HAUCL (light version) | Cheap regularizer that improves embedding discriminability. We avoid VHGAE's overhead. |

### ❌ Components that WILL NOT Work (rejected, with reasoning)

| Component | Origin | Why it is rejected in this design |
|-----------|--------|-----------------------------------|
| **Full VHGAE adaptive hypergraph reconstruction** | HAUCL | The IGM already provides multi-scale flexibility, and the M³Net edge-dependent weights already give fine-grained edge contribution control. Adding VHGAE means a learnable encoder–sampler–decoder per training step (Gumbel-Softmax, KL divergence, reconstruction loss). Complexity is O(N²d²) per sample, and conflicts with the *fixed* hyperedge templates that ConxGNN's HM expects. Combining VHGAE with IGM also produces an unstable optimization surface (two stochastic structural learners on the same nodes). |
| **HRG-SSA's T5 generative decoder** | HRG-SSA | Reformulates ERC as text generation. This is incompatible with the discriminative GNN+hypergraph design and would require an entirely different training regime (autoregressive cross-entropy on label tokens, T5 pretraining/fine-tuning, distinct vocabulary). Out of scope. |
| **Sentiment-laden Semantic Alignment (PSA + MSA)** *as designed* | HRG-SSA | Requires textual sentiment templates encoded with T5 and a separate "historical sentiment" encoder. Tied to text generation paradigm. We replace its motivation (label-aware contrastive) with **CBFC**, which is class-balanced and label-aware but does not need T5. |
| **Two parallel VHGAE views + Gumbel-Softmax** | HAUCL | Doubles model size, doubles forward pass cost. The HAUCL paper itself shows model size 173K vs M³Net's 608K, which is misleading because *we are already adding* the IGM (multiple GNN branches), HM, and multi-frequency block. The compute budget is better spent on the IGM. |
| **Graph Transformer head** at the end of each IGM branch (as in ConxGNN) — **MADE OPTIONAL** | ConxGNN | Adds significant params; on smaller datasets like IEMOCAP it can overfit. We keep it as an optional flag (`--use_graph_transformer`) but disable by default and let ablation decide. |
| **Stacking M³Net's GCN-based multivariate path *and* hypergraph path** | M³Net | M³Net runs hypergraph and frequency in parallel on the same node set but uses a *plain GCN-style* multivariate path. We are replacing that multivariate path with the **richer IGM (multi-window k-GNN)**. Keeping both would be redundant — IGM already covers context aggregation, with explicit multi-scale windows. |

---

## 3. High-Level Architecture (Block Diagram, ASCII)

```
                          ┌─────────────────────────────────┐
                          │   Pre-extracted Features (DONE) │
                          │   text uᵗ, audio uᵃ, visual uᵛ  │
                          └────────────────┬────────────────┘
                                           │
                          ┌────────────────▼────────────────┐
                          │ Unimodal Encoder (Block A)      │
                          │  - Transformer for text         │
                          │  - FC + ReLU for audio / visual │
                          │  - Speaker embedding (add)      │
                          │  → hᵗᵢ, hᵃᵢ, hᵛᵢ ∈ ℝ^{d_h}     │
                          └─────┬───────────────┬───────────┘
                                │               │
            ┌───────────────────┘               └───────────────────┐
            │                                                       │
   ┌────────▼─────────────────────────────┐         ┌───────────────▼──────────────┐
   │ (B) Inception Hyper-Graph Module IGM │         │ (C) Multi-Frequency Module   │
   │ ====================================  │         │ ============================ │
   │  Build n parallel branches; each      │         │  Pairwise graph G = (V, Eg)  │
   │  branch b ∈ {1..n} uses window        │         │  Edges: same modality intra- │
   │  (pᵦ, fᵦ). For each branch:           │         │  dialogue + cross-modal same │
   │   1. Build heterogeneous graph Gᵦ     │         │  utterance.                  │
   │      with edges per ConxGNN R_intra,  │         │  Filters:                    │
   │      R_inter (cosine angular weight). │         │   F_l = I + D⁻½ A D⁻½ (low)  │
   │   2. Inject implicit edges from       │         │   F_h = I - D⁻½ A D⁻½ (high) │
   │      HRG-SSA implicit detector        │         │  Self-gated combination:     │
   │      (FFN + masked softmax > 1/N).    │         │   r_ij^l - r_ij^h =          │
   │   3. Run k-GNN (Morris '19) for N_inc │         │     tanh(W₃(f_i ⊕ f_j))      │
   │      layers per branch.               │         │  Stack K layers → f̄ᵗ, f̄ᵃ, f̄ᵛ│
   │   4. Apply Graph Transformer (opt).   │         │                              │
   │  Aggregate branches by mean → Pᵗ,Pᵃ,Pᵛ│         │                              │
   │                                       │         │                              │
   │  ── HYPERGRAPH SUB-MODULE (HM) ──     │         │                              │
   │  In parallel, build hypergraph H =    │         │                              │
   │  (V_H, E_H) with 3+L hyperedges:      │         │                              │
   │   • 3 modality hyperedges (all nodes  │         │                              │
   │     of a modality across dialogue)    │         │                              │
   │   • L utterance hyperedges (3 nodes   │         │                              │
   │     of utterance i across modalities) │         │                              │
   │  Use M³Net hypergraph conv with edge- │         │                              │
   │  dependent weights ω(e), γ_e(v).      │         │                              │
   │  N_hyp iterations → qᵗ, qᵃ, qᵛ.       │         │                              │
   └─────────────────┬─────────────────────┘         └──────────────┬───────────────┘
                     │                                              │
                     │  (D) Multi-View Concatenation                │
                     └──────────────┬───────────────────────────────┘
                                    │
                          [pᵗ ∥ qᵗ ∥ f̄ᵗ] , [pᵃ ∥ qᵃ ∥ f̄ᵃ] , [pᵛ ∥ qᵛ ∥ f̄ᵛ]
                                    │
                          ┌─────────▼─────────┐
                          │ (E) Cross-Modal   │
                          │ Attention Fusion  │
                          │   CA^{v→t},       │
                          │   CA^{a→t}        │
                          │   f̂ᵗ = fᵗ + sums  │
                          │ then concat       │
                          │   zᵢ = ReLU(...)   │
                          └─────────┬─────────┘
                                    │
                          ┌─────────▼─────────┐
                          │ (F) Classifier    │
                          │   FC → softmax    │
                          │   ŷᵢ = argmax pᵢ  │
                          └─────────┬─────────┘
                                    │
                          ┌─────────▼──────────────┐
                          │ Loss:                  │
                          │   ℒ = ℒ_CBCE +         │
                          │       μ·ℒ_CBFC +       │
                          │       λ·ℒ_DualCL       │
                          └────────────────────────┘
```

---

## 4. Module-by-Module Specification

### 4.1 Unimodal Encoder (Block A)

Reuses ConxGNN's `UnimodalEncoder` design (already present in `ConxGNN/src/model/UnimodalEncoder/`).

- **Text**: 1-layer Transformer encoder over pre-extracted RoBERTa features → `xᵗᵢ ∈ ℝ^{d_h}`.
- **Audio / Visual**: linear projection (`W^τ · uᵢ^τ + b^τ`) → `xᵃᵢ, xᵛᵢ ∈ ℝ^{d_h}`.
- **Speaker embedding**: lookup `sᵢ = Embedding(speaker_id)`, mode `add` (i.e., `hᵢ^τ = sᵢ + xᵢ^τ`).
- Optional unimodal InfoNCE pretraining (ConxGNN does this for the first few epochs — keep as a flag).

**Output**: `{hᵗᵢ, hᵃᵢ, hᵛᵢ}` for `i = 1..L`.

### 4.2 Inception Hyper-Graph Module (Block B) — the heart of HyFIN-Net

This block fuses the **IGM** and **HM** ideas into a unified parallel structure.

#### 4.2.1 Graph construction (per IGM branch)
- Nodes `V_G`: 3L nodes — three per utterance, one per modality.
- Relations (ConxGNN-style):
  - `R_inter`: connects `{hᵗᵢ, hᵃᵢ, hᵛᵢ}` for the same utterance i (cross-modal, same time).
  - `R_intra^past`: connects `hτᵢ ↔ hτⱼ` when `i - pᵦ ≤ j < i`, same modality.
  - `R_intra^future`: connects `hτᵢ ↔ hτⱼ` when `i < j ≤ i + fᵦ`, same modality.
- Edge weight: angular similarity `A_ij = 1 - arccos(cos(hτᵢ, hτⱼ))/π`.
- **NEW (from HRG-SSA)**: per modality η ∈ {t, a, v}, compute implicit-connection scores
  `s_ij^η = LeakyReLU(mask + W_η^T[hᵢ^η ∥ hⱼ^η])`, then softmax over j. Add edge if `α_ij^η > 1/N`.
  Apply the upper-triangular mask exactly as in HRG-SSA to preserve temporal causality.

#### 4.2.2 Inception branches
- Define a set of `n` window pairs `P = {(p₁, f₁), ..., (pₙ, fₙ)}` — same hyper-parameter as ConxGNN.
- For IEMOCAP, recommended: `[(10, 9), (5, 3), (3, 2)]` (ConxGNN's best setting).
- For MELD, recommended: `[(11, 11), (7, 4), (6, 4)]`.
- Per branch b:
  1. Build graph `Gᵦ` with windows `(pᵦ, fᵦ)`.
  2. Inject implicit edges (one-shot at construction).
  3. Stack `N_inc = 3` layers of k-GNN (Morris et al. 2019, higher-order Weisfeiler–Lehman).
  4. Optional final Graph Transformer head (`use_graph_transformer=False` by default).
- **Aggregate across branches**: `pτᵢ = (1/n) · Σ_b o_b,iτ` → `Pτ = [pτ₁, ..., pτ_L]` for `τ ∈ {t, a, v}`.

#### 4.2.3 Hypergraph sub-module (HM)
- Nodes `V_H`: 3L (same as IGM).
- Hyperedges `E_H` (|E_H| = 3 + L):
  - 3 **modality** hyperedges: one per modality covering all L utterances.
  - L **utterance** hyperedges: one per utterance covering its 3 modality nodes.
- Edge weights: learnable scalar `ω(e)` per hyperedge.
- Edge-dependent node weights: learnable `γ_e(v)` (M³Net's incidence matrix `Ĥ`).
- Propagation (M³Net Eq. 5):
  `V^{l+1} = σ( D_H⁻¹ Ĥ W_e B⁻¹ Ĥ^T V^l )`
- `N_hyp = 4` for IEMOCAP, `3` for MELD (matches M³Net's `L=3, K=4` settings).
- Output: `qτᵢ` per modality, per utterance.

### 4.3 Multi-Frequency Module (Block C)

Standalone parallel branch over a plain heterogeneous graph (not the hypergraph).

#### 4.3.1 Graph construction
- Same `V_G` as IGM but **fully-connected within dialogue** per modality (no sliding window — frequency analysis benefits from a global view).
- Edges:
  - Intra-modality: `f_i^x ↔ f_j^x` for all `i, j` in same dialogue.
  - Cross-modality, same utterance: `f_i^x ↔ f_i^z` for `x ≠ z`.
- Build symmetric adjacency `A`; compute normalized Laplacian.

#### 4.3.2 Filters and self-gating (M³Net §3.3)
- Low-pass: `F_l = I + D⁻½ A D⁻½`
- High-pass: `F_h = I - D⁻½ A D⁻½` (Laplacian)
- Self-gating coefficient per edge:
  `r_ij^l - r_ij^h = tanh(W₃ · [f_i ⊕ f_j])`
  with the constraint `r_ij^l + r_ij^h = 1` (so they live on a 1-simplex via reparameterisation).
- Update rule (M³Net Eq. 9):
  `F^{k+1} = F^k + (R^l - R^h) D⁻½ A D⁻½ F^k`
- Stack `K = 4` layers for IEMOCAP, `K = 3` for MELD.

#### 4.3.3 Output
- `f̄τᵢ = f^τ_{i,(K)}` for `τ ∈ {t, a, v}`.

### 4.4 Multi-View Concatenation (Block D)

For each modality `τ` and utterance `i`, concatenate the three views:
```
m_i^τ = [ p_i^τ  ∥  q_i^τ  ∥  f̄_i^τ ]   ∈ ℝ^{3·d_h}
```
- `pτ` = IGM output (multi-scale local context).
- `qτ` = HM output (multivariate higher-arity).
- `f̄τ` = multi-frequency output (high-pass discrepancy + low-pass commonality).

### 4.5 Cross-Modal Attention Fusion (Block E)

Identical to ConxGNN's fusion (text-as-query):
- `CA^{v→t}(i) = softmax((W_Q m_i^t)(W_K m_i^v)^T / √d_h) · W_V m_i^v`
- `CA^{a→t}(i) = softmax((W_Q m_i^t)(W_K m_i^a)^T / √d_h) · W_V m_i^a`
- `f̂_i^t = m_i^t + CA^{v→t}(i) + CA^{a→t}(i)`
- Final: `z_i = ReLU( W_z · [f̂_i^t ∥ m_i^a ∥ m_i^v] + b_z ) ∈ ℝ^{d_z}`.

### 4.6 Classifier (Block F)

Standard MLP head:
- `p_i = softmax(W_τ z_i + b_τ)`
- `ŷ_i = argmax p_i`

---

## 5. Training Objectives

The total loss is a weighted sum of three terms:

```
ℒ = ℒ_CBCE + μ · ℒ_CBFC + λ · ℒ_DualCL
```

### 5.1 Class-Balanced Cross-Entropy (CBCE)
- ConxGNN Eq. 19. Effective-number reweighting: `w_c(i) = (1 - β)/(1 - β^{n_c})`.
- Hyperparameter: `β = 0.999`.

### 5.2 Class-Balanced Focal Contrastive (CBFC)
- ConxGNN Eq. 18. Same `w_c` term, focal modulation `(1 - t_{j,k})^γ`.
- Hyperparameter: `μ = 0.8`.

### 5.3 Dual-View Contrastive (DualCL) — light HAUCL replacement
- During training, run the **hypergraph branch twice** with two independent feature-dropouts (rate 0.1 on input features only).
- Treat the two views as positive pairs for the same node; treat different nodes (within batch) as negatives.
- Standard InfoNCE: `ℒ_DualCL = -(1/N) Σ_i log( exp(sim(v_i^{(1)}, v_i^{(2)})/τ) / Σ_j exp(sim(v_i^{(1)}, v_j^{(2)})/τ) )`.
- Hyperparameter: `λ = 0.1`, `τ = 0.5`.
- **No VHGAE, no Gumbel-Softmax, no KL term** — pure feature-view contrastive, ~10 LOC.

---

## 6. Hyperparameter Summary

| Hyperparameter | IEMOCAP | MELD |
|---------------|---------|------|
| Hidden size `d_h` | 512 | 512 |
| Number of IGM branches `n` | 3 | 3 |
| IGM windows `P` | `[(10,9),(5,3),(3,2)]` | `[(11,11),(7,4),(6,4)]` |
| IGM layers per branch `N_inc` | 3 | 3 |
| HM layers `N_hyp` | 2 | 4 |
| Multi-frequency layers `K` | 4 | 3 |
| Batch size | 16 | 16 |
| Optimizer | Adam | Adam |
| Learning rate | 4e-4 | 4e-4 |
| Epochs | 40 | 40 |
| `β` (CBCE) | 0.999 | 0.999 |
| `μ` (CBFC) | 0.8 | 0.8 |
| `λ` (DualCL) | 0.1 | 0.1 |
| `τ` (contrastive temp) | 0.5 | 0.5 |
| Dropout | 0.3 | 0.4 |

---

## 7. Implementation Plan (file/module layout)

Reuse the **ConxGNN** repository skeleton — it already has hypergraph + IGM scaffolding. Extend in-place.

```
shynet/
├── src/
│   ├── model/
│   │   ├── MainModel.py                  # orchestrate A→B→C→D→E→F (extend)
│   │   ├── UnimodalEncoder/              # KEEP from ConxGNN
│   │   ├── GraphModel/
│   │   │   ├── InceptionGraphModule.py   # NEW — wrap k-GNN branches w/ windows
│   │   │   ├── ImplicitEdgeDetector.py   # NEW — HRG-SSA detector (small FFN)
│   │   │   ├── HyperGraphModule.py       # EXTEND ConxGNN; add γ_e(v) per M³Net
│   │   │   ├── MultiFrequencyModule.py   # NEW — port from M³NET/model.py
│   │   │   └── HypergraphConv.py         # PORT from M3NET/HypergraphConv.py
│   │   ├── Crossmodal/                   # KEEP from ConxGNN
│   │   └── Classifier/                   # KEEP from ConxGNN
│   ├── loss/
│   │   ├── CBConstrastiveLoss.py         # KEEP (CBFC)
│   │   ├── DualViewInfoNCE.py            # NEW — lightweight dual-view contrastive
│   │   └── FocalLoss.py                  # KEEP
│   ├── Coach.py                          # KEEP, add λ·ℒ_DualCL term
│   ├── Config.py                         # EXTEND, add multi-freq / dualcl flags
│   └── Dataset.py                        # KEEP
├── configs/
│   ├── iemocap_hyfin.yaml                # NEW
│   └── meld_hyfin.yaml                   # NEW
├── train.py                              # KEEP (entry point)
└── ablation_study.py                     # EXTEND with new ablation flags
```

### Implementation order (suggested)
1. **Port M³Net's frequency module** into `MultiFrequencyModule.py` as a drop-in block consuming `{hᵗ, hᵃ, hᵛ}` and producing `{f̄ᵗ, f̄ᵃ, f̄ᵛ}`. Validate on toy graph.
2. **Add edge-dependent γ_e(v) weights** to existing ConxGNN HyperGraphModel (M³Net Eq. 5 with `Ĥ`).
3. **Wrap ConxGNN's IGM** with the implicit-edge detector — insert one FFN before each branch's graph construction. Mask upper triangle for causality.
4. **Extend MainModel.forward** to compute `m_i^τ = [pτ ∥ qτ ∥ f̄τ]`, then run the existing cross-modal attention on `m`.
5. **Add DualCL loss** to the trainer — call hypergraph branch twice with dropout in training mode, gather node embeddings, compute InfoNCE.
6. **Hyperparameter grid** over `n`, window sizes, `K`, `μ`, `λ`.
7. **Run ablation matrix** (see §8).

---

## 8. Ablation Plan

Run the following ablations to demonstrate each module's contribution. Use IEMOCAP w-F1 as the headline metric.

| Variant | What is removed |
|---------|----------------|
| **Full HyFIN-Net** | — (baseline) |
| w/o IGM (single branch only) | Replace IGM with one k-GNN layer using a single global window |
| w/o HM | Set `q^τ = 0`; concat reduces to `[p^τ ∥ f̄^τ]` |
| w/o Multi-Frequency | Set `f̄^τ = 0`; concat reduces to `[p^τ ∥ q^τ]` |
| w/o Implicit-Edge Detector | Skip step 4.2.1 last bullet |
| w/o γ_e(v) (M³Net node weights) | Use uniform hyperedge incidence (ConxGNN HM only) |
| w/o Cross-Modal Attention | Use plain concat-and-FC |
| w/o CBFC | Train with CBCE only |
| w/o DualCL | Train with `λ = 0` |
| w/o Class-Balanced reweighting | `w_c = 1` for all c |
| Inception #branches ∈ {1, 2, 3, 4} | Vary `n` |
| Frequency layers `K ∈ {1..7}` | Vary frequency depth |

Expected outcome: every component should contribute ≥ 0.3% w-F1.

---

## 9. Expected Performance (informed estimates)

Based on the union of the individual modules' published gains:

| Dataset | Baseline (best of source papers) | HyFIN-Net target |
|---------|----------------------------------|------------------|
| IEMOCAP w-F1 | 75.47 (HRG-SSA) | **75.8 – 77.0** |
| MELD w-F1 | 67.05 (M³Net) / 68.64 (ConxGNN ours-Acc) / 66.83 (HRG-SSA w-F1) | **67.3 – 68.5** |

The intuition: each source method optimizes one axis (multi-scale, hypergraph multivariate, frequency, implicit edges). HyFIN-Net stacks orthogonal axes and is the first to combine **all four** with the explicit goal of avoiding redundancy.

---

## 10. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| **Parameter explosion** (IGM × HM × Freq + dual-view forward) | Share unimodal-encoder params across branches; cap branches at `n=3`; disable Graph Transformer by default. |
| **Optimization instability** (multi-loss) | Two-stage training — first 5 epochs unimodal InfoNCE pretraining (already in ConxGNN), then joint training with `λ` warm-up over 3 epochs. |
| **Frequency module dominates / dies** (M³Net showed sensitivity to `ε`) | Use M³Net's hyperparameter-free formulation (Eqs. 7–10), not FAGCN's `ε`. |
| **IGM windows overfit small datasets** | Use validation set to select window tuples; clamp `n ≤ 3` for IEMOCAP. |
| **Implicit detector adds noisy edges** | Apply the `α_ij > 1/N` threshold strictly (HRG-SSA Eq. 11); upper-triangular causal mask. |
| **DualCL collapses (trivial solution)** | Use feature-dropout views (not augmentation chains); detach negatives' gradient to one branch. |

---

## 11. What is Explicitly *Not* Included (clear scope)

To keep the architecture tractable and avoid wishful stacking:

- ❌ No T5 / generative decoder (HRG-SSA).
- ❌ No VHGAE encoder–sampler–decoder (HAUCL).
- ❌ No Gumbel-Softmax structural learning.
- ❌ No sentiment-text-based contrastive losses (HRG-SSA PSA/MSA).
- ❌ No KL-divergence reconstruction term.
- ❌ No external knowledge graphs / commonsense (COSMIC-style).
- ❌ No LLM / RoBERTa fine-tuning at the architecture layer (features are pre-extracted as the user specified).

---

## 12. One-Sentence Summary

**HyFIN-Net** combines the **Inception Graph Module** (multi-window k-GNN, ConxGNN) with **edge-weighted hypergraph propagation** (M³Net's γ_e(v) on ConxGNN's modality/utterance hyperedges) and **adaptive multi-frequency message passing** (M³Net), enriched by a lightweight **implicit-edge detector** (HRG-SSA) and **dual-view contrastive** regularization (a stripped-down HAUCL), fused via **text-anchored cross-modal attention** and trained with **class-balanced focal contrastive + cross-entropy** losses — explicitly avoiding the heavyweight VHGAE and generative-decoder components that would otherwise inflate model size without clear gains.
