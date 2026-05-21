# Recent Breakthroughs in Graphs, Transformers, and Contrastive Learning — and Whether They Actually Help HyperShift v2

**Companion document to:** `HyperShift_v2_BMVC_Research_Plan.md`
**Date:** May 17, 2026
**Purpose:** Evaluate 2024–2026 advances and decide which ones genuinely strengthen the contribution vs. which are hype distractions for this specific problem.

---

## How to Read This Document

For every technique below, the format is:

- **What it is** — one-paragraph technical summary with citations.
- **Application to HyperShift v2** — what it would do for *our* pipeline specifically.
- **Verdict** — one of three calls:

| Symbol | Meaning |
|---|---|
| 🟢 **INTEGRATE (core)** | Directly strengthens the contribution. Build into the pipeline. |
| 🟡 **TRY IN ABLATION** | Worth experimenting; may or may not make the final paper. |
| 🔴 **SKIP** | Solves the wrong problem, already published, or is hype not signal. |

The bias of this document is *aggressive pruning*. A BMVC paper has space for ~3 core ideas and ~5 ablations. Anything beyond that becomes noise that reviewers will use to attack the contribution as "kitchen-sink".

---

## 1. Graph Architecture Breakthroughs (2023–2026)

### 1.1 Graph Transformers — GraphGPS / Exphormer

**What it is.** GraphGPS (Rampášek et al., NeurIPS 2022) is a modular framework that combines local message passing with global attention plus positional/structural encodings. Exphormer (Shirzad et al., ICML 2023) is the sparse-attention variant using expander graphs + virtual global nodes; it scales linearly in graph size while preserving theoretical expressivity. Both are now standard backbones in molecular and citation-network learning.

**Application to HyperShift v2.** Replace the SGConv layers inside hypergraph propagation with a GraphGPS-style hybrid: local hypergraph convolution + sparse global attention via virtual speaker nodes. The virtual node could *be* the speaker — a single global node per speaker that attends to all their utterance nodes across modalities.

**Verdict: 🔴 SKIP** — Exphormer is engineered for graphs with 10K–1M+ nodes. MERC dialogues have at most ~M=150 utterances × 3 modalities = ~450 nodes. Importing GraphGPS machinery into a tiny graph adds parameters without commensurate gain, and reviewers will (correctly) ask why a method built for ogbn-arxiv is being deployed on a 50-utterance IEMOCAP dialogue. The one genuinely interesting idea — virtual speaker nodes — is *implementable in a stripped-down form within the hypergraph itself* by adding a speaker hyperedge containing all of that speaker's modality nodes, which is already in the plan. So we already get the benefit without the GraphGPS overhead.

---

### 1.2 Heterogeneous Graph Transformers — HGT / HAN / MAGNN

**What it is.** HGT (Hu et al., WWW 2020) uses meta-relation triplets `(source_type, edge_type, target_type)` to parameterize attention weights so different node-type interactions get different transformations. HAN (Wang et al., WWW 2019) does hierarchical attention at node-level and meta-path-level. These dominate citation-network and recommendation benchmarks where nodes have distinct types.

**Application to HyperShift v2.** Text, visual, and acoustic modality nodes are heterogeneous by construction. An HGT-style attention head could replace the simple MLP in our hyperedge weighting: instead of `α_e = MLP([AU_i, AU_j, AU_k])`, we'd parameterize separate query/key/value matrices per modality-pair and compute meta-relation attention.

**Verdict: 🟡 TRY IN ABLATION** — One ablation row in the paper: "AU-conditioned MLP weighting" vs. "HGT-style meta-relation attention". HGT is the textbook way to handle heterogeneous nodes, so reviewers will ask if you tried it. Run the experiment so you can answer with data, but expect the simpler MLP to win on small graphs because HGT's parameter count balloons fast with three modalities and three hyperedge types. If the MLP wins, that's actually a *good* story: "task-conditioned simple gating beats parameter-heavy heterogeneous attention at this scale".

---

### 1.3 Heterogeneous Hypergraphs with Multi-Granular Attention — MGA-HHN

**What it is.** Multi-Granular Attention based Heterogeneous Hypergraph Neural Network (Jin et al., arXiv:2505.04340, 2025) constructs meta-path-based hypergraphs and applies attention at *both* the node level (within a hyperedge) and the hyperedge level (across hyperedges of different types). Addresses over-squashing in long-range hypergraph message passing.

**Application to HyperShift v2.** Our pipeline has three hyperedge types: triadic-utterance, speaker, and temporal-window. MGA-HHN's two-level attention maps cleanly onto this: node-level attention within each hyperedge (which modality node contributes most), then hyperedge-level attention (does triadic, speaker, or temporal context matter more for this utterance?). This is a stronger formulation than the single attention mechanism in the original draft.

**Verdict: 🟢 INTEGRATE (core)** — Use the *two-level attention pattern* in our hypergraph propagation. The visual-conditioning (AU-driven) sits inside the node-level attention; a separate hyperedge-level gate decides whether the model emphasizes triadic, speaker, or temporal context per utterance. This sharpens the structural contribution without inflating parameter count, and it gives the architecture diagram a clean, papertable structure ("dual-granular attention over a visually-conditioned heterogeneous hypergraph"). Cite MGA-HHN as inspiration; differentiate via the visual conditioning, which it does not have.

---

### 1.4 Variational / Dynamic Hypergraphs — HAUCL

**What it is.** HAUCL (Yi & Shen, ACM MM 2024) uses a Variational Hypergraph Autoencoder (VHGAE) to dynamically reconstruct hyperedges during training, reducing redundancy in long dialogues. Trained with a hypergraph reconstruction loss + contrastive learning loss. Reports SOTA over M3Net on IEMOCAP and MELD.

**Application to HyperShift v2.** This is the closest competitor in the literature. HAUCL learns *which* hyperedges to keep; our plan instead learns *how much weight* to give each hyperedge via AU-conditioning. The two ideas could be combined: VHGAE to prune the candidate hyperedge set, then AU-conditioning for weighting the survivors.

**Verdict: 🟡 TRY IN ABLATION** — The combination is principled but adds an extra loss term and an extra forward pass through the VAE. The bigger risk is that combining your dynamic weighting with their dynamic structure muddies the contribution claim — reviewers will ask "is the win from your weighting or from their structure?". Run a clean three-way ablation: (a) fixed hyperedge structure + AU weights, (b) VHGAE structure + uniform weights, (c) both. If (c) wins by a meaningful margin and (a) is already competitive, you have a strong story. If (a) ≈ (c), drop VHGAE entirely. **Default to (a) for the main result.**

---

### 1.5 Graph Neural ODEs — DGODE

**What it is.** Dynamic Graph Neural ODE Network (Shou et al., COLING 2025, arXiv:2412.02935) models the continuous dynamics of node representations over time via a neural ODE solver. Alleviates over-smoothing, enables deeper networks. Already published on IEMOCAP/MELD for MERC.

**Application to HyperShift v2.** Could replace the bidirectional Mamba as the temporal aggregator. Graph ODE on a sequence of fused utterance representations would model emotion dynamics continuously rather than discretely.

**Verdict: 🔴 SKIP** — DGODE is already a published MERC paper by an active research group (Shou, Meng, Ai, Li have ~10 MERC papers since 2023). Borrowing their core idea invites direct comparison on their home turf, and ODE-based models are notoriously sensitive to hyperparameters and solver choice — high risk for marginal gain. Mamba covers the same conceptual ground ("continuous-time-ish state evolution") with much cleaner engineering and a stronger systems narrative. Keep Mamba.

---

### 1.6 Graph Foundation Models / LLM-on-Graph (Graph-MLLM, Graph-to-Vision)

**What it is.** Emerging line of work treating LLMs as graph reasoners — encoding graph structure as natural language or feeding graph features into LLaVA-style multimodal LLMs (Li & Jiang 2025, Liu et al. 2025).

**Application to HyperShift v2.** Could in principle reformulate MERC as "describe the dialogue graph to an LLM and have it predict emotions".

**Verdict: 🔴 SKIP** — Wrong problem, wrong venue. LLM-based ERC (e.g., InstructERC) is a separate research thread already covered at EMNLP/ACL, not BMVC. Mixing it in would make the paper unfocused.

---

## 2. Transformer Architecture Breakthroughs (2023–2026)

### 2.1 Mamba-2 and the State-Space Duality (SSD) Framework

**What it is.** Mamba-2 (Dao & Gu, 2024) reformulates the selective state-space model with a structured matrix view ("SSD") that unifies attention and SSMs under one framework. Roughly 2–8× faster than Mamba-1 in practice on modern hardware, simpler to implement, and matches or beats Mamba-1 in quality at the same parameter count. RWKV-6 (Eagle/Finch) and DeltaNet are parallel evolutions with similar performance profiles.

**Application to HyperShift v2.** Drop-in replacement for the bidirectional Mamba block over the fused utterance sequence. No conceptual change, but cleaner engineering and faster training.

**Verdict: 🟢 INTEGRATE (core)** — Use Mamba-2 instead of Mamba-1. Same idea, better-supported code, faster. Cite both Gu & Dao 2023 (Mamba) and Dao & Gu 2024 (Mamba-2). Don't bother with RWKV — Mamba has dominant momentum in 2025–2026 vision work (VideoMamba, U-Mamba, etc.) and the audience will recognize it immediately.

---

### 2.2 Multimodal Bottleneck Transformer — MBT

**What it is.** MBT (Nagrani et al., NeurIPS 2021) introduces a small set of latent "bottleneck tokens" through which cross-modal information must flow. Forces each modality to condense its most relevant information before sharing. Outperforms unrestricted cross-attention with lower compute. Cited extensively in 2024–2025 audiovisual fusion work.

**Application to HyperShift v2.** Implement vision-anchoring not just via the contrastive loss but architecturally: a small set of learned "vision-anchor" tokens that text and audio must attend through. The bottleneck forces T and A to channel their relevant content through V-conditioned latents, which is the same inductive bias we want but baked into the architecture rather than only the loss.

**Verdict: 🟡 TRY IN ABLATION** — Strong idea, BMVC-compatible (the original is a vision/audiovisual paper). The risk is overlap with the contrastive alignment block: if `L_align` already does its job, MBT bottlenecks may be redundant. Worth one ablation row: "with bottleneck tokens vs. without". If both are needed, the story is "architectural and loss-level vision anchoring are complementary". If not, drop one and keep the contribution sharper.

---

### 2.3 Cross-Attention Fusion (CrossFuse, ATFusion, gated cross-attention)

**What it is.** A family of 2024–2025 papers refining bidirectional cross-attention between modalities — CrossFuse (Li et al., 2024) uses reversed softmax for complementarity, ATFusion (Yan et al., 2024) separates discrepancy and commonality injection, MSGCA (Zong et al., 2024) adds gated cross-attention for time-series with 6–31% MCC gains.

**Application to HyperShift v2.** Could replace the hypergraph's cross-modal information flow with explicit cross-attention layers between every modality pair.

**Verdict: 🔴 SKIP** — These methods solve the *flat* multimodal fusion problem (where there's no graph structure to exploit) and are direct competitors to the graph-based MERC tradition. Adopting cross-attention as the fusion mechanism would weaken the graph-based contribution argument. The whole point of GraphSmile and its descendants is that the *graph structure encodes the conversational topology* better than flat cross-attention can. Stay on the graph/hypergraph side of the line.

---

### 2.4 Gated Recursive / Sequential Multimodal Fusion (GRF, MulT, Husformer)

**What it is.** Architectures that process modalities sequentially rather than in parallel pairwise, with a recurrent context vector updated by gated cross-attention at each step. GRF (Shihata, 2025) explicitly addresses the O(n²) cost of MulT-style pairwise fusion. Husformer (2024) targets multimodal human-state recognition.

**Application to HyperShift v2.** Linearizes the cross-modal interaction.

**Verdict: 🔴 SKIP** — With only three modalities (T, V, A), the quadratic cost is six pairwise interactions. Six. The scalability problem GRF solves does not exist in MERC. Using a method whose entire motivation is "scaling to many modalities" on a three-modality problem reads as confused.

---

### 2.5 FlashAttention / FlashAttention-2 / FlashAttention-3

**What it is.** Engineering optimizations of attention computation that reduce memory cost and increase wall-clock speed by 2–4× without changing the algorithm.

**Application to HyperShift v2.** Use wherever standard attention appears (likely in MBT bottleneck tokens or any global attention layer).

**Verdict: 🟢 INTEGRATE (engineering only)** — Free speed-up; no paper-level contribution; mention in the implementation section. Modern PyTorch / `torch.nn.functional.scaled_dot_product_attention` uses it automatically.

---

### 2.6 Mixture of Experts (MoE) in Multimodal Transformers

**What it is.** Sparse activation of experts based on input routing (e.g., DeepSeek-V3, Mixtral). In multimodal contexts, recent work like "More Is Better: A MoE-Based Emotion Recognition Framework" (MRAC 2025) explores MoE for emotion-class-specific experts.

**Application to HyperShift v2.** A per-emotion expert head, routed by gating on the fused representation.

**Verdict: 🔴 SKIP** — MoE pays off when (a) data is huge and (b) compute budget rewards parameter scale at constant FLOPs. MERC datasets are tiny (IEMOCAP ~7K utterances, MELD ~13K). With minority classes at 200–400 examples, training class-specific experts is asking to overfit. The class-balanced loss + targeted contrastive learning (below) is the right tool for the imbalance problem, not MoE.

---

## 3. Contrastive Learning Breakthroughs (2023–2026)

### 3.1 Balanced Contrastive Learning (BCL) and Targeted Supervised Contrastive (TSC)

**What it is.** TSC (Li et al., CVPR 2022, arXiv:2111.13998) generates a uniform set of class prototypes on a hypersphere and pulls each class's features toward its assigned target, enforcing uniform feature distribution even under heavy long-tail imbalance. BCL (Zhu et al., CVPR 2022) modifies the SupCon loss to form a regular simplex, correcting the head-class bias in vanilla SupCon. PaCo, KCL, SBC are parallel variants. The "Tale of Two Classes" paper (Mar 2025, arXiv:2503.17024) benchmarks all of these systematically.

**Application to HyperShift v2.** This is the **most important integration in this entire document**. MELD has severe class imbalance (Fear, Disgust at ~2–3% of data). The original HyperShift draft used vanilla SupCon, which is known to favor head classes. Replacing it with TSC or BCL directly targets the minority-class F1 metric that we want to dominate.

**Verdict: 🟢 INTEGRATE (core)** — Replace vanilla SupCon in the `L_emo` loss with **BCL** (preferred — slightly more empirical evidence on imbalanced vision benchmarks) or **TSC** (cleaner geometric story, easier to write up). Either one directly supports the paper's claim of "8–12 point F1 improvement on minority emotions". This is one of the cheapest wins in the entire plan: it's a loss-function swap with no architectural changes, and the BCL/TSC papers are well-cited so reviewers will recognize the choice as principled.

**Concrete recipe:** Use BCL — it has the cleanest theoretical analysis (regular simplex convergence). The combined emotion loss becomes:
```
L_emo = CB-Focal(logits, y) + λ_sup · BCL(features, y)
```

---

### 3.2 SigLIP / SigLIP 2 — Sigmoid Loss for Multimodal Contrastive Learning

**What it is.** SigLIP (Zhai et al., ICCV 2023) and SigLIP 2 (Tschannen et al., Feb 2025, arXiv:2502.14786) replace the softmax-normalized InfoNCE loss with a pairwise sigmoid (binary cross-entropy) loss. Decouples positive/negative pair treatment, supports multiple positives per anchor, and works well with small batches (eliminates SimCLR/CLIP's need for huge batch sizes).

**Application to HyperShift v2.** Two angles. (1) The `L_align` vision-anchored contrastive loss could be sigmoid-based instead of softmax-based, naturally supporting the fact that multiple utterances in a batch can share an emotion label. (2) The visual feature extractor itself could use a SigLIP 2 backbone (still vision-language, but stronger than CLIP-ViT and trained with the new objective).

**Verdict (loss):** 🟡 TRY IN ABLATION — One row: SupCon vs. SigLIP-style sigmoid alignment. SigLIP's main selling point (huge batch sizes) does not apply at MERC scale, but the multi-positive support is genuinely useful when many utterances share emotion labels.
**Verdict (backbone):** 🟢 INTEGRATE — Use SigLIP 2 instead of CLIP-ViT-L/14 or EmoCLIP for the semantic visual stream. SigLIP 2 is published, has open weights, and provides cleaner localization/dense features per its paper. Cite it as the modern replacement for vanilla CLIP. This is one component where "use the newest backbone" is a defensible argument because SigLIP 2 is February 2025 and demonstrably better than CLIP on every downstream vision benchmark.

---

### 3.3 Hard Negative Mining — HCL / DCL / Subspace-Preserving Methods

**What it is.** A family of methods that prioritize negatives close to the anchor in embedding space. Hard CL (HCL, Robinson et al., ICLR 2021) reweights negatives by hardness. Debiased CL (Chuang et al., NeurIPS 2020) removes false negatives. More recent graph variants (GRAPE, WWW 2024) and supervised variants (SCL with hard negatives, arXiv:2209.00078) refine the idea.

**Application to HyperShift v2.** GraphSmile's confusion matrices show that the hardest confusions are *Happy vs. Excited* (IEMOCAP) and *Neutral vs. Surprise/Anger* (MELD). Hard negative mining inside the SupCon/BCL loss could specifically separate these.

**Verdict: 🟡 TRY IN ABLATION** — Worth one experiment: BCL with random negatives vs. BCL with hard-negative-mined negatives (mined by feature-space distance to anchor). The risk is *over-separation*: Happy and Excited are genuinely similar emotions, and forcing strong separation may hurt generalization. Use a moderate temperature and a percentage-based hardness threshold (top-30% hardest negatives) rather than an aggressive top-1. If it improves Happy/Excited F1 without hurting Hap → Sad confusion, keep it.

---

### 3.4 Decoupled Contrastive Learning (DCL)

**What it is.** Yeh et al. (ECCV 2022) remove the positive-pair term from the denominator of the InfoNCE loss, decoupling positive and negative gradients. Improves training stability at small batch sizes.

**Verdict: 🔴 SKIP** — Marginal effect (~0.5 point on ImageNet). Not paper-worthy as a contribution and adds another knob. The benefit overlaps with what SigLIP already provides.

---

### 3.5 Hypergraph Contrastive Learning (HCL, S-Mixup, GRACE-derivatives)

**What it is.** Self-supervised contrastive objectives over augmented hypergraph views — edge-dropping, node-masking, feature-shuffling — used as a pretraining or regularization signal.

**Application to HyperShift v2.** Add a self-supervised reconstruction or view-consistency loss on the hypergraph itself.

**Verdict: 🔴 SKIP** — HAUCL already does this and our differentiator vs. HAUCL is *not* doing more SSL on the graph. Adopting it would erase the cleanest differentiator we have. Stick to supervised contrastive (BCL) and the causal regularizer; let HAUCL keep its hypergraph SSL niche.

---

### 3.6 Cross-Modal Contrastive Learning (CMC, MoCo-v3 for video, LanguageBind)

**What it is.** Contrastive pretraining frameworks that align multiple modalities to a shared space using cross-modal positive pairs. ImageBind (Meta, 2023), LanguageBind (PKU, 2023), and various video-text-audio aligners.

**Application to HyperShift v2.** Pretrain modality encoders contrastively on a large unlabeled corpus before fine-tuning on MERC.

**Verdict: 🟢 INTEGRATE (via choice of pretrained backbones)** — But not as a contribution we claim. We use SigLIP 2 (vision-text aligned), WavLM (audio-aligned-via-pretraining-on-large-corpus), and RoBERTa-Large. All three backbones are already products of cross-modal or large-scale self-supervised training. We benefit from CMC without doing CMC ourselves.

---

## 4. Summary — What Actually Goes Into the Paper

### 4.1 Final integration list (CORE — these become parts of the contribution)

| Component | Replaces / Adds | Why |
|---|---|---|
| **Mamba-2** | Replaces Mamba-1 in the temporal aggregator | Cleaner engineering, free speedup |
| **MGA-HHN's two-level attention** | Adds node-level + hyperedge-level attention to hypergraph | Sharper structural design; differentiates from HAUCL/ConxGNN |
| **SigLIP 2 backbone** | Replaces CLIP-ViT-L/14 / EmoCLIP for visual semantics | Newest, demonstrably better, defensible choice |
| **BCL (Balanced Contrastive Learning)** | Replaces vanilla SupCon in `L_emo` | Directly addresses minority-class F1 — the main story |
| **FlashAttention-2** | Engineering inside attention layers | Free training speedup |

### 4.2 Ablation list (TRY — keep the ones that win, mention the ones that don't)

| Component | Ablation Question |
|---|---|
| HGT-style heterogeneous attention | Does meta-relation attention beat AU-conditioned MLP for hyperedge weighting? |
| MBT bottleneck tokens | Is architectural vision-anchoring complementary to the contrastive loss? |
| HAUCL-style VHGAE structure | Does dynamic structure + AU weights beat static structure + AU weights? |
| Hard negative mining in BCL | Does it improve Happy/Excited separation without hurting other classes? |
| SigLIP sigmoid alignment | Does sigmoid alignment beat softmax SupCon at this scale? |

### 4.3 Skip list (do not waste cycles)

| Component | Why Skip |
|---|---|
| GraphGPS / Exphormer | Built for huge graphs; MERC graphs are tiny |
| DGODE / Graph ODEs | Already published for MERC; risky and overlaps with Mamba |
| LLM-on-Graph / InstructERC-style | Wrong venue; different research thread |
| CrossFuse / ATFusion (flat cross-attention) | Competes with our graph-based premise |
| GRF / sequential multimodal fusion | Solves a 3+ modality scaling problem we don't have |
| MoE multimodal | Datasets too small; overfitting risk |
| DCL (Decoupled CL) | Marginal effect; overlaps with SigLIP |
| Hypergraph SSL (HAUCL-style) | HAUCL owns this niche; erases differentiator |

---

## 5. Updated Pipeline Diagram (Integrating the Verdicts)

```
                  ┌─────────────────────────────────────────────┐
                  │  Modality Encoders (modern backbones)       │
                  │  • Text: RoBERTa-Large                      │
                  │  • Vision: SigLIP 2 + OpenFace 2.0 AUs      │  ← upgraded
                  │  • Audio: WavLM-Large                       │
                  └────────────────────┬────────────────────────┘
                                       │
                  ┌────────────────────▼────────────────────────┐
                  │  Vision-Anchored Contrastive Alignment      │
                  │  L_align = SupCon(z_v, z_t) + SupCon(z_v,z_a) │
                  │  (or SigLIP-style sigmoid — ablation)       │
                  └────────────────────┬────────────────────────┘
                                       │
                  ┌────────────────────▼────────────────────────┐
                  │  Visually-Conditioned Heterogeneous         │
                  │  Hypergraph (HGNN+ formulation)             │
                  │  • Three hyperedge types (triadic,          │
                  │    speaker, temporal-window)                │
                  │  • Two-level attention from MGA-HHN:        │  ← integrated
                  │      node-level AU-conditioned weighting    │
                  │      hyperedge-level type gating            │
                  └────────────────────┬────────────────────────┘
                                       │
                  ┌────────────────────▼────────────────────────┐
                  │  Causal-Counterfactual Training Path        │
                  │  TE = predict(T, V, A)                      │
                  │  NDE = predict(T, V=∅, A=∅)                 │  ← differentiator
                  │  L_causal pushes the model away from        │
                  │  the text-only shortcut (CLEF-style)        │
                  └────────────────────┬────────────────────────┘
                                       │
                  ┌────────────────────▼────────────────────────┐
                  │  Bidirectional Mamba-2 over fused sequence  │  ← upgraded
                  │  (O(M) temporal aggregator)                 │
                  └────────────────────┬────────────────────────┘
                                       │
                  ┌────────────────────▼────────────────────────┐
                  │  Classifier head                            │
                  │  L_emo = CB-Focal + λ · BCL                 │  ← upgraded
                  │  (replaces vanilla SupCon)                  │
                  └─────────────────────────────────────────────┘

Total loss: L = L_emo + λ_align · L_align + λ_caus · L_causal + β · ||Θ||²
```

---

## 6. The Sharp Edge — What This Document Actually Argues

The original HyperShift draft tried to win on five fronts at once. After auditing the 2024–2026 literature, three of those five were already published. This document filters those five into a tighter set of choices: **three core upgrades** (SigLIP 2 backbone, MGA-HHN dual-attention, BCL loss) **plus the one genuinely novel contribution** (causal counterfactual training for text-dominance debiasing in MERC). Mamba-2 and FlashAttention-2 are engineering hygiene.

That is the paper. Everything else is an ablation row or a distraction.

The trap to avoid: a "rich" paper that incorporates everything in this document at once. BMVC reviewers will read a 14-page paper with 8 architectural innovations as either (a) a kitchen-sink with unclear contribution attribution, or (b) a paper that won't ablate cleanly because every component depends on every other. The discipline is to pick the *minimum set of components needed for the claimed contribution* and aggressively cut the rest.

---

## 7. One-Week Action Items from This Document

1. **Pull the BCL repository** (https://github.com/FlamieZhu/BCL) and drop the BCL loss into the existing GraphSmile training script. Run one epoch on IEMOCAP to confirm it works mechanically. *This is a 2-hour task and immediately makes the loss-side contribution concrete.*
2. **Download SigLIP 2 weights** (Google DeepMind, HuggingFace `google/siglip2-so400m-patch14-384` or smaller) and verify you can extract features from a single IEMOCAP video frame. Compare cosine similarity between same-emotion vs. different-emotion utterance pairs vs. the same comparison with the original DenseNet features. *If SigLIP 2 doesn't separate emotions better than DenseNet, the visual-feature story is weaker than assumed.*
3. **Skim the MGA-HHN paper** (arXiv:2505.04340) end-to-end and write a one-paragraph note on how to adapt their two-level attention to your hypergraph. *This pins down the architectural detail before any coding.*
4. **Read CLEF (CVPR 2024, arXiv:2403.05963) carefully**, especially the NDE/TIE formulation in Section 3.3. Sketch the computation graph for the counterfactual forward pass in HyperShift v2 — three input modalities, two predictions (TE and NDE), one combined loss. *This is the most fragile part of the implementation and benefits most from up-front design.*

If those four steps go smoothly, the architectural risk of the paper is largely retired and the next 12 weeks become an engineering and ablation exercise.
