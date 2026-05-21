# Novel Graph-Based MERC Architectures: Research Proposal
*Grounded in literature; prioritizing simplicity, feasibility, and meaningful novelty over unnecessary complexity.*

---

## 0. Preface — What gaps are we actually trying to close?

A survey of ~27 graph-based MERC papers (DialogueGCN→HRG-SSA) reveals the field's **genuinely open problems** are not architectural breadth (we already have hypergraphs, multi-frequency, SSMs, contrastive, etc.) but rather:

| Gap | Why it remains open | Best existing partial solution |
|---|---|---|
| **Implicit graph edges grounded in influence, not feature similarity** | HRG-SSA's implicit-edge module is similarity-based; this captures *correlation* but conflates emotional propagation with mere semantic resemblance | HRG-SSA (IJCAI 2025) — but uses cosine similarity |
| **Multi-band frequency on graphs (beyond binary low/high split)** | M3Net (CVPR 2023) and GS-MCC (2024) both split signal into only two bands. Wavelet decomposition with K=3–5 scales is well-established in graph signal processing but unused in MERC | M3Net — binary; FAGCN — binary |
| **Orthogonal attack on the three hardest empirical problems simultaneously**: modality dominance + class imbalance + similar-emotion confusion | Each is attacked in isolation: Ada²I (modality), DER-GCN/ConxGNN (imbalance), AR-IIGCN/EACL (similar-emotion). No paper combines all three | Ada²I or DER-GCN or EACL (each alone) |
| **Online / streaming MERC** | ALL existing graph MERC methods are offline (rely on full conversation including future utterances) | None (open) |
| **Robustness to missing modalities** | Graph methods generally collapse when audio or video is missing; not addressed in graph backbone | GCNet (Lian et al.) — non-mainstream |

The three core proposals below target the **first three rows** (mature enough to design well, novel enough to justify a paper). Proposal 4 is sketched as a future direction.

A consistent design principle across all proposals: **prefer 1–2 architectural innovations grounded in literature over a long list of contributions**. Reviewer fatigue with "kitchen sink" MERC papers is real (each major venue receives 5–10 incremental MERC submissions per cycle), and most accepted MERC papers in the last two years have one clearly-articulated innovation.

---

## 1. CIEI-MERC: Causal Implicit Edge Inference for MERC

### 1.1 Motivation

HRG-SSA (IJCAI 2025) is the current SoTA on IEMOCAP (75.47 WF1). Its key innovation is the **implicit-edge detector** for each modality:

$$s^\eta_{ij} = \text{LeakyReLU}(\text{mask} + W^T_\eta [H^\eta_i \| H^\eta_j])$$

This is a learned cosine-similarity-style scorer. As the HRG-SSA authors themselves note, this captures **similarity, not causation**. Two utterances may have similar feature representations without one influencing the other (e.g., the speaker's repeated "I'm happy" across a conversation), while an angry outburst U₅ may causally trigger a defensive U₈ whose surface features are dissimilar but whose emotional content is causally linked.

**Hypothesis**: replacing the similarity-based implicit edges with **influence-based (pseudo-causal) edges** captures the actual structure of emotional propagation and yields better same-modality reasoning.

This is a *grounded* hypothesis: in the broader sequential modeling literature, attention has repeatedly been shown to underperform causal-flavored attention in long-range dependency tasks (e.g., Goyal et al., *Neural Production Systems*, 2021; Schölkopf et al., *Towards Causal Representation Learning*, 2021). MERC has not yet benefited from this development.

### 1.2 Architecture overview

```
[Text U_i]    [Audio U_i]    [Visual U_i]
    ↓             ↓               ↓
[RoBERTa-Lg]  [HuBERT-Lg]    [OpenFace 2.0 + CLIP]
    ↓             ↓               ↓
              [Speaker + Position Embedding]
                       ↓
   ┌─────────────────────────────────────┐
   │ Per-modality graph with:            │
   │  - Explicit edges (adjacency ±k,    │
   │     same-speaker, same-utterance)   │
   │  - Implicit CAUSAL edges (NEW)      │
   │    inferred via Counterfactual      │
   │    Attention Drop (CAD) scorer      │
   └─────────────────────────────────────┘
                       ↓
        [Layer-by-layer alternating GAT]
        (GraphSmile-style: alt. inter/intra)
                       ↓
   [Concatenate per-modality outputs] + MLP
                       ↓
   [CE loss + EACL anchor-contrastive]
```

### 1.3 Module-by-module design

#### (a) Modality encoders

| Choice | Rationale | Why not the alternative |
|---|---|---|
| **Text: RoBERTa-Large (fine-tuned)** | M3Net's ablation shows +4–7 WF1 over GloVe-TextCNN; widely used in 2023–2025 papers | GloVe outdated; T5 only needed if generation; DeBERTa-v3 marginally better but adds compute |
| **Audio: HuBERT-Large** | Outperforms Wav2Vec2 on SUPERB emotion benchmark by ~2–3 pts; OpenSmile features lose phonetic detail | OpenSmile = handcrafted, weaker; Wav2Vec2 slightly weaker on emotion-specific evaluation per SUPERB |
| **Visual: OpenFace 2.0 (facial AUs) + CLIP-ViT visual embedding** | OpenFace gives precise facial-action units (linked to FACS emotion theory); CLIP gives scene context. COGMEN showed OpenFace works well | 3D-CNN ignores facial detail; DenseNet-FER trained on FER2013 = small/old; raw frame ViT lacks emotion specificity |

#### (b) Graph construction

**Explicit edges** (same as HRG-SSA, no innovation): adjacency (current ± k=10 utterances), same-speaker, same-utterance cross-modal. No reason to deviate — these are proven and cheap.

**Implicit edges (the actual innovation): Counterfactual Attention Drop (CAD)**.

For each pair (i, j) with j > i in the same modality, compute an **influence score**:

$$\text{Inf}(i \to j) = \| f(H_j \mid \text{ctx}_j) - f(H_j \mid \text{ctx}_j \setminus H_i) \|_2$$

where `ctx_j` is the explicit-edge context of utterance j, and `f` is a small attention block. Concretely: forward-pass twice (with and without H_i in the context), take L2 difference of representations. Apply Gumbel-Softmax to top-k pairs to get differentiable edge sampling.

**Alternatives considered:**

| Alternative | Why rejected (or kept as variant) |
|---|---|
| **Granger causality** (lagged regression test) | Too rigid for dialogue — assumes linear lag structure; doesn't generalize to multimodal features. Could be a baseline comparison. |
| **PCMCI** (causal discovery) | Designed for time series with stationarity assumptions that conversations violate; high compute |
| **NOTEARS** (differentiable DAG learning) | Strict acyclicity constraint loses cyclic dialogue dependencies (rebuttals, repeated themes); too much regularization |
| **Pure self-attention** | Equivalent to HRG-SSA in essence; would not be a novel contribution |
| **Learned do-calculus interventions** | True causal inference requires interventional data, which we don't have. CAD is a *pseudo*-causal proxy and we are transparent about this |

The honest framing for the paper: **"counterfactual attention drop" is not strict causal inference but a representation-level intervention proxy** that captures "if utterance i hadn't existed, would j's representation differ?" — a reasonable surrogate for influence.

#### (c) Graph learning backbone

**Layer-by-layer alternating GAT** (per GraphSmile GSF). Each layer alternates: inter-modal aggregation → intra-modal aggregation. GraphSmile demonstrated this alternation avoids "fusion conflict" (concurrent inter/intra aggregation causes representation collisions).

**Alternatives considered:**

| Alternative | Why not | When it would be considered |
|---|---|---|
| **GCN/GCNII** | Spectral, doesn't easily handle heterogeneous edge types; M3Net showed hypergraph needed for higher-arity | If we want spectral filters (Proposal 2) |
| **RGCN** | Multi-relational but expensive 2M² relation parameters | Only if relation types are few and discrete (Proposal 3) |
| **Hypergraph NN** | Captures higher-arity but overkill if causal edges are pairwise | Proposal 2 uses it |
| **GraphTransformer** | Strong but parameter-heavy; COGMEN shows benefit but compute trade-off significant | Considered for ablation |
| **SSM (Mamba)** | Loses graph structure; orthogonal direction | Future work |

#### (d) Fusion strategy

**Late concatenation** of per-modality graph outputs → MLP. Simple and works (M3Net, GraphSmile, HRG-SSA all converge on similar strategies).

**Alternatives:**
- **Cross-modal attention (CORECT P-CM)** — adds 2–4M parameters; provides a small gain. Worth trying as ablation.
- **Probability-guided fusion (Broad Mamba)** — interesting but unstable for ambiguous cases per Broad Mamba's own ablation.
- **Pair-wise complementary (GraphCFC)** — incremental compute, modest gain.

#### (e) Loss

```
L = L_CE + λ₁ · L_EACL + λ₂ · L_CB-focal
```

- **L_CE**: standard cross-entropy
- **L_EACL** (Anchor-based contrastive, Yu et al. 2023): per-emotion learnable anchor vectors; pull each sample toward its true-emotion anchor, push away from similar-emotion anchors. Specifically attacks the happy↔excited and anger↔frustrated confusions documented in CORECT and COGMEN
- **L_CB-focal** (Class-Balanced Focal, Cui et al. 2019, used in ConxGNN): handles MELD's severe imbalance (anger/sadness/joy >> disgust/fear)

**Alternative loss choices:**
- **Plain focal loss** (MMGCN, MM-DFN): doesn't account for effective number of samples per class; CB-focal strictly better
- **SupCon** (Khosla et al. 2020): general contrastive; less targeted than EACL for similar-emotion problem
- **Triplet loss with hard mining**: similar to EACL but no learned anchors → less stable

### 1.4 Self-review

#### Strengths
- **Genuine novelty axis**: no graph MERC paper has attempted causal/counterfactual edge inference. Even if the actual improvement is modest (1–2 WF1), the conceptual contribution is meaningful and citable.
- **Grounded**: builds directly on HRG-SSA's identified limitation; not speculative.
- **Modest implementation cost**: CAD only doubles forward-pass cost on implicit-edge inference (a small module), not on the whole graph.
- **Compatible with existing strong components**: uses GraphSmile's GSF backbone, EACL anchors, CB-focal — these are all proven.

#### Weaknesses
- **CAD is pseudo-causal, not causal**: true causal inference requires interventional data we don't have. Reviewers will probe this.
- **Top-k Gumbel sampling is hyperparameter-sensitive**: k (number of implicit edges) needs tuning per dataset. Likely k≈3–5 per node based on HRG-SSA's implicit-edge ratio.
- **Double forward pass for CAD** doubles inference latency on the implicit-edge module (acceptable but worth noting).
- **No theoretical guarantee** that CAD recovers any meaningful causal structure — purely empirical.

#### Assumptions
- That emotional propagation is the dominant hidden structure (vs. shared latent topic / world state). This is testable: if CAD edges cluster by topic rather than emotion, the assumption fails.
- That same-modality intra-utterance influence is the main missing signal (i.e., that explicit cross-modal edges already capture cross-modal influence).
- That k=3–5 implicit edges per node is sufficient (HRG-SSA's threshold scheme suggests this is reasonable).

#### Likely failure cases
- **Highly turn-taking dialogues with little emotional contagion** (e.g., transactional dialogue): CAD edges degenerate to noise; method reduces to explicit-edge GAT
- **Very short conversations** (< 5 utterances): not enough context for the counterfactual to be meaningful
- **MELD's multi-party setting**: with 4+ speakers, the counterfactual signal becomes diffuse; gains likely smaller than on IEMOCAP

#### Anticipated reviewer criticisms

| Criticism | Defense |
|---|---|
| *"Is this really causal? Looks like just a more elaborate attention mechanism"* | **Acknowledge directly in the paper**. Frame as "counterfactual *proxy* for influence", not as causal inference. Show ablations that this counterfactual-style attention beats standard self-attention by ≥1 WF1 — proving the inductive bias matters even if not strictly causal. |
| *"Ablation: does Gumbel-Softmax actually help vs. hard top-k?"* | Run this ablation. Likely Gumbel helps by 0.3–0.7 WF1 due to gradient flow. |
| *"Why not just use a bigger Transformer?"* | Show parameter-matched comparison: CIEI-MERC with X params vs. Transformer baseline with X params. Graph priors should win on data-efficiency. |
| *"How does this handle ambiguous cases where multiple utterances jointly influence j?"* | Multi-edge CAD with top-k naturally handles this; can show ablation varying k. |
| *"Compute cost"* | Quantify it. CAD doubles cost only on implicit-edge module, which is a small fraction of total forward pass. |

### 1.5 Expected magnitude of gain

Based on the gap between HRG-SSA's similarity-based implicit edges and reasonable causal-flavored alternatives in adjacent NLP tasks (relation extraction, dialogue act recognition), expected improvement is **+0.5 to +1.5 WF1 on IEMOCAP, +0.3 to +0.8 WF1 on MELD**. Smaller than the +6 WF1 jump from HAUCL → HRG-SSA, but in the typical range of accepted IJCAI/EMNLP MERC contributions.

---

## 2. WaveHyp-MERC: Wavelet-Frequency Hypergraph for MERC

### 2.1 Motivation

Two SoTA graph-MERC papers — M3Net (CVPR 2023) and GS-MCC (2024) — model the conversational graph as having a **binary** frequency structure (low-pass for consistency, high-pass for discrepancy). Both papers explicitly note this is a simplification.

In graph signal processing, **wavelet decomposition** (Hammond et al. 2011, *Wavelets on Graphs via Spectral Graph Theory*; Xu et al. 2019, *Graph Wavelet Neural Networks*) provides multi-band frequency analysis with 3–7 scales. This has demonstrated benefits in node classification and recommendation but is **completely unused in MERC**.

Independently, HAUCL (Yi et al., ACM MM 2024) shows that hypergraph structure (with >pairwise edges) helps MERC. But HAUCL uses standard hypergraph spectral analysis — no multi-band decomposition.

**Hypothesis**: combining (i) hypergraph structure for higher-arity relations and (ii) multi-band wavelet decomposition for finer frequency analysis attacks **two** simultaneous limitations of M3Net and HAUCL respectively.

### 2.2 Architecture overview

```
[Encoders — same as Proposal 1]
        ↓
[Hypergraph Construction]
   - Multimodal hyperedge per utterance (3 nodes: T, A, V)
   - Same-modality contextual hyperedge (N nodes)
   - Same-speaker hyperedge per modality
        ↓
[Graph Wavelet Decomposition]
   - Compute hypergraph Laplacian L_H
   - Chebyshev polynomial approximation of K=4 wavelet scales
     ψ_s(L_H) for s ∈ {0.5, 1, 2, 4}
        ↓
[Per-scale Hypergraph Convolution]
   H_s = ψ_s(L_H) · X · W_s   for each scale s
        ↓
[Scale-Attention Aggregation]
   α = softmax(MLP([H_s for all s]))
   H_out = Σ_s α_s · H_s
        ↓
[Layer-by-layer alternating]
        ↓
[CE + EACL + CB-focal] (same as Prop 1)
```

### 2.3 Module-by-module design

#### (a) Modality encoders — same as Proposal 1.

#### (b) Hypergraph construction

Three hyperedge types:
1. **Multimodal hyperedge per utterance**: connects the 3 modalities of utterance i (text, audio, visual). Same as M3Net.
2. **Same-modality contextual hyperedge**: connects all utterances in same modality within a dialogue. Same as M3Net.
3. **Same-speaker hyperedge per modality**: connects all utterances by the same speaker within same modality. *New addition* over M3Net — captures speaker-specific emotional baseline.

**Alternative considered**: Learnable hyperedges (HAUCL-style VHGAE). Rejected for now because (a) adds variational sampling variance, (b) data-hungry, (c) the three deterministic hyperedge types above already cover the major structural relations. Could be added as Phase 2 extension if results plateau.

#### (c) Wavelet decomposition (core innovation)

Graph wavelet at scale s is defined as ψ_s = U · g(sΛ) · U^T, where U, Λ are the Laplacian eigendecomposition. Full eigendecomposition is O(N³) which doesn't scale. Use **Chebyshev polynomial approximation** (Hammond et al. 2011):

$$\psi_s \approx \sum_{k=0}^{K_c} c_k(s) \cdot T_k(\tilde{L}_H)$$

where T_k are Chebyshev polynomials and K_c = 3–5 polynomial order. This gives O(N · |E|) per scale.

Choose **4 scales** s ∈ {0.5, 1, 2, 4}:
- s=0.5: very-fine-grained (highest frequency) — captures abrupt local emotion shifts
- s=1: medium-high frequency — utterance-to-utterance variation
- s=2: medium-low frequency — multi-utterance trends (e.g., escalating frustration)
- s=4: low frequency — global conversational mood

**Alternative considered:**

| Alternative | Why not |
|---|---|
| **Full eigendecomposition** | O(N³) compute, doesn't scale to MELD's 13K utterances |
| **Diffusion wavelets** (Coifman & Maggioni) | More complex; Chebyshev gives same expressive power with simpler implementation |
| **Learnable polynomial coefficients** (ChebNet) | Could replace fixed c_k with learnable — worth ablating |
| **Wavelet on graph product** | Overkill; pairwise + hypergraph wavelets sufficient |

Number of scales K — **important hyperparameter**. Literature suggests 3–7 scales for graph signal tasks. We pick 4 as a default; ablate {2, 3, 4, 5, 6} in experiments. If gains plateau at K=3, prune for efficiency.

#### (d) Scale-attention aggregation

Per-utterance learnable attention over scales:

$$\alpha_i = \text{softmax}(W_\alpha \cdot [H_i^{s_1} \| H_i^{s_2} \| H_i^{s_3} \| H_i^{s_4}])$$

$$H_i^{\text{out}} = \sum_s \alpha_i^s \cdot H_i^s$$

This allows different utterances to weight scales differently — e.g., a sharp emotion shift weighs s=0.5 more; a steady-state utterance weighs s=4 more. **Per-utterance** attention is critical (per-graph or fixed weights lose this flexibility).

#### (e) Auxiliary loss for scale diversity

To prevent scale-attention collapse (all weight on one scale), add a diversity regularizer:

$$L_{div} = -\frac{1}{N} \sum_i H(\alpha_i)$$

where H is entropy. Small weight (λ_div ≈ 0.01).

**Alternative considered**: Mutual information minimization across scales. More principled but expensive. Entropy regularization is cheaper and works in practice for mixture-of-experts.

#### (f) Loss — same as Proposal 1 (CE + EACL + CB-focal + λ_div · L_div)

### 2.4 Self-review

#### Strengths
- **Principled extension of M3Net**: addresses M3Net's own acknowledged limitation (binary frequency split is coarse).
- **Wavelet decomposition is well-established** in graph signal processing; brings 15+ years of theoretical understanding into MERC.
- **Per-utterance scale attention** allows the model to adapt frequency emphasis based on context — interpretable (one can visualize which scales each utterance attends to).
- **Compatible with Proposal 1's causal edges** — could combine if budget allows.

#### Weaknesses
- **Compute**: Chebyshev approximation is cheap per scale (O(N·|E|·K_c)) but K=4 scales × hypergraph means ~4× the cost of standard hypergraph convolution. Acceptable on IEMOCAP (151 dialogues, 7K utterances) but heavy on MELD (1.4K dialogues, 13K utterances).
- **Hyperparameter burden**: number of scales K, scale values s, Chebyshev order K_c, diversity weight λ_div — four hyperparameters to tune.
- **Wavelet on hypergraph is less mature** than on simple graphs. Hypergraph Laplacian definition varies (Zhou et al. vs. Chan et al.) — must commit to one.
- **No new conceptual insight** beyond "more frequency bands help": the contribution is more methodological than conceptual.

#### Assumptions
- That MERC graphs have meaningful multi-band frequency content (not just dominantly low-pass or noisy).
- That per-utterance scale weighting is more useful than per-graph (testable via ablation).
- That MELD's multi-party graphs have enough algebraic connectivity for wavelets to be informative (might fail on very sparse multi-party dialogues).

#### Likely failure cases
- **Very short conversations**: insufficient nodes for wavelet decomposition to find structure
- **Dominantly low-pass signals**: if emotional dynamics are mostly slow drift, the higher-frequency scales add only noise
- **MELD's multi-party sparseness**: small clique sizes per speaker may not give wavelets enough room to work

#### Anticipated reviewer criticisms

| Criticism | Defense |
|---|---|
| *"Why not just stack more GNN layers? Each layer captures progressively higher-order info."* | GNN layer stacking conflates depth with frequency — wavelet explicitly disentangles them. Ablation: K=4 wavelet vs. L=4 GCN layers, parameter-matched. Wavelet should win on data-efficiency and interpretability. |
| *"Wavelet on graphs is not new (Xu et al. 2019). What's specifically novel for MERC?"* | Two things: (a) on **hypergraphs** specifically (mostly unstudied), (b) the **per-utterance scale attention** that adapts frequency emphasis dynamically. |
| *"Chebyshev approximation introduces error. How is this controlled?"* | Standard error bounds from Hammond et al. (2011); K_c=4 polynomials gives < 1% spectral approximation error in practice. |
| *"Compute"* | Profile and report. Should be 2–4× hypergraph convolution baseline, comparable to ConxGNN. |
| *"Per-utterance scale attention sounds like a fancy gating mechanism"* | It is. The contribution is the **frequency-aware** gating, not the gating itself. |

### 2.5 Expected magnitude of gain

Wavelet-based methods in node classification (e.g., GraphWaveNet) typically give +1–3% accuracy over single-band counterparts. For MERC, expect **+0.7 to +1.8 WF1 on IEMOCAP** (vs M3Net-RoBERTa baseline), **+0.3 to +1.0 on MELD**. Magnitude is modest because M3Net already gets the major benefit from binary frequency split.

---

## 3. ADMR-MERC: Anchor-Debiased Multi-Relational Graph for MERC

### 3.1 Motivation

The three most empirically-significant unsolved problems in MERC — particularly on MELD — are **orthogonal in cause but each is attacked alone**:

| Problem | Best existing solution | Limitation of existing solution |
|---|---|---|
| **Modality dominance** (text overshadows audio/visual) | Ada²I (AFW + AMW + disparity ratio) | Heuristic disparity ratio; only modality balancing |
| **Class imbalance** (MELD's anger=1109 vs disgust=68) | DER-GCN contrastive minority loss; ConxGNN CB-focal | Doesn't address modality or similar-emotion confusion |
| **Similar-emotion confusion** (happy↔excited, anger↔frustrated) | AR-IIGCN adversarial; EACL anchor-contrastive | Doesn't address modality dominance or imbalance |

**No paper combines all three.** This is not because the combination is conceptually hard (each is a drop-in module) but because the field favors single-innovation papers. There is genuine value in a **carefully-ablated, well-tuned combined system** that pushes MELD numbers above 67 WF1.

This proposal is the **least architecturally novel but most empirically valuable** of the three.

### 3.2 Architecture overview

```
[Encoders — same as Proposal 1]
        ↓
[Modality-Specific Subspace Projection]
   - Per-modality FFN to align dimensions
   - Ada²I-style AFW reweighting BEFORE graph
        ↓
[Multi-Relational Graph Construction]
   - 4 edge types:
     R1: temporal adjacency (±k window)
     R2: same-speaker (intra-modal)
     R3: same-utterance (cross-modal, M×M complete)
     R4: emotion-shift candidate edge (NEW)
        ↓
[Relation-Aware GNN]
   - RGCN-style with 4 relation matrices
   - Layer-by-layer alternating: R3 (cross-modal) then R1+R2 (intra-modal)
        ↓
[Ada²I AMW: Adaptive Modality Weighting]
   - Per-modality logit re-weighting
   - Disparity-ratio-controlled training
        ↓
[Late Fusion: Concatenation + MLP]
        ↓
[Multi-component Loss]:
   L = L_CE
       + λ₁ · L_EACL (anchor-contrastive for similar emotions)
       + λ₂ · L_CB-focal (class-balanced focal for imbalance)
       + λ₃ · L_disparity (Ada²I modality balance penalty)
```

### 3.3 Module-by-module design

#### (a) Modality encoders — same as Proposal 1.

#### (b) Ada²I-style Adaptive Feature Weighting (AFW) before graph

For each modality η ∈ {T, A, V}, learn a **per-feature gate**:

$$g^\eta = \sigma(W_g^\eta \cdot \text{pool}(H^\eta))$$

$$\tilde{H}^\eta = g^\eta \odot H^\eta$$

This down-weights uninformative feature dimensions per modality. Ada²I demonstrated this improves modality balance.

**Alternative**: Learnable orthogonal projection per modality (GraphCFC subspaces). Adds parameters; AFW is simpler.

#### (c) Multi-relational graph construction (core innovation)

Four edge types:
- **R1**: temporal adjacency edges (i ↔ j if |i-j| ≤ k=10)
- **R2**: same-speaker edges (intra-modal)
- **R3**: same-utterance cross-modal edges (text-audio, audio-visual, text-visual of same i)
- **R4**: emotion-shift candidate edges — **new** — edge between (i, j) if the *sentiment polarity* of utterance i differs from utterance i-1 (using a lightweight pre-trained sentiment classifier like DistilBERT-SST or VADER on the text modality)

R4 is the **only architectural novelty** in this proposal. It explicitly encodes the "emotion shift" location into the graph topology, complementing GraphSmile's SDP auxiliary task but without requiring sentiment labels for training (R4 uses zero-shot sentiment of text only).

**Alternatives considered for R4:**

| Alternative for emotion-shift representation | Why R4 wins |
|---|---|
| **GraphSmile's SDP auxiliary task** | Requires sentiment ground truth for SDP loss; R4 needs only zero-shot sentiment for edge construction |
| **GAT-CRESA's learned emotion-shift labels** | More complex; supervision-heavy |
| **Explicit emotion shift loss** | Doesn't change graph topology, just objective; R4 changes both |

#### (d) Relation-aware GNN backbone

RGCN-style message passing:

$$h_i^{(l+1)} = \sigma\left(\sum_r \sum_{j \in N_i^r} \frac{1}{|N_i^r|} W_r^{(l)} h_j^{(l)} + W_0^{(l)} h_i^{(l)}\right)$$

with 4 learnable relation matrices W_r.

**Layer-by-layer alternation** (per GraphSmile): even layers aggregate cross-modal (R3), odd layers aggregate intra-modal (R1+R2+R4). Avoids fusion conflict per GraphSmile's argument.

**Alternative considered**: Heterogeneous GAT (with per-relation attention). More expressive but 4× parameters vs. RGCN. RGCN is more parameter-efficient on small graphs.

#### (e) Ada²I-style Adaptive Modality Weighting (AMW)

After graph + per-modality readout, compute **per-modality confidence** then re-weight logits:

$$w^\eta = \text{softmax}(\text{MLP}([\hat{H}^T, \hat{H}^A, \hat{H}^V]))$$

$$\hat{y} = \sum_\eta w^\eta \cdot \text{cls}(\hat{H}^\eta)$$

#### (f) Multi-component loss

```
L = L_CE
    + λ_1 · L_EACL          (anchor-contrastive, similar-emotion separation)
    + λ_2 · L_CB-focal      (class-balanced focal, MELD imbalance)
    + λ_3 · L_disparity     (Ada²I modality balance penalty)
    + λ_4 · L_R4            (auxiliary edge-prediction loss for R4 robustness)
```

**Critical hyperparameter consideration**: tuning four loss weights is hard. Use **uncertainty-weighted multi-task** (Kendall et al. 2018, *Multi-task learning using uncertainty to weigh losses*) to learn λ_i automatically.

**Alternative loss configurations:**
- Drop L_EACL → loses similar-emotion gain
- Drop L_CB-focal → MELD MF1 (macro-F1) plummets on minority classes
- Drop L_disparity → modality dominance returns (text dominates 0.6–0.7 of logit weight)
- Drop L_R4 → R4 edges may be ignored by RGCN; auxiliary loss forces them to carry signal

### 3.4 Self-review

#### Strengths
- **Attacks the three most-cited MERC failure modes simultaneously** — modality dominance, class imbalance, similar-emotion confusion.
- **Highest expected practical impact** of the three proposals: each ingredient is proven; their interaction is plausibly additive.
- **Most directly transferable to industrial settings** — no exotic mechanisms (no causal inference, no wavelets).
- **R4 (emotion-shift edges) is a small but meaningful architectural addition** — uses zero-shot sentiment to anchor topology to emotional discontinuities.

#### Weaknesses
- **Lowest architectural novelty** of the three proposals. Risk of being characterized as "method-engineering" rather than research contribution.
- **Hyperparameter explosion**: 4 loss weights + AFW/AMW gates + 4 relation types + window k + EACL anchor count → ~10 hyperparameters. Tuning burden high.
- **Tight coupling to existing methods**: this paper's success depends on three other groups' methods (Ada²I, EACL, ConxGNN) being correctly implemented and integrated.
- **R4 depends on quality of zero-shot sentiment** — failures of DistilBERT-SST propagate into graph topology.

#### Assumptions
- That the three problems are **truly orthogonal** (no destructive interaction between gains). Plausible but must be ablated rigorously.
- That uncertainty-weighted loss balancing converges; for 4 loss terms this is not guaranteed.
- That AMW doesn't simply learn `w_T = 1, w_A = w_V = 0` (collapse to text-only). Disparity penalty must be strong enough to prevent this.

#### Likely failure cases
- **AMW collapse to text-only**: if disparity penalty is too weak, AMW just learns "text is best, ignore others" — undoes the modality balancing entirely. Must verify modality weights remain non-trivial.
- **EACL anchors collapse** to dataset-level emotion means: similar-emotion separation only works if anchors stay distinguishable. Add an anchor-separation regularizer to prevent.
- **L_disparity dominates training** if λ_3 is too large: model maximizes balance at expense of accuracy.

#### Anticipated reviewer criticisms

| Criticism | Defense |
|---|---|
| *"This is just combining existing methods. Where's the novelty?"* | Acknowledge candidly. Frame the contribution as: **(1) showing these three problems can be attacked simultaneously without destructive interaction** (non-obvious; requires careful loss balancing), **(2) the R4 emotion-shift edge type** as a small but meaningful innovation, **(3) the most thorough ablation in MERC literature** demonstrating each ingredient's marginal contribution. |
| *"Many hyperparameters — how robust is this?"* | Show hyperparameter sensitivity sweeps; demonstrate stability via uncertainty-weighted balancing. |
| *"R4 depends on external sentiment classifier — what about errors?"* | Auxiliary L_R4 loss makes R4 edges robust to ~20% noise (similar to noisy-edge results in graph contrastive literature). |
| *"AMW could collapse to text-only"* | Show w_T, w_A, w_V remain in [0.2, 0.6] range with proper disparity penalty (per Ada²I's own results). |
| *"How does this compare to just using a bigger backbone (e.g., GraphTransformer)?"* | Run parameter-matched comparison. Should win on the three target metrics (modality dominance reduction, minority class F1, similar-emotion separation) while not necessarily winning on overall WF1. |

### 3.5 Expected magnitude of gain

Each ingredient adds +0.3 to +1.0 WF1 in published ablations. With three orthogonal ingredients, expect roughly **+1.5 to +3.0 WF1 on MELD** (where all three problems are most acute), **+0.8 to +1.8 WF1 on IEMOCAP** (where imbalance and modality dominance are smaller).

The MELD-specific gain is the more important sell — it would push GraphSmile's 66.71 toward 68–69, approaching HRG-SSA territory without requiring T5 generation.

---

## 4. Honorable Mentions / Future Directions

These are sketched directions worth considering but not fully designed:

### 4.1 S-OMERC: Streaming Online MERC with Bounded Graph Buffer

- **Motivation**: All current graph MERC methods are offline (use future context). Real applications (live chatbots, customer service, mental health) need online inference.
- **Sketch**: Maintain a bounded buffer (K=20–50 utterances). New utterances added incrementally; old utterances compressed into a learned summary node (Compressive Transformer-style). Graph backbone is **SSM (Mamba)** over the graph, not just over the sequence — i.e., Mamba aggregates over graph neighbors, not just temporal neighbors.
- **Why not a full proposal**: streaming MERC benchmarks don't exist; would require creating a new evaluation protocol. Risky for a single paper. Would suit a workshop paper or applied venue first.

### 4.2 LLM-SAG: LLM-Distilled Sentiment-Aware Graph

- **Motivation**: GraphSmile's SDP module is bottlenecked by noisy ground-truth sentiment labels. LLMs (Emotion-LLaMA, GPT-4) can provide higher-quality utterance-level sentiment via zero-shot or few-shot inference.
- **Sketch**: Use Emotion-LLaMA as a teacher to label sentiment for all training utterances. Train a GraphSmile-style architecture with SDP loss against distilled sentiment labels.
- **Why not a full proposal**: dependent on which LLM is used; risks looking like "LLM teaching helps everything" which is a tired result by 2026.

### 4.3 MR-MERC: Missing-Modality Robust MERC

- **Motivation**: Graph methods generally collapse when audio or video is missing. GCNet (Lian et al. 2023) addresses this but isn't widely adopted.
- **Sketch**: Train with random modality dropout; use learned "modality-absent" embedding (Lian-style) for missing modalities; combine with Ada²I modality balancing.
- **Why not a full proposal**: missing-modality MERC is a sub-niche; could be a complementary ablation in Proposals 1–3 rather than a standalone paper.

### 4.4 CGD-MERC: Causal Graph Discovery for MERC

- **Motivation**: True causal discovery (NOTEARS, DAGMA) is more principled than CIEI-MERC's pseudo-causal CAD.
- **Sketch**: Apply differentiable DAG learning to discover causal structure across utterances; constrain edges to respect causal ordering.
- **Why not a full proposal**: DAGMA-style methods have unresolved scalability issues; emotion contagion is plausibly **cyclic** (mutual escalation), violating DAG assumption.

---

## 5. Comparative Summary

| Aspect | Proposal 1: CIEI-MERC | Proposal 2: WaveHyp-MERC | Proposal 3: ADMR-MERC |
|---|---|---|---|
| **Type of novelty** | Conceptual (causal edges) | Methodological (multi-band frequency) | Empirical (orthogonal problem combo) |
| **Implementation complexity** | Medium | Medium-High | Medium |
| **Compute cost vs SoTA** | 1.2–1.5× | 2–4× | 1.3–1.6× |
| **Expected IEMOCAP gain** | +0.5 to +1.5 WF1 | +0.7 to +1.8 WF1 | +0.8 to +1.8 WF1 |
| **Expected MELD gain** | +0.3 to +0.8 WF1 | +0.3 to +1.0 WF1 | +1.5 to +3.0 WF1 |
| **Novelty axis is** | Causal/counterfactual edge inference | Graph wavelet decomposition | Joint solution to 3 problems |
| **Strongest target venue** | EMNLP / NAACL | ICML / NeurIPS / ICASSP | ACM MM / TAFFC |
| **Reviewer risk** | "Is it truly causal?" | "Wavelet GNNs aren't new" | "Just combining existing methods" |
| **Practical impact** | Medium | Medium | High |
| **Architectural novelty** | High | Medium | Low |

### 5.1 Recommended priority

For a PhD student with **1–2 years of remaining time** and a **strong publication target**:

1. **Highest expected publication probability**: **Proposal 1 (CIEI-MERC)** — strongest novelty axis (causal-flavored edge inference is meaningful in 2025/2026 MERC), feasible implementation, modest compute cost. EMNLP/NAACL likely.

2. **Highest expected practical impact** (and best if industry-facing): **Proposal 3 (ADMR-MERC)** — boosts MELD from 66 → ~68 WF1, attacks real deployment issues. ACM MM / TAFFC likely.

3. **Best if combined with a strong theoretical background in signal processing**: **Proposal 2 (WaveHyp-MERC)** — most rigorous methodologically, but reviewer pool for graph wavelet + MERC is narrow. ICASSP / Neurocomputing likely.

If the budget allows **two papers**: do **CIEI-MERC first** (higher novelty risk to establish quickly) then **ADMR-MERC** (lower-risk follow-up demonstrating mature engineering).

---

## 6. Implementation Considerations

### 6.1 Datasets and benchmarks
- **Primary**: IEMOCAP (6-way classification; sessions 1–4 train, session 5 test) and MELD (standard split).
- **Secondary**: CMU-MOSEI (CORECT/Ada²I evaluate here) as a complementary benchmark; useful especially for Proposal 1's causal edges since MOSEI has longer monologues.
- **For Proposal 1**: also report on a **subset of IEMOCAP filtered for explicit emotional contagion** (manually identify dialogues where one speaker's emotion clearly drives the other's). Strengthens the causal-edge story.

### 6.2 Reproducibility hygiene
- Use the **HRG-SSA author's feature pipeline** (T5-base for text, OpenSmile for audio, DenseNet/3D-CNN for visual) for ALL baseline comparisons — eliminates the feature-extractor confound that plagues the field (see corrected survey, §Notes on Reproducibility).
- Report **5-seed mean ± std**, not single-run best.
- Always report **macro-F1 alongside weighted-F1** on MELD — weighted-F1 is dominated by majority classes and hides minority-class failures.
- Release implementation with exact hyperparameter configs, not just final numbers.

### 6.3 Ablation strategy
- **Per-module ablation** for each proposal (drop one component at a time, measure WF1 drop).
- **Synthetic-data probes**: for Proposal 1, generate synthetic dialogues with controlled emotional contagion vs. independence; verify CAD edges capture contagion but not independence.
- **Per-emotion-class breakdown** (especially MELD): show whether gains are concentrated in minority classes or distributed.
- **Modality-ablation**: report T-only, T+A, T+V, T+A+V configurations to verify modality balance.

### 6.4 Compute budget
- IEMOCAP: ~6h training on 1× RTX 3090 / A6000 for all proposals.
- MELD: ~12h training on 1× A6000 for Proposals 1, 3; ~24h for Proposal 2 (wavelet cost).
- Total experimental budget: 200–400 GPU-hours per proposal including ablations.

### 6.5 Risks and contingencies
- **Risk**: Proposal 1's CAD doesn't beat plain attention. **Contingency**: pivot to a hybrid (CAD + self-attention) and frame as "complementary signals". Document the negative result honestly.
- **Risk**: Proposal 2's wavelet adds compute but no measurable gain. **Contingency**: reduce to 2 scales, present as "principled binary frequency split with theoretical grounding".
- **Risk**: Proposal 3's loss balancing fails to converge. **Contingency**: use uncertainty-weighted multi-task balancing (Kendall et al. 2018); if still fails, drop one component (most likely L_disparity if AMW already provides modality balance).

---

## 7. Concluding Notes

Each proposal builds on **identified, named limitations** of existing methods rather than on speculation:
- CIEI-MERC addresses HRG-SSA's acknowledged "similarity-based implicit edges" limitation.
- WaveHyp-MERC addresses M3Net's and GS-MCC's binary frequency simplification.
- ADMR-MERC addresses the documented orthogonal-problem gap (no paper attacks modality + imbalance + similar-emotion simultaneously).

A meta-observation worth keeping in mind: the **most cited MERC papers of 2023–2025 (M3Net, HRG-SSA, GraphSmile) each have exactly one clearly-articulated innovation**. Resist the temptation to combine all three proposals into one super-architecture — reviewers consistently reject "kitchen sink" MERC submissions in favor of single-axis innovations with rigorous ablation.

The field is maturing. Strong contributions in 2026 will likely come from:
1. **New conceptual angles** (causal, counterfactual, theory-grounded) — what CIEI attempts.
2. **Principled multi-band frequency or spectral analysis** — what WaveHyp attempts.
3. **Empirically rigorous combinations** with thorough ablation — what ADMR attempts.
4. **Out-of-distribution settings**: missing modalities, streaming, multilingual, cross-cultural — what the honorable mentions point at.

Pure architectural complexity has plateaued; the next 1–2 WF1 points on IEMOCAP/MELD will come from inductive biases, not bigger networks.

---

*This proposal is grounded in the corrected survey of 27+ graph-based MERC papers (IEMOCAP/MELD coverage), with all design choices traceable to published findings. Each proposal is sized for a single PhD-level project deliverable (a top-venue paper). Implementation, ablation, and risk plans are outlined to support a 6–9 month project timeline per proposal.*
