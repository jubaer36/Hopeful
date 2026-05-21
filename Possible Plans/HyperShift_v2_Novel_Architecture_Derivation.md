# A Novel Graph-Based Architecture for MERC — Module-by-Module Derivation

**Companion to:** `HyperShift_v2_BMVC_Research_Plan.md`, `HyperShift_v2_Breakthrough_Analysis.md`
**Date:** May 17, 2026
**Goal:** Derive a single, defensible, reasonably novel graph-based architecture by evaluating 2–4 evidence-backed options per module, then re-checking the synthesized whole.

---

## 0. Design Principles (Before Anything Else)

Five constraints that bound every choice below:

1. **Graph must be the protagonist.** This is a graph-based MERC paper. If a non-graph alternative wins a module, it must be one that lives *inside* a graph framework.
2. **Visual modality must be structurally distinguished**, not just one of three equal streams. The whole story is that current MERC under-uses vision.
3. **Differentiate from ConxGNN, HAUCL, GraphSmile, M3Net, DGODE.** Anything that obviously overlaps with these gets rejected.
4. **No more than ~3 new ideas total.** A BMVC paper can defend three contributions. Four becomes a kitchen sink; reviewers attack contribution attribution.
5. **Every choice must be ablatable.** If a component cannot be isolated and removed for an ablation row, it cannot be in the paper.

---

## 1. The Modules

A MERC pipeline has eleven decision points. The architecture is the cross product of choices at each.

| # | Module | Decision |
|---|---|---|
| M1 | Visual encoder | Which feature extractor for the visual stream |
| M2 | Audio encoder | Which feature extractor for the audio stream |
| M3 | Text encoder | Which feature extractor for the text stream |
| M4 | Graph node structure | What does each node represent |
| M5 | Hyperedge construction | How is the hypergraph topology built |
| M6 | Hyperedge weighting | How are hyperedge contributions weighted |
| M7 | Propagation | How does information flow through the graph |
| M8 | Cross-modal interaction | How modalities exchange information |
| M9 | Temporal aggregation | How utterance-level dependencies are captured |
| M10 | Auxiliary objective | What extra supervision shapes the embedding space |
| M11 | Primary loss | How the final classifier is trained |

For each, I evaluate 2–4 candidate options with evidence and assign a verdict.

---

## 2. Module-by-Module Options

### M1 — Visual Encoder

| Option | Evidence | Verdict |
|---|---|---|
| **(a) DenseNet pre-trained on FER+** | GraphSmile baseline; established but from 2017 | 🔴 Stale; everyone uses this; no contribution |
| **(b) CLIP-ViT-L/14** | Standard modern vision backbone; used in many 2024 multimodal works | 🟡 Better than DenseNet but predictable choice |
| **(c) SigLIP 2 + OpenFace 2.0 AUs (dual stream)** | SigLIP 2 (Tschannen et al., Feb 2025) outperforms CLIP on all downstream vision benchmarks; AUs give structured facial features that no MERC method uses today | 🟢 Modern semantic features + structured facial cues |
| **(d) EmoCLIP + AUs** | EmoCLIP (Bondi et al., BMVC 2024) is CLIP fine-tuned for emotion | 🟡 Strong but narrower; loses the "modern general backbone" framing |

**Chosen: (c) SigLIP 2 + OpenFace 2.0 AUs.** Two reasons. (1) SigLIP 2 is the strongest open vision backbone as of Feb 2025; using it is defensible as "current state of the art". (2) OpenFace AUs give the model a *structured* visual signal that none of the competitors have — and that structure is what enables module M6 (AU-conditioned weighting) to be genuinely novel.

---

### M2 — Audio Encoder

| Option | Evidence | Verdict |
|---|---|---|
| **(a) OpenSMILE IS10 features** | GraphSmile baseline; hand-crafted from 2010 | 🔴 Stale |
| **(b) WavLM-Large** | SOTA speech SSL backbone (Chen et al., 2022); strong on SER benchmarks | 🟢 Standard modern choice |
| **(c) HuBERT-Large** | Comparable to WavLM | 🟡 Either works; WavLM has slightly more SER citations |
| **(d) Whisper encoder** | Strong but trained on ASR, not emotion | 🔴 Wrong objective for SER |

**Chosen: (b) WavLM-Large.** Uncontroversial modern choice. The audio side is not where the contribution lives.

---

### M3 — Text Encoder

| Option | Evidence | Verdict |
|---|---|---|
| **(a) RoBERTa-Large** | GraphSmile baseline; still strong | 🟢 Keep — matching baseline avoids "you changed text encoder" criticism |
| **(b) DeBERTa-v3** | +0.3 typical gain | 🔴 Tiny benefit; looks like padding |
| **(c) Frozen LLM features (Llama-3)** | Strong but expensive; opens unrelated debate | 🔴 Distracts from graph contribution |

**Chosen: (a) RoBERTa-Large.** Same as every recent MERC paper. The text side is deliberately conservative — the paper is a vision/graph contribution, not a text contribution.

---

### M4 — Graph Node Structure

| Option | Evidence | Verdict |
|---|---|---|
| **(a) Modality nodes only (per-modality, per-utterance)** | GraphSmile, ConxGNN, M3Net, MMGCN — universal in MERC | 🟡 Standard; no novelty |
| **(b) Fused utterance nodes (one per utterance after early fusion)** | Loses fine-grained cross-modal control | 🔴 Throws away the modality structure that vision-anchoring needs |
| **(c) Modality nodes + speaker meta-nodes** | Speaker meta-nodes are common in heterogeneous graph literature; no MERC method has explicitly used them (qmask is loaded but ignored, even by GraphSmile) | 🟢 Adds a structural element that consumes qmask properly |
| **(d) Modality nodes + speaker meta-nodes + emotion prototype nodes** | Prototype-based classification is well-established (TSC, NCM); no MERC method uses prototypes as graph nodes | 🟡 Genuinely novel but adds complexity; risk of underfitting on small classes |

**Chosen: (c) Modality nodes + speaker meta-nodes.** Option (d) is more novel but adds a second axis of contribution that competes with the causal-debiasing story (M10). Option (c) is *enough* novelty — no published MERC method explicitly puts speaker nodes in the graph — and it keeps the architecture defensible. Save (d) as a future-work extension.

**Concrete spec.** For a dialogue of M utterances with K speakers: 3M modality nodes (t, v, a per utterance) + K speaker meta-nodes. Total ~3M + K nodes. For IEMOCAP dyadic dialogues, K=2, so this adds only 2 nodes — negligible memory cost, but structurally meaningful.

---

### M5 — Hyperedge Construction

| Option | Evidence | Verdict |
|---|---|---|
| **(a) Pairwise + window-based edges** | GraphSmile's three pairwise modality graphs | 🔴 Reproduces GraphSmile; not novel |
| **(b) Static triadic + speaker + temporal hyperedges** | Triadic hyperedge per utterance = HAUCL-adjacent but cleaner; speaker hyperedge per speaker; temporal-window hyperedge per local context | 🟢 Three orthogonal sources of structure; ablatable |
| **(c) VHGAE-learned hyperedges** | HAUCL (ACM MM 2024) does this | 🔴 Directly overlaps with HAUCL |
| **(d) AU-similarity-driven dynamic hyperedges** | Utterances with co-activated AUs share a hyperedge; novel | 🟡 Interesting but introduces a threshold hyperparameter; partly redundant with M6 |

**Chosen: (b) Static triadic + speaker + temporal hyperedges.** The dynamism that HAUCL gets from VHGAE structure-learning, we get from M6's weighting. Keeping the *structure* static and the *weights* dynamic is a cleaner contribution claim: it tells reviewers exactly what is learned and what is fixed.

**Concrete spec.**
- **Triadic hyperedge `e_uni_i`** per utterance `i`: `{n_t_i, n_v_i, n_a_i}` — connects the three modality nodes of a single utterance.
- **Speaker hyperedge `e_spk_k`** per speaker `k`: connects all modality nodes belonging to utterances spoken by `k`, *plus* the speaker meta-node `s_k`.
- **Temporal hyperedge `e_tmp_(i,W)`** per utterance window of size `W`: connects all 3W modality nodes in the window.

Total: M + K + M hyperedges = 2M + K hyperedges. Manageable.

---

### M6 — Hyperedge Weighting

| Option | Evidence | Verdict |
|---|---|---|
| **(a) Uniform (W=I)** | HGNN baseline | 🔴 Loses the per-edge importance signal |
| **(b) Learned scalar per edge type** | Three scalars: w_uni, w_spk, w_tmp | 🟡 Too coarse — same weight regardless of content |
| **(c) AU-conditioned MLP, dual-granular (MGA-HHN style)** | Node-level attention (within-hyperedge contribution) + hyperedge-level gating (across edge types); AU vector drives the gating | 🟢 Inspired by MGA-HHN (arXiv:2505.04340), specialized via visual content; this is the headline contribution |
| **(d) Multi-head attention over node features** | More parameters, less interpretable | 🟡 Possible but heavier |

**Chosen: (c) AU-conditioned MLP with dual-granular attention.** This is the key novel mechanism. Two specifications:

**Node-level (within-hyperedge):**
```
α_{n,e} = softmax_n( MLP_node([h_n, AU_n]) )    for each node n ∈ e
```
Each modality node in a hyperedge gets a softmax-normalized weight conditioned on its current representation and (if it's a visual node) its AU vector. Text and audio nodes use a zero AU vector — so the AU signal only flows in when visual content is present.

**Hyperedge-level (across edge types):**
```
β_e = sigmoid( MLP_edge([avg(h_n for n ∈ e), AU_summary_e, edge_type_embed]) )
```
Each hyperedge gets a gating scalar that decides how much it contributes to propagation. The gating sees a summary of AU activations across the hyperedge — so a triadic hyperedge where the speaker's face shows strong AU4+AU15 (sadness markers) gets a stronger weight than one where AUs are neutral.

**Why this is genuinely novel.** ConxGNN's hypergraph weighting is content-agnostic. HAUCL's is structure-level (reconstruct hyperedges) not weight-level. No published MERC method uses facial-AU activations to drive hyperedge weights. This is the cleanest defensible novelty in the entire pipeline.

---

### M7 — Propagation

| Option | Evidence | Verdict |
|---|---|---|
| **(a) Standard HGNN+ low-pass smoothing** | Gao et al. TPAMI 2022; what HAUCL uses | 🟢 Well-understood and stable |
| **(b) M3Net-style multi-frequency** | High-frequency component for sentiment-shift discrimination | 🟡 Already in M3Net; overlap risk |
| **(c) DGODE-style continuous propagation** | COLING 2025 | 🔴 Already a published MERC method; risky to copy |
| **(d) Heterogeneous Graph Transformer attention** | HGT (Hu et al., WWW 2020) | 🟡 Parameter-heavy on small graphs |

**Chosen: (a) Standard HGNN+ propagation.** With residual connections, as GraphSmile shows residuals prevent over-smoothing up to ~7 layers. The novelty is in M6 (weighting), not in M7 (propagation). Reusing the standard formulation keeps the contribution tight and reviewers can match-check the math directly.

**Concrete spec.**
```
H^(l+1) = σ( D_v^(-1/2) H W^(l) D_e^(-1) H^T D_v^(-1/2) X^(l) Θ^(l) ) + X^(l)
```
where `W^(l) = diag(β_e^(l))` from M6 and `H` is the incidence matrix. Residual connection added per GraphSmile's finding.

---

### M8 — Cross-Modal Interaction

| Option | Evidence | Verdict |
|---|---|---|
| **(a) Implicit via hypergraph (triadic edges)** | The triadic hyperedge already forces T-V-A interaction | 🟢 Built-in; no extra module |
| **(b) Explicit pairwise cross-attention before graph** | MulT-style | 🔴 Competes with graph premise |
| **(c) MBT bottleneck tokens** | NeurIPS 2021; latent fusion units | 🟡 Adds parameters and another loss path |
| **(d) Vision-anchored contrastive alignment** | Pulls T-V and A-V pairs together with shared emotion label | 🟢 Loss-level enforcement of vision-anchoring |

**Chosen: (a) + (d) — the triadic hyperedge handles structural cross-modal interaction, contrastive alignment handles representation-space cross-modal alignment.** Skip (b) and (c) entirely. The alignment loss is a single line added to the total loss; the triadic hyperedge costs no extra module.

**Concrete spec.**
```
L_align = SupCon(z_v, z_t; y_emo) + SupCon(z_v, z_a; y_emo)
```
with vision as the anchor in both terms. Same emotion label → pull; different → push.

---

### M9 — Temporal Aggregation

| Option | Evidence | Verdict |
|---|---|---|
| **(a) Bidirectional Mamba-2** | Dao & Gu 2024; O(M) memory; clean implementation | 🟢 Modern, fast, well-supported |
| **(b) Transformer encoder** | Standard but O(M²) | 🟡 Fine but unremarkable |
| **(c) Bidirectional LSTM** | Used as a baseline in older MERC papers | 🟡 Keep as baseline ablation |
| **(d) Graph-ODE** | DGODE | 🔴 Already published for MERC |

**Chosen: (a) Bidirectional Mamba-2.** Replaces GraphSmile's SDP module entirely. The Mamba block sits *after* the hypergraph propagation: hypergraph outputs a sequence of M fused utterance representations, Mamba processes them as a sequence, classifier reads each position. O(M) cost, no shift-classifier bookkeeping needed.

**Important framing.** Mamba is *not* presented as a contribution. It's the efficient sequence aggregator. Three Mamba-for-MERC papers were published in 2025 already (Section 1.1 of the previous breakthrough analysis), so the paper says "we use bidirectional Mamba-2 as the temporal aggregator following recent practice [cite]". Acknowledge prior art, move on.

---

### M10 — Auxiliary Objective

| Option | Evidence | Verdict |
|---|---|---|
| **(a) Sentiment shift classification (SDP)** | GraphSmile auxiliary task | 🔴 Already in GraphSmile; not novel |
| **(b) Counterfactual / NDE-debiasing loss** | CLEF (CVPR 2024, arXiv:2403.05963); D2CL (IEEE 2025); CIDer (arXiv:2506.10452). All for image emotion or CAER, none for MERC | 🟢 No MERC paper has done this; directly attacks text-shortcut |
| **(c) Hypergraph reconstruction loss** | HAUCL does this | 🔴 Overlaps with HAUCL |
| **(d) Trajectory continuity regression** | Original HyperShift v1 idea | 🟡 Weaker than (b); less narrative power |

**Chosen: (b) Counterfactual NDE-debiasing.** This is the second novel contribution. Specification:

**Total Effect (TE):** standard forward pass with all three modalities present.
**Natural Direct Effect (NDE):** counterfactual forward pass where vision and audio inputs are replaced with mean-feature placeholders (representing "modality absent"), so only the text-through-graph pathway contributes.

```
L_causal = CE(TE_logits, y) + λ_nde · KL( NDE_logits || U )
```
where `U` is a uniform distribution over emotion classes. The KL term says: "the text-only pathway should *not* be confidently predictive of emotion — if it is, the model is taking the text shortcut and we penalize it". At inference, use `TE_logits` directly (option to subtract `α · NDE_logits` for explicit debiasing, as an ablation).

**Why this fits BMVC.** CLEF, D2CL, and CIDer are all vision/CV-side causal-emotion papers. Bringing the same methodology to conversational MERC is a clean cross-pollination story that BMVC reviewers will recognize as principled rather than ad-hoc.

---

### M11 — Primary Loss

| Option | Evidence | Verdict |
|---|---|---|
| **(a) Cross-entropy** | GraphSmile default | 🔴 Known to underperform on imbalanced classes |
| **(b) Class-Balanced Focal** | Cui et al. CVPR 2019; standard imbalance fix | 🟡 Good but not enough on its own |
| **(c) CB-Focal + Balanced Contrastive Learning (BCL)** | BCL (Zhu et al., CVPR 2022); regular-simplex feature distribution; targets MELD-like long-tail directly | 🟢 Strongest empirical option for class imbalance |
| **(d) CB-Focal + Targeted Supervised Contrastive (TSC)** | Li et al., CVPR 2022; pre-defined hypersphere targets | 🟡 Comparable to BCL; BCL has cleaner theoretical analysis |

**Chosen: (c) CB-Focal + BCL.** This is the third (and quietest) contribution: replacing GraphSmile's plain CE with a proven long-tail-friendly composite loss. The paper does not claim this as a methodological novelty — BCL is two years old — but it is *the* loss recipe that makes the minority-class F1 numbers work, and reviewers will accept it as a principled choice with citation.

**Total loss:**
```
L = L_emo + λ_align · L_align + λ_causal · L_causal + β · ||Θ||²
where  L_emo = CB-Focal(logits, y) + λ_sup · BCL(features, y)
```

Four terms. Each is independently ablatable.

---

## 3. The Synthesized Architecture

Stitching the chosen options together yields the following pipeline. I'll call it **SC-VAH**: **Speaker-Coupled Vision-Anchored Hypergraph** (working name — pick a better one for the paper).

```
                ┌────────────────────────────────────────────┐
                │ Modality Encoders                          │
                │  Text  → RoBERTa-Large            (frozen) │
                │  Video → SigLIP 2 + OpenFace AUs  (frozen) │
                │  Audio → WavLM-Large              (frozen) │
                └────────────────────┬───────────────────────┘
                                     │
                ┌────────────────────▼───────────────────────┐
                │ Per-modality dim-reduction MLPs            │
                │ → h_t_i, h_v_i, h_a_i ∈ R^d  per utterance │
                └────────────────────┬───────────────────────┘
                                     │
                ┌────────────────────▼───────────────────────┐
                │ Build hypergraph                           │
                │   Nodes: 3M modality nodes + K speaker     │
                │          meta-nodes                        │
                │   Hyperedges:                              │
                │     • Triadic per utterance  (M edges)     │
                │     • Speaker per speaker    (K edges)     │
                │     • Temporal-window        (M edges)     │
                └────────────────────┬───────────────────────┘
                                     │
                ┌────────────────────▼───────────────────────┐
                │ Dual-granular AU-conditioned weighting     │
                │   α_{n,e} = node-level softmax (with AU)   │
                │   β_e     = hyperedge-level gate (with AU) │
                └────────────────────┬───────────────────────┘
                                     │
                ┌────────────────────▼───────────────────────┐
                │ HGNN+ propagation, L layers, with residual │
                │   H^(l+1) = σ(D_v^-½ H W^(l) D_e^-1 ...    │
                │            ... H^T D_v^-½ X^(l) Θ^(l))     │
                │            + X^(l)                         │
                └────────────────────┬───────────────────────┘
                                     │
                ┌────────────────────▼───────────────────────┐
                │ Pool to per-utterance features             │
                │ Bidirectional Mamba-2 over M utterances    │
                └────────────────────┬───────────────────────┘
                                     │
                ┌────────────────────▼───────────────────────┐
                │ Classifier head → logits per utterance     │
                └────────────────────────────────────────────┘

Training signals:
  • L_emo     = CB-Focal + λ_sup · BCL                  (primary)
  • L_align   = SupCon(z_v,z_t) + SupCon(z_v,z_a)       (vision anchor)
  • L_causal  = CE(TE) + λ_nde · KL(NDE || U)           (causal debiasing)
  • L2 reg
```

**Three contributions, exactly:**

1. **Dual-granular AU-conditioned hypergraph weighting** (M5 + M6). The graph topology consumes facial action units to weight cross-modal and inter-utterance information flow. No prior MERC method conditions graph structure on facial AUs.

2. **Counterfactual debiasing of the text shortcut** (M10). NDE-style auxiliary loss penalizes the model when text alone is enough to predict the emotion. First application of causal inference to MERC.

3. **Speaker meta-nodes that consume `qmask` structurally, not as concatenated features** (M4 + M5). The hypergraph's speaker hyperedges + speaker meta-nodes give the model a principled way to use speaker identity. Prior MERC models load `qmask` but ignore it in graph topology.

Everything else (modern feature backbones, Mamba-2, BCL loss, SupCon alignment) is engineering hygiene with citations.

---

## 4. Reiteration — Is Each Module Actually Suitable?

Now I re-check the chosen architecture against the design principles from Section 0.

### 4.1 Module-by-module suitability audit

| Module | Choice | Suits the contribution? | Risk |
|---|---|---|---|
| M1 Visual | SigLIP 2 + AUs | Yes — AUs are required by M6 | If SigLIP 2 features are not better than DenseNet on emotion separability, the "modern backbone" story weakens. **Mitigation:** verify in Week 3 of the roadmap with a feature-separability test. |
| M2 Audio | WavLM-Large | Yes — modern, neutral, doesn't compete | Negligible |
| M3 Text | RoBERTa-Large | Yes — same as baseline | Negligible |
| M4 Nodes | Modality + speaker meta-nodes | Yes — enables M5 speaker hyperedges | Speaker meta-nodes may be underused if K=2 (IEMOCAP). **Mitigation:** ablate by removing speaker meta-nodes; if MELD (more speakers) shows bigger gain than IEMOCAP, that supports the claim. |
| M5 Hyperedges | Triadic + speaker + temporal | Yes — three orthogonal structural sources | Risk that the three edge types' contributions overlap. **Mitigation:** three-way ablation table. |
| M6 Weighting | Dual-granular AU-conditioned | **This is the contribution** | Heaviest implementation risk. AU vectors must be valid. **Mitigation:** verify OpenFace 2.0 outputs on a sample of utterances before integration. |
| M7 Propagation | HGNN+ with residual | Yes — well-understood baseline | None unless we go too deep; cap at 7 layers per GraphSmile's finding |
| M8 Cross-modal | Triadic hyperedge + L_align | Yes — no extra module | If L_align dominates training, the hypergraph contribution is masked. **Mitigation:** small λ_align (start with 0.3) and ablate. |
| M9 Temporal | Bidirectional Mamba-2 | Yes — efficient, modern, non-headline | None; well-supported implementation |
| M10 Auxiliary | NDE counterfactual | **This is the contribution** | Counterfactual forward pass may be unstable to train. **Mitigation:** warm-up schedule; freeze backbone for first 5 epochs; small λ_nde initially. |
| M11 Loss | CB-Focal + BCL | Yes — directly fixes minority-class | None — BCL is well-tested |

### 4.2 Coherence check

**Does the contribution-set hang together as one paper?**

- C1 (AU-conditioned hypergraph) and C2 (causal debiasing) attack the same underlying problem from two directions: text-dominance over vision. C1 says "use vision structurally"; C2 says "penalize models that ignore vision". They are complementary, not redundant.
- C3 (speaker meta-nodes) is a smaller structural contribution that supports C1 by enriching the hypergraph. Reviewers may or may not accept it as a "contribution" — it can also be presented as part of C1's design.

**Is there overlap with published work?**

| Published Work | Their Contribution | Our Differentiator |
|---|---|---|
| GraphSmile (TPAMI 2025) | Three pairwise modality graphs + SDP shift task | We use one hypergraph, no SDP, AU-conditioned weighting, causal loss |
| ConxGNN (ICASSP 2025) | Hypergraph + multi-scale graph + re-weighting | They don't use AUs for weighting; no causal training; no speaker hyperedges |
| HAUCL (ACM MM 2024) | Dynamic hyperedge structure via VHGAE + contrastive | We keep structure static, dynamize weights via AUs; no VAE machinery; causal loss |
| M3Net (CVPR 2023) | Multi-frequency graph propagation | Different propagation (we use standard HGNN+); we use AUs for weighting instead of frequency decomposition |
| DGODE (COLING 2025) | Graph ODE temporal evolution | We use Mamba-2 (different paradigm); also we contribute via M6/M10, not propagation |

The differentiation is clean.

### 4.3 Three-axis novelty assessment

For a BMVC paper, novelty is judged along: **method**, **insight**, **systems**.

- **Method:** AU-conditioned dual-granular hyperedge weighting is genuinely new. ✓
- **Insight:** Text-shortcut as a causal-debiasing problem in MERC is a fresh framing. ✓
- **Systems:** Modern visual backbone (SigLIP 2) + AU stream + Mamba-2 is a reasonable systems update for the MERC field, which has been recycling 2018-era features. ✓

All three axes carry weight, but the paper's headline rests on **Method** (M6) and **Insight** (M10). The systems aspect is supporting evidence.

### 4.4 What could kill this paper

Six failure modes ranked by likelihood:

1. **AU features turn out to be too sparse or unreliable on MELD.** MELD's TV-show videos have occlusion, fast cuts, and group shots. If OpenFace 2.0 fails on >20% of utterances, M6 cannot use them as designed. **Mitigation:** test on a MELD subset in Week 3; if AU coverage is bad, switch to SigLIP 2 attention-pooled features as the conditioning signal instead.
2. **SigLIP 2 features don't help over CLIP / DenseNet.** Possible — emotion may not benefit from general visual semantic features. **Mitigation:** verify in Week 3 with a simple linear-probe on per-frame emotion classification.
3. **Causal NDE branch unstable to train.** Counterfactual forward passes can produce degenerate behavior. **Mitigation:** warm-up + careful λ_nde scheduling + freeze backbone initially.
4. **Speaker hyperedges add zero performance on IEMOCAP (K=2).** Possible because with two speakers, the speaker hyperedge nearly partitions the dialogue. **Mitigation:** position the paper to show MELD/MOSEI gains on speaker hyperedges, not IEMOCAP; this is honest and ablatable.
5. **BCL + CB-Focal interact badly.** Both shape the loss for imbalance; double-correcting could hurt head classes. **Mitigation:** ablation with each alone and combined; choose the winner empirically.
6. **Reviewers see this as ConxGNN++ with a causal twist.** This is the biggest framing risk. **Mitigation:** lead the abstract with the causal/text-shortcut framing, not the hypergraph design. The hypergraph is the implementation; the *insight* is the contribution.

### 4.5 Honest verdict on complexity

The design principle said "no more than ~3 new ideas". The synthesized architecture has:
- 1 structurally novel mechanism (AU-conditioned dual-granular weighting on hypergraph)
- 1 conceptually novel objective (causal NDE debiasing)
- 1 minor structural addition (speaker meta-nodes)
- 3 engineering choices (SigLIP 2, BCL, Mamba-2)

That's exactly at the limit. Anything more would push it into kitchen-sink territory.

The reasonable-and-rational check: **yes**, this is buildable in 4–6 months, has clear ablations, and each contribution is independently meaningful. The dependencies between modules are clean — M6 needs AUs (M1); M10 needs the forward pass infrastructure but is otherwise modular; M11 plugs in independently.

---

## 5. Practical Next Step

If this design survives the reading-week (per the previous roadmap document), the implementation order should be:

1. **Week 3–5:** Build the SigLIP 2 + OpenFace 2.0 feature pipeline. Verify AU coverage on a MELD subset *first* — if it fails, the M6 design changes here, not at week 9.
2. **Week 6–8:** Implement the hypergraph + dual-granular weighting + HGNN+ propagation on cached features. Get to a reproducible baseline number on IEMOCAP-6 *before* adding the causal branch.
3. **Week 9–12:** Add the causal NDE branch. This is the highest-risk component — give it time.
4. **Week 13–16:** Ablate everything (Section 4.4 of this document is a pre-built ablation grid).
5. **Week 17–20:** Write.

That timeline matches the BMVC 2027 / ACM MM 2027 plan from the first document.

---

## 6. One-Sentence Summary

> *We propose a graph-based MERC framework that uses facial action units to drive hypergraph structure (via dual-granular AU-conditioned weighting), explicitly debiases the text-modality shortcut via a counterfactual NDE auxiliary loss, and structurally consumes speaker identity through speaker meta-nodes — three orthogonal contributions targeting the same diagnosis that current MERC models under-use the visual stream.*

That is the paper.
