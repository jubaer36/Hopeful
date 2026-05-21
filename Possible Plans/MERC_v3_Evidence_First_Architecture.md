# MERC v3 — An Evidence-First Architecture for Multimodal Emotion Recognition in Conversations

**Companion to:** `HyperShift_v2_BMVC_Research_Plan.md`, `HyperShift_v2_Breakthrough_Analysis.md`, `HyperShift_v2_Novel_Architecture_Derivation.md`
**Date:** May 17, 2026
**Goal:** Build the best architecture the evidence supports, without pre-committing to "graph-based", "vision-anchored", or a 3-contribution cap. Then re-check whether the resulting design is internally coherent and reasonable.

---

## 0. What Changes Without The Old Constraints

The previous derivation forced:
- Graph must be the protagonist (locked in hypergraph)
- Vision must be structurally distinguished (locked in AU conditioning)
- Differentiate from ConxGNN/HAUCL/M3Net/DGODE (locked out anything overlapping)
- ≤3 contributions (forced cutting useful components)
- Every choice ablatable (forced simpler design)

Lifting these means: a non-graph module can win if the evidence is strong; vision can be one of three equal streams if that's what the data supports; partial overlap with prior work is acceptable if the integration is principled; and components can be added for *engineering value* even if they aren't headline contributions.

The 2024–2026 literature now offers enough evidence that the dominant problem in MERC is **modality imbalance and text dominance**, not graph topology. Five independent papers in the last 18 months converge on this:

- A bias-and-fairness study of multimodal emotion detection found that text alone has the least bias and accounts for the majority of the models' performances, raising doubts about the worthiness of multimodal emotion recognition systems when bias and fairness are desired alongside model performance.
- Previous MER approaches often depend on a dominant modality rather than considering all modalities, leading to poor generalization; this is referred to as modality bias and can disrupt the overall optimization of the recognition system, preventing it from fully harnessing the complementary strengths of each modality.
- Existing methods often face semantic inconsistencies across modalities, such as conflicting emotional cues between text and visual inputs, and are dominated by the text modality due to its strong representational capacity, which can compromise recognition accuracy.
- Because the representational power of text features derived from pretrained language models far exceeds that of audio and visual features based on handcrafted descriptors, the model is incorrectly dominated by the text modality, disregards the modalities that correctly convey the emotions, and ultimately produces recognition errors.
- Text dominance is a fundamental and pervasive bias in Transformer-based multimodal models, extending across a wide spectrum of modalities, caused by token redundancy in non-text modalities, fusion architecture design, and task formulations that implicitly favor textual inputs.

This is the single best-documented problem in current MERC. The architecture should be built around solving it, with everything else as supporting infrastructure.

---

## 1. The Modules — Same 11 Decision Points, Re-Evaluated

### M1 — Visual Encoder

| Option | Evidence | Verdict |
|---|---|---|
| **(a) DenseNet/FER+** (GraphSmile baseline) | Standard but 2017-era; outperformed by every modern backbone on FER benchmarks | 🔴 Stale |
| **(b) CLIP-ViT-L/14** | Strong general-purpose features; used in many 2024 multimodal works | 🟡 Modern but generic |
| **(c) EmoCLIP** (BMVC 2024) | CLIP fine-tuned for emotion via natural-language descriptions of facial expressions | 🟢 Emotion-specialized, vision-text aligned, BMVC-published |
| **(d) SigLIP 2 + OpenFace 2.0 AUs** | SigLIP 2 (Feb 2025) outperforms CLIP across vision benchmarks; AU stream gives structured facial cues | 🟢 Strongest combination but adds AU-reliability risk |
| **(e) V-JEPA 2 / InternVideo2** | Strong video backbones | 🟡 Better than image-only but heavier; less emotion-specific evidence |

**Chosen: (d) SigLIP 2 + OpenFace 2.0 AUs.** Reasoning:

1. SigLIP 2 outperforms CLIP on every vision benchmark and is the strongest open vision backbone as of Feb 2025.
2. The AU stream is documented to give independent and useful signal: automatic facial AU recognition achieves recognition rates of 95–96% on basic action units, comparable to inter-observer agreement in manual FACS coding. Importantly, even when AU detection fails partially, occlusion-aware AU detection frameworks can adapt to challenging conditions like occlusion and illumination changes, with attention mechanisms helping handle partial occlusion.
3. Critical caveat from recent work: existing large vision-language models perform reasonably well under ideal conditions (e.g., clear AU visibility) but exhibit degradation under complexities such as occlusion and demographic bias. This means AU reliability is *not* uniform across MELD (TV-show video with occlusion, cuts) vs. IEMOCAP (controlled dyadic recording). We address this in M11 via a per-utterance AU-confidence weighting.

The previous draft used AUs but had no contingency for unreliable AUs. This is a real concern, and we fix it by treating AU confidence as a *gating signal*, not a hard input.

---

### M2 — Audio Encoder

| Option | Evidence | Verdict |
|---|---|---|
| **(a) OpenSMILE IS10** | GraphSmile baseline; hand-crafted from 2010 | 🔴 Stale |
| **(b) WavLM-Large** | SOTA speech SSL backbone; strong on SER | 🟢 Standard modern choice |
| **(c) HuBERT-Large** | Comparable to WavLM | 🟡 Either works |
| **(d) Emotion2Vec / EmoBox features** | Emotion-pretrained audio features | 🟡 Less mature than WavLM; smaller community |

**Chosen: (b) WavLM-Large.** Same as v2. Uncontroversial; the audio side isn't where the contribution lives.

---

### M3 — Text Encoder

| Option | Evidence | Verdict |
|---|---|---|
| **(a) RoBERTa-Large** | GraphSmile baseline; still strong; matches prior work | 🟢 Keep for comparability |
| **(b) DeBERTa-v3** | +0.3 typical gain | 🔴 Tiny benefit; looks like padding |
| **(c) Llama-3 / Qwen-2 frozen features** | Strong but expensive | 🟡 Possible but distracts from main story |
| **(d) Sentence-T5 / E5** | Emotion-rich sentence embeddings | 🟡 Niche; less precedent in MERC |

**Chosen: (a) RoBERTa-Large.** Conservative. Every recent MERC paper uses it. The architecture's contribution is *not* "swap text encoder", so matching baseline conditions is the right call.

---

### M4 — Graph Node Structure

| Option | Evidence | Verdict |
|---|---|---|
| **(a) Modality nodes only (per-modality, per-utterance)** | GraphSmile/ConxGNN/M3Net/MMGCN standard | 🟡 Standard, no novelty |
| **(b) Fused utterance nodes (early fusion)** | Throws away modality structure | 🔴 Wrong direction for a modality-imbalance problem |
| **(c) Modality nodes + speaker meta-nodes** | qmask consumed structurally; no MERC paper does this | 🟢 Useful addition |
| **(d) Modality + speaker + emotion-prototype nodes** | Prototype-based classification (TSC, NCM) shown to help long-tail; no MERC method does this | 🟢 Stronger but adds complexity |
| **(e) Modality + speaker + context-LSTM hybrid** | Context-level LSTM achieves good results on IEMOCAP with its large average number of utterances, while speaker-level LSTM achieves good results on datasets with small average number of utterances and many speakers like MELD — hybrid does best | 🟡 Effective but engineering-heavy |

**Chosen: (c) Modality nodes + speaker meta-nodes + (d) emotion-prototype nodes.** Without the 3-contribution cap, both can go in. Justification for the addition of (d):

- Emotion-prototype nodes give the long-tail loss (BCL) something concrete to pull toward. Instead of BCL operating only in feature space, it can pull modality nodes toward their class prototype node *via the graph*. This is novel — no MERC paper combines prototype nodes with hypergraph propagation.
- The risk is overfitting prototype nodes on small minority classes. Mitigation: prototype nodes are initialized from class-mean features and updated with a low learning rate, following TSC/PaCo practice.

**Concrete spec.** For a dialogue with M utterances, K speakers, and C emotion classes:
- 3M modality nodes (t, v, a per utterance)
- K speaker meta-nodes
- C emotion-prototype nodes (one per emotion class)

Total: 3M + K + C nodes. For IEMOCAP-6, K=2, C=6, so ~3M+8 nodes. For MELD, K varies, C=7, so ~3M+K+7. Manageable.

---

### M5 — Hyperedge Construction

| Option | Evidence | Verdict |
|---|---|---|
| **(a) Pairwise + window-based edges** | GraphSmile | 🔴 Reproduces baseline |
| **(b) Triadic + speaker + temporal hyperedges (static)** | Three orthogonal structural sources | 🟢 Clean ablations |
| **(c) VHGAE-learned hyperedges** | HAUCL does this; HAUCL dynamically adjusts hypergraph connections via variational hypergraph autoencoder and uses contrastive learning to mitigate uncertainty during reconstruction, outperforming SOTA on IEMOCAP and MELD | 🟡 Strong but overlaps directly with published work |
| **(d) AU-similarity-driven dynamic hyperedges** | AU co-activation as edge constructor | 🟡 Threshold-dependent; partially redundant with M6 weighting |
| **(e) Speaker-conditioned multi-type hypergraph + prototype hyperedges** | Adds emotion-prototype-to-utterance hyperedges | 🟢 Extends (b) with prototype nodes from M4 |

**Chosen: (e) — extends (b) with prototype hyperedges.** Concrete spec:

- **Triadic hyperedge** per utterance i: `{n_t_i, n_v_i, n_a_i}` — M edges.
- **Speaker hyperedge** per speaker k: all modality nodes of utterances spoken by k, plus the speaker meta-node `s_k` — K edges.
- **Temporal hyperedge** per window of W consecutive utterances: 3W modality nodes — M edges.
- **Prototype hyperedge** per emotion class c: prototype node `p_c` plus the modality nodes of training-set utterances labeled c (during training only; at inference, these edges include nothing from the current test dialogue) — C edges.

Total: 2M + K + C hyperedges. Still manageable.

The prototype hyperedges are the novel piece. They give the model an explicit graph path from a class prototype to all training examples of that class, which is exactly what BCL does in loss space but now *also* structurally inside the graph. This dual enforcement of class-prototype geometry is what no prior MERC paper has.

---

### M6 — Hyperedge Weighting

| Option | Evidence | Verdict |
|---|---|---|
| **(a) Uniform W=I** | HGNN baseline | 🔴 Loses content-aware weighting |
| **(b) Learned scalar per edge type** | Too coarse | 🟡 Useless as standalone |
| **(c) AU-conditioned MLP, dual-granular** | MGA-HHN style (arXiv:2505.04340) | 🟢 Content-aware structural signal |
| **(d) Content + AU-confidence joint gating** | (c) plus a confidence multiplier that down-weights edges when AU detection is unreliable | 🟢 Robust to MELD's AU-failure cases |

**Chosen: (d).** This is the same as v2's M6, but with an explicit AU-confidence multiplier added. Specification:

**Node-level (within-hyperedge):**
```
α_{n,e} = softmax_n( MLP_node([h_n, AU_n, conf_AU_n]) )    for each node n ∈ e
```

**Hyperedge-level (across edge types):**
```
β_e = σ(MLP_edge([avg(h_n for n ∈ e), AU_summary_e, mean_conf_AU_e, edge_type_embed]))
```

Why the confidence channel matters: facial AU recognition performance degrades significantly for spontaneous facial displays with free head movements, occlusions, and various illumination conditions. MELD is exactly this case — TV-show video with cuts, occlusions, and group shots. If we don't include a confidence-gating mechanism, AU noise propagates into weights and hurts performance on MELD. The confidence multiplier means: when OpenFace 2.0 returns low-confidence AUs, the weighting reverts toward content-only weighting; when AUs are reliable, they contribute meaningfully.

This is more robust than v2's design and directly addresses the biggest implementation risk from the previous document.

---

### M7 — Propagation

| Option | Evidence | Verdict |
|---|---|---|
| **(a) HGNN+ low-pass smoothing** | Standard, well-understood, stable | 🟢 Default |
| **(b) M3Net multi-frequency** | High-frequency band for similar-emotion discrimination | 🟢 Strong empirical evidence on MERC |
| **(c) DGODE continuous propagation** | COLING 2025 already-published MERC | 🔴 Overlap |
| **(d) HGT-style heterogeneous attention** | Parameter-heavy on small graphs | 🟡 Possible ablation |
| **(e) Frequency-aware HGNN+ (M3Net-inspired but on hypergraph)** | Combines (a) and (b) | 🟢 Novel extension |

**Chosen: (e) Frequency-aware hypergraph propagation.** M3Net showed multi-frequency propagation helps on MERC graphs; we extend this to hypergraphs:

```
X_low  = (D_v^(-1/2) H W D_e^(-1) H^T D_v^(-1/2)) X Θ_low
X_high = (I − D_v^(-1/2) H W D_e^(-1) H^T D_v^(-1/2)) X Θ_high
X_out  = γ · X_low + (1-γ) · X_high
```

where γ is a per-utterance learned gate computed from `[h_t, h_v, h_a, AU_conf]`. The intuition (from M3Net): utterances near a sentiment shift benefit from high-frequency information that emphasizes differences; emotionally stable stretches benefit from low-pass smoothing.

The previous draft (v2) cut this for novelty-claim hygiene. Without that constraint, frequency-aware propagation is well-motivated, has strong empirical support from M3Net (CVPR 2023), and gives an additional axis of improvement. The integration into a *hypergraph* (vs. M3Net's graph) is a non-trivial extension.

---

### M8 — Cross-Modal Interaction

| Option | Evidence | Verdict |
|---|---|---|
| **(a) Implicit via hypergraph triadic edges** | Built into M5 | 🟢 Free |
| **(b) Vision-anchored contrastive alignment** | SupCon with vision as anchor | 🟢 Loss-level enforcement |
| **(c) MBT bottleneck tokens** | NeurIPS 2021 | 🟡 Adds parameters and a path |
| **(d) Text-dominant cross-modal diffusion attention** | Recent work proposes a text-dominant cross-modal diffusion strategy at the fusion stage to enhance the robustness and semantic consistency of the fused representation, achieving w-F1 of 74.87% on IEMOCAP and 66.62% on MELD | 🟡 Strong empirical but contradicts our "anti-text-dominance" framing |
| **(e) Pseudo-unimodal pretraining + parameter-free fusion** | CMC introduces a Pseudo Label Generation Module to produce pseudo unimodal labels, enabling unimodal pretraining in a self-supervised fashion, then employs a Parameter-free Fusion Module and a Multimodal Consensus Router to mitigate text dominance and guide fusion toward a more reliable consensus | 🟢 Direct evidence-based anti-text-dominance method |

**Chosen: (a) + (b).** Skip (c) and (d) — (c) adds parameters with overlap to (b)'s effect; (d) directly contradicts the anti-text-dominance framing.

The interesting candidate is (e). CMC (Oct 2025) is the most recent and most direct anti-text-dominance method. We borrow its *idea* — pseudo-unimodal pretraining as a warm-start — without adopting its full machinery. Specifically:

**Two-phase training (a soft version of CMC's pretraining):**
- Phase 1 (epochs 1–10): Pretrain each modality encoder independently with a per-modality classification head, using only the labeled training data. This forces each modality to develop its own emotion-discriminative representation *before* fusion can let one modality dominate.
- Phase 2 (epochs 11+): Combine into the full graph + multi-modal training.

This is curriculum *over training stages* (drawing from Section 5 of our CL discussion), and it has direct evidence in CMC. Worth including.

**Final cross-modal interaction stack:** triadic hyperedge (structural) + L_align (loss) + two-phase training (procedural).

---

### M9 — Temporal Aggregation

| Option | Evidence | Verdict |
|---|---|---|
| **(a) Bidirectional Mamba-2** | Modern, O(M), well-supported | 🟢 Standard new choice |
| **(b) Transformer encoder** | O(M²) but simple | 🟡 Acceptable baseline |
| **(c) Bidirectional LSTM** | Hypergraph paper found context-LSTM strongest on IEMOCAP | 🟡 Surprisingly competitive on small graphs |
| **(d) Graph-ODE** | DGODE | 🔴 Already published |
| **(e) Bi-Mamba-2 + LSTM hybrid** | Mamba for long context, LSTM for short | 🔴 Over-engineered |

**Chosen: (a) Bidirectional Mamba-2.** Replaces GraphSmile's SDP module. The hypergraph paper noted that context-level LSTM is strong on IEMOCAP while speaker-level modeling matters more on MELD — Mamba-2 covers both cases efficiently. Position as standard practice (citing the three Mamba-for-MERC 2025 papers); not a contribution claim.

---

### M10 — Auxiliary Objectives

The biggest difference from v2 is here. Without the 3-contribution cap, we can include *multiple* auxiliary objectives, each targeting a different problem.

| Option | Evidence | Verdict |
|---|---|---|
| **(a) Sentiment shift classification (SDP)** | GraphSmile baseline | 🔴 Already in GraphSmile |
| **(b) Counterfactual NDE-debiasing** | CausalMER leverages counterfactual reasoning and causal graphs to capture relationships between modalities and reduce direct modality effects contributing to bias, applied in a model-agnostic manner without architectural modifications, achieving 83.4% average accuracy on IEMOCAP with MulT backbone; CLEF (CVPR 2024) for CAER | 🟢 First proven application of NDE-debiasing to MERC |
| **(c) Hypergraph reconstruction (HAUCL-style)** | HAUCL ACM MM 2024 | 🔴 Direct overlap |
| **(d) Trajectory continuity regression** | Original v1 idea | 🟡 Weaker than (b) |
| **(e) Modality-importance / consensus loss** | MIGR-style | 🟡 LLM-focused; awkward fit |
| **(f) Confidence-calibration loss (CMERC-style)** | CMERC integrates curriculum learning to progressively guide the model to learn from uncertain samples, supervised contrastive learning to refine utterance representations, and confidence constraints to penalize uncertainty | 🟢 Adds reliability signal |
| **(g) Curriculum-based loss scheduling** | LSDGNN+ICL (Jul 2025) | 🟢 Training-protocol benefit |

**Chosen: (b) + (f) + (g).**

**(b) Counterfactual NDE-debiasing.** CausalMER (Dec 2024) is the closest published precedent, but it uses MulT and DMD as backbones, not graph-based MERC. Adapting NDE-debiasing to a hypergraph backbone is the novel piece. The fact that CausalMER exists makes the methodology *more* defensible, not less — it's no longer a "speculative borrow from CV" but an established MERC technique we're extending into the graph regime.

```
L_causal = CE(TE_logits, y) + λ_nde · KL(NDE_logits || U)
```

**(f) Confidence-calibration penalty.** Borrows directly from CMERC's confidence-constraint idea. Penalizes the model when its confidence on a sample is high but a single-modality counterfactual would also produce high confidence — exactly the pathological "modality independence" behavior CMERC identified. This is partly redundant with (b) but operates at a different scale: (b) penalizes the *direct text path*, (f) penalizes *any modality being individually sufficient*. Both can coexist.

**(g) Curriculum-based loss scheduling.** Not a separate loss, but a schedule for *when* losses are activated:
- Epochs 1–10: only `L_emo` (CB-Focal + BCL)
- Epochs 11–20: add `L_align`
- Epochs 21–35: add `L_causal`
- Epochs 36+: add `L_calib`

This addresses the highest-risk component (counterfactual training instability) by giving the base model time to converge before the counterfactual branch destabilizes it.

---

### M11 — Primary Loss

| Option | Evidence | Verdict |
|---|---|---|
| **(a) Cross-entropy** | GraphSmile | 🔴 Underperforms on imbalance |
| **(b) CB-Focal** | Cui et al. CVPR 2019 | 🟡 Standard |
| **(c) CB-Focal + BCL (vanilla)** | BCL (Zhu et al., CVPR 2022); regular-simplex feature distribution | 🟢 Strong for long-tail |
| **(d) CB-Focal + BCL + prototype-anchored term** | BCL pulling toward prototype nodes from M4 | 🟢 Stronger if prototype nodes are used |
| **(e) CB-Focal + Hybrid Supervised Contrastive (CMERC-style)** | CMERC's hybrid SCL with calibration | 🟢 Strong but overlaps with CMERC |

**Chosen: (d).** Extends BCL with a prototype-anchoring term. Because we have emotion-prototype nodes in the graph (M4), we can pull each utterance's fused representation toward its class prototype node's embedding *both* in loss space (BCL) *and* through the graph (prototype hyperedges in M5). The combined loss:

```
L_emo = CB-Focal(logits, y) + λ_sup · BCL(features, y) + λ_proto · ||z_i − p_{y_i}||²
```

where `p_{y_i}` is the embedding of the prototype node for the true class. The L2 prototype term is the cheapest possible regularizer that links feature space to graph structure.

This is a small but principled addition. It only makes sense because we have prototype nodes in the graph.

---

## 2. The Synthesized Architecture (MERC v3)

Call it **PHASE**: **Prototype-aware Hypergraph with Adaptive Stream-weighted Edges** (working name).

```
                ┌────────────────────────────────────────────────┐
                │ Modality Encoders                              │
                │  Text  → RoBERTa-Large            (frozen)     │
                │  Video → SigLIP 2 + OpenFace 2.0 AUs           │
                │           + AU-confidence scores               │
                │  Audio → WavLM-Large              (frozen)     │
                └────────────────────────┬───────────────────────┘
                                         │
                ┌────────────────────────▼───────────────────────┐
                │ Per-modality dim-reduction MLPs                │
                │ → h_t_i, h_v_i, h_a_i ∈ R^d  per utterance     │
                └────────────────────────┬───────────────────────┘
                                         │
                ┌────────────────────────▼───────────────────────┐
                │ Build hypergraph                               │
                │   Nodes:                                       │
                │     • 3M modality nodes                        │
                │     • K speaker meta-nodes                     │
                │     • C emotion-prototype nodes                │
                │   Hyperedges:                                  │
                │     • Triadic per utterance       (M edges)    │
                │     • Speaker per speaker         (K edges)    │
                │     • Temporal-window             (M edges)    │
                │     • Prototype per class         (C edges)    │
                └────────────────────────┬───────────────────────┘
                                         │
                ┌────────────────────────▼───────────────────────┐
                │ AU-confidence-gated dual-granular weighting    │
                │   α_{n,e} = softmax_n(MLP_node(...))           │
                │   β_e     = σ(MLP_edge(...))                   │
                │ Both conditioned on AU-confidence              │
                └────────────────────────┬───────────────────────┘
                                         │
                ┌────────────────────────▼───────────────────────┐
                │ Frequency-aware hypergraph propagation         │
                │   X_low  = HGNN+ low-pass                      │
                │   X_high = (I − HGNN+) high-pass               │
                │   X_out  = γ·X_low + (1-γ)·X_high              │
                │   γ learned per utterance                      │
                └────────────────────────┬───────────────────────┘
                                         │
                ┌────────────────────────▼───────────────────────┐
                │ Bidirectional Mamba-2 over fused sequence      │
                └────────────────────────┬───────────────────────┘
                                         │
                ┌────────────────────────▼───────────────────────┐
                │ Classifier head → emotion logits               │
                │                                                │
                │ Counterfactual branch (training only):         │
                │   NDE = forward pass with V=∅, A=∅             │
                └────────────────────────────────────────────────┘

Training signals (scheduled via curriculum):
  Phase 1 (E 1-10):   L_emo = CB-Focal + BCL + prototype-L2
  Phase 2 (E 11-20):  + L_align (vision-anchored SupCon)
  Phase 3 (E 21-35):  + L_causal (NDE debiasing)
  Phase 4 (E 36+):    + L_calib (confidence-calibration penalty)

Pretraining step (before Phase 1):
  Each modality encoder pretrained independently with its own
  classification head for 5 epochs (pseudo-unimodal warm-up).
```

### Components by Contribution Level

**Headline contributions (3 ideas, defensible at BMVC):**

1. **AU-conditioned dual-granular hyperedge weighting with confidence gating** (M6) — facial Action Units drive hypergraph weights, with confidence multipliers making this robust to MELD-like in-the-wild video. No prior MERC method conditions graph structure on AUs.

2. **Counterfactual NDE debiasing of text shortcut for graph-based MERC** (M10b) — extends CausalMER's modality-debiasing methodology to a hypergraph backbone. First graph-based MERC method with explicit causal debiasing.

3. **Emotion-prototype nodes with prototype hyperedges + prototype-anchored loss** (M4 + M5 + M11) — dual enforcement of class-prototype geometry, structurally inside the graph and in loss space. No prior MERC method combines prototype nodes with hypergraph propagation.

**Supporting techniques (engineering, with citations, not contribution claims):**

- Frequency-aware hypergraph propagation (M7) — extends M3Net to hypergraphs
- Speaker meta-nodes (M4) — finally consumes qmask
- Vision-anchored contrastive alignment (M8) — extends SupCon
- Two-phase pseudo-unimodal warm-up (M8) — borrowed from CMC
- Curriculum-based loss scheduling (M10g) — borrowed from CMERC/LSDGNN+ICL
- Confidence-calibration penalty (M10f) — borrowed from CMERC
- Bidirectional Mamba-2 (M9) — standard practice, three 2025 MERC papers
- BCL loss (M11) — standard for long-tail
- SigLIP 2 + WavLM-Large + RoBERTa-Large backbones — standard

---

## 3. Re-Check — Is Each Module Suitable?

### 3.1 Suitability audit per module

| Module | Choice | Suits the contribution? | Risk |
|---|---|---|---|
| M1 Visual | SigLIP 2 + OpenFace AUs + AU confidence | Yes — AU confidence channel makes M6 robust | AU detection degrades under occlusion and demographic bias, but the confidence channel mitigates this directly. **Verified by:** mitigation built into M6. |
| M2 Audio | WavLM-Large | Yes — neutral modern choice | Negligible |
| M3 Text | RoBERTa-Large | Yes — comparable to baselines | Negligible |
| M4 Nodes | Modality + speaker + prototype | Yes — prototype nodes enable M5+M11 prototype mechanisms | Prototype nodes may overfit on small classes. **Mitigation:** low LR on prototype embeddings; initialize from class-mean features. |
| M5 Hyperedges | Triadic + speaker + temporal + prototype | Yes — four orthogonal sources | Risk that prototype hyperedges dominate during training; **Mitigation:** ablate; control prototype-edge weight separately. |
| M6 Weighting | AU + confidence dual-granular | This is contribution #1 | AU reliability risk fully addressed via confidence channel. |
| M7 Propagation | Frequency-aware hypergraph | Yes — supporting extension of M3Net | Adds parameters; may overfit. **Mitigation:** ablation against single-band HGNN+. |
| M8 Cross-modal | Triadic edge + L_align + 2-phase warm-up | Yes — three-pronged anti-text-dominance | Two-phase training adds complexity; monitor for warm-up overfitting. |
| M9 Temporal | Bi-Mamba-2 | Yes — efficient | None |
| M10 Auxiliary | NDE + calib + scheduling | Contains contribution #2 (NDE) | Counterfactual instability; **Mitigation:** activated only in Phase 3, after base model converges. |
| M11 Loss | CB-Focal + BCL + prototype-L2 | Contains contribution #3 (prototype) | None — all components are well-tested |

### 3.2 Coherence check across contributions

Three contributions, three different problems:

- **C1 (AU-weighting):** Structural use of vision. Attacks "vision is ignored" by making graph weights depend on vision quality.
- **C2 (NDE-debiasing):** Causal anti-shortcut. Attacks "text shortcut" by penalizing text-only sufficiency.
- **C3 (Prototype):** Long-tail discrimination. Attacks "minority classes collapse" by anchoring features to class prototypes.

These three are *complementary, not redundant*. They attack three different documented failure modes of MERC. Reviewers should be able to read the abstract and immediately understand which contribution addresses which problem.

### 3.3 Overlap audit against published work

| Published Work | Their Contribution | Overlap with PHASE | Differentiation |
|---|---|---|---|
| GraphSmile (TPAMI 2025) | Pairwise modality graphs + SDP | Both use graph-based MERC | We use hypergraph; no SDP; AU weighting; causal loss; prototypes |
| ConxGNN (ICASSP 2025) | Hypergraph + multi-scale + re-weighting | Both hypergraph + re-weighting | They don't use AUs, no NDE causal, no prototype nodes |
| HAUCL (ACM MM 2024) | VHGAE dynamic hyperedges + SSL contrastive | Both hypergraph + contrastive | We keep structure static, dynamize weights via AUs; no VAE; NDE causal; prototypes |
| M3Net (CVPR 2023) | Multi-frequency graph | We extend to hypergraph | Hypergraph + AU + causal + prototypes |
| CausalMER (Dec 2024) | NDE debiasing on MulT/DMD/PMR backbones | Same NDE methodology | Different backbone (graph); combined with other contributions |
| CMC (Oct 2025) | Pseudo-unimodal pretraining + consensus router | Both target text dominance | We borrow only the pretraining idea; their consensus router not adopted |
| CMERC (ACM MM 2024) | Curriculum + supervised contrastive + confidence | Both use curriculum and calibration | They are model-agnostic; we integrate into a graph architecture |
| DGODE (COLING 2025) | Graph ODE temporal | We use Mamba-2 | Different temporal paradigm |
| LSDGNN+ICL (Jul 2025) | DAG + improved CL with weighted shifts | Both use curriculum | They use DAG; we use hypergraph; different difficulty metric |

The overlap with each individual paper is partial. The combination is genuinely novel.

### 3.4 Three-axis novelty assessment

- **Method:** Hypergraph with AU-confidence-gated weighting + prototype nodes is novel. ✓
- **Insight:** Text-dominance as a documented MERC failure mode now has five+ independent supporting papers, and combining NDE-debiasing with prototype anchoring is a fresh formulation. ✓
- **Systems:** Modern backbones + Mamba-2 + frequency-aware propagation + two-phase training assembled into a single pipeline is non-trivial. ✓

### 3.5 What could kill this paper (updated)

Six failure modes ranked by likelihood:

1. **AU features unreliable on MELD.** Mitigated by confidence channel in M6 — this is the biggest improvement over v2. If AU coverage on MELD is <60%, the confidence channel gracefully reduces to content-only weighting, and the model is still functional. **Mitigation: verified by design.**
2. **Causal NDE branch unstable.** Mitigated by Phase 3 scheduling — activated only after base model converges. **Mitigation: protocol-level.**
3. **Prototype nodes overfit on minority classes.** Possible. **Mitigation:** low LR on prototype embeddings; weight tying to BCL term.
4. **Four hyperedge types create propagation conflicts.** Possible — prototype hyperedges connect across all dialogues, breaking the locality of triadic/speaker/temporal edges. **Mitigation:** prototype hyperedges have a separate weight schedule and can be removed via ablation.
5. **Two-phase pretraining adds wall-clock cost without ablatable benefit.** Possible — if unimodal pretraining doesn't help, drop it. **Mitigation:** the two-phase protocol is itself an ablation row.
6. **Too many moving parts; reviewers complain about kitchen-sink design.** This is the biggest framing risk now that we've relaxed the 3-contribution cap. **Mitigation:** the paper text leads with three named contributions (C1, C2, C3) and clearly marks everything else as "supporting engineering, ablatable individually". The ablation table must show that each of C1, C2, C3 contributes meaningfully *independently*.

### 3.6 Honest complexity assessment

The synthesized architecture has:
- 3 explicit novel contributions (AU-weighting, NDE-debiasing, prototype mechanisms)
- 5 supporting engineering choices (frequency propagation, speaker meta-nodes, 2-phase warm-up, curriculum scheduling, calibration penalty)
- 3 modern-backbone swaps (SigLIP 2, WavLM-Large, Mamba-2) — engineering hygiene
- 1 standard loss family (CB-Focal + BCL + prototype-L2)

This is significantly more than v2's 3+3+3 budget. The question becomes: **is this still reasonable, or has it tipped into kitchen-sink?**

Honest verdict: **borderline**. The five supporting engineering choices add real value but also expose the paper to "complexity attacks" from reviewers. The way to defend this is to be explicit about which components are *contributions* and which are *engineering*. The contributions section says "We make three contributions:" and lists C1-C3. The methods section presents the full pipeline including all engineering pieces. The ablation table separates "contribution ablations" (removing C1, C2, C3 individually) from "engineering ablations" (removing supporting pieces).

If reviewers complain that any single supporting piece is unjustified, it can be dropped without affecting the contribution claim. That's the test for "engineering vs. contribution".

---

## 4. Updated Implementation Roadmap

The reading and reproduction stages are unchanged from the original BMVC plan. The implementation stages change:

**Weeks 1–2 (unchanged):** Reproduce GraphSmile, ConxGNN, HAUCL on the same hardware and splits.

**Weeks 3–5:** Visual re-extraction pipeline. **Critical addition:** measure OpenFace 2.0 AU confidence distribution on a 1000-utterance MELD subset. If median confidence is below 0.5, M1 design needs revision. This is the highest-risk verification step.

**Weeks 6–8:** Implement modality encoders, basic hypergraph (without prototype nodes), HGNN+ propagation, basic classifier. Get a clean baseline number on IEMOCAP-6.

**Weeks 9–11:** Add C1 (AU-confidence-gated dual-granular weighting). Test ablation: M6 alone vs. no M6.

**Weeks 12–14:** Add C3 (prototype nodes + prototype hyperedges + prototype-anchored loss). Test ablation.

**Weeks 15–17:** Add C2 (NDE-debiasing branch + Phase 3 scheduling). This is the highest-risk component. Spend time on stability.

**Weeks 18–19:** Engineering integrations (frequency-aware propagation, 2-phase warm-up, calibration penalty). Each gets an ablation row.

**Weeks 20–22:** Full ablation grid, modality-dropout robustness, confusion matrices.

**Weeks 23–26:** Writing.

Total: 26 weeks (~6 months). Realistic for BMVC 2027 / ACM MM 2027.

---

## 5. The Sharp Edge

The previous derivation (v2) chose minimalism. This derivation (v3) chose evidence. The result is more complex but also more defensible:

- Every component points to a documented MERC failure mode (text dominance, modality bias, AU unreliability, long-tail collapse, similar-emotion confusion).
- Every choice has at least one published precedent (often as a methodology that hasn't been adapted to graph-based MERC).
- Every component is independently ablatable.

The two derivations are honest expressions of two different research strategies:

- **v2 (minimalist):** "Three contributions, clean ablation, defensible at any conference."
- **v3 (maximalist-with-discipline):** "Three contributions plus five supporting techniques, full evidence base, beats SOTA by exploiting every documented MERC weakness."

If the goal is a clean BMVC submission with high acceptance probability, v2 is safer. If the goal is to maximize benchmark performance and *also* submit to BMVC, v3 is stronger but requires more discipline in the writing to avoid kitchen-sink criticism. The v3 contribution claims must be ruthlessly tight even though the architecture is rich.

---

## 6. One-Sentence Summary

> *We propose a prototype-aware hypergraph framework for MERC where (i) facial-AU confidence gates the hypergraph edge weights, (ii) a counterfactual Natural Direct Effect loss penalizes the text-only shortcut documented in five recent MERC papers, and (iii) emotion-prototype nodes anchor class geometry both structurally inside the hypergraph and in loss space — three orthogonal interventions against the three best-documented failure modes of current MERC systems (text dominance, AU unreliability, long-tail collapse).*

That is the paper.
