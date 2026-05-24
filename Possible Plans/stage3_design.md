# Stage 3 Design — Graph Construction + Model

*Grounded in your actual on-disk features. Every backbone/module choice is argued on merit, not defaulted. The temporal `(3, dim)` visual signal is preserved as you specified — never pooled to a single vector.*

---

## 0. What we are designing against (your real features)

| Modality | Encoder | Shape per utterance | Notes |
|---|---|---|---|
| Text | RoBERTa-Large (CLS, frozen) | `(1024,)` | one vector |
| Audio | WavLM-Large (masked mean-pool) | `(1024,)` | one vector. WavLM-Large UA 77.41 on IEMOCAP SER ≥ HuBERT-Large 76.92, so your primary choice is the right one |
| Visual | CLIP ViT-L/14 | `(3, 768)` | begin/mid/end |
| Visual | SigLIP 2 so400m | `(3, 1152)` | begin/mid/end |
| Visual | OpenFace-AU (OF3) | `(3, 8)` | begin/mid/end |

**Your visual spec (honored exactly):**
```
visual_seg[t] = OpenFaceAU[t]  ⊕  ( SigLIP2[t]  OR  CLIP[t] )   for t ∈ {begin, mid, end}
```
- SigLIP 2 path → `(3, 1160)`
- CLIP path → `(3, 776)`

The three timeline rows are kept as **three separate things** all the way into the graph. No mean-pool, no flatten-to-one.

**SigLIP 2 vs CLIP — recommendation, not assumption.** Default to **SigLIP 2** as the semantic stream and keep CLIP as a one-row ablation. Reason: SigLIP 2 (Tschannen et al., Feb 2025) is a strict architectural successor to CLIP, trained with a sigmoid objective plus dense/localization-aware losses, and outperforms CLIP on essentially all downstream vision transfer benchmarks. There is no published MERC-specific head-to-head, so this stays an ablation axis rather than a settled fact — but the prior strongly favors SigLIP 2, and your higher feature dim (1152 vs 768) carries more facial-semantic signal. OpenFace-AU is **kept in both paths** because the 8 action-unit intensities are FACS-grounded (directly tied to emotion theory) and are information the CLIP/SigLIP semantic stream does not explicitly encode.

---

## 1. The thing almost nobody does (your asset)

Mainstream graph-MERC — MMGCN, MM-DFN, M3Net, GraphSmile, CORECT, HGF-MiLaG, HGLER — uses **one node per (utterance, modality)** and pools the visual stream to a single vector before the graph ever sees it. The only work that treats sub-utterance temporal segments as structure is (a) single-utterance speech-emotion graphs (frames-as-nodes, not conversational) and (b) MLLM approaches like MicroEmo (not graph-based, not benchmarked in WF1 on IEMOCAP/MELD).

That gap is the lever. Your pipeline already produced begin/mid/end visual segments — the **expression arc (onset → apex → offset)**. Preserving it as something the model can reason over is a clean, defensible novelty axis that is *yours by construction*, not borrowed from any single paper. This is also why the backbone choice below is **not** GraphSmile by default.

---

## 2. Backbone bake-off — chosen on merit

The previous drafts leaned on GraphSmile. That was unearned. Here is the honest comparison for **your** constraints: (i) must accommodate a variable node count per utterance (3 visual sub-nodes + text + audio), (ii) must reach top-tier WF1, (iii) must be implementable/reproducible by one person.

| Backbone | Core mechanism | Fits 3 visual sub-nodes? | Reported strength | Verdict for us |
|---|---|---|---|---|
| **Hypergraph (M3Net-style)** | Hyperedges connect >2 nodes; multi-frequency (low/high-pass) filtering | **Yes — natively.** A hyperedge holds any number of nodes, so {text, audio, vis_b, vis_m, vis_e} is a single 5-node hyperedge with zero hacks | M3Net strong on IEMOCAP/MELD; hypergraph captures higher-arity cross-modal relations | **Lead choice.** The variable-arity property is exactly what the temporal sub-nodes need |
| **GraphSmile GSF (TPAMI 2025)** | Layer-by-layer **alternating** inter-/intra-modal aggregation; connects cross-modal nodes across *different* utterances; avoids "fusion conflict" | Awkwardly — it is built around one node per modality per utterance; 3 visual nodes break its pairwise modality bookkeeping | Among the strongest pairwise graph results; clean anti-fusion-conflict idea | **Borrow the idea, not the scaffold.** The alternating inter/intra schedule is worth keeping; the rigid node layout is not |
| **HRG-SSA (IJCAI 2025)** | T5 encoder–decoder, MERC as generation; implicit per-modality edges | No — the engine is a generative decoder, not a node set you extend | Current IEMOCAP high-water mark (75.47 WF1) | **Reference ceiling, not our base.** Buys into T5 features (you extracted RoBERTa) + generation overhead; doesn't exploit temporal sub-nodes |
| **HAUCL (ACM MM 2024)** | Variational hypergraph autoencoder + contrastive | Yes (hypergraph) but VHGAE adds sampling variance + is data-hungry | SOTA-ish over M3Net | **Ablation, not base.** Dynamic structure muddies attribution of any temporal-node gain |
| **MM-DFN (ICASSP 2022)** | Deep GCN + gated dynamic fusion across layers | Partially (pairwise graph) | Solid, very reproducible (public code, well-understood) | **Safety-net base for the low-risk plan** |
| **Non-graph (SDT / transformer fusion)** | Cross-modal attention | No graph topology to exploit; temporal segments handled by attention | Competitive but discards conversational structure | **Skip as primary** — throws away the structure that is the point |

**Conclusion:** the temporal-preservation requirement *itself* selects the hypergraph as the most natural backbone, with GraphSmile's alternating schedule kept as a mechanism we adopt inside it. This is the merit-based answer to "why not just GraphSmile."

---

## 3. Three plans, decreasing novelty — all preserve the 3 segments

The ladder varies **how much structural status** the temporal arc gets, not whether it survives. Every plan keeps begin/mid/end.

### Plan A — Temporal Heterogeneous Hypergraph (highest novelty, highest ceiling)

**One-line idea:** promote the 3 visual segments to **first-class graph nodes**, so the expression arc is structure the hypergraph reasons over — something no MERC graph paper does.

**Node set per utterance _i_ (5 nodes):**
- `t_i` text node — RoBERTa-Large 1024 → projected to d
- `a_i` audio node — WavLM-Large 1024 → projected to d
- `v_i^b, v_i^m, v_i^e` three visual nodes — each `OpenFaceAU ⊕ SigLIP2` (1160) → projected to d

A shared per-modality projection (one Linear+LayerNorm+GELU per modality, audio and visual following HRG-SSA's two-projection alignment-to-common-dim convention) maps all nodes to a common width d (256 for IEMOCAP, 384 for MELD, matching MM-DFN/GraphSmile hidden sizes).

**Hyperedge set:**

| Hyperedge | Members | Captures | Source of idea |
|---|---|---|---|
| Multimodal-utterance | {t_i, a_i, v_i^b, v_i^m, v_i^e} | within-utterance cross-modal binding incl. the visual arc | M3Net multimodal hyperedge, extended with temporal sub-nodes (new) |
| Visual-arc (directed chain) | v_i^b → v_i^m → v_i^e | onset→apex→offset dynamics of the face | new; motivated by MicroEmo's micro-expression argument |
| Contextual (per modality, windowed) | all nodes of one modality within window w (past w_p, future w_f) | intra-modal conversational context | M3Net contextual hyperedge / MM-DFN window |
| Speaker | all utterance-nodes of one speaker (per modality) | speaker emotional baseline; uses the qmask that GraphSmile loads-but-ignores | speaker meta-structure (under-used in MERC) |

**Propagation:** hypergraph convolution with **two-level attention** (node-level: which member of a hyperedge matters; hyperedge-level: which hyperedge type matters for this node), adapted from MGA-HHN, run on an **alternating inter-/intra schedule** borrowed from GraphSmile GSF to avoid fusion conflict. Optionally add M3Net's low/high-pass split as one ablation.

**Readout & fusion:** after K layers, pool the 3 visual nodes of each utterance by attention (learned weights over begin/mid/end — this is where the arc gets summarized, *after* the graph has used it structurally), then concatenate {text, audio, visual} per utterance → small fusion MLP.

**Loss:** `L = CB-Focal(logits, y) + λ_sup · BCL(features, y)`. CB-Focal (Cui et al. 2019) for MELD imbalance; BCL (Balanced Contrastive Learning, Zhu et al. CVPR 2022) replaces vanilla SupCon to fix head-class bias — the cheapest minority-class-F1 win available. Optionally GraphSmile's sentiment-shift auxiliary task if you have sentiment labels.

**Why it can top the table:** it adds a signal axis (visual temporal arc) orthogonal to everything current SOTA optimizes, on top of a hypergraph that already matches M3Net-class performance, with a modern imbalance-aware loss.

---

### Plan B — Temporal-Encoded Nodes + Alternating Heterogeneous Graph (medium novelty, safer)

**One-line idea:** keep the arc, but compress it inside a node rather than as separate nodes. A tiny temporal encoder turns the 3 segments into one **temporally-aware** visual node; the backbone is then a strong *pairwise* heterogeneous graph.

**Visual node construction:** feed the 3 segments `(3, 1160)` through a **2-layer Transformer encoder** (or BiGRU — ablation) over the 3 timesteps; take the sequence summary as the visual node feature. The arc is preserved *in the features* (the encoder sees ordering) but the graph stays one-node-per-modality, so it slots into proven backbones unchanged.

**Backbone:** heterogeneous graph with **GraphSmile-style alternating inter-/intra-modal layers** — here the GSF schedule is appropriate because the node layout is pairwise. Cross-modal edges connect a node to same- and different-utterance cross-modal nodes (GraphSmile's key fix over MMGCN). This is where GraphSmile *earns* its place rather than being a default.

**Fusion / loss:** same as Plan A (attention fusion + CB-Focal + BCL).

**Why choose it:** ~80% of Plan A's temporal benefit at a fraction of the implementation risk; directly comparable to published pairwise-graph numbers; the temporal-encoder-vs-pooling ablation is itself a clean paper figure.

---

### Plan C — Temporal Sub-Nodes in a Plain Spectral GCN (lowest novelty, de-risked baseline)

**One-line idea:** still keep the 3 visual nodes (per your constraint), but drop the hypergraph and the two-level attention — use the simplest proven message passing so you have a **reliable, reproducible floor** that already beats naive single-vector baselines.

**Construction:** 5 nodes per utterance as in Plan A, but edges are ordinary pairwise (fully-connected within modality + same-utterance cross-modal, MMGCN/MM-DFN style), spectral GCN propagation (MM-DFN's gated deep GCN). Visual arc enters as begin↔mid↔end pairwise edges.

**Fusion / loss:** concatenation + CB-Focal (skip BCL initially to keep it minimal; add later).

**Why have it:** this is your safety net and your ablation anchor. If Plan A/B underperform, Plan C isolates whether the gain is from *temporal nodes* (C vs single-vector MM-DFN) or from *the hypergraph + attention* (A vs C). It also de-risks the whole project: you get a working, defensible number early.

---

## 4. Module-by-module summary across plans

| Module | Plan A | Plan B | Plan C |
|---|---|---|---|
| Visual feature | OpenFaceAU ⊕ SigLIP2, `(3,1160)` kept | same, encoded to 1 node | same, kept as 3 nodes |
| Temporal handling | 3 visual **nodes** + arc edge | temporal **encoder** → 1 node | 3 visual **nodes**, pairwise edges |
| Node projection | per-modality Linear+LN+GELU → d | same | same |
| Graph type | heterogeneous **hypergraph** | heterogeneous **pairwise** graph | pairwise spectral graph |
| Propagation | 2-level attention, alternating inter/intra | GraphSmile alternating GSF | MM-DFN gated deep GCN |
| Speaker info | speaker hyperedge (uses qmask) | speaker edge type | speaker embedding add |
| Fusion | attention pool + MLP | attention pool + MLP | concat + MLP |
| Loss | CB-Focal + BCL (+SDP opt.) | CB-Focal + BCL | CB-Focal |
| Novelty | temporal arc as structure (new) | temporal-encoded node (mild) | temporal nodes, plain (low) |
| Risk | medium-high | low-medium | low |

---

## 5. Recommended path

1. **Build Plan C first** (1–2 weeks). It validates the data path end-to-end and gives a reproducible baseline + the ablation anchor. Low risk, immediate number.
2. **Promote to Plan A** as the headline model. The hypergraph is the merit-based backbone for temporal sub-nodes, and the temporal-arc-as-structure idea is the cleanest novelty you have.
3. **Keep Plan B in the paper as the controlled middle** — it answers "does the arc need to be *structure*, or is encoding it in features enough?", which is exactly the ablation a reviewer will demand.

The three plans are designed so that running all three *is itself the ablation study*, not extra work.

---

## 6. Honest risks & open checks

- **Node-count blow-up.** 5 nodes/utterance × up to ~110 utterances (IEMOCAP) ≈ 550 nodes/dialogue; on MELD's shorter dialogues it's fine. Hypergraph incidence stays sparse, so this is tractable, but worth profiling before scaling layers.
- **Does the arc actually carry signal?** Some utterances have flat affect (no onset→apex). If the visual-arc edge rarely helps, Plan A degrades gracefully toward Plan C. The C-vs-A ablation will tell you directly — don't assume the arc helps until measured.
- **SigLIP 2 vs CLIP for *faces* specifically.** Both are scene/object-trained; neither is face-expression-specialized. OpenFace-AU is the expression-grounded stream and may carry most of the emotional load. Run the AU-only vs AU⊕SigLIP2 vs AU⊕CLIP ablation early — it's possible the heavy semantic stream adds little over AUs for this task.
- **MELD face coverage.** ~99% expected, but utterances with no detected face get zero visual nodes — decide now whether those collapse to a learned "visual-absent" embedding (recommended) or drop the visual nodes for that utterance.
- **BCL integration.** Confirm the BCL loss runs mechanically on one IEMOCAP epoch before committing — it's a loss swap, cheap to verify, and de-risks the minority-class story.

---

*Backbone selected by fit to your temporal-preservation requirement and reported strength, not by default. Hypergraph leads because variable-arity hyperedges natively hold the 3 visual sub-nodes; GraphSmile's alternating schedule is retained as an internal mechanism where the node layout is pairwise. All visual handling honors: OpenFace-AU ⊕ (SigLIP2 | CLIP), three temporal segments preserved, never pooled to one.*
