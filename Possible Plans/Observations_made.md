Openface 3.0 might be a superior option instead of openface 2.0

Why Angular Similarity, Not Raw Cosine?
This is the subtle design choice worth understanding. Cosine similarity lives in [-1, +1]. If you used it directly as an edge weight, negative values would mean negative edge weights — which destabilizes graph convolution (negative weights cause oscillating or diverging aggregations in GNNs).

Not all papers are speaker aware
HRG-SSA is weakly speaker-aware
Graphsmile is not speaker-aware