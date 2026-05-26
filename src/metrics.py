"""Classification metrics: WA, weighted-F1, macro-F1."""

from typing import List


def compute_metrics(preds: List[int], labels: List[int], num_classes: int) -> dict:
    n    = len(labels)
    corr = sum(p == l for p, l in zip(preds, labels))
    wa   = corr / n if n > 0 else 0.0

    f1_per_class = []
    support      = []
    for c in range(num_classes):
        tp = sum(1 for p, l in zip(preds, labels) if p == c and l == c)
        fp = sum(1 for p, l in zip(preds, labels) if p == c and l != c)
        fn = sum(1 for p, l in zip(preds, labels) if p != c and l == c)
        sup = tp + fn
        prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_per_class.append(f1)
        support.append(sup)

    total_sup = sum(support) or 1
    wf1 = sum(f1_per_class[c] * support[c] for c in range(num_classes)) / total_sup
    uf1 = sum(f1_per_class) / num_classes

    return {"wa": wa, "wf1": wf1, "uf1": uf1, "f1_per_class": f1_per_class}
