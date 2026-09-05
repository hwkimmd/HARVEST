"""Fixed RARE26 PPV@90% recall evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

TARGET_RECALL = 0.90
SIMULATIONS = 1000
SEED = 20260830


def _arrays(y_true, y_score):
    y = np.asarray(y_true, dtype=np.int8).reshape(-1)
    s = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if y.shape != s.shape or not np.isin(y, (0, 1)).all() or not np.isfinite(s).all():
        raise ValueError("labels and finite scores must be aligned one-dimensional arrays")
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("both classes are required")
    return y, s


def interpolated_ppv_at_90_recall(y_true, y_score):
    y, s = _arrays(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y, s)
    return float(np.interp(TARGET_RECALL, recall[::-1], precision[::-1]))


def fp_at_90_recall(y_true, y_score):
    y, s = _arrays(y_true, y_score)
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    last = np.r_[np.flatnonzero(np.diff(ss)), len(ss) - 1]
    tp, fp = tp[last], fp[last]
    needed = int(np.ceil(TARGET_RECALL * y.sum() - 1e-9))
    index = min(int(np.searchsorted(tp, needed)), len(tp) - 1)
    return int(fp[index])


@dataclass(frozen=True)
class RareMetric:
    median_ppv: float
    ppv_p5: float
    ppv_p95: float
    ppv_p2_5: float
    ppv_p97_5: float
    n_iter: int
    n_negative: int
    n_positive_sampled: int
    seed: int

    def as_dict(self):
        return asdict(self)


def simulate_rare26_metric(y_true, y_score):
    y, s = _arrays(y_true, y_score)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    n_positive = max(1, int(len(negative) / 100))
    labels = np.r_[
        np.ones(n_positive, dtype=np.int8),
        np.zeros(len(negative), dtype=np.int8),
    ]
    draws = np.random.default_rng(SEED).integers(
        0, len(positive), size=(SIMULATIONS, n_positive)
    )
    values = np.empty(SIMULATIONS, dtype=np.float64)
    for index, draw in enumerate(draws):
        scores = np.r_[s[positive[draw]], s[negative]]
        values[index] = interpolated_ppv_at_90_recall(labels, scores)
    return RareMetric(
        median_ppv=float(np.median(values)),
        ppv_p5=float(np.percentile(values, 5)),
        ppv_p95=float(np.percentile(values, 95)),
        ppv_p2_5=float(np.percentile(values, 2.5)),
        ppv_p97_5=float(np.percentile(values, 97.5)),
        n_iter=SIMULATIONS,
        n_negative=len(negative),
        n_positive_sampled=n_positive,
        seed=SEED,
    )


def evaluate_predictions(y_true, y_score):
    y, s = _arrays(y_true, y_score)
    result = simulate_rare26_metric(y, s).as_dict()
    result.update(
        {
            "auroc": float(roc_auc_score(y, s)),
            "auprc": float(average_precision_score(y, s)),
            "raw_interpolated_ppv_at_90r": interpolated_ppv_at_90_recall(y, s),
            "raw_fp_at_90r": fp_at_90_recall(y, s),
            "n_positive": int(y.sum()),
            "n_negative": int((y == 0).sum()),
            "rule": "sklearn precision_recall_curve + numpy.interp",
        }
    )
    return result
