import numpy as np
from sklearn.metrics import precision_recall_curve

from harvest.metrics import interpolated_ppv_at_90_recall, simulate_rare26_metric


def test_interpolation_matches_released_formula():
    y = np.array([1, 0, 1, 0, 0, 1, 0])
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    precision, recall, _ = precision_recall_curve(y, score)
    expected = np.interp(0.9, recall[::-1], precision[::-1])
    assert interpolated_ppv_at_90_recall(y, score) == expected


def test_perfect_ranking_simulation_is_one():
    y = np.r_[np.ones(10), np.zeros(1000)]
    score = np.r_[np.ones(10), np.zeros(1000)]
    result = simulate_rare26_metric(y, score)
    assert result.median_ppv == 1.0
    assert result.n_positive_sampled == 10
