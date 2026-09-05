"""Calibration-free logit alignment and robust HARVEST aggregation."""
from __future__ import annotations
import numpy as np

COMPONENTS = ("vitl_ssl224_twoview", "vitl_ssl384_oneview")


def fit_affine(logits):
    values = np.asarray(logits, dtype=np.float64)
    mean, std = float(values.mean()), float(values.std(ddof=0))
    if not np.isfinite([mean, std]).all() or std <= 0:
        raise ValueError("validation logits require finite non-zero variance")
    return mean, std


def standardize(logits, mean, std):
    if not np.isfinite([mean, std]).all() or std <= 0:
        raise ValueError("invalid validation statistics")
    return (np.asarray(logits, dtype=np.float64) - mean) / std


def middle_three_mean(member_logits):
    values = np.asarray(member_logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 5:
        raise ValueError("each HARVEST component requires [5 folds, N images]")
    if not np.isfinite(values).all():
        raise ValueError("member logits contain NaN or infinity")
    return np.sort(values, axis=0)[1:4].mean(axis=0)


def stable_sigmoid(logits):
    x = np.asarray(logits, dtype=np.float64)
    output = np.empty_like(x)
    positive = x >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exponent = np.exp(x[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def aggregate_component(member_logits, validation_means, validation_stds):
    values = np.asarray(member_logits, dtype=np.float64)
    if values.shape[0] != 5:
        raise ValueError("exactly five fold members are required")
    means = np.asarray(validation_means, dtype=np.float64).reshape(5, 1)
    stds = np.asarray(validation_stds, dtype=np.float64).reshape(5, 1)
    return middle_three_mean((values - means) / stds)


def aggregate_harvest(components):
    if set(components) != set(COMPONENTS):
        raise ValueError(f"HARVEST requires exactly {COMPONENTS}")
    component_logits = [aggregate_component(*components[name]) for name in COMPONENTS]
    return stable_sigmoid(0.5 * component_logits[0] + 0.5 * component_logits[1])
