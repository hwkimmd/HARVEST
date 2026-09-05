"""Deterministic exact-composition samplers used by HARVEST."""
from __future__ import annotations
import math
import numpy as np
from torch.utils.data import Sampler


class ExactCompositionBatchSampler(Sampler):
    def __init__(self, strata, composition, batches_per_epoch=None, seed=20260830):
        self.strata = np.asarray(strata).astype(str)
        self.composition = {str(k): int(v) for k, v in composition.items()}
        self.pools = {key: np.flatnonzero(self.strata == key) for key in self.composition}
        if any(len(pool) == 0 for pool in self.pools.values()):
            raise ValueError("every requested sampling stratum must be non-empty")
        self.batches_per_epoch = batches_per_epoch or math.ceil(len(self.strata) / sum(self.composition.values()))
        self.seed, self.epoch = int(seed), 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.batches_per_epoch):
            batch = np.concatenate([rng.choice(self.pools[key], count, replace=True) for key, count in self.composition.items()])
            rng.shuffle(batch)
            yield batch.tolist()


def pooled_batch_sampler(labels, centers, batch_size=32, seed=20260830):
    if batch_size <= 0 or batch_size % 8:
        raise ValueError("batch_size must be a positive multiple of eight")
    labels = np.asarray(labels).astype(int)
    centers = np.asarray(centers).astype(str)
    strata = np.char.add(np.char.add(centers, "_"), np.where(labels == 1, "positive", "negative"))
    scale = batch_size // 8
    return ExactCompositionBatchSampler(strata, {
        "center_1_positive": scale, "center_2_positive": scale,
        "center_1_negative": 3 * scale, "center_2_negative": 3 * scale,
    }, seed=seed)


def two_view_batch_sampler(labels, seed=20260830):
    strata = np.where(np.asarray(labels).astype(int) == 1, "positive", "negative")
    return ExactCompositionBatchSampler(strata, {"positive": 16, "negative": 48}, seed=seed)
