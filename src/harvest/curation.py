"""Final GastroNet-5M curation utilities used by HARVEST."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset

CLUSTERING_SEED = 20260822
SAMPLING_SEED = 20260830
EXPECTED_IMAGES = 4_823_974
REVIEW_CLUSTERS = 200
CATEGORY_QUOTAS = {
    "egj": 50_000,
    "other_esophagus": 25_000,
    "stomach": 25_000,
}
CATEGORY_CLUSTER_COUNTS = {
    "egj": 22,
    "other_esophagus": 8,
    "stomach": 35,
}
CORESET_SIZE = sum(CATEGORY_QUOTAS.values())


def scan_gastronet(root: str | Path) -> pd.DataFrame:
    """Recursively index all PNG files in deterministic order."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"GastroNet root not found: {root}")
    paths = []
    for directory, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for filename in sorted(filenames):
            if filename.lower().endswith(".png"):
                paths.append(Path(directory, filename).relative_to(root).as_posix())
    if len(paths) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"expected {EXPECTED_IMAGES:,} images, found {len(paths):,}; "
            "refusing to curate an incomplete corpus"
        )
    return pd.DataFrame(
        {
            "row": np.arange(len(paths), dtype=np.int64),
            "image_id": [Path(path).with_suffix("").as_posix() for path in paths],
            "relative_path": paths,
            "shard": [Path(path).parts[0] for path in paths],
        }
    )


class GastroNetImages(Dataset):
    """Return image tensors and stable embedding row indices."""

    def __init__(self, index: pd.DataFrame, data_root: str | Path, transform):
        self.rows = index["row"].to_numpy()
        root = Path(data_root).expanduser().resolve()
        self.paths = [root / str(path) for path in index["relative_path"]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def _load(self, path):
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))

    def __getitem__(self, index):
        path = self.paths[index]
        try:
            return self._load(path), int(self.rows[index]), True
        except Exception:
            previous = ImageFile.LOAD_TRUNCATED_IMAGES
            try:
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                return self._load(path), int(self.rows[index]), True
            except Exception:
                import torch

                return torch.zeros(3, 224, 224), int(self.rows[index]), False
            finally:
                ImageFile.LOAD_TRUNCATED_IMAGES = previous


def normalized_float32(array: np.ndarray, chunk_size: int = 500_000):
    output = np.empty(array.shape, dtype=np.float32)
    for start in range(0, len(array), chunk_size):
        chunk = np.asarray(array[start : start + chunk_size], dtype=np.float32)
        norm = np.linalg.norm(chunk, axis=1, keepdims=True)
        np.divide(
            chunk,
            np.maximum(norm, 1e-12),
            out=output[start : start + len(chunk)],
        )
    return output


def spherical_kmeans(features: np.ndarray, clusters: int):
    """Run the fixed 25-iteration FAISS spherical k-means."""
    import faiss

    kmeans = faiss.Kmeans(
        d=features.shape[1],
        k=clusters,
        niter=25,
        seed=CLUSTERING_SEED,
        spherical=True,
        verbose=True,
        max_points_per_centroid=256,
    )
    kmeans.train(features)
    index = faiss.IndexFlatIP(features.shape[1])
    index.add(np.ascontiguousarray(kmeans.centroids, dtype=np.float32))
    similarity, labels = index.search(
        np.ascontiguousarray(features, dtype=np.float32), 1
    )
    return (
        kmeans.centroids.reshape(clusters, features.shape[1]),
        labels[:, 0].astype(np.int32),
        similarity[:, 0].astype(np.float32),
    )


def cluster_summary(labels, similarity, index):
    frame = pd.DataFrame(
        {
            "cluster": labels,
            "similarity": similarity,
            "shard": index["shard"].to_numpy(),
        }
    )
    grouped = frame.groupby("cluster")
    return pd.DataFrame(
        {
            "cluster": grouped.size().index,
            "size": grouped.size().to_numpy(),
            "mean_similarity": grouped.similarity.mean().to_numpy(),
            "min_similarity": grouped.similarity.min().to_numpy(),
            "shards": grouped.shard.nunique().to_numpy(),
        }
    ).sort_values("size", ascending=False)


def review_representatives(labels, similarity, cluster):
    """Select 24 nearest, 24 random, and 16 farthest review images."""
    members = np.flatnonzero(labels == cluster)
    ordered = members[np.argsort(-similarity[members])]
    near = ordered[:24]
    far = ordered[-16:]
    pool = np.setdiff1d(members, np.r_[near, far])
    rng = np.random.default_rng(CLUSTERING_SEED + int(cluster))
    random_rows = rng.choice(pool, size=min(24, len(pool)), replace=False)
    return {"nearest": near, "random": np.sort(random_rows), "farthest": far}


def _sqrt_quota(cluster_ids, capacities, budget):
    """Allocate an exact sqrt(capacity) budget with stable cluster-ID ties."""
    ids = np.asarray(cluster_ids, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    if ids.ndim != 1 or capacities.shape != ids.shape:
        raise ValueError("cluster_ids and capacities must be aligned")
    if int(capacities.sum()) < budget:
        raise RuntimeError(f"capacity {capacities.sum()} is below quota {budget}")
    quotas = np.zeros_like(capacities)
    weights = np.sqrt(capacities.astype(np.float64))
    remaining = int(budget)
    while remaining:
        room = capacities - quotas
        active = np.flatnonzero(room > 0)
        raw = remaining * weights[active] / weights[active].sum()
        additions = np.minimum(np.floor(raw).astype(np.int64), room[active])
        if int(additions.sum()):
            quotas[active] += additions
        else:
            order = np.lexsort((ids[active], -raw))
            quotas[active[order[:remaining]]] += 1
        remaining = budget - int(quotas.sum())
    return quotas

def reviewed_coreset(
    index,
    review,
    coarse_labels,
    coarse_similarity,
):
    """Apply anatomical/view review and build the fixed 50k/25k/25k corpus."""
    required = {
        "cluster",
        "decision",
        "anatomical_view_category",
        "notes",
    }
    if missing := required - set(review.columns):
        raise ValueError(f"review CSV missing columns: {sorted(missing)}")
    review = review.copy()
    review["decision"] = review["decision"].fillna("").str.strip().str.upper()
    review["anatomical_view_category"] = (
        review["anatomical_view_category"].fillna("").str.strip().str.lower()
    )
    if len(review) != REVIEW_CLUSTERS or set(review["cluster"]) != set(
        range(REVIEW_CLUSTERS)
    ):
        raise ValueError("review must contain each K=200 cluster exactly once")
    decisions = {"KEEP", "EXCLUDE", "UNCERTAIN"}
    if not review["decision"].isin(decisions).all():
        raise ValueError("all clusters require KEEP, EXCLUDE, or UNCERTAIN")
    if review["notes"].fillna("").str.strip().eq("").any():
        raise ValueError("every cluster requires a clinician review note")
    kept = review.loc[review["decision"] == "KEEP"]
    if len(kept) != 65:
        raise ValueError("the recorded HARVEST review retained exactly 65 clusters")
    if not kept["anatomical_view_category"].isin(CATEGORY_QUOTAS).all():
        raise ValueError(
            "every KEEP cluster requires an anatomical/view category: "
            "egj, other_esophagus, or stomach"
        )
    if review.loc[review["decision"] != "KEEP", "anatomical_view_category"].ne("").any():
        raise ValueError("only KEEP clusters may receive a selection category")
    category_counts = kept["anatomical_view_category"].value_counts().to_dict()
    if category_counts != CATEGORY_CLUSTER_COUNTS:
        raise ValueError(
            f"anatomical/view cluster counts must be {CATEGORY_CLUSTER_COUNTS}; "
            f"received {category_counts}"
        )

    rng = np.random.default_rng(SAMPLING_SEED)
    selected_frames = []
    for category, budget in CATEGORY_QUOTAS.items():
        cluster_ids = kept.loc[
            kept["anatomical_view_category"] == category, "cluster"
        ].astype(int).sort_values().to_numpy()
        pools = {}
        for cluster in cluster_ids:
            members = np.flatnonzero(coarse_labels == cluster)
            ordered = members[np.argsort(coarse_similarity[members], kind="stable")]
            drop = int(round(0.20 * len(members)))
            pools[cluster] = ordered[drop:]
        coarse_quota = _sqrt_quota(
            cluster_ids,
            [len(pools[cluster]) for cluster in cluster_ids],
            budget,
        )
        category_rows = []
        for cluster, target in zip(cluster_ids, coarse_quota):
            rows = pools[cluster]
            category_rows.append(
                rng.choice(rows, size=int(target), replace=False)
            )
        rows = np.concatenate(category_rows)
        if len(rows) != budget or len(np.unique(rows)) != budget:
            raise RuntimeError(f"{category} sampling invariant failed")
        selected_frames.append(
            pd.DataFrame(
                {
                    "row": rows,
                    "anatomical_view_category": category,
                    "category_quota": budget,
                    "sampling_seed": SAMPLING_SEED,
                }
            )
        )

    selected = pd.concat(selected_frames, ignore_index=True)
    selected = selected.iloc[rng.permutation(len(selected))].reset_index(drop=True)
    if len(selected) != CORESET_SIZE or not selected["row"].is_unique:
        raise RuntimeError("100,000-image uniqueness invariant failed")
    rows = selected["row"].to_numpy()
    output = index.iloc[rows][
        ["row", "image_id", "relative_path", "shard"]
    ].reset_index(drop=True)
    output["review_cluster"] = coarse_labels[rows]
    output = pd.concat(
        [
            output,
            selected[
                [
                    "anatomical_view_category",
                    "category_quota",
                    "sampling_seed",
                ]
            ],
        ],
        axis=1,
    )
    counts = output["anatomical_view_category"].value_counts().to_dict()
    if counts != CATEGORY_QUOTAS:
        raise RuntimeError(f"category quota invariant failed: {counts}")
    return output
