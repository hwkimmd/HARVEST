#!/usr/bin/env python3
"""Run the K=200 spherical FAISS clustering used by HARVEST."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from harvest.curation import (
    CLUSTERING_SEED,
    REVIEW_CLUSTERS,
    cluster_summary,
    normalized_float32,
    spherical_kmeans,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")
    import faiss

    faiss.omp_set_num_threads(args.threads)
    array = np.load(args.embedding_dir / "embeddings.f16.npy", mmap_mode="r")
    index = pd.read_parquet(args.embedding_dir / "index.parquet")
    metadata = json.loads((args.embedding_dir / "metadata.json").read_text())
    if metadata.get("architecture") != "dinov3_vits16plus":
        raise ValueError("HARVEST curation requires DINOv3 ViT-S+/16 embeddings")
    features = normalized_float32(array)
    args.output_dir.mkdir(parents=True)
    for clusters in (REVIEW_CLUSTERS,):
        output = args.output_dir / f"k{clusters}"
        output.mkdir()
        centroids, labels, similarity = spherical_kmeans(features, clusters)
        np.save(output / "centroids.npy", centroids)
        np.save(output / "labels.npy", labels)
        np.save(output / "similarity.npy", similarity)
        cluster_summary(labels, similarity, index).to_csv(
            output / "cluster_summary.csv", index=False
        )
        (output / "metadata.json").write_text(
            json.dumps(
                {
                    "clusters": clusters,
                    "iterations": 25,
                    "seed": CLUSTERING_SEED,
                    "spherical": True,
                    "embedding_checkpoint_sha256": metadata["checkpoint_sha256"],
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
