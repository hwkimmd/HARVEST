#!/usr/bin/env python3
"""Build the fixed HARVEST 100k SSL manifest from completed clinical review."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from harvest.curation import reviewed_coreset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--clustering-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    index = pd.read_parquet(args.index)
    review = pd.read_csv(args.review)
    coarse = args.clustering_dir / "k200"
    output = reviewed_coreset(
        index,
        review,
        np.load(coarse / "labels.npy"),
        np.load(coarse / "similarity.npy"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"wrote {len(output):,} reviewed images -> {args.output}")


if __name__ == "__main__":
    main()
