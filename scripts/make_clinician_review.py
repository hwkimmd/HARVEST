#!/usr/bin/env python3
"""Create local-only K=200 contact sheets and an anatomical review CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from harvest.curation import REVIEW_CLUSTERS, review_representatives

GROUP_COLORS = {
    "nearest": (46, 125, 50),
    "random": (21, 101, 192),
    "farthest": (198, 40, 40),
}


def contact_sheet(groups, paths, thumb=128, columns=8):
    label_height = 22
    rows = sum((len(groups[name]) + columns - 1) // columns for name in groups)
    sheet = Image.new(
        "RGB",
        (columns * thumb, rows * thumb + 3 * label_height),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    y = 0
    for name, selected in groups.items():
        draw.rectangle(
            (0, y, sheet.width, y + label_height),
            fill=GROUP_COLORS[name],
        )
        draw.text((5, y + 4), name.upper(), fill="white")
        y += label_height
        for position, row in enumerate(selected):
            try:
                with Image.open(paths[row]) as image:
                    image = image.convert("RGB")
                    image.thumbnail((thumb, thumb))
                    tile = Image.new("RGB", (thumb, thumb), (235, 235, 235))
                    tile.paste(
                        image,
                        (
                            (thumb - image.width) // 2,
                            (thumb - image.height) // 2,
                        ),
                    )
            except Exception:
                tile = Image.new("RGB", (thumb, thumb), (50, 50, 50))
            sheet.paste(
                tile,
                (
                    (position % columns) * thumb,
                    y + (position // columns) * thumb,
                ),
            )
        y += ((len(selected) + columns - 1) // columns) * thumb
    return sheet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--clustering-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")
    index = pd.read_parquet(args.index)
    labels = np.load(args.clustering_dir / "k200" / "labels.npy")
    similarity = np.load(args.clustering_dir / "k200" / "similarity.npy")
    root = args.data_root.expanduser().resolve()
    paths = [root / str(path) for path in index["relative_path"]]
    sheets = args.output_dir / "contact_sheets"
    sheets.mkdir(parents=True)
    for cluster in range(REVIEW_CLUSTERS):
        groups = review_representatives(labels, similarity, cluster)
        contact_sheet(groups, paths).save(
            sheets / f"cluster_{cluster:03d}.jpg", quality=90
        )
    pd.DataFrame(
        {
            "cluster": range(REVIEW_CLUSTERS),
            "decision": "",
            "anatomical_view_category": "",
            "notes": "",
        }
    ).to_csv(args.output_dir / "clinician_review.csv", index=False)
    print(
        "Review every sheet with KEEP, EXCLUDE, or UNCERTAIN. "
        "For KEEP only, record egj, other_esophagus, or stomach. "
        "These are anatomical/view categories, not diagnostic labels."
    )


if __name__ == "__main__":
    main()
