#!/usr/bin/env python3
"""Embed the full GastroNet-5M index with frozen DINOv3 ViT-S+/16."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from harvest.curation import EXPECTED_IMAGES, GastroNetImages
from harvest.dinov3_adapter import load_official_vits16plus

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
BATCH_SIZE = 128


def transform():
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(256, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(MEAN, STD),
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dinov3-source", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise SystemExit("set CUDA_VISIBLE_DEVICES explicitly")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    index = pd.read_parquet(args.index)
    if len(index) != EXPECTED_IMAGES or not np.array_equal(
        index["row"].to_numpy(), np.arange(len(index))
    ):
        raise ValueError("index is incomplete or row ordering is invalid")
    device = torch.device("cuda:0")
    model, checkpoint_sha = load_official_vits16plus(
        args.dinov3_source,
        args.official_checkpoint,
        args.checkpoint_sha256,
        device,
    )
    model.eval()
    dataset = GastroNetImages(index, args.data_root, transform())
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    output = np.lib.format.open_memmap(
        args.output_dir / "embeddings.f16.npy",
        mode="w+",
        dtype=np.float16,
        shape=(len(index), int(model.embed_dim)),
    )
    failed = []
    with torch.inference_mode():
        for images, rows, ok in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                features = model(images.to(device, non_blocking=True))
            output[rows.numpy()] = features.float().cpu().numpy().astype(np.float16)
            failed.extend(rows[~ok].tolist())
    output.flush()
    if failed:
        raise RuntimeError(
            f"{len(failed)} images failed to decode; zero rows remain in the embedding"
        )
    index.to_parquet(args.output_dir / "index.parquet", index=False)
    metadata = {
        "architecture": "dinov3_vits16plus",
        "checkpoint_sha256": checkpoint_sha,
        "images": len(index),
        "dimension": int(model.embed_dim),
        "preprocessing": "resize_short_side_256_center_crop_224_imagenet_normalization",
        "dtype": "float16",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
