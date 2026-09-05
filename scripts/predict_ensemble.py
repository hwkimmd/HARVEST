#!/usr/bin/env python3
"""Run the complete ten-member HARVEST ensemble."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from harvest.data import build_eval_transform, tta_views
from harvest.dinov3_adapter import load_domain_vitl16
from harvest.ensemble import COMPONENTS, aggregate_harvest
from harvest.models import LayerNormLinearHead, VitFeatures

SPECS = {
    "vitl_ssl224_twoview": {"ssl": "vitl_ssl224", "size": 224},
    "vitl_ssl384_oneview": {"ssl": "vitl_ssl384", "size": 384},
}


class Images(Dataset):
    def __init__(self, rows, root, transform):
        self.rows = rows
        self.paths = [Path(root) / path for path in rows.relative_path]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as handle:
            image = handle.copy()
        return self.transform(image), index


@torch.inference_mode()
def predict_member(source, ssl_checkpoint, member_checkpoint, rows, root, device, workers):
    member = torch.load(member_checkpoint, map_location="cpu", weights_only=False)
    component = member["component"]
    spec = SPECS[component]
    backbone, ssl_payload = load_domain_vitl16(source, ssl_checkpoint, device)
    if ssl_payload.get("model") != spec["ssl"]:
        raise RuntimeError("component and SSL checkpoint do not match")
    features = VitFeatures(backbone).to(device)
    features.load_state_dict(member["backbone_tail"], strict=False)
    head = LayerNormLinearHead(features.output_dim, 0.10).to(device)
    head.load_state_dict(member["head"], strict=True)
    features.eval()
    head.eval()
    loader = DataLoader(
        Images(rows, root, build_eval_transform(spec["size"])),
        batch_size=64 if spec["size"] == 224 else 16,
        shuffle=False, num_workers=workers, pin_memory=True,
        persistent_workers=workers > 0,
    )
    result = np.empty(len(rows), dtype=np.float64)
    for images, indices in loader:
        images = images.to(device, non_blocking=True)
        logits = []
        for view in tta_views(images):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits.append(head(features(view)).float())
        result[indices.numpy()] = torch.stack(logits).mean(0).cpu().numpy()
    return result, float(member["validation_mean"]), float(member["validation_std"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dinov3-source", type=Path, required=True)
    parser.add_argument("--vitl-ssl224-checkpoint", type=Path, required=True)
    parser.add_argument("--vitl-ssl384-checkpoint", type=Path, required=True)
    parser.add_argument("--members-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    rows = pd.read_csv(args.manifest)
    if not {"image_id", "relative_path"}.issubset(rows.columns):
        raise ValueError("manifest requires image_id and relative_path")
    checkpoints = {
        "vitl_ssl224_twoview": args.vitl_ssl224_checkpoint,
        "vitl_ssl384_oneview": args.vitl_ssl384_checkpoint,
    }
    device = torch.device("cuda:0")
    components = {}
    for component in COMPONENTS:
        member_logits, means, stds = [], [], []
        for fold in range(5):
            path = args.members_root / component / f"fold{fold}.pth"
            logits, mean, std = predict_member(
                args.dinov3_source, checkpoints[component], path,
                rows, args.data_root, device, args.workers,
            )
            member_logits.append(logits)
            means.append(mean)
            stds.append(std)
        components[component] = (np.stack(member_logits), np.asarray(means), np.asarray(stds))
    scores = aggregate_harvest(components)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame({"image_id": rows.image_id, "score": scores})
    if "label" in rows:
        output.insert(1, "label", rows.label.to_numpy())
    output.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
