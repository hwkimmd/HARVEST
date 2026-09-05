#!/usr/bin/env python3
"""Train one of the two HARVEST domain-SSL backbones."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from harvest.dinov3_adapter import DINO_ARCH, load_official_vitl16
from harvest.ssl import HarvestDomainSSLTrainer
from harvest.ssl.data import MultiCropManifestDataset, build_dino_augmentation, collate_multicrop

MODELS = {"vitl_ssl224": 224, "vitl_ssl384": 384}
SEED = 20260830
UPDATES = 12000
BATCH_SIZE = 128
BACKBONE_LR = 1e-5
HEAD_LR = 1e-3
LAYER_DECAY = 0.98
WEIGHT_DECAY = 0.04


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dinov3-source", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--smoke-batches", type=int, default=0)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unfreeze_last_eight(backbone):
    backbone.requires_grad_(False)
    for block in list(backbone.blocks)[-8:]:
        block.requires_grad_(True)
    backbone.norm.requires_grad_(True)


def schedule(step, total, warmup, peak, end):
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup - 1, 1)
    return end + (peak - end) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def main():
    args = arguments()
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        raise SystemExit("CUDA_VISIBLE_DEVICES must select exactly one GPU")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    image_size = MODELS[args.model]
    set_seed(SEED)
    device = torch.device("cuda:0")

    probe, source_sha = load_official_vitl16(
        args.dinov3_source, args.official_checkpoint,
        args.checkpoint_sha256, "cpu"
    )
    embed_dim = int(probe.embed_dim)
    del probe

    def build_backbone():
        model, _ = load_official_vitl16(
            args.dinov3_source, args.official_checkpoint,
            args.checkpoint_sha256, "cpu"
        )
        unfreeze_last_eight(model)
        return model

    trainer = HarvestDomainSSLTrainer(build_backbone, embed_dim, args.dinov3_source).to(device)
    transform = build_dino_augmentation(args.dinov3_source, image_size)
    dataset = MultiCropManifestDataset(args.manifest, args.data_root, transform)
    loader_kwargs = dict(
        dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_multicrop,
        generator=torch.Generator().manual_seed(SEED), persistent_workers=args.workers > 0,
    )
    if args.workers:
        loader_kwargs["prefetch_factor"] = 4
    loader = DataLoader(**loader_kwargs)
    if not len(loader):
        raise RuntimeError("dataset is smaller than one drop-last batch")

    groups = []
    for name, parameter in trainer.student.backbone.named_parameters():
        if not parameter.requires_grad:
            continue
        multiplier = 1.0 if name.startswith("norm") else LAYER_DECAY ** (23 - int(name.split(".")[1]))
        decay = 0.0 if parameter.ndim <= 1 or name.endswith((".bias", ".gamma")) else WEIGHT_DECAY
        groups.append({"params": [parameter], "mult": multiplier, "weight_decay": decay, "kind": "backbone"})
    for name, parameter in trainer.student.head.named_parameters():
        decay = 0.0 if parameter.ndim <= 1 or name.endswith(".bias") else WEIGHT_DECAY
        groups.append({"params": [parameter], "mult": 1.0, "weight_decay": decay, "kind": "head"})
    optimizer = torch.optim.AdamW(groups)
    total = args.smoke_batches or UPDATES
    warmup = max(1, round(0.10 * total))
    iterator = iter(loader)
    trainable = [parameter for parameter in trainer.student.parameters() if parameter.requires_grad]
    metadata = {
        "model": args.model, "arch": DINO_ARCH, "image_size": image_size,
        "local_crop_size": 96, "loss_version": "cross_view_v2",
        "updates": total, "valid_cross_view_terms": 18,
        "source_checkpoint_sha256": source_sha, "seed": SEED,
    }
    (args.output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    started = time.time()
    for step in range(total):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        progress = step / max(total - 1, 1)
        backbone_lr = schedule(step, total, warmup, BACKBONE_LR, BACKBONE_LR * 0.1)
        head_lr = schedule(step, total, warmup, HEAD_LR, HEAD_LR * 0.001)
        for group in optimizer.param_groups:
            group["lr"] = (backbone_lr if group["kind"] == "backbone" else head_lr) * group["mult"]
        global_crops = batch["global_crops"].to(device, non_blocking=True)
        local_crops = batch["local_crops"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = trainer(global_crops, local_crops, progress)
        output["loss"].backward()
        if step < len(loader):
            trainer.student.head.last_layer.weight.grad = None
        torch.nn.utils.clip_grad_norm_(trainable, 3.0)
        optimizer.step()
        trainer.update_teacher(trainer.ema_momentum(progress))
        if step == 0 or (step + 1) % 20 == 0:
            with (args.output / "train.jsonl").open("a") as handle:
                handle.write(json.dumps({
                    "step": step + 1, "loss": float(output["loss"].detach()),
                    "dino": float(output["dino"]), "koleo": float(output["koleo"]),
                    "elapsed_seconds": time.time() - started,
                }) + "\n")

    torch.save({
        **metadata, "step": total,
        "teacher_backbone": {k: v.detach().cpu() for k, v in trainer.teacher.backbone.state_dict().items()},
    }, args.output / "checkpoint.pth")


if __name__ == "__main__":
    main()
