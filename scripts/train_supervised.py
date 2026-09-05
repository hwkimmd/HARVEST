#!/usr/bin/env python3
"""Five-fold fine-tuning for the two HARVEST components only."""
from __future__ import annotations

import argparse
import copy
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from harvest.data import (
    ManifestDataset,
    build_eval_transform,
    build_train_transform,
    pooled_batch_sampler,
    tta_views,
    two_view_batch_sampler,
)
from harvest.dinov3_adapter import load_domain_vitl16
from harvest.ensemble import fit_affine
from harvest.models import LayerNormLinearHead, VitFeatures

MODELS = {
    "vitl_ssl224_twoview": {"size": 224, "ssl": "vitl_ssl224", "views": 2},
    "vitl_ssl384_oneview": {"size": 384, "ssl": "vitl_ssl384", "views": 1},
}
SEED = 20260830


class TwoViewDataset(Dataset):
    def __init__(self, rows, root, transform):
        self.rows = rows.reset_index(drop=True)
        self.paths = [Path(root) / path for path in self.rows.relative_path]
        self.labels = self.rows.label.to_numpy(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as handle:
            image = handle.copy()
        return self.transform(image), self.transform(image), self.labels[index], index


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dinov3-source", type=Path, required=True)
    parser.add_argument("--ssl-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=15)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(rows, root, transform, workers, batch_size=64, batch_sampler=None):
    dataset = ManifestDataset(rows, transform, Path(root))
    kwargs = {"dataset": dataset, "num_workers": workers, "pin_memory": True, "persistent_workers": workers > 0}
    if batch_sampler is None:
        kwargs.update(batch_size=batch_size, shuffle=False)
    else:
        kwargs["batch_sampler"] = batch_sampler
    return DataLoader(**kwargs)


@torch.inference_mode()
def predict(features, head, loader, device, rotations=1):
    features.eval()
    head.eval()
    logits, indices = [], []
    for images, _, batch_indices in loader:
        images = images.to(device, non_blocking=True)
        views = tta_views(images) if rotations == 4 else [images]
        scores = []
        for view in views:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                scores.append(head(features(view)).float())
        logits.append(torch.stack(scores).mean(0).cpu().numpy())
        indices.append(batch_indices.numpy())
    return np.concatenate(logits), np.concatenate(indices)


def auprc(rows, logits, order):
    return float(average_precision_score(rows.label.to_numpy()[order], logits))


def unfreeze_tail(features):
    features.backbone.requires_grad_(False)
    for block in list(features.backbone.blocks)[20:24]:
        block.requires_grad_(True)
    features.backbone.norm.requires_grad_(True)


def trainable_state(module):
    names = {name for name, parameter in module.named_parameters() if parameter.requires_grad}
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items() if name in names}


def partial_optimizer(features, head):
    groups = []
    for name, parameter in features.backbone.named_parameters():
        if not parameter.requires_grad:
            continue
        lr = 3e-6
        if name.startswith("blocks."):
            lr *= 0.85 ** (23 - int(name.split(".")[1]))
        decay = 0.0 if parameter.ndim == 1 or name.endswith((".bias", ".gamma")) else 0.02
        groups.append({"params": [parameter], "lr": lr, "base_lr": lr, "weight_decay": decay})
    for name, parameter in head.named_parameters():
        decay = 0.02 if name == "classifier.weight" else 0.0
        groups.append({"params": [parameter], "lr": 1e-4, "base_lr": 1e-4, "weight_decay": decay})
    return torch.optim.AdamW(groups)


def lr_scale(step, total, warmup):
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup - 1, 1)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def load_features(args, device):
    backbone, payload = load_domain_vitl16(args.dinov3_source, args.ssl_checkpoint, device)
    if payload.get("model") != MODELS[args.model]["ssl"]:
        raise RuntimeError("fine-tuning model and SSL checkpoint do not match")
    if payload.get("step") != 12000 or payload.get("loss_version") != "cross_view_v2":
        raise RuntimeError("HARVEST requires a completed 12,000-update SSL checkpoint")
    return VitFeatures(backbone).to(device), payload


def frozen_probe(args, train, validation, device, size, seed):
    features, ssl_payload = load_features(args, device)
    features.requires_grad_(False).eval()
    head = LayerNormLinearHead(features.output_dim, 0.10).to(device)
    sampler = pooled_batch_sampler(train.label, train.center, batch_size=32, seed=seed)
    train_loader = make_loader(train, args.data_root, build_train_transform(size, 0.40), args.workers, batch_sampler=sampler)
    validation_loader = make_loader(validation, args.data_root, build_eval_transform(size), args.workers, batch_size=64 if size == 224 else 16)
    optimizer = torch.optim.AdamW([
        {"params": [head.classifier.weight], "weight_decay": 0.01},
        {"params": [head.norm.weight, head.norm.bias, head.classifier.bias], "weight_decay": 0.0},
    ], lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    criterion = nn.BCEWithLogitsLoss()
    best, bad = {"score": -math.inf}, 0
    for epoch in range(args.max_epochs):
        sampler.set_epoch(epoch)
        head.train()
        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                embeddings = features(images)
            loss = criterion(head(embeddings), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 3.0)
            optimizer.step()
        scheduler.step()
        logits, order = predict(features, head, validation_loader, device)
        score = auprc(validation, logits, order)
        if score > best["score"]:
            best = {"score": score, "head": copy.deepcopy({k: v.cpu() for k, v in head.state_dict().items()}), "epoch": epoch + 1}
            bad = 0
        else:
            bad += 1
        if epoch + 1 >= 5 and bad >= 4:
            break
    return best, ssl_payload


def train_partial(args, train, validation, device, size, views, frozen, seed):
    features, _ = load_features(args, device)
    unfreeze_tail(features)
    head = LayerNormLinearHead(features.output_dim, 0.10).to(device)
    head.load_state_dict(frozen["head"], strict=True)
    transform = build_train_transform(size, 0.40 if views == 2 else 0.20)
    if views == 2:
        sampler = two_view_batch_sampler(train.label, seed=seed)
        dataset = TwoViewDataset(train, args.data_root, transform)
        train_loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    else:
        sampler = pooled_batch_sampler(train.label, train.center, batch_size=32, seed=seed)
        train_loader = make_loader(train, args.data_root, transform, args.workers, batch_sampler=sampler)
    validation_loader = make_loader(validation, args.data_root, build_eval_transform(size), args.workers, batch_size=64 if size == 224 else 16)
    optimizer = partial_optimizer(features, head)
    criterion = nn.BCEWithLogitsLoss()
    total = args.max_epochs * len(train_loader)
    warmup = len(train_loader)
    best, bad, step = {"score": -math.inf}, 0, 0
    trainable = [p for module in (features, head) for p in module.parameters() if p.requires_grad]
    for epoch in range(args.max_epochs):
        sampler.set_epoch(epoch)
        features.train()
        head.train()
        for batch in train_loader:
            scale = lr_scale(step, total, warmup)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * scale
            optimizer.zero_grad(set_to_none=True)
            if views == 2:
                image1, image2, labels, _ = batch
                image1, image2 = image1.to(device), image2.to(device)
                labels = labels.to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = 0.5 * (criterion(head(features(image1)).float(), labels) + criterion(head(features(image2)).float(), labels))
            else:
                images, labels, _ = batch
                images, labels = images.to(device), labels.to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = criterion(head(features(images)).float(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 3.0)
            optimizer.step()
            step += 1
        logits, order = predict(features, head, validation_loader, device)
        score = auprc(validation, logits, order)
        if score > best["score"]:
            best = {
                "score": score, "epoch": epoch + 1,
                "backbone_tail": trainable_state(features),
                "head": copy.deepcopy({k: v.cpu() for k, v in head.state_dict().items()}),
            }
            bad = 0
        else:
            bad += 1
        if epoch + 1 >= 5 and bad >= 4:
            break
    features.load_state_dict(best["backbone_tail"], strict=False)
    head.load_state_dict(best["head"])
    logits, order = predict(features, head, validation_loader, device, rotations=4)
    best["validation_mean"], best["validation_std"] = fit_affine(logits)
    best["validation_auprc_tta4"] = auprc(validation, logits, order)
    return best


def main():
    args = arguments()
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        raise SystemExit("CUDA_VISIBLE_DEVICES must select exactly one GPU")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    spec = MODELS[args.model]
    rows = pd.read_csv(args.split_manifest)
    required = {"image_id", "relative_path", "label", "center", "partition", "cv_fold"}
    if missing := required - set(rows.columns):
        raise ValueError(f"split manifest missing columns: {sorted(missing)}")
    development = rows.loc[rows.partition == "development"]
    train = development.loc[development.cv_fold != args.fold].reset_index(drop=True)
    validation = development.loc[development.cv_fold == args.fold].reset_index(drop=True)
    if train.empty or validation.empty:
        raise ValueError("empty development train or validation fold")
    output = args.output_root / args.model / f"fold{args.fold}.pth"
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    seed = SEED + 1000 * args.fold
    set_seed(seed)
    device = torch.device("cuda:0")
    frozen, ssl_payload = frozen_probe(args, train, validation, device, spec["size"], seed)
    final = train_partial(args, train, validation, device, spec["size"], spec["views"], frozen, seed)
    torch.save({
        "model_name": "HARVEST", "component": args.model, "fold": args.fold,
        "ssl_model": spec["ssl"], "ssl_source_checkpoint_sha256": ssl_payload["source_checkpoint_sha256"],
        "image_size": spec["size"], "views": spec["views"],
        "initialized_from_same_fold_frozen_probe": True,
        "frozen_best_epoch": frozen["epoch"], **final,
    }, output)


if __name__ == "__main__":
    main()
