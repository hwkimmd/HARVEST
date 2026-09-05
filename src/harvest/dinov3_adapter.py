
"""Strict adapter to a user-supplied local Meta DINOv3 checkout.

No Meta source or weights are distributed with HARVEST. This module performs
no network access and never downloads a checkpoint.
"""
from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path

import torch

DINO_ARCH = "dinov3_vitl16"
CURATION_ARCH = "dinov3_vits16plus"


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def activate_dinov3(source: str | Path) -> Path:
    source = Path(source).expanduser().resolve()
    required = (source / "hubconf.py", source / "dinov3", source / "LICENSE.md")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "A complete user-supplied Meta DINOv3 checkout is required. "
            f"Missing: {missing}. See docs/DINOV3_SETUP.md."
        )
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return source


def build_vitl16(source: str | Path) -> torch.nn.Module:
    activate_dinov3(source)
    backbones = importlib.import_module("dinov3.hub.backbones")
    return getattr(backbones, DINO_ARCH)(pretrained=False)


def build_vits16plus(source: str | Path) -> torch.nn.Module:
    activate_dinov3(source)
    backbones = importlib.import_module("dinov3.hub.backbones")
    return getattr(backbones, CURATION_ARCH)(pretrained=False)


def _unwrap_state(payload):
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    return payload


def load_official_vitl16(
    source: str | Path,
    checkpoint: str | Path,
    expected_sha256: str | None = None,
    device: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, str]:
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DINOv3 checkpoint not found: {checkpoint}")
    actual = sha256_file(checkpoint)
    if expected_sha256 and actual.lower() != expected_sha256.lower():
        raise RuntimeError("DINOv3 checkpoint SHA-256 mismatch")
    model = build_vitl16(source)
    state = _unwrap_state(torch.load(checkpoint, map_location="cpu", weights_only=True))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"DINOv3 state mismatch: {len(missing)} missing, "
            f"{len(unexpected)} unexpected"
        )
    return model.to(device), actual


def load_official_vits16plus(
    source: str | Path,
    checkpoint: str | Path,
    expected_sha256: str,
    device: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, str]:
    """Load the official ViT-S+/16 used only for GastroNet curation."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DINOv3 checkpoint not found: {checkpoint}")
    actual = sha256_file(checkpoint)
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError("DINOv3 checkpoint SHA-256 mismatch")
    model = build_vits16plus(source)
    state = _unwrap_state(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"DINOv3 state mismatch: {len(missing)} missing, "
            f"{len(unexpected)} unexpected"
        )
    return model.to(device), actual


def load_domain_vitl16(
    source: str | Path,
    checkpoint: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, dict]:
    checkpoint = Path(checkpoint).expanduser().resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("arch") != DINO_ARCH:
        raise RuntimeError("domain checkpoint architecture is not dinov3_vitl16")
    if payload.get("loss_version") != "cross_view_v2":
        raise RuntimeError("domain checkpoint does not use cross_view_v2")
    model = build_vitl16(source)
    model.load_state_dict(payload["teacher_backbone"], strict=True)
    return model.to(device), payload
