"""Fixed HARVEST feature extraction for DINOv3 ViT-L/16."""
from __future__ import annotations

import torch
from torch import nn


class VitFeatures(nn.Module):
    """Concatenate normalized CLS and mean patch-token features."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    @property
    def output_dim(self) -> int:
        return 2 * int(self.backbone.embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.backbone.forward_features(images)
        cls_token = output["x_norm_clstoken"]
        patch_mean = output["x_norm_patchtokens"].mean(dim=1)
        return torch.cat((cls_token, patch_mean), dim=-1)
