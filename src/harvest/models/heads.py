"""Small batch-safe classifier head for low-positive fine-tuning."""
from __future__ import annotations

import torch
from torch import nn


class LayerNormLinearHead(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.10):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(features.float())
        return self.classifier(self.dropout(normalized)).squeeze(-1)
