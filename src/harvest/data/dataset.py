"""Manifest-backed labeled image dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ManifestDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, transform, repo_root: Path):
        required = {"image_id", "relative_path", "label"}
        missing = required - set(rows.columns)
        if missing:
            raise ValueError(f"manifest missing columns: {sorted(missing)}")
        self.image_ids = rows.image_id.astype(str).tolist()
        self.paths = [repo_root / path for path in rows.relative_path]
        self.labels = rows.label.to_numpy(dtype=np.float32)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as handle:
            image = handle.copy()
        return self.transform(image), self.labels[index], index
