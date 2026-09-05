"""Fixed HARVEST multi-crop input pipeline using a local DINOv3 checkout."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from harvest.dinov3_adapter import activate_dinov3


def build_dino_augmentation(dinov3_source: str | Path, image_size: int):
    """Build the selected DINO augmentation for 224- or 384-pixel SSL."""
    if image_size not in {224, 384}:
        raise ValueError("HARVEST SSL image_size must be 224 or 384")
    activate_dinov3(dinov3_source)
    from dinov3.data.augmentations import DataAugmentationDINO

    return DataAugmentationDINO(
        global_crops_scale=(0.32, 1.0),
        local_crops_scale=(0.05, 0.32),
        local_crops_number=8,
        global_crops_size=image_size,
        local_crops_size=96,
        patch_size=16,
    )


class MultiCropManifestDataset(Dataset):
    def __init__(self, manifest, data_root, transform):
        rows = pd.read_csv(manifest)
        if "relative_path" not in rows:
            raise ValueError("SSL manifest requires relative_path")
        root = Path(data_root).expanduser().resolve()
        self.paths = [root / str(path) for path in rows.relative_path]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def _open(path):
        with Image.open(path) as image:
            return image.convert("RGB")

    def __getitem__(self, index):
        path = self.paths[index]
        try:
            image = self._open(path)
        except Exception:
            previous = ImageFile.LOAD_TRUNCATED_IMAGES
            try:
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                image = self._open(path)
            finally:
                ImageFile.LOAD_TRUNCATED_IMAGES = previous
        return self.transform(image)


def collate_multicrop(batch):
    n_global = len(batch[0]["global_crops"])
    n_local = len(batch[0]["local_crops"])
    global_crops = torch.stack(
        [sample["global_crops"][view] for view in range(n_global) for sample in batch]
    )
    local_crops = torch.stack(
        [sample["local_crops"][view] for view in range(n_local) for sample in batch]
    )
    return {"global_crops": global_crops, "local_crops": local_crops}
