"""Full-frame transforms used by HARVEST fine-tuning."""
from __future__ import annotations
import random
import torch
from PIL import ImageOps
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RandomQuarterTurn:
    def __call__(self, image):
        return image.rotate(random.choice((0, 90, 180, 270)), expand=True)


class WeakAcquisitionAugment:
    def __init__(self, master_probability=0.40, max_operations=2):
        self.master_probability = float(master_probability)
        self.max_operations = int(max_operations)
        if not 0 <= self.master_probability <= 1:
            raise ValueError("master_probability must be in [0, 1]")
        self.operations = (
            (0.30, self._color_jitter), (0.15, self._gamma),
            (0.10, self._blur), (0.10, self._jpeg),
            (0.05, self._noise), (0.10, self._sharpness),
        )

    @staticmethod
    def _color_jitter(x):
        return v2.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.08, hue=0.01)(x)

    @staticmethod
    def _gamma(x):
        return F.adjust_gamma(x, random.uniform(0.90, 1.10))

    @staticmethod
    def _blur(x):
        sigma = random.uniform(0.10, 1.00)
        return F.gaussian_blur(x, [5, 5], [sigma, sigma])

    @staticmethod
    def _jpeg(x):
        return v2.JPEG(quality=random.randint(75, 100))(x)

    @staticmethod
    def _noise(x):
        std = random.uniform(0.0, 0.01) * 255.0
        return (x.float() + torch.randn_like(x, dtype=torch.float32) * std).round().clamp(0, 255).to(torch.uint8)

    @staticmethod
    def _sharpness(x):
        return F.adjust_sharpness(x, random.uniform(0.85, 1.15))

    def __call__(self, x):
        if random.random() >= self.master_probability:
            return x
        selected = [op for probability, op in self.operations if random.random() < probability]
        if len(selected) > self.max_operations:
            selected = random.sample(selected, self.max_operations)
        random.shuffle(selected)
        for operation in selected:
            x = operation(x)
        return x


def _decode_rgb(image):
    return ImageOps.exif_transpose(image).convert("RGB")


def build_train_transform(size, photometric_master_probability):
    return v2.Compose([
        _decode_rgb, RandomQuarterTurn(), v2.ToImage(),
        v2.Resize((size, size), interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
        WeakAcquisitionAugment(photometric_master_probability),
        v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_eval_transform(size):
    return v2.Compose([
        _decode_rgb, v2.ToImage(),
        v2.Resize((size, size), interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
        v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def tta_views(x):
    return [torch.rot90(x, k, dims=(-2, -1)) for k in range(4)]
