# Third-party notices

The MIT License in `LICENSE` applies only to original HARVEST code and documentation in this repository. It does not apply to Meta DINOv3, model weights, RARE data, GastroNet data, or other third-party material.

## Meta DINOv3

HARVEST uses a user-supplied Meta DINOv3 checkout and official ViT-S+/16 and ViT-L/16 checkpoints. DINOv3 and its weights are governed by the separate [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md); they are not relicensed under MIT.

This repository contains no files copied from the DINOv3 repository and no official or derivative DINOv3 weights. HARVEST calls `DINOHead`, `DINOLoss`, `KoLeoLoss`, and `DataAugmentationDINO` only at runtime from the user-supplied checkout; those components remain governed by the DINOv3 License. In particular it contains no pretrained, domain-SSL, teacher, student, backbone, trunk, tail, head, fine-tuned, ensemble, Docker, or submission artifacts.

Users must obtain DINOv3 directly from Meta, review its current license, and keep the checkout and every weight file outside version control. The HARVEST adapter performs no download.

## Data

RARE and GastroNet images are not redistributed. Users must obtain authorized access and comply with the applicable dataset terms. Committed example manifests contain synthetic rows only.
