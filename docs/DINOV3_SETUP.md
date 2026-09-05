# DINOv3 setup

DINOv3 is intentionally not vendored.

Review Meta's current repository and license first. Then place a checkout in an ignored local directory:

```bash
git clone https://github.com/facebookresearch/dinov3.git third_party/dinov3
git -C third_party/dinov3 checkout 6876159a11b4df116f30f667f8c9888617df0751
pip install -e third_party/dinov3
```

Acquire the official ViT-S+/16 and ViT-L/16 checkpoints through Meta's authorized mechanism and store them under `private_weights/`. Record SHA-256 digests locally and pass them explicitly to HARVEST commands.

Never add the checkout, official checkpoints, SSL checkpoints, fine-tuned members, archives, or container layers to this repository. The adapter validates the checkout shape and checkpoint digest and never downloads code or weights.
