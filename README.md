# HARVEST

**Human-curated Adaptation of Representations via Endoscopic Self-supervision and Transfer**

Repository: https://github.com/hwkimmd/HARVEST

This repository contains the original MIT-licensed code needed to reproduce the HARVEST pipeline for the RARE26 task. It covers GastroNet-5M curation, dual-resolution domain self-supervised learning, five-fold fine-tuning, and final ensemble prediction. Only the final HARVEST path is exposed; experimental ablations are intentionally absent.

## Critical license boundary

The repository itself is MIT-licensed. Meta DINOv3 is not. No DINOv3 source, official checkpoint, adapted checkpoint, fine-tuned weight, submission archive, or dataset image is included or licensed by this repository. Obtain DINOv3 independently from Meta and comply with its current license.

Read [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [docs/LICENSE_BOUNDARY.md](docs/LICENSE_BOUNDARY.md), and [docs/DINOV3_SETUP.md](docs/DINOV3_SETUP.md) before use. Keep the user-supplied DINOv3 checkout and all weights in ignored locations such as `third_party/dinov3/` and `private_weights/`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[curation,test]"
```

Install the CUDA-compatible PyTorch build appropriate for the local system if the default package is unsuitable. DINOv3 must be obtained and installed separately as described in [docs/DINOV3_SETUP.md](docs/DINOV3_SETUP.md).

## Method

The commands below show the complete path from source images to final prediction. Replace placeholder paths with authorized local data and checkpoint locations. SHA-256 values are computed locally from the downloaded checkpoint files; they are integrity identifiers, not credentials or an independent attestation from Meta.

### 1. Index GastroNet-5M

```bash
python scripts/index_gastronet.py \
  --data-root /path/to/gastronet \
  --output private_data/gastronet.index.parquet
```

### 2. Embed GastroNet-5M with ViT-S+/16

First record the digest of the authorized local checkpoint:

```bash
sha256sum private_weights/dinov3_vits16plus.pth
```

Then pass the printed digest explicitly:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/embed_gastronet.py \
  --index private_data/gastronet.index.parquet \
  --data-root /path/to/gastronet \
  --dinov3-source third_party/dinov3 \
  --official-checkpoint private_weights/dinov3_vits16plus.pth \
  --checkpoint-sha256 YOUR_LOCAL_SHA256 \
  --output-dir outputs/gastronet_embeddings
```

### 3. Run spherical FAISS K=200 clustering

```bash
python scripts/cluster_gastronet.py \
  --embedding-dir outputs/gastronet_embeddings \
  --output-dir outputs/gastronet_k200
```

### 4. Generate the clinician-review material

```bash
python scripts/make_clinician_review.py \
  --index outputs/gastronet_embeddings/index.parquet \
  --data-root /path/to/gastronet \
  --clustering-dir outputs/gastronet_k200 \
  --output-dir outputs/clinician_review
```

A clinician completes `outputs/clinician_review/clinician_review.csv` using the generated contact sheets.

### 5. Build the reviewed 100,000-image corpus

```bash
python scripts/build_reviewed_coreset.py \
  --index outputs/gastronet_embeddings/index.parquet \
  --review outputs/clinician_review/clinician_review.csv \
  --clustering-dir outputs/gastronet_k200 \
  --output private_data/harvest_100k.csv
```

### 6. Train the 224- and 384-pixel domain-SSL backbones

Compute the local ViT-L/16 checkpoint digest with `sha256sum`, then run both configurations:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_domain_ssl.py \
  --model vitl_ssl224 \
  --manifest private_data/harvest_100k.csv \
  --data-root /path/to/gastronet \
  --dinov3-source third_party/dinov3 \
  --official-checkpoint private_weights/dinov3_vitl16.pth \
  --checkpoint-sha256 YOUR_LOCAL_SHA256 \
  --output runs/vitl_ssl224

CUDA_VISIBLE_DEVICES=0 python scripts/train_domain_ssl.py \
  --model vitl_ssl384 \
  --manifest private_data/harvest_100k.csv \
  --data-root /path/to/gastronet \
  --dinov3-source third_party/dinov3 \
  --official-checkpoint private_weights/dinov3_vitl16.pth \
  --checkpoint-sha256 YOUR_LOCAL_SHA256 \
  --output runs/vitl_ssl384
```

### 7. Prepare the RARE25 labeled split

```bash
python scripts/prepare_splits.py \
  --manifest private_data/rare_labeled.csv \
  --output private_data/rare_split.csv
```

### 8. Fine-tune five folds for both final components

```bash
for fold in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=0 python scripts/train_supervised.py \
    --model vitl_ssl224_twoview \
    --fold "$fold" \
    --split-manifest private_data/rare_split.csv \
    --data-root /path/to/rare \
    --dinov3-source third_party/dinov3 \
    --ssl-checkpoint runs/vitl_ssl224/checkpoint.pth \
    --output-root runs/members

done

for fold in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=0 python scripts/train_supervised.py \
    --model vitl_ssl384_oneview \
    --fold "$fold" \
    --split-manifest private_data/rare_split.csv \
    --data-root /path/to/rare \
    --dinov3-source third_party/dinov3 \
    --ssl-checkpoint runs/vitl_ssl384/checkpoint.pth \
    --output-root runs/members

done
```

### 9. Generate final HARVEST predictions

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/predict_ensemble.py \
  --manifest private_data/test.csv \
  --data-root /path/to/images \
  --dinov3-source third_party/dinov3 \
  --vitl-ssl224-checkpoint runs/vitl_ssl224/checkpoint.pth \
  --vitl-ssl384-checkpoint runs/vitl_ssl384/checkpoint.pth \
  --members-root runs/members \
  --output outputs/predictions.csv
```

If the prediction manifest contains a `label` column, evaluate the output with:

```bash
python scripts/evaluate_predictions.py \
  --predictions outputs/predictions.csv
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for exact commands and [docs/METHOD.md](docs/METHOD.md) for the algorithm.

## Repository contents

- `src/harvest/`: installable HARVEST package;
- `scripts/`: command-line entry points for the end-to-end pipeline;
- `configs/harvest.json`: machine-readable fixed settings;
- `examples/`: synthetic input schemas only;
- `tests/`: metric, split, ensemble, and license-boundary checks;
- `docs/`: method, reproducibility, curation, and third-party setup documentation.

All committed CSV files contain synthetic rows. Real review decisions, image paths, data, hashes tied to private artifacts, and all weights remain local and ignored.

This is research code, not a medical device.
