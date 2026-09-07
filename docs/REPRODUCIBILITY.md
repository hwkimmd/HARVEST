# Reproducibility

This document gives the executable HARVEST pipeline from authorized local data to final predictions. The commands use repository-relative output locations that are excluded by `.gitignore`.

## 1. Environment and external dependency

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[curation,test]"
```

Follow [DINOV3_SETUP.md](DINOV3_SETUP.md) to place the user-supplied Meta DINOv3 checkout at `third_party/dinov3/`. Store authorized ViT-S+/16 and ViT-L/16 checkpoints under `private_weights/`. Neither the checkout nor any checkpoint is distributed here.

Calculate each local checkpoint digest:

```bash
sha256sum private_weights/dinov3_vits16plus.pth
sha256sum private_weights/dinov3_vitl16.pth
```

Use the corresponding printed value for every `--checkpoint-sha256` argument. This detects accidental checkpoint substitution and records provenance. It does not independently prove that a checkpoint originated from Meta.

## 2. GastroNet-5M indexing and embedding

Build a deterministic image index:

```bash
python scripts/index_gastronet.py \
  --data-root /path/to/gastronet \
  --output private_data/gastronet.index.parquet
```

The released pipeline expects 4,823,974 indexed images. Embed them in stable index order using the frozen, user-supplied ViT-S+/16 checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/embed_gastronet.py \
  --index private_data/gastronet.index.parquet \
  --data-root /path/to/gastronet \
  --dinov3-source third_party/dinov3 \
  --official-checkpoint private_weights/dinov3_vits16plus.pth \
  --checkpoint-sha256 YOUR_VITS16PLUS_LOCAL_SHA256 \
  --output-dir outputs/gastronet_embeddings
```

The output contains a float16 embedding matrix, the aligned index, and metadata recording architecture, preprocessing, image count, dimension, and checkpoint digest.

## 3. K=200 clustering and clinician review

Run deterministic 25-iteration spherical FAISS K-means:

```bash
python scripts/cluster_gastronet.py \
  --embedding-dir outputs/gastronet_embeddings \
  --output-dir outputs/gastronet_k200 \
  --threads 32
```

Generate review contact sheets and a review table:

```bash
python scripts/make_clinician_review.py \
  --index outputs/gastronet_embeddings/index.parquet \
  --data-root /path/to/gastronet \
  --clustering-dir outputs/gastronet_k200 \
  --output-dir outputs/clinician_review
```

A clinician records `KEEP`, `EXCLUDE`, or `UNCERTAIN` and assigns a category to every retained cluster. The required schema is illustrated by `examples/manifests/`; real review decisions are private and are not distributed.

Build the fixed 100,000-image manifest:

```bash
python scripts/build_reviewed_coreset.py \
  --index outputs/gastronet_embeddings/index.parquet \
  --review outputs/clinician_review/clinician_review.csv \
  --clustering-dir outputs/gastronet_k200 \
  --output private_data/harvest_100k.csv
```

This removes the farthest 20% within each retained cluster and performs seeded, without-replacement sampling under the category targets in `configs/harvest.json`.

## 4. Dual-resolution domain SSL

Run the two fixed configurations independently. Both start from the same authorized ViT-L/16 checkpoint and use the reviewed 100K corpus:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_domain_ssl.py \
  --model vitl_ssl224 \
  --manifest private_data/harvest_100k.csv \
  --data-root /path/to/gastronet \
  --dinov3-source third_party/dinov3 \
  --official-checkpoint private_weights/dinov3_vitl16.pth \
  --checkpoint-sha256 YOUR_VITL16_LOCAL_SHA256 \
  --output runs/vitl_ssl224

CUDA_VISIBLE_DEVICES=0 python scripts/train_domain_ssl.py \
  --model vitl_ssl384 \
  --manifest private_data/harvest_100k.csv \
  --data-root /path/to/gastronet \
  --dinov3-source third_party/dinov3 \
  --official-checkpoint private_weights/dinov3_vitl16.pth \
  --checkpoint-sha256 YOUR_VITL16_LOCAL_SHA256 \
  --output runs/vitl_ssl384
```

Each output directory contains the resulting `checkpoint.pth` and run metadata. These derived checkpoints remain governed by the applicable DINOv3 terms and must not be committed.

## 5. RARE25 split and five-fold fine-tuning

The labeled input manifest `private_data/rare_labeled.csv` must contain the fields shown in `examples/manifests/labeled.example.csv`. RARE25-specific directory and annotation parsing is kept outside the training pipeline; users should locally map their authorized copy of RARE25 to this standardized manifest format.

`relative_path` is interpreted relative to the directory passed via `--data-root`. For example, `relative_path=center_1/negative/example_0001.png` with `--data-root /path/to/rare` refers to `/path/to/rare/center_1/negative/example_0001.png`.

Create the fixed internal-test/development assignment and five development folds:

```bash
python scripts/prepare_splits.py \
  --manifest private_data/rare_labeled.csv \
  --output private_data/rare_split.csv
```

Train the final two components over folds 0–4:

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

Each component produces five fold members. Model selection uses only development-fold predictions; the fixed internal test partition is reporting-only.

## 6. Ensemble prediction and optional evaluation

The prediction manifest must contain `image_id` and `relative_path`. Run the fixed HARVEST ensemble:

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

If labels are present, calculate the released official-style metrics:

```bash
python scripts/evaluate_predictions.py \
  --predictions outputs/predictions.csv
```

## 7. Fixed controls and expected boundaries

- Global seed: `20260830`.
- Clustering: spherical FAISS K=200.
- Domain SSL: two global and eight local crops, batch 128, 12,000 updates.
- Fine-tuning: five development folds and one untouched internal fixed partition.
- Inference: four quarter-turn views per fold member, five members per component, two components.
- No network download is performed by HARVEST code.
- All DINOv3 code and weights are supplied separately by the user.
- Dataset-specific runtime and memory depend on storage throughput, GPU architecture, and local PyTorch/CUDA builds.

The complete fixed settings are available in `configs/harvest.json`; the algorithmic rationale is described in [METHOD.md](METHOD.md).
