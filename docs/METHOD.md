# HARVEST method

HARVEST means **Human-curated Adaptation of Representations via Endoscopic Self-supervision and Transfer**. The method turns a large unlabeled endoscopy collection into a clinician-guided domain-SSL corpus, adapts two image-resolution backbones, fine-tunes complementary supervised components, and combines them into one prediction.

## 1. GastroNet-5M indexing and representation extraction

GastroNet-5M is indexed once in deterministic row order. Every image is resized and center-cropped to 224 pixels, normalized with ImageNet mean and standard deviation, and embedded with a frozen, user-supplied official DINOv3 ViT-S+/16 checkpoint. Embeddings are stored in float16 and aligned one-to-one with the image index.

HARVEST does not contain or download the DINOv3 implementation or checkpoint. The external adapter imports the separately installed checkout at runtime and verifies the locally recorded checkpoint SHA-256.

## 2. Spherical clustering and clinician review

The L2-normalized embeddings are partitioned using 25-iteration spherical FAISS K-means with K=200 and seed 20260822. For each cluster, the review material contains centroid-nearest, seeded-random, and centroid-farthest examples so that both the core appearance and cluster heterogeneity can be assessed.

A clinician assigns `KEEP`, `EXCLUDE`, or `UNCERTAIN` and labels the anatomical/view category of retained clusters. The finalized review retains 65 clusters: 22 EGJ, 8 other-esophagus, and 35 stomach/broader-upper-GI clusters. Review decisions and image-level manifests remain private data.

## 3. Reviewed 100K corpus construction

Sampling is performed independently inside every retained K=200 cluster:

1. rank members by cosine similarity to the assigned centroid;
2. remove `round(0.20 × cluster size)` least-similar members;
3. allocate each category target among eligible clusters proportional to the square root of eligible cluster size, subject to capacity and deterministic cluster-ID tie breaking;
4. sample the allocated quota uniformly without replacement with seed 20260830.

The category targets are 50,000 EGJ, 25,000 other-esophagus, and 25,000 stomach/broader-upper-GI images. The result is exactly 100,000 unique images. No secondary K=1000 clustering is used.

## 4. Dual-resolution domain SSL

Two DINOv3 ViT-L/16 backbones are adapted on the same reviewed corpus:

- `vitl_ssl224`: 224-pixel global crops and 96-pixel local crops;
- `vitl_ssl384`: 384-pixel global crops and 96-pixel local crops.

Both use two global crops, eight local crops, the corrected 18-term cross-view DINO objective excluding same-view global pairs, KoLeo weight 0.1, last-eight-block plus final-norm unfreezing, batch 128, AdamW, 12,000 updates, and seed 20260830. The DINO head, centering loss, KoLeo loss, and augmentation are imported at runtime from the user's local DINOv3 checkout. HARVEST supplies the training orchestration and fixed experiment contract, not a copy of those Meta components.

## 5. Labeled split and supervised adaptation

The RARE25 labeled manifest is divided once into a 20% fixed internal test partition and five center-by-label-stratified development folds. The fixed partition is excluded from model selection.

Each component and fold first fits a frozen linear probe with batch 32 and weak full-frame augmentation with photometric master probability 0.40. The selected frozen probe initializes partial fine-tuning, where blocks 20–23 and the final norm are unfrozen.

- `vitl_ssl224_twoview`: two independently augmented views of each sampled source image; the two BCE losses are averaged before backpropagation. Training uses batch 64 with 16 positive and 48 random-negative samples and photometric probability 0.40.
- `vitl_ssl384_oneview`: one augmented view per sampled image, center-by-label-balanced batch 32, and photometric probability 0.20.

Both components use exact random quarter turns, direct bicubic square resize, ImageNet normalization, at most 15 epochs, a minimum of 5 epochs, patience 4, and development AUPRC for checkpoint selection. Two-view training still uses one shared backbone and classifier; it does not create two independently parameterized networks.

## 6. Fold-member inference

At inference, every fold member receives four deterministic views produced by 0°, 90°, 180°, and 270° rotations. The four logits are averaged for that member. This test-time augmentation is applied to both the 224- and 384-pixel components.

Each raw member logit is then standardized using the corresponding fold-validation mean and population standard deviation (`ddof=0`). Within each five-fold component, the lowest and highest standardized logits are removed per image and the middle three are averaged.

## 7. Final dual-resolution ensemble

HARVEST averages the aggregated 224-pixel two-view component and the aggregated 384-pixel one-view component with weights 0.5 and 0.5. A single numerically stable sigmoid converts the final combined logit into the reported probability.

In compact form:

```text
image
 ├─ vitl_ssl224_twoview × 5 folds ─ 4-rotation TTA ─ validation standardization ─ middle-three mean ─┐
 └─ vitl_ssl384_oneview × 5 folds ─ 4-rotation TTA ─ validation standardization ─ middle-three mean ─┤
                                                                                                      └─ 0.5/0.5 logit mean ─ sigmoid
```

The implementation IDs describe pipeline components only. The public model name is HARVEST.
