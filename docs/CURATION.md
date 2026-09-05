# GastroNet-100k curation

HARVEST indexes the 4,823,974 GastroNet PNG files deterministically and embeds them with a frozen, user-supplied official DINOv3 ViT-S+/16 model. Embeddings are L2-normalized and clustered by 25-iteration spherical FAISS K-means with K=200 and seed 20260822.

For every cluster, the review sheet contains 24 centroid-nearest, 24 seeded-random, and 16 centroid-farthest images. A clinician records one of KEEP, EXCLUDE, or UNCERTAIN and assigns each KEEP cluster to an anatomical/view category.

The finalized review contains 65 KEEP clusters: 22 EGJ, 8 other-esophagus, and 35 stomach/broader-upper-GI clusters. The review CSV itself is private data and is not distributed.

Selection is performed independently within every KEEP cluster:

1. sort members by cosine similarity to the assigned centroid;
2. remove `round(0.20 * cluster_size)` least-similar members;
3. allocate each category target across its eligible clusters proportional to the square root of eligible cluster size, with capacity caps and stable cluster-ID tie breaking;
4. uniformly sample each cluster quota without replacement using seed 20260830.

Targets are 50,000 EGJ, 25,000 other-esophagus, and 25,000 stomach/broader-upper-GI images. The result contains exactly 100,000 unique images. No secondary clustering is used.
