# License boundary

`LICENSE` grants MIT rights only to original HARVEST files committed here.

The boundary is deliberately strict:

- included: HARVEST orchestration, curation logic, training integration, inference aggregation, metrics, tests, and documentation;
- excluded: Meta DINOv3 source and training/inference implementation;
- excluded: official DINOv3 checkpoints and every checkpoint derived from them;
- excluded: GastroNet, RARE, clinician-review records, predictions, Docker images, and submission archives.

The import adapter requires a local DINOv3 checkout supplied by the user. Domain SSL delegates the DINO head, loss, KoLeo regularizer, and augmentation to that checkout rather than redistributing or reimplementing those components. That dependency remains under Meta's DINOv3 License. Keeping the dependency outside this repository prevents the MIT grant from appearing to cover it. Do not commit files from `third_party/`, `private_weights/`, `private_data/`, `outputs/`, or `runs/`.
