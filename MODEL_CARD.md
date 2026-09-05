# HARVEST model card

## Model

HARVEST is a dual-resolution, ten-member ensemble for Barrett's neoplasia image classification. It combines five 224-pixel two-view ViT-L folds and five 384-pixel one-view ViT-L folds.

## Training path

Both components start from independently acquired official DINOv3 ViT-L/16 weights and undergo domain SSL on the same clinician-guided GastroNet-100k corpus. Each labeled fold first fits a frozen linear probe. The 224-pixel component then performs partial fine-tuning with two independently augmented views and BCE only; the 384-pixel component performs one-view partial fine-tuning. No contrastive, hard-negative, consistency, or alternative-backbone branch is part of HARVEST.

## Inference

Each member averages logits over four exact quarter-turn views. Member logits are standardized using that fold's development-validation mean and population standard deviation. The middle three of five standardized fold logits are averaged within each resolution. The two component logits are then averaged with equal weights and passed through one sigmoid.

## Intended use and limitations

The model is intended for research evaluation on the RARE26 task. It is not validated for clinical deployment. Dataset shift, center effects, correlated video frames, small positive counts, and label uncertainty may materially affect performance. Output scores are ranking scores, not clinically calibrated probabilities.

## Distribution

This repository distributes code only. DINOv3 code and all official, adapted, and fine-tuned weights are excluded and remain governed by Meta's DINOv3 License. RARE and GastroNet data are also excluded.
