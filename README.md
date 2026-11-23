# Axis-Aware Local–Global Feature Fusion for Robust Deceptive Chart Detection

**Proposed method:** SALF-DINO (DINOv3-based structure/axis-aware local–global feature fusion).

This repository provides the **verified frozen-backbone, three-seed** HVPD-DVCS experiment
artifacts underlying a research manuscript. It is **not** a faithful rerun of the
full-fine-tuned DINOv3-L/16 experiments reported by Maciborski et al. (IEEE Access, 2026).

## Audit status — 23 November 2026

- **15/15 model–seed experiments strongly validated** against local checkpoints, predictions, metrics and hashes.
- Five configurations (DINOv3-L/16 global pooling, SwinV2 Tiny, SALF-DINO, axis-only, local-only) × three seeds (42, 123, 2025).
- Source: 24,480 HVPD-DVCS synthetic chart images; **19,544/2,465/2,471** group-disjoint train/validation/test.
- Test-set identity and source-series grouping fixed across models; **733 test groups**.
- Manifest SHA-256: `c8fc29542853326611668612b0444b7c044853e9ca483c591fba65bbc86b4d7a`.
- Pretrained backbones frozen; 256×256 RGB; optimizer AdamW; up to 20 epochs; validation macro-F1 selection, checkpointed recovery.

## Held-out test macro-F1 (mean ± sample SD across three seeds)

| Method | Macro-F1 | Tested seeds |
|---|---:|---|
| **SALF-DINO (proposed)** | **97.53 ± 0.21%** | 42, 123, 2025 |
| Axis-only | 97.33 ± 0.29% | 42, 123, 2025 |
| Local-only | 96.87 ± 0.24% | 42, 123, 2025 |
| DINOv3 global average | 90.13 ± 0.20% | 42, 123, 2025 |
| SwinV2 Tiny | 87.99 ± 0.19% | 42, 123, 2025 |

Paired **2000-replicate test-group bootstrap**, averaged over matched training seeds:
SALF-DINO vs DINOv3, **+7.41 pp** (95% CI: +6.38 to +8.45);
vs SwinV2, **+9.54 pp** (+8.42 to +10.71);
vs axis-only, **+0.20 pp** (-0.15 to +0.54);
vs local-only, **+0.66 pp** (+0.32 to +1.02).

The axis-only comparison interval includes zero. These bootstrap CIs describe sampling of the
**fixed held-out test groups conditional on the trained networks**, not uncertainty over datasets
or an independent significance assessment of training-seed variability.

## Key limitation and robustness qualification

SALF-DINO has higher **absolute** macro-F1 than the frozen-backbone controls under all three
tested perturbations, but does **not** show the smallest clean-to-corrupted drop on every
perturbation. Specifically, Gaussian blur and downsampling reduce its score, and the DINOv3
global baseline loses fewer *absolute points* under those two conditions. Performance on this
mostly synthetic collection is **not** proof of generalization to live online charts.
No gate-disabled ablation, alternative axis layout study, or external dataset is claimed.

## Research package

- `data/split_manifest.csv`: fixed, audited train/validation/test identities (not image files).
- `results/per_seed_metrics.csv`, `model_summary.csv`, `paired_group_bootstrap.csv`:
  audit-ready quantitative results.
- `results/category_error_summary.csv`: by-category errors; single-class `colNoScl`
  evaluated by deceptive recall instead of within-category binary macro-F1.
- `results/corruption_summary.csv`: JPEG, blur, and downsample perturbations.
- `results/raw/`: **90** original per-run artifact files (results only, not checkpoint weights).
- `figures/Fig4_* ... Fig6_*`: 600-dpi generated research figures.
- `docs/RESULTS_AND_DISCUSSION.md`, `docs/RQ_CONTRIBUTIONS.md`:
  evidence-aligned interpretation and scope.
- `docs/RELEASE_SHA256.json`: reproducible checksums.

The raw chart-image archive, private credentials, author manuscript, and large GPU
checkpoints are intentionally excluded. The source checkpoint directory is preserved on the
researcher's D drive.

## Underlying data and prior art

Source HVPD resource: https://huggingface.co/datasets/WieszczUrsynowa/HVPD-HarmfulVisualPlotsDataset

Maciborski et al., *Combating Visual Misinformation: Automated Detection of Deceptive Chart
Design Using Vision Transformers*, IEEE Access (2026), DOI: 10.1109/ACCESS.2026.3697128.

## Interpretation and reproducibility

Use the included scripts/notebooks only with the appropriate dataset and model licenses.
The `results/raw/` files are the experiment outputs, not regenerated or fabricated
predictions. Inferences should stay within the frozen-backbone / group-disjoint test protocol.

Scientific code and figures are provided for review; attribution of original dataset,
prior work and image examples must be maintained.
