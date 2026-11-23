# SALF-DINO — experimentally verified state, 16 Nov 2026

## Kaggle
- Kaggle Version 7: stopped at cloud timeout after 8 completed DINOv3-L16 seed-42 epochs.
- Kaggle Version 8: COMPLETE, resumed exactly at epoch 9 and completed through epoch 12. The notebook ended as a partial stage rather than the 20-epoch final evaluation.
- Validation macro-F1 by epoch 9–12: 0.88391, 0.88512, 0.8890950055547673, 0.88638. Best model is at epoch 11.
- No final held-out test macro-F1 is available for the 20-epoch DINOv3-L16 run.
- Other main models/seeds and six branch ablation runs remain uncompleted; do not report inferred performance.

## Downloaded and verified local evidence
- Dataset ZIP: 1,190,593,622 bytes, SHA256 d9f624d2cafb054c73c772a458e53135172529fb8a86fe7831169f5ebe4ab9ff.
- Extracted archive: 24,667 image files; 24,480 synthetic images evaluated in the core benchmark.
- The Kaggle-original split_manifest.csv is now on D:; SHA256 c8fc29542853326611668612b0444b7c044853e9ca483c591fba65bbc86b4d7a.
- Partition counts: train 19,544; validation 2,465; test 2,471. Group count 7,267; no source-group leakage.
- Version 8 outputs: best.pt (epoch 11), last.pt (epoch 12), training history, stage-result JSON, label audit, manifest and excluded-web-example list.
- Output checkpoint pair copied to the local checkpoints directory, matching the saved Kaggle Version 8 files byte for byte.
- Exact original pretrained DINOv3-L16 and SwinV2 weights cached on D:, deliberately excluded from public GitHub artifacts.

## Local continuation
- CUDA virtualenv on drive D: includes PyTorch 2.6.0+cu124, torchvision, timm, Jupyter kernel and matching scientific dependencies.
- SALF_DINO_GPU_Laptop_Resume.ipynb selects only one model/seed, defaults to RUN_TRAINING=False and preserves the original 20-epoch scheduler.
- Local saved-manifest loading has been adjusted to retain the original Kaggle CSV row order, not regenerate or reorder training samples.
- The thermal guard refuses training when the RTX 3070 Ti GPU is hotter than 78 C or already using over 2,048 MiB.
- During preparation the GPU was 81–84 C while an unrelated local Python process was active; laptop GPU training was NOT started.
- The obsolete Kaggle automatic new-stage launcher is disabled. Version 8 has completed; the cloud-to-D export completed successfully, and the corresponding one-shot export task is disabled.
- The Word manuscript still contains older pHash terminology; adjust scientific Methods before publication, keeping Mendeley fields intact.

This status document is evidence-only. A claimed paper-ready result requires completed main and ablation experiments on the same validated held-out split.
