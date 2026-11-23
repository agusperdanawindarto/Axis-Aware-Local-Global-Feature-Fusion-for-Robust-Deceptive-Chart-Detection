# Local GPU continuation for SALF-DINO

Use the laptop notebook in notebooks/SALF_DINO_GPU_Laptop_Resume.ipynb, and set SALF_DINO_PROJECT_ROOT to the D-drive experiment directory. The dataset and pretrained weights are stored outside Git.

## Completed setup
1. HVPD DVCS.zip downloaded to the D-drive dataset directory (1,190,593,622 bytes, SHA-256 d9f624d2cafb054c73c772a458e53135172529fb8a86fe7831169f5ebe4ab9ff); all 24,667 images extracted. The core benchmark is 24,480 synthetic plots.
2. Python virtualenv at env/.venv has CUDA-enabled PyTorch 2.6.0+cu124, torchvision 0.21.0, timm, Pandas and evaluation packages. The locally installed CUDA driver is compatible.
3. DINOv3-L16 and SwinV2 pretrained weights cached locally; do not publish their weight files, and respect the upstream weight licenses.
4. Saved frozen-backbone DINOv3-L16 seed-42 training states from Kaggle Version 7 have been copied and verified in the D-drive checkpoints directory.
5. Local Kaggle-outcome sync task is configured. The newer Kaggle Version 8 must finish before its outputs can be transferred.
6. The laptop GPU is an 8 GB GeForce RTX 3070 Ti; GPU thermal overload has been observed (81–84 C). **Never start a training run when another GPU workload is active or the GPU exceeds 78 C.**

## Scientific preflight
Run 05_CHECK_LOCAL_GPU_RESUME.cmd. Training can begin only when all checks pass, including exact source-series/SHA-256 split manifestation, paired checkpoint integrity, CUDA availability, and terminal Kaggle status. Do not substitute a new split with matching class counts for the actual Kaggle manifest if the checksum differs.

The notebook defaults RUN_TRAINING=False. Set True only after a successful preflight; a local continuation stage is limited to two epochs and uses the original total 20-epoch optimizer/scheduler schedule. The final three-seed main comparisons and both component ablations still require empirical evaluation.

## GitHub privacy
The GitHub-ready ZIP contains notebooks, scripts, figures, and documentation only. It excludes pretrained weights, large checkpoints, dataset images, credentials, and the original Word manuscript. The publication manuscript requires a correction from earlier transitive pHash grouping to actual source-series and exact SHA-256 grouping.
