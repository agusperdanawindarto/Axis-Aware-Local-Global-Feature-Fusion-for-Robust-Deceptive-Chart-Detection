# Automatic Kaggle-to-laptop export

A Windows Task Scheduler entry named SALF_DINO_Kaggle_Result_Sync was installed to check Kaggle every 30 minutes while the user session is active.

1. If main Kaggle Version 7 is RUNNING, it only checks status and does not download the dataset.
2. After Version 7 finalizes, the script downloads only checkpoint, per-image predictions, metric, history, robustness and split-manifest files to local_artifacts/kaggle_v7 (ignored by Git).
3. It verifies the exact split-manifest SHA-256 and requires all 9 model-seed completion records before starting ablations.
4. It submits a separate PRIVATE Kaggle notebook under agusperdana/salf-dino-hvpd-branch-ablations for 6 branch ablation runs if and only if the main study passed the above checks.
5. After that separate notebook completes, it downloads ablation results to local_artifacts/kaggle_ablation_v2 (ignored by Git), verifies 6 completion records and the identical split, then disables its own scheduled task.

The poller does not guarantee Kaggle will complete within free GPU quotas. If the main job errors or expires without all 9 completed model-seed records, automatic ablations will NOT launch. Its status is written to local_artifacts/monitor_status.json and a local log to the project KAGGLE folder.

To manually check Kaggle and export:
python scripts/sync_kaggle_results.py --version 7

To manually run the status monitor:
python scripts/monitor_full_study.py

Never upload private_manuscript, local_artifacts, Kaggle credentials, raw HVPD images, or large model weights to a public GitHub repository.
