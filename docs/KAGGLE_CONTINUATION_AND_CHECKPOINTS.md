# Kaggle staged continuation and checkpoints

Status on 16 November 2026:
- Kaggle Version 7 reached 8 DINOv3-L16 seed-42 epochs, then hit runtime limit.
- Epoch 8 best validation macro-F1 was 0.8846008785406361, not a held-out test metric.
- Best and last head checkpoints were downloaded, SHA-256 audited and uploaded to a private Kaggle checkpoint dataset.
- Version 8 was submitted for bounded epoch 9 to 12 continuation. A RUNNING status is not a completed-epoch confirmation.
- Stage-manager code is in the KAGGLE/stage_manager directory and its Windows task name is SALF_DINO_Kaggle_Staged_Training.
- Its configured plan has 9 main and 6 ablation model-seed combinations. It checks cloud status every 20 minutes while Windows is active, downloads results when Kaggle finalizes a version, validates checkpoint and split integrity, updates the private dataset and submits the next bounded stage.
- Kaggle quotas and execution duration can stop this workflow. If a stage is incomplete or checkpoint validation fails, the manager blocks further submission rather than inventing outcomes.
- Closing the laptop does not stop an already-running Kaggle GPU notebook. Windows synchronization and new-stage submission pause until the machine is active.
- Checkpoint files and original raw dataset are excluded from public Git. Source code and nonsensitive reproduction documentation may be published separately.
- The frozen 256px experiment is not the source article's 384px full fine-tuning. Main test and ablation results remain unconfirmed until completion files are validated.
