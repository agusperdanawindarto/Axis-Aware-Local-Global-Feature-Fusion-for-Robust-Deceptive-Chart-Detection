# Alignment with Section III of the actual Word manuscript

Source: SALF-DINO_IEEE.docx, locally copied without editing into the excluded private_manuscript directory. The original Mendeley field codes are preserved.

- Three models: DINOv3-L/16 global baseline; SwinV2 Tiny; proposed SALF-DINO.
- Three seeds 42, 123, 2025; maximum 20 epochs; early stopping patience 5.
- Input 256x256, RGB normalization, common mild training augmentation.
- AdamW, head learning rate 0.0005, weight decay 0.05, mini-batch 2, gradient accumulation 8, mixed precision.
- Held-out binary macro-F1, accuracy, balanced accuracy, deceptive recall, srtAxY errors, confusion matrices.
- JPEG quality 40, Gaussian blur and downsampling perturbations; paired group bootstrap 2000 iterations.
- Axis-only and local-only ablations must be completed for the manuscript's proposed component analysis.

MATERIAL DIFFERENCE requiring correction in the Word manuscript before publication:
The Word Section III-A describes pHash near-duplicate merging with Hamming threshold four.
The verified Kaggle split instead groups numeric-series identities and exact SHA-256 duplicates.
The earlier pHash transitive merging made a bad split (24474/2/4); those results are excluded.
The correct split is 19544/2465/2471 (7267 groups). Do not claim pHash grouping in final Methods.

The full Kaggle sweep provides an architecture-controlled FROZEN-encoder study, not the 384px full-backbone fine-tuning of the original IEEE Access study. Pilot DINOv3-B performance is not a paper-model outcome.

Fig. 1 is the research workflow; Fig. 2 can show reconstructed prior architecture and Fig. 3 the proposed one. Ensure the final Word captions and references match before submission.
