# Research Questions and Main Contributions

**Title:** Axis-Aware Local–Global Feature Fusion for Robust Deceptive Chart Detection
**Proposed method inside the paper:** SALF-DINO.

## RQ1 — Does task-specific local–global fusion improve classification and axis-related error patterns?
Evidence: 3-seed matched frozen-backbone comparison on 2,471 test charts (733 groups),
paired group-bootstrap differences, per-category srtAxY errors/FN/FP, and independent SwinV2 comparator.
**C1:** A chart-border-aware, local-patch-fused DINOv3 classification head evaluated against matched frozen
DINOv3 and SwinV2. Do not claim superiority over the original authors' fully fine-tuned DINOv3 score.

## RQ2 — What is the incremental evidence for the axis-region and local-refinement pathways?
Evidence: axis-only and local-only ablations, held fixed pretrained encoder, split, and three seeds;
bootstrap comparisons and observed negative/non-decisive results.
**C2:** A controlled branch-ablation study shows a +0.66-point difference against local-only (95% paired
test-group CI +0.32 to +1.02), but only +0.20 against axis-only (CI -0.15 to +0.54).
The latter cannot establish a reliable additional full-fusion benefit. No gate-disabled ablation exists.

## RQ3 — Is the absolute gain preserved under common corruptions, and what are the computational trade-offs?
Evidence: JPEG quality 40, Gaussian blur, and downsample/upsample with held-out samples,
mean macro-F1 and degradation; parameter counts, trainable-head counts and inference throughput.
**C3:** Corruption-specific analysis distinguishes higher absolute performance from inconsistent
relative degradation. The full head has 8.404 M trainable parameters vs 0.00205 M for the frozen
global-pooling baseline. External-image generalization and mobile efficiency are not established.

## Evidence matrix
RQ1 ⇢ Results Table IV, Figure 4, category error table and Figure 5.
RQ2 ⇢ same-seed branch ablations and 2,000-replicate test-group bootstrap confidence intervals.
RQ3 ⇢ corruption Table VI / Figure 6 plus model capacity and hardware-dependent throughput.

**Limitations:** synthetic HVPD-DVCS, coarse border priors (not learned axis segmentation),
no alternative-axis-layout/generalization test, no gate-isolated arm, frozen encoders vs
full-fine-tuning in prior work, only three training seeds.
