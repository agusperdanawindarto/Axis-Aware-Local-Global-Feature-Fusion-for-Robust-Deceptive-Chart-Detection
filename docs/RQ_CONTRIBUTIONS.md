# Research Questions and Main Contributions — Evidence-Aligned Novelty

**Title:** Axis-Aware Local–Global Feature Fusion for Robust Deceptive Chart Detection
**Proposed method:** SALF-DINO (discussed inside the paper).
**Evidence:** 15/15 fully verified runs; five architectures × seeds 42/123/2025; frozen pretrained encoders; 24,480 synthetic HVPD-DVCS charts; source-group-disjoint 19,544/2,465/2,471 split (733 held-out groups).

## Research Gap

The defensible gap is **not** that previous studies ignored axes or that attention, convolution, or local–global fusion had never been used. The cited HVPD DINOv3 reference does not establish whether an *annotation-free chart-border prior* reduces errors in distinguishing deceptive Y-axis manipulations from legitimate but unusual axis layouts within a frozen-representation protocol. Overall macro-F1 alone cannot distinguish false alarms from missed deceptive designs or identify incremental structural-versus-local pathway contributions. The proposed investigation therefore targets a specific failure mode, a pathway-level hypothesis, and an explicit robustness–complexity trade-off.

## Research Questions

**RQ1.** Under a fixed group-disjoint HVPD-DVCS protocol, does an axis-adjacent local–global representation improve held-out macro-F1 and reduce srtAxY false-positive and false-negative errors relative to a frozen DINOv3 global descriptor and an independent SwinV2 comparator?

**RQ2.** How much additional evidence is contributed by the axis-region and local-patch pathways when each is isolated in axis-only and local-only variants, and which incremental differences remain uncertain under paired group-bootstrap evaluation?

**RQ3.** Does SALF-DINO retain stronger absolute detection performance under JPEG recompression, Gaussian blur, and downsampling, and what changes in trainable parameters and recorded evaluation throughput accompany this performance?

## Main Contributions

**C1 (architectural contribution; RQ1).** A task-specific, annotation-free chart-border fusion head that combines attention-weighted left, right, and lower border descriptors with global DINOv3 context and depthwise-refined neighboring patches through gated residual adaptation. Novelty is claimed for this task-oriented integration and testable structural prior, not for attention, convolution, local–global fusion, or gating individually.

**C2 (failure-mode evidence; RQ1).** A structure-focused classification analysis that resolves aggregate improvements into deceptive misses and false alarms on truncated-Y-axis charts. Matched-seed evaluation shows that the most consequential reduction occurs in false-positive predictions on non-deceptive axis layouts, thereby targeting a failure mode that overall macro-F1 alone would obscure.

**C3 (incremental pathway evidence; RQ2).** A controlled, three-seed comparison against axis-only and local-only variants, supplemented by paired held-out-group bootstrap intervals. The findings support a measurable contribution of chart-border information relative to local-only features but do not establish a decisive advantage of the full model over axis-only; no independent effect is attributed to the gate without a gate-disabled experiment.

**C4 (robustness–cost evidence; RQ3).** A matched corruption and complexity study that separates high absolute macro-F1 under JPEG degradation, Gaussian blur, and downsampling from the magnitude of performance loss relative to clean inputs, and reports trainable-head capacity and evaluation throughput. This establishes a transparent robustness–cost boundary rather than claiming invariance, efficiency superiority, or natural-chart generalization.

## RQ–Contribution–Evidence Matrix

| RQ | Contribution | Experiments | Verified outcome and boundary |
|---|---|---|---|
| RQ1 | C1, C2 | Frozen DINOv3 vs SALF-DINO vs independent SwinV2; three seeds; per-category false positives and false negatives | SALF-DINO 97.53 ± 0.21% test macro-F1 vs DINOv3 90.13 ± 0.20% and SwinV2 87.99 ± 0.19%; srtAxY n=872, average errors 89.7 → 26.7 and false positives 73.7 → 14.0 relative to DINOv3. Does not isolate an equal-capacity generic head. |
| RQ2 | C3 | Axis-only, local-only, three seeds; 2,000 paired bootstrap resamples of 733 test groups | SALF minus local-only +0.66 percentage points (95% CI [+0.32,+1.02]); SALF minus axis-only +0.20 points (CI [−0.15,+0.54], crosses zero). No gate-disabled arm; branch ablations retain the same implemented trainable parameter budget. |
| RQ3 | C4 | JPEG quality 40, Gaussian blur, downsampling/upsampling; parameter counts and descriptive throughput | SALF corrupted-test macro-F1 95.21%, 95.21%, 93.21%; 8.404M trainable parameters versus ≈0.00205M in the frozen DINOv3 global head. Relative degradation and efficiency do not universally improve. |

## Claim Boundaries

- Structural chart-border regions are a **fixed spatial prior**, not predicted or annotated axes.
- The proposed method is not demonstrated to outperform the **fully fine-tuned** DINOv3 previously published by Maciborski et al.; cross-study evaluation settings differ.
- The held-out test-group bootstrap is conditional on fitted models, not an independent uncertainty analysis of training seeds, new datasets, or deployment distributions.
- The measured advantage cannot conclusively separate architectural mechanism from increased head capacity; no capacity-matched generic MLP control was tested.
- The benefit of gating is not isolated, and the incremental benefit beyond axis-only is uncertain.
- The synthetic HVPD-DVCS results do not establish generalization to independent natural charts, unseen axis placements, or adversarial attacks.

## Evidence and Reproducibility

Evidence: `results/raw/`, `results/per_seed_metrics.csv`, `results/category_error_summary.csv`, `results/corruption_summary.csv`, `results/paired_group_bootstrap.csv`, `results/evidence_full.json`, and `data/split_manifest.csv`. Group-disjoint manifest SHA-256: `c8fc29542853326611668612b0444b7c044853e9ca483c591fba65bbc86b4d7a`.

Reference predecessor: Maciborski, K. J., Struniawski, K., and Wierzbicki, A. (2026). *Combating Visual Misinformation: Automated Detection of Deceptive Chart Design Using Vision Transformers*. IEEE Access. https://doi.org/10.1109/ACCESS.2026.3697128
