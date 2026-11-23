# Research Questions and Main Contributions — Paper-Aligned

**Manuscript:** *Axis-Aware Local–Global Feature Fusion for Robust Deceptive Chart Detection*  
**Proposed architecture:** **SALF-DINO**  
**Evidence:** 15 fully validated runs; five model configurations × seeds **42, 123, 2025**; frozen pretrained encoders; HVPD-DVCS group-disjoint **19,544/2,465/2,471** split; **733** held-out groups.

This document is synchronized with the **three** contributions and **three** research questions in the revised manuscript. It supersedes the earlier four-contribution mapping; the prior failure-mode contribution is now part of **C1**, not a fourth independent novelty claim.

## Research gap

The claim is **not** that prior deceptive-chart studies ignored axes, or that attention, depthwise convolution, local–global fusion, and gating were individually invented. The focused hypothesis concerns whether an **annotation-free fixed chart-border spatial prior** improves the separation of deceptive axis manipulations from legitimate but unusual axis layouts within a frozen-vision-backbone protocol.

## Main contributions

**C1 — Axis-aware local–global feature fusion (RQ1).** SALF-DINO integrates attention-weighted left/lower/right chart-border descriptors, depthwise local patch refinement, and global DINOv3 features through gated residual fusion. The architectural contribution is its **task-specific structural integration**; the principal diagnostic evidence is fewer false-positive classifications on non-deceptive truncated-Y-axis charts.

**C2 — Component-level validation and uncertainty analysis (RQ2).** Three-seed **axis-only** and **local-only** ablations with paired held-out-group bootstrap separate supported contributions from unresolved incremental gains. The chart-border pathway has clearer incremental evidence relative to local-only than extra local refinement adds beyond axis-only. The gate has **no isolated causal claim**, since no gate-disabled model was evaluated.

**C3 — Robustness and computational trade-off evaluation (RQ3).** SALF-DINO is examined under JPEG compression, Gaussian blur, and downsampling, with trainable parameters and observed evaluation throughput. These experiments establish **higher absolute performance under specified image perturbations**, but not superior relative retention under every corruption or computational-efficiency dominance.

## Research questions

**RQ1 (C1).** To what extent does axis-aware local–global fusion improve overall classification performance and reduce truncated-Y-axis detection errors compared with DINOv3 and SwinV2 baselines under a shared group-disjoint evaluation protocol?

**RQ2 (C2).** What are the individual contributions of axis-aware attention and local patch refinement, and how statistically reliable are their incremental improvements across three random seeds?

**RQ3 (C3).** How robust is SALF-DINO to image-quality degradation, and what computational trade-offs arise in terms of trainable parameters and inference throughput?

## Evidence mapping

| RQ | Main contribution | Verified evidence | Claim boundary |
|---|---|---|---|
| **RQ1** | **C1** | **Table IV:** SALF-DINO **97.53 ± 0.21%** test macro-F1 vs frozen DINOv3 **90.13 ± 0.20%** and SwinV2 **87.99 ± 0.19%**. **Table VI:** `srtAxY` mean errors **89.7 → 26.7**, false positives **73.7 → 14.0** compared with DINOv3. **Figures 3–4; Figure S2.** | Synthetic HVPD-DVCS, frozen-backbone only; prior full fine-tuning not a matched control. |
| **RQ2** | **C2** | **Table V:** full vs local-only **+0.66 pp**, group-bootstrap CI **+0.32 to +1.02**; full vs axis-only **+0.20 pp**, CI **−0.15 to +0.54** (crosses zero). 2,000 paired held-out-group bootstrap resamples over 733 groups. | CIs conditional on trained networks; ablations retain identical head capacity; no gate-disabled control. |
| **RQ3** | **C3** | **Table VII/Figure 5:** JPEG **95.21%**, blur **95.21%**, downsampling **93.21%** for SALF-DINO. **Table VIII:** 311.48 M total, **8.404 M trainable**; evaluation **13.3 images/s** recorded mean. | Greater trainable-head cost; relative degradation not always smallest; throughput not an isolated, controlled speed benchmark. |

## Paper figures and supplementary images

The repository provides the **exact embedded images** from the final manuscript, under [`figures/paper/`](../figures/paper/):

- **Figure 1:** [Research framework](../figures/paper/Figure_01_Research_Framework.png)
- **Figure 2:** [SALF-DINO proposed method](../figures/paper/Figure_02_SALF_DINO_Architecture.png)
- **Figure 3:** [Three-seed Macro-F1](../figures/paper/Figure_03_Test_MacroF1_3Seeds.png)
- **Figure 4:** [srtAxY misclassifications](../figures/paper/Figure_04_srtAxY_Misclassifications.png)
- **Figure 5:** [Controlled corruption robustness](../figures/paper/Figure_05_Controlled_Corruption_Robustness.png)
- **Figure S1:** [Genuine training chart examples](../figures/paper/Figure_S1_Genuine_HVPD_DVCS_srtAxY_Samples.png)
- **Figure S2:** [Actual held-out corrected classifications](../figures/paper/Figure_S2_Heldout_srtAxY_Corrected_Errors_Seed42.png)

Exact image SHA-256 hashes, pixel dimensions, captions, and embedded media mappings are in [`PAPER_FIGURE_SOURCE_MAP.json`](PAPER_FIGURE_SOURCE_MAP.json).

## Research limitations

The fixed chart-border regions are a **spatial layout prior**, not annotated axes or OCR. The model is evaluated on a group-disjoint synthetic dataset, not an independently collected real-chart benchmark. No gate-disabled or generic capacity-matched fusion-head control was executed. The +0.20 pp gain over axis-only is uncertain; bootstrap intervals quantify held-out group sampling uncertainty **conditional on the trained models**, not generalization to new datasets. Throughput measurements are descriptive rather than results of synchronized profiling.

The cited predecessor by Maciborski et al., *Combating Visual Misinformation: Automated Detection of Deceptive Chart Design Using Vision Transformers* ([DOI](https://doi.org/10.1109/ACCESS.2026.3697128)), used a fully fine-tuned protocol and is not treated as a directly matched numerical baseline.

## Audit sources

[`../results/raw/`](../results/raw/) · [`../results/model_summary.csv`](../results/model_summary.csv) · [`../results/paired_group_bootstrap.csv`](../results/paired_group_bootstrap.csv) · [`../results/category_error_summary.csv`](../results/category_error_summary.csv) · [`../results/corruption_summary.csv`](../results/corruption_summary.csv) · [`../data/split_manifest.csv`](../data/split_manifest.csv).

Split SHA-256: `c8fc29542853326611668612b0444b7c044853e9ca483c591fba65bbc86b4d7a`.
