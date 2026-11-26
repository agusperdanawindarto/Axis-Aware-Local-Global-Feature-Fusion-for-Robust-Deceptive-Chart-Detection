## Research questions and main contributions

RQ1 (C1) asks whether axis-aware local–global fusion improves overall classification and reduces truncated-Y-axis errors relative to frozen-backbone baselines under a shared group-disjoint protocol. RQ2 (C2) examines the contributions of border aggregation, local refinement and multiplicative fusion through pathway ablations, equal-capacity controls and paired group-bootstrap uncertainty. RQ3 (C3) investigates absolute and relative performance under controlled image degradation and associated parameter and throughput trade-offs.

- **C1 — Axis-aware structural feature fusion:** a task-specific chart-border spatial prior integrated with local and global frozen DINOv3 features, without axis-coordinate annotation (binary supervision is still required).
- **C2 — Capacity-controlled component validation:** three-seed pathway ablations and two equally parameterized controls, with paired held-out-group inference. The controls were designed after the initial results, and reuse the test benchmark; they are exploratory follow-up evidence.
- **C3 — Robustness and computational trade-offs:** matched JPEG, blur and resolution degradation plus parameter counts and recorded throughput.

The novelty is the chart-specific structural integration and its controlled evidence, not attention, convolution or residual fusion individually.
