# Model Architecture

## Computational graph

<p align="center">
  <img src="assets/figures/sepsis-most-framework.png" width="100%" alt="Sepsis-MOST computational graph">
</p>

<p align="center"><em>Onset-anchored feature construction, SOFAE graph learning, hybrid temporal encoding, graph-guided expert routing and multi-task prediction.</em></p>

## Module contract

| Stage | Representation | Function |
|:--|:--|:--|
| Clinical feature constructor | `X ∈ R^(21×12)`, `s ∈ R^116` | Aligns temporal and static variables to the sepsis-onset landmark. |
| SOFAE graph constructor | 9 nodes, 3 directed relation families | Encodes temporal precedence, co-occurrence and syndrome triggering from the development split. |
| Spectral graph encoder | 8-dimensional node embeddings | Compresses six relation matrices with truncated SVD and freezes them before evaluation. |
| Temporal encoder | hidden width 96 | Combines three residual causal TCN blocks with a selective state-space layer. |
| KG-MoE router | 6 experts, top-2 shared routing | Fuses the temporal state with the SOFAE embedding to construct shared context. |
| Organ streams | 6 dedicated expert pathways | Preserves organ-specific representations independent of shared top-2 routing. |
| Competing-risk heads | 6 × 8 × 3 logits | Models organ failure, ICU death before failure and event-free survival in each time bin. |
| Adverse-event heads | 3 probabilities | Predicts ICU mortality, septic shock and MODS from the shared representation. |

## SOFAE graph

The graph is estimated from the MIMIC-IV training split only and remains fixed during validation. Edges supported by fewer than 50 patients are removed. The retained relation matrices and their transposes are jointly factorised to produce organ embeddings for expert routing. A column-normalised temporal-precedence matrix supplies a non-negative downstream hazard bias to each organ head.

## Optimisation objective

The joint objective combines:

1. discrete-time competing-risk negative log-likelihood;
2. pairwise ranking loss over cumulative incidence;
3. expert load balancing;
4. organ-expert specialisation;
5. binary cross-entropy for the three adverse events.

The deployed configuration uses AdamW, cosine learning-rate scheduling, early stopping on validation mean C-index and post-hoc isotonic calibration.

## Release layout

```text
SepsisMOST/
├── assets/
│   └── figures/                 manuscript-ready figures
├── configs/
│   ├── cohorts/                 cohort-specific feature mappings
│   ├── evaluation/              horizons, metrics and bootstrap settings
│   └── model/                   encoder, router and objective configuration
├── results/
│   └── tables/                  aggregate, non-patient-level results
├── scripts/
│   ├── build/                   cohort, endpoint and graph construction
│   ├── evaluate/                internal and external validation entry points
│   └── train/                   training and calibration entry points
├── src/sepsis_most/
│   ├── clinical/
│   │   ├── cohorts/             cohort selection and split contracts
│   │   ├── endpoints/           organ failure and adverse-event definitions
│   │   ├── features/            temporal and static feature construction
│   │   └── harmonization/       cross-database variable alignment
│   ├── graph/
│   │   ├── relations/           precedence, co-occurrence and trigger matrices
│   │   ├── embedding/           frozen spectral graph representation
│   │   └── coupling/            graph-derived downstream hazard coupling
│   ├── model/
│   │   ├── encoders/            causal TCN and selective state-space blocks
│   │   ├── experts/             organ-specific expert networks
│   │   ├── routing/             graph-biased sparse routing
│   │   └── heads/               competing-risk and adverse-event heads
│   ├── objectives/              likelihood, ranking and routing losses
│   ├── training/                optimisation, checkpoint and calibration logic
│   ├── evaluation/
│   │   ├── discrimination/      C-index, time-dependent AUC and AUPRC
│   │   ├── calibration/         Brier score, ICI, slope and intercept
│   │   ├── clinical_utility/    operating points, lead time and net benefit
│   │   └── external_validation/ cohort transfer evaluation
│   └── interpretability/        SHAP, expert alignment and graph attribution
└── tests/
    ├── unit/                     module-level invariants
    ├── integration/              end-to-end tensor and output contracts
    └── statistical/             metric and resampling checks
```

The directory hierarchy is published as the planned source-release contract. Its implementation is intentionally withheld from this branch.
