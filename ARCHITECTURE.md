# Model Architecture

## Computational graph

<p align="center">
  <img src="assets/figures/sepsis-most-framework.png" width="100%" alt="Sepsis-MOST computational graph">
</p>

<p align="center"><em>Onset-anchored feature construction, SOFAE graph learning, hybrid temporal encoding, graph-guided expert routing and multi-task prediction.</em></p>

## Module contract

| Stage | Representation | Function |
|:--|:--|:--|
| Clinical feature constructor | `X ∈ R^(12×21)`, `s ∈ R^116` | Aligns temporal and static variables to the sepsis-onset landmark (upstream of this release; see `docs/DATA_FORMAT.md`). |
| SOFAE graph constructor | 9 nodes, 3 relation families | Encodes temporal precedence, co-occurrence and syndrome triggering from event onset times (`graph/relations`). |
| Spectral graph encoder | 8-dimensional node embeddings | Compresses the three relation matrices and their transposes with truncated SVD and freezes the result (`graph/embedding`). |
| Temporal encoder | hidden width 96 | Combines three residual causal TCN blocks with a selective state-space layer (`model/encoders`). |
| KG-MoE router | 6 experts, top-2 shared routing | Fuses the temporal state with the SOFAE embedding to construct shared context (`model/routing`, `model/experts`). |
| Organ streams | 6 dedicated expert pathways | Preserves organ-specific representations independent of shared top-2 routing. |
| Competing-risk heads | 6 × 8 × 3 logits | Models organ failure, ICU death before failure and event-free survival in each time bin (`model/heads`). |
| Adverse-event heads | 3 probabilities | Predicts ICU mortality, septic shock and MODS from the shared representation. |

## SOFAE graph

The graph is estimated once from the MIMIC-IV development data and remains fixed in every evaluation; the external cohorts do not contribute to it. Edges supported by fewer than 50 patients are removed. The retained relation matrices and their transposes are jointly factorised to produce organ embeddings for expert routing. A column-normalised temporal-precedence matrix supplies a non-negative downstream hazard bias to each organ head.

## Optimisation objective

The objective is the competing-risk likelihood of the organ outcomes plus four weighted auxiliary terms, L = L<sub>cr</sub> + Σ<sub>a</sub> λ<sub>a</sub> L<sub>a</sub> with a ∈ {syn, rank, lb, spec}:

1. discrete-time competing-risk negative log-likelihood (main term, `objectives/competing_risks.py`);
2. binary cross-entropy for the three adverse events (λ<sub>syn</sub> = 0.5);
3. pairwise ranking loss over the 7-day cumulative incidence (λ<sub>rank</sub> = 0.15, `objectives/ranking.py`);
4. expert load balancing (λ<sub>lb</sub> = 0.01, computed in `model/routing/kg_moe.py`);
5. organ-expert specialisation (λ<sub>spec</sub> = 0.2, `objectives/specialisation.py`).

Training uses AdamW with a cosine learning-rate schedule and early stopping on the validation mean C-index (`training/trainer.py`). The released configuration trains one network for each of the seeds 42, 7 and 2024 and averages their predictions (`training/ensemble.py`). An optional post-hoc isotonic calibration of the 7-day organ CIF, fitted on the validation split, is available in `training/calibration.py`.

## Source layout

```text
src/sepsis_most/
├── constants.py                 organs, adverse events, graph nodes and time bins
├── graph/
│   ├── relations/               precedence, co-occurrence and trigger matrices, edge pruning
│   ├── embedding/               spectral node embeddings
│   ├── coupling/                column-normalised precedence matrix for organ coupling
│   └── sofae.py                 graph construction, storage and edge list
├── model/
│   ├── encoders/                causal TCN blocks and selective state-space layer
│   ├── experts/                 organ-specific expert networks
│   ├── routing/                 graph-biased top-k routing and load balancing
│   ├── heads/                   discrete-time competing-risk decoding
│   ├── pooling.py               attention pooling
│   └── sepsis_most.py           full network
├── objectives/                  likelihood, ranking, specialisation and joint objective
├── training/                    configuration, data interface, augmentation, trainer, frozen model, calibration
└── evaluation/discrimination/   cause-specific concordance index used for model selection
```

Entry points are in `scripts/build`, `scripts/train` and `scripts/predict`; the input files are specified in `docs/DATA_FORMAT.md`.
