# Results

The reported evaluation uses a model trained on MIMIC-IV and applied without retraining to eICU-CRD and PKUPH. Organ-specific analyses treat ICU death before organ failure as a competing event.

## Internal organ-failure prediction

Across the six organ systems, the mean cause-specific C-index was 0.795. Performance was strongest for liver and coagulation failure; kidney and heart failure also exceeded a C-index of 0.75. The mean IPCW integrated Brier score was 0.103.

<table align="center">
  <thead>
    <tr>
      <th align="left">Organ</th>
      <th align="center">C-index (95% CI)</th>
      <th align="center">AUC 24 h</th>
      <th align="center">AUC 72 h</th>
      <th align="center">AUC 168 h</th>
      <th align="center">IPCW-IBS</th>
      <th align="center">ICI</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Kidney</td><td align="center">0.777 (0.767–0.787)</td><td align="center">0.814</td><td align="center">0.717</td><td align="center">0.632</td><td align="center">0.152</td><td align="center">0.064</td></tr>
    <tr><td>Lung</td><td align="center">0.730 (0.720–0.740)</td><td align="center">0.734</td><td align="center">0.664</td><td align="center">0.627</td><td align="center">0.157</td><td align="center">0.087</td></tr>
    <tr><td>Heart</td><td align="center">0.761 (0.750–0.772)</td><td align="center">0.787</td><td align="center">0.704</td><td align="center">0.675</td><td align="center">0.068</td><td align="center">0.019</td></tr>
    <tr><td>Brain</td><td align="center">0.723 (0.714–0.731)</td><td align="center">0.751</td><td align="center">0.634</td><td align="center">0.357</td><td align="center">0.150</td><td align="center">0.046</td></tr>
    <tr><td>Coagulation</td><td align="center">0.877 (0.867–0.887)</td><td align="center">0.887</td><td align="center">0.875</td><td align="center">0.843</td><td align="center">0.069</td><td align="center">0.042</td></tr>
    <tr><td>Liver</td><td align="center">0.905 (0.900–0.911)</td><td align="center">0.893</td><td align="center">0.895</td><td align="center">0.869</td><td align="center">0.022</td><td align="center">0.012</td></tr>
    <tr><td><strong>Mean</strong></td><td align="center"><strong>0.795</strong></td><td align="center"><strong>0.811</strong></td><td align="center"><strong>0.748</strong></td><td align="center"><strong>0.667</strong></td><td align="center"><strong>0.103</strong></td><td align="center"><strong>0.045</strong></td></tr>
  </tbody>
</table>

<p align="center">
  <img src="assets/figures/internal-organ-roc-pr.png" width="100%" alt="Internal organ-failure ROC and precision-recall analysis">
</p>

<p align="center"><em>Internal organ-failure discrimination across the six prediction targets.</em></p>

## Internal adverse-event prediction

The three adverse-event heads are evaluated jointly with the organ-failure tasks while retaining outcome-specific discrimination analyses.

<p align="center">
  <img src="assets/figures/internal-adverse-event-roc-pr.png" width="100%" alt="Internal adverse-event ROC and precision-recall analysis">
</p>

<p align="center"><em>Internal discrimination for ICU mortality, septic shock and MODS.</em></p>

## Calibration

Calibration is assessed for both organ-specific cumulative incidence and adverse-event probability outputs.

<p align="center">
  <img src="assets/figures/internal-calibration.png" width="100%" alt="Calibration of organ-failure and adverse-event predictions">
</p>

<p align="center"><em>Calibration across the nine prediction targets.</em></p>

## Cross-cohort validation

Organ-level discrimination remained highest for liver and coagulation failure across both external cohorts. Adverse-event prediction transferred more consistently than the mean organ-level C-index, with ICU mortality AUROC remaining above 0.80 in all three cohorts.

<table align="center">
  <thead>
    <tr>
      <th align="left">Outcome</th>
      <th align="center">Metric</th>
      <th align="center">MIMIC-IV</th>
      <th align="center">eICU-CRD</th>
      <th align="center">PKUPH</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Mean organ failure</td><td align="center">C-index</td><td align="center">0.795</td><td align="center">0.709</td><td align="center">0.727</td></tr>
    <tr><td>ICU mortality</td><td align="center">AUROC</td><td align="center">0.801</td><td align="center">0.811</td><td align="center">0.814</td></tr>
    <tr><td>Septic shock</td><td align="center">AUROC</td><td align="center">0.917</td><td align="center">0.904</td><td align="center">0.865</td></tr>
    <tr><td>MODS</td><td align="center">AUROC</td><td align="center">0.918</td><td align="center">0.856</td><td align="center">0.820</td></tr>
    <tr><td>ICU mortality</td><td align="center">AUPRC</td><td align="center">0.315</td><td align="center">0.385</td><td align="center">0.307</td></tr>
    <tr><td>Septic shock</td><td align="center">AUPRC</td><td align="center">0.809</td><td align="center">0.737</td><td align="center">0.727</td></tr>
    <tr><td>MODS</td><td align="center">AUPRC</td><td align="center">0.937</td><td align="center">0.833</td><td align="center">0.714</td></tr>
  </tbody>
</table>

## Component analysis

The ablation study separates the contributions of graph guidance, expert specialisation, temporal encoding, static features, ranking supervision and competing-risk modelling. Removing the SOFAE graph produced the largest decline in mean organ C-index among the tested component removals, while the full model retained the strongest joint performance across organ failure and adverse-event tasks.

<p align="center">
  <img src="assets/figures/component-ablation.png" width="100%" alt="Ablation of Sepsis-MOST components">
</p>

<p align="center"><em>Component ablation across mean organ C-index and adverse-event AUROC.</em></p>

## Learned organ relations

The SOFAE graph represents three distinct data-derived relations: whether one organ tends to fail before another, how frequently outcomes co-occur and how organ failure relates to septic shock or MODS. These frozen relations enter the network through the expert router and the organ-coupling term.

<p align="center">
  <img src="assets/figures/sofae-relations.png" width="100%" alt="Temporal precedence, co-occurrence and syndrome-trigger relations in the SOFAE graph">
</p>

<p align="center"><em>Temporal precedence, co-occurrence and syndrome-trigger relation matrices.</em></p>

## External adverse-event discrimination

On eICU-CRD, the frozen model achieved AUROC values of 0.811 for ICU mortality, 0.904 for septic shock and 0.856 for MODS. The curves below show discrimination after direct transfer from MIMIC-IV without cohort-specific retraining.

<p align="center">
  <img src="assets/figures/eicu-adverse-event-roc.png" width="100%" alt="Adverse-event ROC curves in eICU-CRD">
</p>

<p align="center"><em>External adverse-event discrimination in eICU-CRD.</em></p>

## Feature attribution

Task-specific SHAP analyses separate the signals used for six organ-failure outputs and three adverse-event outputs. The resulting profiles show organ-dependent attribution patterns while preserving a shared representation across tasks; they describe model behaviour and do not establish causal effects.

<p align="center">
  <img src="assets/figures/shap-attributions.png" width="100%" alt="SHAP feature attributions across nine Sepsis-MOST prediction targets">
</p>

<p align="center"><em>Feature attributions across six organ-failure and three adverse-event targets.</em></p>

Patient-level attribution complements the global analysis by decomposing individual predictions into additive feature contributions. It is presented as a model-behaviour view rather than a causal explanation.

<p align="center">
  <img src="assets/figures/patient-level-attribution.png" width="100%" alt="Patient-level feature attribution across Sepsis-MOST targets">
</p>

<p align="center"><em>Patient-level decomposition of the nine model outputs.</em></p>

## Interpretation boundaries

The study is retrospective. PKUPH neurological failure uses a disorder-of-consciousness proxy because numeric GCS was not routinely available, and its cardiovascular endpoint relies on reconstructed vasopressor exposure. These endpoints are therefore not directly equivalent to their MIMIC-IV and eICU-CRD definitions. Prospective clinical utility has not been established.

## Aggregate tables

- [`internal_organ_performance.csv`](results/tables/internal_organ_performance.csv)
- [`cross_cohort_performance.csv`](results/tables/cross_cohort_performance.csv)
