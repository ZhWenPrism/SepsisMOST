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

## Interpretation boundaries

The study is retrospective. PKUPH neurological failure uses a disorder-of-consciousness proxy because numeric GCS was not routinely available, and its cardiovascular endpoint relies on reconstructed vasopressor exposure. These endpoints are therefore not directly equivalent to their MIMIC-IV and eICU-CRD definitions. Prospective clinical utility has not been established.

## Aggregate tables

- [`internal_organ_performance.csv`](results/tables/internal_organ_performance.csv)
- [`cross_cohort_performance.csv`](results/tables/cross_cohort_performance.csv)
