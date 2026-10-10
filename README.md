# Sepsis-MOST

**Knowledge-Graph-Guided Mixture-of-Experts for Joint Prediction of Multi-Organ Failure, Septic Shock, and ICU Mortality in Sepsis**

![Tasks](https://img.shields.io/badge/tasks-6%20organs%20%2B%203%20adverse%20events-315E6B?style=flat-square)
![Horizon](https://img.shields.io/badge/horizon-7%20days-B76E5A?style=flat-square)
![Validation](https://img.shields.io/badge/validation-2%20external%20cohorts-697F4B?style=flat-square)
![Code](https://img.shields.io/badge/source-release%20staged-6B7280?style=flat-square)

Sepsis-MOST is a multi-task prognostic framework for early, organ-resolved risk estimation in sepsis. From a 12-hour window centred on sepsis onset, the model estimates the 7-day cumulative incidence of kidney, lung, heart, brain, coagulation and liver failure, while jointly predicting septic shock, multiple organ dysfunction syndrome and ICU mortality.

The framework combines dilated causal temporal convolutions, a selective state-space encoder and six organ-specific experts. A Sepsis-associated Organ Failure and Adverse Event graph learned only from the development split guides expert routing and downstream organ coupling. ICU death is modelled as a competing event for every organ-specific outcome.

## Framework

The end-to-end design links the sepsis-onset landmark, temporal and static feature streams, training-only SOFAE graph, hybrid temporal encoder, graph-guided expert router and nine prediction targets in one computational pathway.

<p align="center">
  <img src="assets/figures/sepsis-most-framework.png" width="100%" alt="Sepsis-MOST study design and model architecture">
</p>

<p align="center"><em>Sepsis-MOST study design and model architecture.</em></p>

## Core design

- **Onset-anchored prediction** — 21 hourly variables across 12 hours and 116 static features define the prediction state at sepsis onset + 6 hours.
- **SOFAE graph** — temporal precedence, co-occurrence and syndrome-trigger relations are compressed into frozen spectral node embeddings.
- **Organ-specialised routing** — a graph-biased top-2 router builds shared context while every organ head retains its dedicated expert stream.
- **Competing-risk outputs** — eight discrete-time bins jointly represent organ failure, ICU death before organ failure and event-free survival.
- **Unified multi-task learning** — six cumulative-incidence outputs and three adverse-event probabilities are optimised in one network.

## Study design

<table align="center">
  <thead>
    <tr>
      <th align="left">Role</th>
      <th align="center">Cohort</th>
      <th align="center">Patients</th>
      <th align="left">Use</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Development</td><td align="center">MIMIC-IV</td><td align="center">28,800</td><td>Training, validation and internal testing</td></tr>
    <tr><td>External validation</td><td align="center">eICU-CRD</td><td align="center">24,662</td><td>Frozen-model evaluation</td></tr>
    <tr><td>External validation</td><td align="center">PKUPH</td><td align="center">4,849</td><td>Frozen-model evaluation in an independent health system</td></tr>
  </tbody>
</table>

The same onset-anchored eligibility logic was applied across the development and external-validation cohorts. The development cohort was split by patient into training, validation and internal test sets; both external cohorts were reserved for frozen-model evaluation.

<p align="center">
  <img src="assets/figures/cohort-selection.png" width="100%" alt="Cohort selection for MIMIC-IV, eICU-CRD and PKUPH">
</p>

<p align="center"><em>Cohort selection and analysis roles across the three study populations.</em></p>

## Evaluation

Internal testing produced a mean organ-level C-index of 0.795. The three adverse-event outputs retained AUROC values above 0.80 in the internal test set and both external cohorts.

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
    <tr><td>Mean organ failure</td><td align="center">C-index</td><td align="center"><strong>0.795</strong></td><td align="center">0.709</td><td align="center">0.727</td></tr>
    <tr><td>ICU mortality</td><td align="center">AUROC</td><td align="center">0.801</td><td align="center">0.811</td><td align="center">0.814</td></tr>
    <tr><td>Septic shock</td><td align="center">AUROC</td><td align="center">0.917</td><td align="center">0.904</td><td align="center">0.865</td></tr>
    <tr><td>MODS</td><td align="center">AUROC</td><td align="center">0.918</td><td align="center">0.856</td><td align="center">0.820</td></tr>
  </tbody>
</table>

The component analysis supports the combined graph, temporal encoder, expert-routing and competing-risk design. Removing graph guidance produced the largest decline in mean organ C-index among the evaluated component removals.

<p align="center">
  <img src="assets/figures/component-ablation.png" width="100%" alt="Ablation of Sepsis-MOST components">
</p>

<p align="center"><em>Component ablation across organ-failure and adverse-event tasks.</em></p>

Complete organ-level metrics, external validation results and interpretation figures are available in [`RESULTS.md`](RESULTS.md).

## Repository

This repository contains the model specification, release-oriented package layout, manuscript figures and aggregate result tables. Patient-level data, trained weights and source implementation are not distributed here.

The planned source tree is documented in [`ARCHITECTURE.md`](ARCHITECTURE.md). Every implementation directory is reserved as an empty release scaffold; no source file is included.

## Data governance

MIMIC-IV and eICU-CRD are available through PhysioNet under credentialed access and their respective data-use agreements. The independent institutional cohort is not redistributed. No protected health information or patient-level derivatives are included in this repository.
