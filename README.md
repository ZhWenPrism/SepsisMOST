<div align="center">

# Sepsis-MOST

**Knowledge-Graph-Guided Mixture-of-Experts for Organ-Resolved Sepsis Forecasting**

[![Status](https://img.shields.io/badge/Status-research_project-315E6B?style=flat-square)](#research-overview)
[![Clinical AI](https://img.shields.io/badge/Clinical_AI-sepsis_prognosis-B76E5A?style=flat-square)](#clinical-scope)
[![Validation](https://img.shields.io/badge/Validation-multicohort-697F4B?style=flat-square)](#cohort-design)
[![Code](https://img.shields.io/badge/Code-model_and_training-6B7280?style=flat-square)](#code)

<sub>longitudinal EHR · organ crosstalk · competing risks · multi-task learning · external validation</sub>

</div>

<p align="center">
  <img src="assets/figures/sepsis-most-framework.png" width="100%" alt="Sepsis-MOST study design and model architecture">
</p>

Developed in collaboration with **Peking University People's Hospital**, Sepsis-MOST is a clinical forecasting framework for modelling how sepsis progresses across interconnected organ systems. The model uses longitudinal intensive-care records around the onset of sepsis to estimate the short-term risk of kidney, lung, cardiovascular, neurological, coagulation and liver failure while jointly predicting septic shock, multiple organ dysfunction syndrome and ICU mortality.

The project is motivated by a practical limitation of conventional bedside assessment. Scores such as SOFA describe the organ dysfunction that is already present, but they do not directly indicate which organ is most likely to fail next, when that failure may occur or how deterioration in one organ changes the risk of deterioration elsewhere. Existing prediction systems also tend to focus on a single outcome or build separate models for each endpoint, leaving the physiological dependence between organ systems largely unused.

Sepsis-MOST addresses this gap with a unified architecture that combines onset-aligned temporal modelling, a data-derived organ-event graph, six organ-specialised experts and competing-risk prediction. The graph represents recurrent temporal and co-occurrence patterns among organ failures and adverse events; it then guides expert routing and downstream organ coupling. Each organ retains a dedicated prediction pathway, so cross-organ information can be shared without collapsing all outcomes into one undifferentiated risk score.

This repository contains the source code used to construct and train Sepsis-MOST: the SOFAE graph constructor, the network, the training objective, and the optimisation and inference procedures. It also provides the manuscript figures and aggregate result tables. Cohort extraction, outcome labelling and feature construction are performed upstream of this code and are not part of the present release. No patient-level data, derived datasets or trained weights are distributed.

## At a glance

<table align="center">
  <thead>
    <tr align="center">
      <th>Clinical setting</th>
      <th>Observation window</th>
      <th>Prediction horizon</th>
      <th>Outputs</th>
    </tr>
  </thead>
  <tbody>
    <tr align="center">
      <td>Adult ICU patients<br>meeting Sepsis-3</td>
      <td>6 h before to 6 h after<br>sepsis onset</td>
      <td>7 days from<br>onset + 6 h</td>
      <td>Six organ failures<br>three adverse events</td>
    </tr>
  </tbody>
</table>

<table align="center">
  <thead>
    <tr align="center">
      <th>Development cohort</th>
      <th>Internal test</th>
      <th>External validation</th>
      <th>Clinical collaboration</th>
    </tr>
  </thead>
  <tbody>
    <tr align="center">
      <td>MIMIC-IV<br><b>28,800 patients</b></td>
      <td>MIMIC-IV holdout<br><b>4,321 patients</b></td>
      <td>eICU-CRD <b>24,662</b><br>PKUPH <b>4,849</b></td>
      <td>Peking University<br>People's Hospital</td>
    </tr>
  </tbody>
</table>

<table align="center">
  <thead>
    <tr align="center">
      <th>Temporal representation</th>
      <th>Structured knowledge</th>
      <th>Specialised learning</th>
      <th>Risk formulation</th>
    </tr>
  </thead>
  <tbody>
    <tr align="center">
      <td>Dilated causal TCN<br>selective state-space layer</td>
      <td>SOFAE graph<br>three relation families</td>
      <td>Six organ experts<br>graph-guided routing</td>
      <td>Discrete-time<br>competing risks</td>
    </tr>
  </tbody>
</table>

## Research overview

Sepsis is not a single static condition. Its clinical course is shaped by a sequence of interacting organ injuries that may stabilise, recover or progress toward shock, multiple organ dysfunction and death. A useful prognostic system must therefore do more than produce one mortality score. It should preserve the identity of individual organ systems, recognise their shared physiological context and represent the fact that some outcomes prevent others from being observed.

Sepsis-MOST formulates this problem as nine related tasks. Six tasks estimate the cumulative incidence of new organ failure over the following seven days, and three tasks estimate the probability of septic shock, MODS and ICU mortality over the same horizon. The organ tasks are modelled separately because failure of the kidney, lungs, heart, brain, coagulation system and liver has different predictors and clinical consequences. They are trained together because the deterioration of these systems is physiologically connected.

The prediction landmark is fixed at six hours after sepsis onset. Information is collected from the preceding 12-hour window, spanning six hours before to six hours after onset. Events already present at the landmark are masked for the corresponding task, while the patient can still contribute to other outcomes that have not yet occurred. This task-specific landmarking prevents an existing event from being presented as a future prediction and allows the model to retain clinically informative patients rather than removing an entire admission because one endpoint is already present.

For every organ, ICU death before organ failure is treated as a competing event. This matters because a patient who dies can no longer experience a subsequently recorded organ failure. Treating such deaths as ordinary censoring would overstate organ-failure risk. Sepsis-MOST instead estimates organ failure, death before failure and event-free survival within a common discrete-time formulation.

## Research contributions

1. **Organ-resolved multi-task prognosis** — one network produces six organ-specific risk trajectories and three adverse-event probabilities while retaining a separate head for every clinical endpoint.
2. **Data-derived organ-event structure** — the SOFAE graph encodes temporal precedence, co-occurrence and syndrome-trigger relations learned from the development data rather than relying on handwritten interaction rules.
3. **Graph-guided expert specialisation** — six organ experts preserve task identity, while graph-biased routing constructs a shared context from the experts most relevant to the current physiological state.
4. **Competing-risk estimation** — ICU death before organ failure is modelled explicitly, yielding organ-specific cumulative-incidence outputs rather than conventional binary probabilities.
5. **Hybrid temporal representation** — causal temporal convolutions capture local clinical change, and a selective state-space layer models longer dependencies across the onset-centred sequence.
6. **Multicohort evaluation** — the frozen development model is evaluated in eICU-CRD and the independent PKUPH cohort without site-specific retraining or recalibration.
7. **Task-level interpretation** — learned graph relations, component analyses and global and patient-level attribution views expose how the model organises shared and endpoint-specific information.

## Clinical scope

The prediction space covers six organ systems routinely represented in sepsis severity assessment. Kidney failure is defined through renal dysfunction and renal-replacement criteria; lung failure through severe oxygenation impairment under ventilatory support; cardiovascular failure through advanced vasopressor or mechanical-support requirements; neurological failure through severe impairment of consciousness; coagulation failure through thrombocytopenia; and liver failure through marked hyperbilirubinaemia.

The three adverse-event outputs represent clinically consequential states that often trigger escalation of care. Septic shock captures circulatory and metabolic deterioration, MODS represents concurrent failure of multiple organs, and ICU mortality provides a terminal outcome shared across the organ-specific competing-risk tasks. These outputs are estimated jointly but remain distinct, allowing performance and calibration to be assessed separately for each endpoint.

The framework is intended for early risk stratification research. It does not prescribe treatment, replace clinician assessment or establish that an intervention based on its predictions improves patient outcomes. Prospective evaluation would be required before clinical use.

<p align="center">
  <img src="assets/figures/clinical-scope.png" width="100%" alt="Clinical scope of Sepsis-MOST">
  <br>
  <sub>Six organ-failure targets and the adverse-event definitions used by the study.</sub>
</p>

## Cohort design

The development cohort is constructed from MIMIC-IV. After applying the study's sepsis and landmark eligibility criteria, 28,800 patients are included and split into training, validation and internal test subsets in a 70:15:15 ratio, stratified by ICU mortality. The internal test set contains 4,321 patients and is not used for parameter estimation or model selection.

External evaluation uses two distinct settings. The eICU Collaborative Research Database contributes 24,662 eligible patients from a multicentre US critical-care environment. The institutional PKUPH cohort contributes 4,849 eligible patients from a Chinese health system and provides the clinical collaboration component of the project. Both cohorts receive the frozen MIMIC-IV model directly, without cohort-specific retraining or recalibration.

The three cohorts differ in case mix, illness severity, event prevalence, missingness and how some clinical variables are recorded. These differences make external evaluation more informative than a random split from one database, but they also mean that an apparent performance change cannot be attributed to geography alone. Variation may reflect population shift, measurement practice, endpoint availability or differences in clinical workflow.

<p align="center">
  <img src="assets/figures/cohort-selection.png" width="100%" alt="Cohort selection for MIMIC-IV, eICU-CRD and PKUPH">
  <br>
  <sub>Cohort selection and analysis roles across the development and external-validation populations.</sub>
</p>

## Prediction task and inputs

The observation window extends from six hours before to six hours after sepsis onset. The prediction time is therefore fixed at onset + 6 h, after which the model forecasts incident outcomes over the next seven days. Organ-failure time is represented in eight intervals ending at 12, 24, 48, 72, 96, 120, 144 and 168 hours. This provides a risk trajectory rather than a single undifferentiated seven-day label.

Two complementary input representations are constructed from the same observation window. The temporal tensor contains 21 variables at each of 12 hourly steps. It combines organ-level SOFA information with severity indicators, vital signs, laboratory measurements, treatment signals and demographic context. This sequence preserves how the patient's physiological state changes around sepsis onset.

The static branch contains 116 summary features. Repeated measurements are represented through their recent value, range, mean, measurement count and within-window trend, with additional descriptors for urine output, neurological status, ventilatory support and related clinical context. The static branch therefore captures distributional and trajectory summaries that may be difficult to recover from a short hourly sequence alone.

Standardisation parameters are estimated on the training split and carried forward unchanged to validation, internal testing and external evaluation. Keeping the landmark, transformation and task-masking logic fixed is essential to separating model transfer from site-specific refitting.

## Model design

### Onset-aligned clinical representation

The first stage converts heterogeneous ICU observations into aligned temporal and static inputs. Measurements recorded more than once in an hour are consolidated for the sequence representation, while window-level summaries preserve minima, maxima, means, counts, recent values and trends. Organ scores remain on their clinical scales, and the remaining variables are transformed using parameters estimated from the training data.

This design gives the model two views of the same clinical period. The temporal stream focuses on the order and direction of change, whereas the static stream provides a robust summary of the overall physiological state and measurement intensity. The two representations are fused only after the temporal and graph-guided expert stages have constructed their respective latent features.

### SOFAE graph constructor

The Sepsis-associated Organ Failure and Adverse Event graph contains nine nodes: six organ failures and three syndrome nodes (sepsis, septic shock and MODS). It is an empirical organ-event graph learned from the order and joint occurrence of outcomes in the MIMIC-IV development data. It is not a literature ontology and does not import external biomedical knowledge.

Three relation families are represented. Temporal-precedence relations describe how often one organ failure appears before another. Co-occurrence relations quantify how frequently outcomes are observed in the same patient. Syndrome-trigger relations connect organ dysfunction with septic shock and MODS. Weakly supported edges are pruned, and the retained relations are compressed into fixed spectral node embeddings.

The graph is estimated once and then frozen. The external cohorts do not contribute to its edges, weights or embeddings, so no outcome information from external evaluation enters the routing structure.

### Hybrid temporal encoder

Each hourly vector is projected into a common latent space and passed through three residual dilated causal convolution blocks. Their increasing dilation expands the temporal receptive field while preserving directionality, so later steps cannot leak information backward into earlier representations. These blocks are designed to capture short-range changes such as a rapid rise in lactate, a fall in blood pressure or the initiation of organ support.

A selective state-space layer follows the convolutional blocks. Its input-dependent dynamics provide a compact way to model longer interactions across the sequence, complementing the local pattern extraction of the TCN. Gating and residual connections preserve the original temporal signal while allowing the encoder to emphasise clinically relevant changes.

### Graph-guided mixture of experts

The KG-MoE predictor contains six expert networks aligned with the six organ systems. At every time step, the router combines the patient representation with the current graph-derived organ context. It then assigns weights to the experts and retains the two strongest routes when building the shared cross-organ representation.

The top-two gate controls only the shared context. Every organ head also receives the stream from its corresponding expert, regardless of the shared router. This distinction allows the network to consult other physiologically relevant experts without depriving an organ of its own specialised representation. The result is a balance between shared learning and organ-specific decision boundaries.

A separate specialisation objective encourages correspondence between each expert and its designated organ. The graph therefore influences which experts contribute to shared reasoning, while the dedicated streams preserve interpretable organ ownership.

### Organ coupling and prediction heads

The graph also contributes a downstream coupling signal. When upstream organs are dysfunctional and the learned precedence structure supports a relationship, the model can increase the failure hazard of the corresponding downstream organ. Unsupported relationships do not receive this adjustment, and the coupling is constrained so that it cannot lower the downstream hazard.

Each organ head combines three information sources: its dedicated expert representation, the shared graph-guided context and the static clinical embedding. For every future time interval, the head predicts organ failure, ICU death before failure and event-free survival. These state probabilities are accumulated into an organ-specific cumulative-incidence function.

The three adverse-event heads read the shared representation and estimate the seven-day probabilities of ICU mortality, septic shock and MODS. All nine tasks are optimised jointly with competing-risk likelihood, temporal ranking, expert-balancing, specialisation and adverse-event objectives.

<p align="center">
  <img src="assets/figures/sepsis-most-framework.png" width="100%" alt="Sepsis-MOST computational framework">
  <br>
  <sub>Onset-aligned feature construction, empirical organ-event graph learning, hybrid temporal encoding and graph-guided multi-task prediction.</sub>
</p>

The module contracts and the source layout are documented in [`ARCHITECTURE.md`](ARCHITECTURE.md); the implementation is described under [Code](#code).

## Evaluation overview

Evaluation is organised around four questions: whether the model separates higher-risk from lower-risk patients, whether its probabilities agree with observed frequencies, whether performance transfers to new clinical environments and whether the proposed components contribute meaningfully to the final system. Organ failures are evaluated with time-to-event and competing-risk metrics; adverse events are evaluated as seven-day binary outcomes.

### Internal organ-failure prediction

In the held-out MIMIC-IV test set, the mean organ-level C-index reported in the current manuscript is 0.795. Discrimination is strongest for liver failure and coagulation failure, with C-indices of 0.905 and 0.877. Kidney and cardiovascular failure reach 0.777 and 0.761, while lung and neurological failure are more difficult, at 0.730 and 0.723.

Performance is also assessed at multiple horizons rather than only at the seven-day endpoint. Across organs, discrimination is highest near the earlier prediction windows and generally declines as the horizon becomes more distant. This pattern is especially relevant for neurological failure, where the available onset-centred record contains less information about events occurring several days later.

<p align="center">
  <img src="assets/figures/internal-organ-roc-pr.png" width="100%" alt="Internal organ-failure ROC and precision-recall analysis">
  <br>
  <sub>Internal organ-failure discrimination across the six organ-specific outputs.</sub>
</p>

### Internal adverse-event prediction

The three adverse-event heads share the graph-guided multi-expert representation with the organ tasks but retain separate output layers. The current manuscript reports internal AUROC values of 0.918 for MODS, 0.917 for septic shock and 0.801 for ICU mortality. The corresponding precision-recall results are interpreted together with event prevalence because ICU mortality is substantially less common than MODS.

Joint training is intended to let organ-level deterioration inform broader syndrome prediction while avoiding a single composite endpoint. Septic shock, MODS and death therefore remain separately inspectable and can be associated with different operating thresholds in future evaluation.

<p align="center">
  <img src="assets/figures/internal-adverse-event-roc-pr.png" width="100%" alt="Internal adverse-event ROC and precision-recall analysis">
  <br>
  <sub>Internal discrimination for ICU mortality, septic shock and MODS.</sub>
</p>

### Calibration

Discrimination alone does not establish that a predicted probability is clinically meaningful. Sepsis-MOST therefore evaluates calibration for every organ and adverse event. The current manuscript reports a mean organ-level integrated Brier score of 0.103 and an integrated calibration index no greater than 0.087 for any organ in the internal test set.

For the three adverse events, calibration slopes are reported near the ideal value of one in the internal cohort. These analyses assess whether predicted risks track observed event frequencies across the probability range, complementing ROC and precision-recall analysis.

<p align="center">
  <img src="assets/figures/internal-calibration.png" width="100%" alt="Calibration of organ-failure and adverse-event predictions">
  <br>
  <sub>Calibration across organ-failure and adverse-event targets.</sub>
</p>

### External transfer

The frozen model is applied directly to eICU-CRD and PKUPH. Brain failure was not evaluated at PKUPH because numeric Glasgow Coma Scale scores were rarely recorded there, so organ discrimination is compared over the five organs evaluated in all three cohorts: the mean C-index is 0.810 internally, 0.719 in eICU-CRD and 0.736 at PKUPH. Adverse-event discrimination transfers more consistently: the reported AUROCs for ICU mortality, septic shock and MODS are 0.811, 0.904 and 0.856 in eICU-CRD and 0.814, 0.865 and 0.832 at PKUPH, where MODS is counted over the five evaluated organs.

Transfer is not uniform across organ systems. Differences in vasopressor documentation, neurological assessment, missingness, clinical workflow and event prevalence can change both the available predictors and the operational meaning of an endpoint. External results should therefore be read at the task level rather than reduced to a single average.

<p align="center">
  <img src="assets/figures/eicu-adverse-event-roc.png" width="100%" alt="External adverse-event ROC curves in eICU-CRD">
  <br>
  <sub>External adverse-event discrimination in eICU-CRD using the frozen development model.</sub>
</p>

<p align="center">
  <img src="assets/figures/pkuph-organ-roc-pr.png" width="100%" alt="Organ-failure ROC and precision-recall curves in PKUPH">
  <br>
  <sub>Organ-failure discrimination for the five organs evaluated at PKUPH, with 95% bootstrap confidence bands.</sub>
</p>

<p align="center">
  <img src="assets/figures/pkuph-adverse-event-roc-pr.png" width="85%" alt="Adverse-event ROC and precision-recall curves in PKUPH">
  <br>
  <sub>Adverse-event discrimination at PKUPH, with 95% bootstrap confidence bands.</sub>
</p>

## Model analysis

### Component analysis

The component study examines the contribution of graph guidance, expert specialisation, temporal encoding, static features, ranking supervision and competing-risk modelling. The purpose is not only to compare overall performance, but also to test whether each architectural choice plays the role assigned to it in the full framework.

The current manuscript panel supports the combined design at a structural level, with graph removal producing the largest reported reduction in mean organ discrimination among the principal component removals. Extended baseline and ablation material remains part of the manuscript evaluation record and is not presented by this repository as an independently reproduced benchmark.

<p align="center">
  <img src="assets/figures/component-ablation.png" width="100%" alt="Ablation of Sepsis-MOST components">
  <br>
  <sub>Component analysis across organ-failure and adverse-event tasks.</sub>
</p>

### Learned organ relations

The SOFAE graph exposes the structure used by the router and organ-coupling mechanism. Temporal-precedence matrices show directional ordering among organ failures, co-occurrence matrices describe shared patient-level burden, and syndrome-trigger matrices relate organ dysfunction to shock and MODS.

These matrices are model inputs rather than causal graphs. A strong edge indicates a recurrent relationship in the development data; it does not prove that one organ failure biologically causes another. Their value is to provide a stable, inspectable structure for shared prediction while preserving the distinction between statistical association and mechanism.

<p align="center">
  <img src="assets/figures/sofae-relations.png" width="100%" alt="Relations learned by the SOFAE graph">
  <br>
  <sub>Temporal precedence, co-occurrence and syndrome-trigger relations learned from the development split.</sub>
</p>

### Global feature attribution

Task-wise SHAP analysis examines how the model distributes feature importance across the six organ failures and three adverse events. The shared encoder does not force every task to rely on the same inputs: organ scores, vital signs, laboratory measurements, interventions and trajectory summaries can receive different importance depending on the target.

This view is useful for detecting whether an endpoint is dominated by clinically plausible information or by a narrow documentation pattern. It remains a description of model behaviour and should not be interpreted as evidence that changing an attributed feature would causally change the outcome.

<p align="center">
  <img src="assets/figures/shap-attributions.png" width="100%" alt="Global feature attributions across Sepsis-MOST targets">
  <br>
  <sub>Global task-wise attribution across six organ failures and three adverse events.</sub>
</p>

### Patient-level explanation

The patient-level view decomposes each of the nine outputs into feature contributions that increase or decrease the predicted risk relative to the model baseline. Displaying all outputs together makes it possible to see why the same physiological state can raise the risk of one endpoint while reducing or leaving unchanged the risk of another.

These local explanations are intended for audit and model inspection. They do not establish causal effects, treatment recommendations or counterfactual outcomes, and they require prospective human-factors evaluation before being incorporated into a clinical workflow.

<p align="center">
  <img src="assets/figures/patient-level-attribution.png" width="100%" alt="Patient-level Sepsis-MOST attribution example">
  <br>
  <sub>Patient-level decomposition of the six organ and three adverse-event predictions.</sub>
</p>

Complete aggregate tables and additional interpretation boundaries are available in [`RESULTS.md`](RESULTS.md).

## Code

### Installation

```bash
git clone https://github.com/ZhWenPrism/SepsisMOST.git
cd SepsisMOST
pip install -e .                  # or: pip install -r requirements.txt
pip install -e ".[test]" && pytest  # optional: run the test suite
```

The code requires Python 3.9 or later and PyTorch 2.0 or later. It was developed with Python 3.12, PyTorch 2.13, NumPy 1.26 and pandas 2.2, and runs on CPU, CUDA and Apple MPS devices.

### Inputs

Training uses a temporal tensor with the competing-risks labels, a static matrix and a table of event onset times for the SOFAE graph. The arrays, shapes and label coding are specified in [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md). The code that derives these files from the source databases is not included in this release.

### Usage

```bash
# 1. Build and freeze the SOFAE graph from organ-failure and syndrome onset times
python scripts/build/build_sofae_graph.py --onsets onsets.parquet --datetime --out runs/sofae_graph.npz

# 2. Train Sepsis-MOST
python scripts/train/train_sepsis_most.py \
    --temporal tensors.npz --static static.npz --graph runs/sofae_graph.npz \
    --config configs/model/sepsis_most.yaml --out runs/sepsis_most.pt

# 3. Score a cohort with the frozen model, without refitting
python scripts/predict/predict_sepsis_most.py --checkpoint runs/sepsis_most.pt \
    --temporal external_tensors.npz --static external_static.npz --out runs/predictions.npz
```

Without `pip install`, prefix the commands with `PYTHONPATH=src`. The training script saves the checkpoint together with the row indices of the data split (`*.split.npz`) and a summary of the validation C-index for every epoch (`*.summary.json`). The frozen model can also be used from Python:

```python
from sepsis_most.training import SepsisMOSTEnsemble

model = SepsisMOSTEnsemble.load("runs/sepsis_most.pt")
pred = model.predict(X, S)   # pred["cif"]: (N, 6, 8) organ CIFs; pred["event_prob"]: (N, 3)
```

### Configuration

All settings are read from [`configs/model/sepsis_most.yaml`](configs/model/sepsis_most.yaml).

| Setting | Value |
|:--|:--|
| Hidden size, state size, static embedding | 96, 16, 64 |
| TCN blocks | 3, dilations 1, 2, 4, kernel size 3 |
| Organ experts | 6 two-layer networks, hidden size 128, top-2 gate |
| Loss weights | λ<sub>syn</sub> 0.5, λ<sub>rank</sub> 0.15, λ<sub>lb</sub> 0.01, λ<sub>spec</sub> 0.2 |
| Optimiser | AdamW, learning rate 2 × 10<sup>-3</sup>, weight decay 2 × 10<sup>-4</sup>, cosine schedule |
| Training | batch size 256, up to 60 epochs, early stopping on the validation mean C-index (patience 10), input jitter 0.1 |
| Data split | 70:15:15, stratified by ICU mortality, seed 42 |
| Seeds | 42, 7 and 2024; one network per seed, predictions averaged |

### Manuscript components and source files

| Manuscript component | Source |
|:--|:--|
| SOFAE Graph Constructor (Eqs. 1–4) | [`src/sepsis_most/graph/`](src/sepsis_most/graph) |
| Temporal Feature Extractor (Eqs. 5–7) | [`src/sepsis_most/model/encoders/`](src/sepsis_most/model/encoders) |
| KG-MoE Predictor (Eqs. 8–13) | [`model/routing/`](src/sepsis_most/model/routing), [`model/experts/`](src/sepsis_most/model/experts), [`model/pooling.py`](src/sepsis_most/model/pooling.py) |
| Organ coupling and prediction heads (Eqs. 14–17) | [`model/sepsis_most.py`](src/sepsis_most/model/sepsis_most.py), [`model/heads/`](src/sepsis_most/model/heads) |
| Training objective (Eqs. 18–23) | [`src/sepsis_most/objectives/`](src/sepsis_most/objectives) |
| Optimisation, model selection and inference | [`src/sepsis_most/training/`](src/sepsis_most/training), [`evaluation/discrimination/`](src/sepsis_most/evaluation/discrimination) |

## Repository

```text
SepsisMOST/
├── assets/figures/                 # manuscript figures
├── configs/model/sepsis_most.yaml  # model, objective and optimisation settings
├── docs/DATA_FORMAT.md             # specification of the input files
├── results/tables/                 # aggregate, non-patient-level results
├── scripts/
│   ├── build/build_sofae_graph.py  # SOFAE graph construction
│   ├── train/train_sepsis_most.py  # training
│   └── predict/predict_sepsis_most.py  # scoring with a frozen model
├── src/sepsis_most/
│   ├── constants.py                # organs, adverse events, graph nodes, time bins
│   ├── graph/                      # relations, spectral embedding, organ coupling
│   ├── model/                      # encoders, experts, routing, heads, full network
│   ├── objectives/                 # likelihood, cross entropy, ranking, balancing, specialisation
│   ├── training/                   # configuration, data interface, trainer, frozen model
│   └── evaluation/discrimination/  # concordance index for model selection
├── tests/                          # unit and integration tests on synthetic data
├── pyproject.toml
├── requirements.txt
├── ARCHITECTURE.md
├── RESULTS.md
└── README.md
```

The directories `src/sepsis_most/clinical/`, the remaining `src/sepsis_most/evaluation/` subdirectories, `src/sepsis_most/interpretability/`, `configs/cohorts/`, `configs/evaluation/` and `scripts/evaluate/` are reserved for the cohort, endpoint, feature construction, evaluation and interpretation code of subsequent releases.

The repository does not include raw ICU records, patient-level derivatives, protected health information, trained weights, checkpoints, credentials or private configuration. MIMIC-IV and eICU-CRD remain subject to PhysioNet credentialing and their respective data-use agreements, and the PKUPH cohort is not redistributed.

## Interpretation and validation boundaries

This is a retrospective multicohort modelling study, not a prospective clinical trial. The reported results establish performance within the analysed datasets but do not establish clinical effectiveness, workflow benefit or patient-outcome improvement. Decision thresholds, alert burden, treatment response and prospective transportability require separate evaluation.

Endpoint harmonisation is also imperfect across databases. In the PKUPH cohort, numeric Glasgow Coma Scale scores were rarely recorded, so brain failure was not evaluated and MODS was counted over the other five organs. Cardiovascular failure at PKUPH depends on vasopressor rates reconstructed from order totals. These endpoints are not fully equivalent to their MIMIC-IV and eICU-CRD counterparts and should be interpreted as cohort-specific operationalisations.

Sepsis-MOST is therefore presented as a research framework for organ-resolved prognosis, multicohort validation and model interpretation. It is not a medical device and is not intended for direct clinical decision-making in its current form.
