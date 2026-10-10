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

| Role | Cohort | Patients | Use |
|:--|:--|--:|:--|
| Development | MIMIC-IV | 28,800 | Training, validation and internal testing |
| External validation | eICU-CRD | 24,662 | Frozen-model evaluation |
| External validation | PKUPH | 4,849 | Frozen-model evaluation in an independent health system |

## Repository

This repository contains the model specification, release-oriented package layout, manuscript figures and aggregate result tables. Patient-level data, trained weights and source implementation are not distributed here.

Detailed architecture and evaluation records are provided in [`ARCHITECTURE.md`](ARCHITECTURE.md) and [`RESULTS.md`](RESULTS.md).

## Data governance

MIMIC-IV and eICU-CRD are available through PhysioNet under credentialed access and their respective data-use agreements. The independent institutional cohort is not redistributed. No protected health information or patient-level derivatives are included in this repository.
