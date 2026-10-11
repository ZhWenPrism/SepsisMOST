# Input data format

Sepsis-MOST is trained from three files prepared from the source databases. The code that builds these files from MIMIC-IV, eICU-CRD or hospital records is not part of this release, and no data are distributed with the repository.

All arrays are NumPy arrays stored in `.npz` files. Rows of the temporal and static files refer to the same patients in the same order. `N` is the number of patients, `T = 12` the number of hourly steps, `F = 21` the number of temporal features, `P = 116` the number of static features and `K = 8` the number of time bins.

## Temporal tensor with labels (`--temporal`)

| Array | Shape | Type | Content |
|:--|:--|:--|:--|
| `X` | (N, T, F) | float32 | Hourly features over the observation window, from 6 h before to 6 h after sepsis onset. Channels 0 to 5 are the SOFA subscores of the kidney, lung, heart, brain, coagulation and liver, in this order, and channel 6 is the total SOFA score. These seven channels keep their native scale; all other channels are standardised by the training code. |
| `atrisk` | (N, 6) | int | 1 if the patient is alive and free of failure of the organ at the time of prediction (onset + 6 h), else 0. Patients with 0 are masked from that organ's likelihood. |
| `cause` | (N, 6) | int | 0 = failure of the organ, 1 = ICU death before failure of the organ, -1 = censored. |
| `ebin` | (N, 6) | int | Index (0 to K - 1) of the time bin in which the event or censoring falls. |
| `etime` | (N, 6) | float32 | Hours from the time of prediction to the event or censoring. |
| `evt` | (N, 6) | int | 1 if failure of the organ was observed, else 0. |
| `death_h`, `shock_h`, `mods_h` | (N,) | int | 1 if ICU death, septic shock or MODS occurred within 7 days of the time of prediction. |
| `bin_edges` | (K,) | float | Right-closed bin edges in hours: 12, 24, 48, 72, 96, 120, 144, 168. |

An optional integer array `stay_id` may be stored in both the temporal and the static file; when present, the training code checks that the rows are aligned.

## Static matrix (`--static`)

| Array | Shape | Type | Content |
|:--|:--|:--|:--|
| `S` | (N, P) | float32 | Window summaries (last, minimum, maximum, mean, count and linear trend of each laboratory measurement) and additional clinical features. Missing values are imputed before training; at scoring time, a missing value is replaced with the training mean. |

## Event onset table (`--onsets`, SOFAE graph)

A CSV or Parquet table with one row per patient and the columns `kidney`, `lung`, `heart`, `brain`, `coagulation`, `liver`, `sepsis` and `septic_shock`. Each column holds the onset time of the event, either as a datetime (use `--datetime`) or in hours on a common time axis within each patient, and is empty when the event did not occur. MODS onset is derived as the second earliest organ failure.

## Split file (optional, `--split-file`)

An `.npz` file with integer arrays `train`, `val` and `test` holding row indices. Without it, the rows are split 70:15:15 at random, stratified by ICU death (`death_h`), with seed 42. The split used in a run is saved next to the checkpoint.
