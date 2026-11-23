# Pascal signature transfer across motion datasets

The [pre-signature transform study](transforms/README.md) extends the experiments with complete nested eight-person DSADS evaluation, richer NinaPro observations, and matched transform controls across all six datasets.

The [observable-first follow-up](followup/README.md) investigates the NinaPro and DSADS failures, transfers the later HAR geometry, occupation and event ideas, and adds a protocol-qualified literature review. The original experiment below is preserved.

This experiment transfers the raw and spectral Pascal constructions from
[UCI HAR](../uci_har/README.md) to NATOPS and the sibling tensorsig examples:
DSADS, UTD-MHAD, KIMORE and NinaPro. The same generic feature recipe is applied
to each dataset. Dataset-specific phone posture, gravity and sensor-rotation
features are not part of this transfer.

**The transfer is useful but uneven.** Adding spectral Pascal features improves
both raw Pascal variants on all three classification test sets. Raw Pascal
coding itself improves its matched uncoded controls on NATOPS and UTD-MHAD,
but hurts on DSADS. On KIMORE, coding improves the uncoded signature controls,
yet conventional moments have lower MAE. NinaPro supplies a clear failure case:
the transferred Pascal maps have substantially worse squared error than simple
moment and lagged-MAV controls. These statements concern the fixed recipes and
splits below, not every possible tuned signature model.

The run covers **32 evaluation units, 8,784 validation-grid fits and 484 final
fits**. All model results, including failures, are retained in the
[summary CSV](results/summary.csv) and [paired contrasts](results/contrasts.json).

## Classification results

Test accuracy in percent; each row chooses its own ridge penalty using only
training-side validation. P1 and P3 use the same integrated observations.

| Representation | NATOPS | UTD-MHAD | DSADS |
|---|---:|---:|---:|
| Conventional moments | 85.00 | 73.26 | 87.19 |
| Ordinary position-path log | 90.00 | 77.21 | 83.77 |
| P1 integrated log | 81.67 | 62.56 | 77.54 |
| P3 raw log | 87.22 | 71.40 | 73.42 |
| P3 raw plus joint spectral log | 91.67 | 80.93 | 85.44 |
| P1 integrated tensor | 91.11 | 63.02 | 77.72 |
| P3 raw tensor | 92.78 | 66.74 | 73.25 |
| P3 raw plus spectral tensor | 93.33 | 78.14 | 88.16 |

Adding spectra to the raw tensor removes 1, 49 and 170 test errors on NATOPS,
UTD-MHAD and DSADS respectively. The corresponding log comparison removes
8, 41 and 137 errors. These are paired descriptive differences after independent
penalty selection, not significance tests or equal-dimension comparisons.

Selecting the representation itself by validation gives different choices:

| Dataset | Validation-selected representation | Kept features | Validation accuracy | Test accuracy |
|---|---|---:|---:|---:|
| NATOPS | Spectral pool plus raw Pascal tensor | 5,686 | 93.89 | 92.78 |
| UTD-MHAD | Spectral pool plus raw Pascal log | 17,370 | 80.28 | 80.70 |
| DSADS | Factorized spectral Pascal log | 315 | 97.81 | 80.35 |

NATOPS's highest observed score, 93.33%, differs from the selected model by
one test case. UTD-MHAD's joint raw/spectral log likewise beats its selected
model by one case. DSADS is the important caution: validation on subject 7
favors a compact spectral representation that transfers poorly to subject 8.
The observed 88.16% tensor result must not be presented as the outcome of
validation-based model selection. Conventional moments reach 87.19% there.

![All fixed classification models and validation selections](results/classification.png)

These new scores do not exceed the earlier repository results of 96.11% for
[NATOPS offset features](../../paper/evidence/natops/offset_learning/README.md),
81.63% for the [UTD-MHAD joint-local offset model](../../../tensorsig/examples/utd_mhad/README.md),
or 88.86% for [DSADS mean/std](../../../tensorsig/examples/dsads/results/holdout.md).
Those historical runs use different features or tuning; they are context,
not matched controls in this experiment.

## KIMORE regression results

Each exercise pools its five held-out subject folds before calculating metrics;
the table then averages the five exercise metrics equally. Clinical TS remains
on its original 0–50 scale. Lower MAE and RMSE are better.

| Representation | Mean exercise MAE | Mean exercise RMSE |
|---|---:|---:|
| Conventional moments plus duration | 6.813 | 9.842 |
| P1 integrated tensor | 8.112 | 10.400 |
| P3 raw tensor | 7.633 | 9.527 |
| P3 raw plus spectral tensor | 7.302 | 9.517 |
| P3 spectral tensor alone | 7.251 | 9.092 |
| Conventional spectral pool | 7.256 | 9.401 |
| Representation selected separately in each validation fold | 7.242 | 10.501 |

Coding and spectral features help relative to the uncoded tensor, but the
conclusion depends on the loss: moments have the lowest mean exercise MAE,
while the spectral tensor has the lowest mean exercise RMSE among the fixed
maps. The validation-selected mixture chooses moments in 11 of 25 units and
has worse held-out RMSE than either fixed alternative. Selecting among many
representations from small validation groups is not reliably beneficial here.

![KIMORE mean exercise MAE](results/kimore.png)

## NinaPro regression results

These are subject-1 acquisition-transfer results. The four columns change the
number of acquisition-1 calibration repetitions per motion; validation and
test acquisitions remain fixed. Table entries are test RMSE across all 18
glove outputs, in the supplied calibrated units.

| Representation | 1 repetition | 2 repetitions | 5 repetitions | 10 repetitions |
|---|---:|---:|---:|---:|
| Conventional moments | 25.96 | 24.98 | 24.70 | 23.97 |
| Lagged MAV | 26.99 | 26.36 | 25.07 | 25.67 |
| P1 integrated tensor | 53.33 | 44.66 | 43.44 | 43.13 |
| P3 raw tensor | 80.65 | 43.05 | 45.70 | 31.57 |
| P3 spectral tensor | 29.42 | 33.46 | 27.60 | 27.19 |
| P3 raw plus spectral tensor | 70.72 | 38.19 | 39.55 | 36.18 |

Validation selects lagged MAV, moments, P1 tensor and P1 tensor at the four
budgets respectively. The last two choices generalize poorly to acquisition 3.
At the full budget, raw plus spectral log features reduce MAE from 17.55 to
16.02 relative to raw log features, but RMSE remains 35.86 versus 25.67 for
lagged MAV and 23.97 for moments. Reporting MAE alone would hide these failures.
Independent LSQR fits reproduce the high errors of the full-budget raw and
raw/spectral tensors; they are not a discrepancy between ridge solvers.

![NinaPro RMSE across calibration budgets](results/ninapro_rmse.png)

## What is transferred

Three orthonormal Pascal columns weight each signal by constant, linear and
quadratic functions of sample position. The raw construction treats each
observation divided by the number of observations as an increment. Its first
signature level therefore retains weighted averages, including signal levels;
its second level adds interactions between observations in their temporal
order. This reuses the actual UCI HAR implementation.

The spectral construction first computes one-sided FFT power, with the correct
doubling of interior bins, and normalizes each channel's power to sum to one.
It then applies the same Pascal codes and depth-two signature **in increasing
frequency order**. These areas describe spectral shape and relationships
between channels' spectra; they do not retain the original time ordering or
Fourier phase. Energy and `log(1 + energy)` are included separately so normalization
does not discard all amplitude information. Zero-energy spectra have zero
mass. Frequencies are fractions of Nyquist, without dataset-specific band tuning.

The fixed menu contains these controls and comparisons:

| Family | Meaning |
|---|---|
| `moments` | Eight conventional summaries per channel |
| `ordinary_path_log` | Observed positions, a clock and a prepended origin; ordinary depth-two logsignature |
| `ordinary_raw_log`, `ordinary_raw_tensor` | P1 integrated-observation signatures; matched controls for adding Pascal codes |
| `raw_log`, `raw_tensor` | P3 coded observations without a spectral transform |
| `spectral_pool` | Eighteen conventional spectral summaries per channel |
| `spectral_factorized_log` | P3 spectral logsignatures keeping only within-channel areas, plus energy |
| `spectral_log`, `spectral_tensor` | P3 joint spectral signatures, including cross-channel interactions, plus energy |
| `raw_spectral_factorized_log`, `raw_spectral_log`, `raw_spectral_tensor` | Concatenated raw and spectral Pascal features |
| `pool_raw_log`, `pool_raw_tensor` | Conventional spectral pool concatenated with raw Pascal features |

Tensor features expose endpoint products directly to the linear ridge model.
Logsignature features retain endpoints and signed areas. At depth two, each
determines the other by a nonlinear formula, but a linear classifier cannot
perform that conversion. Both are reported rather than treated as equivalent
linear feature sets. Coding also increases dimension: with 24 NATOPS channels,
P3 has 2,628 log coordinates or 5,256 tensor coordinates before adding spectra.
The comparisons do not control for equal feature count.

## Evaluation

The [protocol](protocol.json) fixed the feature menu and selection rules before
the first new model scores. Earlier benchmark outcomes in these repositories
were already known. These are exploratory transfer experiments, not new
untouched benchmark claims.

| Dataset | Inputs and target | Training and evaluation |
|---|---|---|
| NATOPS | 51 frames, 24 landmark coordinates; six gestures | Official 180/180; five stratified folds within training, then refit all 180 |
| UTD-MHAD | Existing 32-frame, 60-coordinate skeleton preprocessing; 27 actions | 431 odd-subject training trials, 430 even-subject test trials; inner leave-one-training-subject-out selection |
| DSADS | 125 frames, 45 sensor channels; 19 activities | Subjects 1–6 train, subject 7 validates, subject 8 tests; 6,840/1,140/1,140 windows |
| KIMORE | Existing 128-frame, 75-coordinate skeleton preprocessing; clinical score | Five subject folds and five exercises, full training budget; next fold validates, remaining three train |
| NinaPro DB8 | Six causal MAV observations, 16 channels; 18 glove targets | Subject 1 only; acquisitions 1/2/3 train/validate/test; 1, 2, 5 or 10 calibration repetitions per motion |

All axes are scaled using training observations only. Signal scaling, constant
column removal and final feature scaling are refitted inside every inner fold.
Classification uses unweighted ridge and the eleven HAR alpha values from
0.01 to 1,000 in the summed-error objective. Regression preserves the sibling
examples' mean-error penalty grids: six values for NinaPro, and 18 choices for
KIMORE including the exact training-mean limit. Larger penalties break ties.
Validation selects each representation's penalty, then optionally selects the
representation using the fixed menu order to break exact ties. Choices are
saved before computing test features and scores. DSADS, KIMORE and NinaPro
never refit on their validation observations.

Eleven of fifteen UTD-MHAD variants and eight DSADS variants select the upper
classification penalty of 1,000. The grid was not expanded after test scores
were observed; stronger regularization could change these comparisons. NATOPS
does not select a boundary penalty. Its archive does not establish a verified
subject-disjoint split, whereas the UTD-MHAD and DSADS protocols explicitly
separate subjects. No repeated deterministic fits are used to claim uncertainty.

KIMORE retains the original loader's strict exclusions, score identity checks,
torso normalization and subject folds. Duration is appended to every feature
map; the ordinary position-path control also uses the original elapsed-time
clock. KIMORE spectra describe normalized trial time rather than a common
physical frequency grid, because recording durations differ. NinaPro
additionally includes the original lagged-MAV linear control. Its spectrum
has only four bins and is a spectrum of the six-point **MAV
history**, not the original 2 kHz EMG. Overlapping NinaPro windows are not
independent repetitions, and subject 1 cannot establish population transfer.

## Reproduce

Run from the application repository root, with its sibling tensorsig checkout:

```sh
../tensorsig/.venv/bin/python examples/pascal_transfer/benchmark.py --self-check
../tensorsig/.venv/bin/python examples/pascal_transfer/benchmark.py \
  --datasets natops utd_mhad dsads --download --output /tmp/pascal-classification
```

Classification downloads are hash-checked against the previously recorded
official archives. To repeat the regression experiments, first download the
original data using the [KIMORE downloader](../../../tensorsig/examples/kimore/download.py)
and [NinaPro instructions](../../../tensorsig/examples/ninapro/README.md), placing
them in `examples/pascal_transfer/data/kimore` and `data/ninapro` respectively:

```sh
../tensorsig/.venv/bin/python ../tensorsig/examples/kimore/download.py \
  examples/pascal_transfer/data/kimore --workers 6
../tensorsig/.venv/bin/python examples/pascal_transfer/benchmark.py \
  --datasets kimore ninapro --output /tmp/pascal-regression
```

Existing unit output directories are never overwritten. Raw data are cached
under ignored `data/`. DSADS keeps several gigabytes of feature matrices in
memory; the full run is intended for a workstation. No neural training or
external classifier scores are substituted for local measurements.

Every result directory includes the full validation grid, selected settings,
test predictions, metrics, split indices, data/loader hashes and timings.
The [feature and ridge self-check](checks.json) checks direct signature
accumulation, Parseval scaling, zero spectra, held-out perturbations and
independent ridge solvers. The [source snapshot](results/source_at_run.zip)
preserves the code used for the initial runs.

The [classification verification](results/verification.json) checks all 45
reported model rows and 1,650 grid rows. It independently reconstructs all
15 feature maps on real NATOPS/UTD-MHAD samples, reproduces all 30 final
classifiers with scikit-learn LSQR, and checks all 135 selected-penalty inner
fold fits. For DSADS, independent feature reconstruction and three solvers
check the validation-selected compact model on all 9,120 records; the other
DSADS models receive saved-metric and selection audits.

The [regression verification](results/regression_verification.json) checks
all 439 model rows and 7,134 grid entries, refits all 29 validation-selected
models with an independent solver, and checks the full-budget NinaPro tensor
failures separately. The [adapter verification](results/regression_adapter_verification.json)
confirms that KIMORE retained all 363 original accepted recordings and 27
exclusions, with identical subject folds, and that all NinaPro source hashes
and calibration counts match the sibling example.

To audit a fresh run and rebuild summaries:

```sh
../tensorsig/.venv/bin/python examples/pascal_transfer/verify_results.py \
  --datasets natops utd_mhad dsads --results /tmp/pascal-classification
../tensorsig/.venv/bin/python examples/pascal_transfer/verify_regression.py \
  --results /tmp/pascal-regression
../tensorsig/.venv/bin/python examples/pascal_transfer/summary.py \
  --results examples/pascal_transfer/results
```

Plot regeneration additionally needs Matplotlib; it is optional for numerical
summaries. The numerical runs used one thread and took about 0.7 seconds for
NATOPS, 6.8 seconds for UTD-MHAD and 303 seconds for DSADS, excluding dataset
loading and downloads. These are complete menu timings with shared feature
extraction, not timings for one selected model.
