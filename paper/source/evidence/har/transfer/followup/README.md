# Observable-first Pascal transfer: diagnosis and experiments

The subsequent [pre-signature transform study](../transforms/README.md) adds a complete nested eight-person DSADS evaluation and matched transform experiments on all six datasets. The results below retain this earlier follow-up's protocol.

The useful transfer from the later UCI HAR work is to **choose physically meaningful, bounded observations before computing coded averages and signatures**, and to control extrapolation at the ridge readout. Increasing the raw polynomial feature set alone does not address the observed failures. The new experiments improve the validation-selected results on UTD-MHAD, KIMORE and three of four NinaPro calibration budgets. They also retain failures: NATOPS selection gets worse, and small KIMORE validation sets can still choose unstable models.

This is an exploratory follow-up designed after inspecting the [original transfer results](../README.md). The new [model menu and protocol](protocol.json) were fixed before the new outcomes. All feature calibration and model selection exclude each unit's test observations and labels, but the test sets are reused from the earlier investigation. These results are evidence for further replication, not an untouched confirmation or a SOTA claim.

## What went wrong

**NinaPro: nonlinear extrapolation concentrated in a few trials.** At the full calibration budget, MAV coordinates reach 30.1 training standard deviations on test; raw Pascal tensor coordinates reach 348.0. Movement 7, repetition 2 contains only 7.1% of test windows but accounts for 74.7% of the old validation-selected tensor model's squared error. Projecting features into their training ranges with the original fitted coefficients changes that model's RMSE from 43.13 to 24.64. This is a posthoc sensitivity experiment, independently reproduced, and not itself a validation-selected result. Glove targets also shift between acquisitions; the data do not establish a unique physical cause.

**DSADS: one validation person gives an overly favorable ranking.** The old spectral winner scores 97.81% on person 7 but 80.35% on person 8. Both people have exactly 60 windows in each of 19 classes. Person 8 has much larger class-specific acceleration-axis shifts, especially for standing and elevator activities. A changed pose or sensor orientation is plausible, but not established. Holding out each training person among 1–6, while keeping final fitting on 1–6, selects moments among three original compact controls and gives 86.14% on person 8. Thus changing selection alone helps, without using person 7 or 8 to fit the model.

The [diagnostic report](DIAGNOSTICS.md) contains the class-level errors, amplitude shifts, exact prediction decomposition, limits of the explanation, and reproducible evidence.

## The mathematical transfer

Let \(x_t\) be an observation and \(q_a(t)\) a fixed Pascal clock. Given an observable \(\phi\), its coded occupation is

\[
C_{a,\phi}=\frac1N\sum_{t=1}^N q_a(t)\phi(x_t).
\]

A constant clock measures how much of the observation occurs; a changing clock distinguishes when it occurs. These are first signature levels of increments \(q(t)\otimes\phi(x_t)/N\). Taking depth-two logs adds signed areas between the coded streams. Choosing a nonlinear \(\phi\) makes its particular question directly available to a linear ridge readout. We do **not** claim that signatures of transformed observations are linear coordinates of the original raw signature.

The fixed additions are:

- **Physical geometry.** Skeleton bone norms, adjacent-bone cosines and dot products; DSADS norms and pairings of acceleration/gyro/magnetometer vectors within each device. These use physical coordinates before independent axis scaling. Common-rotation invariance holds for norms and dots within the stated frame, not for arbitrary independent sensor rotations. NinaPro instead uses bounded activation products and signed increments.
- **Bounded occupations.** Unit bone or sensor directions, their squares, and three fixed hinges, each with two Pascal clocks. NinaPro uses \(u=\log(1+\mathrm{MAV}/r)/(1+\log(1+\mathrm{MAV}/r))\), with training RMS \(r\). HAR uses gravity direction. These expose posture or activation occupancy directly.
- **Local Pascal logs.** Matched per-channel P3, depth-two maps on the whole window and its two halves, using either training-standardized observations \(z\) or \(\tanh(z/2)\). The halves give local clocks; the matched pair tests bounding without changing coordinates or dimension.
- **Event memory.** The HAR soft-reset threshold rule on bounded charges, with decay 0 or 0.75, P3-coded spike rates, and final state. State starts at zero for each window. The decay-zero control separates state from a static threshold transformation; this is a history-dependent observable, not a polynomial word of the raw path.
- **Spectral observations.** Per-channel P3 logs of normalized one-sided power of the bounded streams, with energy summaries. NinaPro also gets a separately identified branch from the actual trailing 256 raw EMG samples. It has the same causal 128 ms endpoint window as the current MAV, but retains information absent from six MAV vectors.

Raw second signature levels scale quadratically with standardized amplitude. Bounding observations limits this growth, but a coordinate with tiny training variance can still become large after feature standardization. Therefore a separate candidate projects each readout coordinate into its fitted training interval \([a,b]\):

\[
\operatorname{clip}_{[a,b]}(f)=a+(f-a)_+-(f-b)_+.
\]

This is an exact hinge readout, using training-derived knots. It leaves training rows unchanged and limits held-out extrapolation coordinatewise. The projected vector need not itself be a realizable signature, and being inside a coordinate box does not guarantee in-distribution prediction. It is evaluated as a readout transformation, not as a signature identity.

“Bounded” in the model names describes the added observation streams. The base pool still includes unbounded moments and energy; only the separate readout projection bounds every retained query coordinate by its training range.

## Selection and reproducibility

The base is moments plus the generic spectral pool; standalone moments and spectral-pool controls are retained. Each concatenated base and extra group is standardized using training data and divided by the square root of its retained width. An extra-group weight \(\gamma\in\{0,0.1,1\}\) is applied **after** scaling, so zero exactly recovers the appropriate base. Ridge minimizes mean squared error plus \(\lambda\|W\|^2\), with a free intercept, over \(10^{-8},\ldots,10^2,\infty\). This broadens and normalizes the earlier penalty search; baseline changes must not be attributed solely to new features. For classification, the squared-error targets are one-versus-all codes, while validation minimizes classification errors.

All five original datasets retain their data, outer test partitions and final training pools. DSADS primary selection uses six training-person holdouts, and the old person-7 selector is separately evaluated with the same new menu. Both choices are saved before computing person-8 features. KIMORE retains 25 exercise/fold units and appends duration to every base. NinaPro retains subject 1, all eligible windows, four A1 repetition budgets, A2 validation and A3 test. UCI HAR is added as a sixth sanity check using its **official subject split**, unlike the random-window split behind the near-99% local HAR results.

The run covers **33 evaluation units, 21,384 recorded validation-grid entries and 526 final model/selector fits**. Nested gamma-zero grid predictions are cached and reused; the grid-entry count is not a count of distinct matrix decompositions. Every unit saves the complete grid, choices, predictions, calibration metadata and source hashes under [results/](results/). No test-optimal model is substituted for the validation choice. Full model and ablation tables are generated by [summarize.py](summarize.py).

## Results

The headline is the selected procedure, including representation selection. KIMORE pools outer-fold predictions separately for each exercise, then averages the five exercise metrics. NinaPro errors are in the supplied 18 glove-channel units; they are not the published five-actuator-angle metric.

| Dataset / metric | Original validation-selected | New validation-selected |
|---|---:|---:|
| NATOPS accuracy | 92.78% | 90.00% |
| UTD-MHAD accuracy | 80.70% | 83.49% |
| DSADS person-8 accuracy | 80.35% | 82.72% |
| KIMORE mean exercise MAE | 7.242 | 6.331 |
| KIMORE mean exercise RMSE | 10.501 | 9.783 |
| NinaPro 1 repetition RMSE | 26.986 | 22.778 |
| NinaPro 2 repetitions RMSE | 24.979 | 25.101 |
| NinaPro 5 repetitions RMSE | 43.443 | 22.182 |
| NinaPro 10 repetitions RMSE | 43.125 | 21.804 |
| UCI HAR official subject-split accuracy | No matched old run | 94.44% |

**NinaPro improves beyond fixing a bad selector.** At ten repetitions, the new selected bounded/event/spectral map with clipping has MAE 14.063, RMSE 21.804 and R² 0.479. The stronger original moments baseline has MAE 15.223, RMSE 23.971 and R² 0.370. Thus the RMSE improvement against that baseline is about 9%, versus the much larger change from the unstable old selected model.

Matched follow-up comparisons clarify the gain. At ten repetitions, pool + geometry + occupations + unbounded local Pascal gives RMSE 25.014; replacing those local observations by bounded ones gives 23.109. The richest bounded map gives 23.132, and adding the training-range readout gives 21.804. Stateful versus static events is a small change, 23.269 versus 23.351. Actual raw-EMG spectra added to MAV give 25.563, and their selection at two repetitions produces 25.101, slightly worse than the original moments baseline. True spectral input did not supply a universal final improvement.

**KIMORE benefits from the bounded features, while selection remains fragile.** The fixed clipped rich map gives mean exercise MAE 6.046 and RMSE 7.863; clipped moments gives 6.174 and 7.866. A fixed bounded-local map gives 6.102 and 8.093, versus 6.734 and 11.195 for its unbounded counterpart. These are predeclared model-family results with validation-tuned hyperparameters, not overall representation-selected scores. In exercise 1/fold 1, the selector still picks the unbounded-local model and obtains RMSE 35.47. Its effect remains in the headline; the selected mean exercise R² is negative despite improved aggregate MAE.

**DSADS selection is still unresolved with the wider menu.** Training-person CV selects the clipped rich map at 94.01% validation accuracy, but person 8 gets 82.72%. Using person 7 to select from the identical new menu chooses moments and gives 87.19%. The compact three-control diagnostic above therefore does not establish that training-person CV will improve selection after expanding the hypothesis set. The richest map wins CV by just six correct predictions out of 6,840 relative to pool + geometry; the latter gets 84.30% on person 8. Geometry and occupations help their new pool baseline, but neither they nor the selection change beat the original moments result of 87.19%, or the original fixed raw-plus-spectral tensor result of 88.16%. There is no demonstrated DSADS last-mile gain.

**There is no universal last feature.** NATOPS improves from the new pool baseline's 77.22% to 92.78% with geometry and occupations; stateful events reach 93.89%, but the validation selector chooses occupations alone and gets 90.00%. On UTD, bounded local Pascal is selected at 83.49%; the clipped rich model reaches 85.12% without winning validation. On official-split HAR, the rich bounded model reaches 96.91%, but selection chooses unbounded local Pascal at 94.44%. These test maxima illustrate selection uncertainty and are not promoted to headline scores.

![Classification results with validation selections](results/classification.png)

![NinaPro RMSE across calibration budgets](results/ninapro_rmse.png)

The [KIMORE plot](results/kimore.png), [NinaPro MAE plot](results/ninapro_mae.png), [247-row summary](results/summary.csv), [paired contrasts](results/contrasts.json) and [old/new selected comparisons](results/old_selected_comparisons.json) retain the complete results.

## Where this sits relative to the literature

The [literature review](literature.md) and [source ledger](literature_sources.json) cover all six datasets through 12 October 2026. NATOPS FIC-TSC reports 98.9% with universal settings, above either local transfer. UTD Dynamic Edge CNN reports 99.53% using the same nominal odd/even subject split, although its hyperparameter-selection independence needs reproduction. DSADS subject-transfer references include MTSDNet at 92.16% and CCIL at 90.7%, averaging different subject/window protocols.

KIMORE papers use different segmentation, targets, modalities and subject separation; none inspected establishes SOTA for our exact whole-recording manifest and folds. NinaPro's recent optimized TCN reports 5.45° and multitask LoRA 5.33°, but predicts five mapped coordinates with A1+A2 training and roughly six seconds of context across participants. Those are useful future replication targets, not numerical rivals to our single-participant, A1-only, 18-output experiment. An official-split HAR CNN/SVM reference reports 97.66% while using both raw signals and the supplied 561 engineered features; near-99% spiking reports need separate split qualification. The cited review provides primary-source links and details for every number.

## Run and check

From the repository root, with the sibling tensorsig environment and existing caches:

```bash
../tensorsig/.venv/bin/python examples/pascal_transfer/followup/run.py --output /tmp/pascal-followup-replay
../tensorsig/.venv/bin/python examples/pascal_transfer/followup/check_readout.py
../tensorsig/.venv/bin/python examples/pascal_transfer/followup/verify_followup.py
../tensorsig/.venv/bin/python examples/pascal_transfer/followup/summarize.py
```

The extraction checks cover direct signature formulas, held-out independence, physical geometry, constant paths and bounded tails. The readout audit compares 144 synthetic configurations against sklearn SVD, including primal/dual, clipping, rank deficiency, classification, regression and infinite shrinkage; the largest regression difference is 2.5×10⁻⁹. The independent saved-output audit passes all 33 units, 526 metric rows, 21,384 grid entries and 1,211 unique source/data/cache hashes. Selected real-data sklearn LSQR checks give identical classification labels and regression prediction differences below 3.7×10⁻⁷. It does not independently rebuild every feature map or every grid fit. See [feature checks](feature_checks.json), [readout checks](readout_checks.json), [verification](results/verification.json), [environment and dependency hashes](environment.json), and the [source snapshot](source_snapshot.zip).
