# Pre-signature transforms and complete DSADS subject evaluation

**Neither NinaPro nor DSADS has an established state-of-the-art result from this work.** NinaPro now improves further on our local baseline, but still covers one participant and 18 glove channels. Published leaders generally predict five mapped actuator angles across participants, with more training data and longer temporal support. DSADS now receives a complete eight-person outer evaluation; its literature comparisons still require matching windows, channels and selection protocols.

This study tests transforms **before** raw and spectral Pascal signatures. The main lesson is that removing or compressing an unwanted variation can matter more than adding signature coordinates. However, a transform that removes sensor orientation can also remove the posture being classified. Matched controls and all-person testing are needed to tell these cases apart.

These are exploratory follow-ups on previously used cohorts. The [menu](protocol.json) was fixed before this round's model outcomes. Every training-dependent transform and readout is fitted inside its training fold. Nested selection prevents using an outer person's labels to choose that person's model; it does not undo the earlier use of this dataset to develop research ideas.

## What “full DSADS scores” means here

There are 9,120 released five-second segments: eight people, 19 activities, 60 segments per person/activity. For each outer fold, one person supplies 1,140 test segments and the other seven supply 7,980 training segments. Seven inner folds each fit six people and validate on the remaining training person. Representation, ridge penalty and clipping are selected inside these folds, then refitted on all seven outer-training people.

The 28 possible six-person training sets serve 56 directional inner folds. A pair excluding people a and b fits neither person: predictions for b can inform the selector testing a, and predictions for a can inform the selector testing b. This sharing saves computation without changing the nested split. All eight selections are recorded before any final outer fit or score.

The [earlier per-person diagnostic scores](data_diagnostics/prior_subject_scores.csv) mixed five- and six-person fits, with selection reused on the inner people. They are documented in [DATA_DIAGNOSTICS.md](DATA_DIAGNOSTICS.md) and are **not** the new balanced outer-fold results.

| Held-out person | Accuracy % | Macro-F1 % | Inner accuracy % | Errors / 1,140 | Selected family |
|---:|---:|---:|---:|---:|---|
| 1 | 91.40 | 89.89 | 93.72 | 98 | asinh + joint P3 log |
| 2 | 95.44 | 94.92 | 94.04 | 52 | Prior rich observations |
| 3 | 90.09 | 88.41 | 92.72 | 113 | Prior rich observations |
| 4 | 88.33 | 86.62 | 92.98 | 133 | Prior rich observations |
| 5 | 97.19 | 97.15 | 94.04 | 32 | Prior rich observations |
| 6 | 93.42 | 91.65 | 92.36 | 75 | Robust scaling + joint P3 log |
| 7 | 85.09 | 82.84 | 93.77 | 170 | Unit directions + joint P3 log |
| 8 | 89.65 | 89.19 | 94.10 | 118 | Prior rich observations |
| Mean | **91.33** | **90.08** |  | **791 / 9,120 total** | |
| Between-person sample SD | **3.93 pp** | **4.53 pp** | | | |

Pooled accuracy is also **91.33%**, since people have equal test sizes. Pooled macro-F1 is **91.23%**; it differs from the mean person-wise macro-F1 of **90.08%** because F1 is nonlinear. The SD describes variation across eight people, not a confidence interval based on 9,120 independent windows. All selected readouts use training-range clipping.

[Every family/person score](summary/dsads_people.csv), [the full accuracy matrix](summary/dsads_accuracy_matrix.csv), [family distributions](summary/dsads_model_distribution.csv), [selected class recalls](summary/dsads_selected_class_recall.csv) and [pooled confusion counts](summary/dsads_selected_confusion.csv) retain the complete results.

![All eight DSADS people](summary/dsads_people.png)

### Why validation selection still fails

The richer protocol improves coverage but does not eliminate changes in model ranking across people. Mean selected inner accuracy is **93.46%**, versus **91.33%** on outer people. Inner-versus-test rank correlations over all 68 families range from **0.56 to 0.80**: validation is useful overall, yet the top choice can still transfer badly. For person 7, unit-direction logs score **93.77%** in inner validation but **85.09%** on the held-out person; their test rank is 37 of 68, while the fixed prior-rich map gives **94.04%** for that person. Person 8's selected rich map falls from **94.10%** inner accuracy to **89.65%** test accuracy. The weakest person therefore depends on the selected representation; person 8 is not universally the difficult case.

There are seven validation people per outer fold, not thousands of independent validation domains. Several winning margins are tiny: person 1's winning family leads by only **4 of 7,980** inner predictions, and person 6's by **11**. Between-person shifts and near-tied family choices make a large menu fragile. These observations diagnose ranking instability; they do not establish a unique physical cause or prove that reducing the menu would improve a future cohort. The [validation/test diagnostics](summary/dsads_validation_test.csv) report every outer fold.

The fixed prior-rich family averages **92.52%**, versus **91.33%** for representation selection. The highest fixed-family test mean is **93.08%** for asinh plus joint P3 tensor features, followed by **92.91%** for robust-asinh joint logs. These families were specified beforehand and each tunes its parameters within the inner folds. Selecting the best family after inspecting its eight outer scores would nevertheless be test-based selection, so neither number replaces the headline.

### Which DSADS transforms help

The following averages use all eight outer people, with each row/view tuning its own readout inside training. The endpoint and P1 columns combine raw and spectral branches just like the joint P3 column.

| Transform | Raw P3 log % | Spectral P3 log % | Joint P3 log % | Joint P3 endpoints % | Joint P1 log % |
|---|---:|---:|---:|---:|---:|
| Standard scaling | 87.84 | 82.61 | 92.09 | 88.89 | 87.62 |
| asinh | 87.19 | 82.37 | 92.83 | 89.06 | 89.16 |
| tanh | 86.62 | 82.64 | 92.42 | 89.75 | 89.43 |
| Differences | 71.79 | 84.80 | 84.91 | 82.31 | 74.40 |
| Window centering | 71.04 | 86.05 | 87.18 | 82.15 | 79.53 |
| Whitening | 87.59 | 81.44 | 91.86 | 88.42 | 87.94 |
| Robust scaling | 87.95 | 84.46 | 92.45 | 88.83 | 88.40 |
| Robust + asinh | 87.08 | 84.35 | 92.91 | 89.64 | 89.52 |
| Vector magnitudes | 74.06 | 77.88 | 82.46 | 77.12 | 73.56 |
| Unit directions | 88.30 | 81.73 | 92.13 | 89.20 | 87.40 |
| Canonical frame | 74.50 | 88.44 | 88.09 | 87.50 | 80.20 |
| Within-device Gram | 80.18 | 88.51 | 89.75 | 87.94 | 86.62 |
| Sum of axis Fourier powers | — | 81.36 | — | — | — |
| Fourier Gram | — | 87.62 | — | — | — |

Combining time and frequency matters more here than either branch alone. Joint asinh P3 logs reach **92.83%**, against **89.06%** for matched endpoints and **89.16%** for matched P1 logs. Thus the fitted construction benefits from both the extra clocks and area coordinates on average, without isolating which branch supplies the benefit. Compression improves the joint tensor family from **91.93%** to **93.08%**.

Orientation removal is not a general cure: magnitudes, canonical frames and Gram streams underperform the direction-preserving combined maps. Centering and differencing severely hurt raw classification, consistent with losing static posture information. In contrast, canonical and Gram spectral maps improve on the basic normalized axis spectra, but their joint maps still fall short. These outcomes favor preserving complementary posture and motion information rather than imposing every plausible invariance.

![DSADS transform comparison](summary/dsads_transforms.png)


## The transform experiment

Write the transformed stream as \(y_t=\Phi(x_{1:N})_t\): some transforms act pointwise, while differences use history and centering or frame estimation uses the complete window. Raw P3 signatures use increments \(q(t)\otimes y_t/N\). Spectral P3 signatures first form the Parseval-weighted one-sided power, normalize each transformed channel across frequency, and use increments \(q(k)\otimes p_k\). Depth-two log coordinates contain endpoints and signed areas. Each transform receives:

- raw P3 logs;
- spectral P3 logs;
- their concatenation;
- the same combined P3 endpoints without areas;
- combined P1 logs, which retain within-group cross-channel areas but have no nonconstant Pascal clock.

Identity and `asinh` also receive tensor-signature versions, as does NinaPro's log-MAV map. These retain symmetric second-level products as well as area information. A common compressed-energy sidecar is included exactly once in every matched map, so shape normalization does not silently discard all amplitude information. The energy sidecar comes from the original input, not from whichever transform happens to be compared.

The eight shared transforms are standard scaling, `asinh`, `tanh`, backward differences, window centering, shrinkage whitening, robust scaling, and robust scaling followed by `asinh`. DSADS additionally tests vector magnitudes, unit directions, a window-fixed gravity/magnetic coordinate frame, and within-device Gram streams. NinaPro additionally tests uncentered RMS scaling, log-MAV, common-gain normalization, and two raw-EMG descriptor histories.

Signatures are joint within XYZ triads for DSADS, HAR and the skeleton datasets, and joint across all 16 electrodes for NinaPro. DSADS magnitude/Gram/Fourier-Gram groups follow the three co-located modalities within each device. This restores interactions missing from the preceding scalar-only local maps, while avoiding a quadratic expansion across every device. Cross-triad raw areas are still omitted. Consequently, these are explicitly grouped constructions, not a rerun of every original full-channel tensor model.

The [mathematical definitions and checks](TRANSFORMS.md) specify scaling, groups, zero cases, spectral conventions and invariance assumptions. [streams.py](streams.py) computes exact depth-two signatures and is checked against the repository's `LocalRealization` implementation.

### Compression and normalization

For robust compression, fit a median \(m\) and scale \(s=\mathrm{IQR}/1.349\) on training observations, with a training-SD fallback for degenerate IQR, then use

\[
\phi(x)=\operatorname{asinh}\bigl((x-m)/s\bigr).
\]

Near zero this is approximately linear; for large magnitude it grows logarithmically and keeps the sign. Unlike `tanh`, it does not saturate to a fixed bound. The paired robust-identity map separates compression from changing the fitted scale.

NinaPro common-gain normalization divides the RMS-scaled six-step, 16-electrode history by one RMS for that history. It preserves relative activation across electrodes and time while removing a common multiplicative factor. This is different from separately normalizing each electrode. The invariance belongs to the normalized stream: the original-energy sidecar deliberately keeps the full feature map sensitive to amplitude. A positive common gain cancels exactly from separately normalized spectral power, so the corresponding spectral map duplicates the uncentered RMS spectral control; that equality is checked rather than counted as independent evidence.

Whitening uses a fixed 10% identity shrinkage within each signature group. For centered raw streams it is an invertible linear change of basis, and complete within-group raw signature levels transform accordingly. It can change ridge regularization and clipping geometry, but does not add raw information. This statement does not automatically apply to recentering observation increments or to taking normalized power after mixing channels.

### Orientation and order of operations

DSADS invariants are computed from physical vectors before independent axis scaling. Within-device norms and dot products are invariant to a common rotation of that device. Unit directions instead remove magnitude and still rotate with the input; they are not rotation-invariant. The canonical frame uses mean acceleration as a gravity proxy and the horizontal component of mean magnetic field; degenerate frames emit zero coordinates and validity flags. It assumes a constant proper rotation within the window, and does not establish that mean acceleration equals gravity during motion. Its frame uses the complete five-second classification window.

Canonicalization can remove useful static orientation. For person 8, standing left-leg acceleration changes mean direction by about 25.7 degrees relative to the other seven people, with norm nearly unchanged; elevator-standing right-arm acceleration changes by about 62.7 degrees. These observations support testing orientation transforms, but do not prove sensor misplacement rather than genuine pose differences. Centering away every window's DC component has the same risk of deleting posture information.

We also distinguish **FFT of a magnitude** from **sum of vector-axis Fourier powers**. If \(v(t)=(\cos\omega t,\sin\omega t,0)\), then \(\|v(t)\|=1\): its magnitude spectrum is all DC. The axis-power spectrum still detects the rotation frequency. A second invariant spectral branch uses complex Fourier Gram entries among acceleration, gyro and magnetometer vectors within each device, retaining relative spectral phase through real and imaginary cross terms. These signed cross streams are normalized by the geometric mean of the two modality energies; they are not probability masses. It is invariant to constant common rotations, not arbitrary time-varying rotations. Same-time Gram streams alone can discard motion that these spectral invariants retain.

### Causal raw-EMG observations

The original six MAV vectors are already rectified, averaged observations, so another transform of that history cannot recover the EMG carrier. We therefore extract RMS and mean absolute adjacent-sample difference from the raw trailing 256-sample windows at exactly the same six endpoints. The complete support remains 756 samples, or 378 ms; no future sample or cross-trial state is introduced. Each descriptor is training-RMS-scaled and log-compressed before coding. They remain lossy activation/variation summaries, not reconstructed carrier waveforms. All histories, targets and boundaries match the original loader, and direct raw-slice calculations verify the cache. See [extraction](emg_observations.py) and [alignment checks](emg_observations_verification.json).

## NinaPro results

Validation selects the following maps; test uses acquisition 3 of subject 1, with all 1,940 eligible histories and the original 18 targets.

| A1 repetitions per movement | Previous follow-up RMSE | New selected RMSE | MAE | Selected transform / raw P3 log |
|---:|---:|---:|---:|---|
| 1 | 22.778 | 22.959 | 15.218 | Common-gain normalization |
| 2 | 25.101 | 22.660 | 14.735 | Common-gain normalization |
| 5 | 22.182 | 20.260 | 12.958 | Common-gain normalization |
| 10 | 21.804 | 19.601 | 12.507 | Robust scaling + asinh |

All four selected readouts use training-range clipping. At ten repetitions R² is 0.579, compared with 0.479 in the previous follow-up. The full-budget RMSE improvement is about 10.1% over that follow-up and 18.2% over the original moments baseline of 23.971. At one repetition the larger menu selects slightly worse than the previous clipped-moments result; that failure remains in the headline.

The matched full-budget comparison is informative:

| Map | Test RMSE |
|---|---:|
| Robust identity, raw P3 log | 26.400 |
| Robust asinh, raw P3 log, validation-selected | 19.601 |
| Robust asinh, spectral P3 log | 23.257 |
| Robust asinh, combined P3 log | 19.593 |
| Robust asinh, combined P3 endpoints only | 22.614 |
| Robust asinh, combined P1 log | 22.816 |

Thus compression matters even after clipping, and the matched combined-map controls support a benefit from both the nonconstant clocks and the area coordinates on this split. Every row tunes its own ridge penalty and clipping; these are comparisons of fitted procedures, not frozen-coefficient causal interventions. The 19.593 combined-map result is not substituted for the representation-selected raw-map result.

The additional raw-EMG descriptors do not consistently improve this result: full-budget combined log-RMS and log-waveform-length maps give RMSE 22.748 and 21.817. Raw log-MAV gives a different transform of the MAV data; the corresponding combined map reaches 20.309. More input detail is not automatically a better transfer under this calibration budget and linear head.

![NinaPro transform comparison](summary/ninapro_transforms.png)

## Checks on the other datasets

The shared transforms also ran on NATOPS, UTD-MHAD, all 25 KIMORE exercise/fold units, and official-split UCI HAR. Their original outer splits and data preprocessing are unchanged.

| Dataset / selected metric | Previous follow-up | This transform study |
|---|---:|---:|
| NATOPS accuracy | 90.00% | 92.22% |
| UTD-MHAD accuracy | 83.49% | 87.91% |
| KIMORE mean exercise MAE | 6.331 | 6.783 |
| KIMORE mean exercise RMSE | 9.783 | 9.489 |
| UCI HAR official-split accuracy | 94.44% | 96.81% |

NATOPS selects whitened endpoints, UTD selects differences with joint raw/spectral logs, and HAR selects the preceding rich feature map under the new common head. Thus the HAR change is not evidence for a newly winning transform. KIMORE worsens on MAE despite a modest RMSE improvement, showing continued instability in choosing among many families from small validation cohorts. Its metrics pool outer-fold predictions within each exercise, then average the five exercise metrics.

The fixed-family controls make the limitations clearer. On KIMORE, the prior rich map under the new head has mean exercise RMSE **7.799**, better than the representation-selected **9.489**. On HAR it reaches **96.81%**, whereas the best new signature family by test accuracy reaches **92.98%**. NATOPS has several fixed endpoint maps at **95.00%**, above the selected **92.22%**. These descriptive test comparisons diagnose selection and representation limits; choosing their winners after reading the test table would not be a valid replacement for validation selection.

## Literature and SOTA interpretation

The [focused literature review](literature.md) checks DSADS and NinaPro through 13 October 2026; the [preceding six-dataset review](../followup/literature.md) covers the broader context.

For DSADS, MTSDNet reports 92.16% accuracy averaged over eight subject holdouts, but uses 32-sample windows. TRI-HAR reports macro-F1 89.07 ± 4.49 across eight LOSO folds and tests independent device rotations, but uses acceleration plus gyro, 30 channels, and a different overlapping-window protocol. Our 45-channel, released-segment, fully nested procedure is not a matched reproduction of either. A numerically higher score would still not establish SOTA without matching these conditions and checking the broader field.

For NinaPro DB8, the recent optimized TCN reports MAE 5.45 degrees and multitask LoRA 5.33 degrees on five mapped actuator angles, using A1+A2 and roughly six seconds of context across participants. Our single-participant, A1-only, 18-output errors are in the supplied glove-channel units and cannot be ranked against those numbers. Further local improvement is not a SOTA claim.

## Reproduce and inspect

The [study runner](study.py) reproduces the complete study sequentially from the existing cached data:

```bash
../tensorsig/.venv/bin/python examples/pascal_transfer/transforms/study.py --output /tmp/pascal-transform-replay
../tensorsig/.venv/bin/python examples/pascal_transfer/transforms/streams.py
../tensorsig/.venv/bin/python examples/pascal_transfer/transforms/verify.py
../tensorsig/.venv/bin/python examples/pascal_transfer/transforms/summarize.py
```

The recorded run parallelizes disjoint DSADS inner pairs and final folds using [pair_worker.py](pair_worker.py) and [outer_worker.py](outer_worker.py). Both call the frozen study's feature, ridge, loss and selection functions; all choices are frozen before outer fitting. There is no model change between sequential and parallel execution.

The menu has 68 DSADS families, 73 NinaPro families and 46 families for each other dataset, including controls. The same 12 mean-SSE ridge penalties and clipped/unclipped query alternatives are selected for every family. Earlier moments, spectral pool, scalar spectral Pascal and rich-map controls are retained. The rich map now uses one common column-scaled feature group, rather than the previous base/extra gamma search; its old and new results must not be described as an exact identical-head replay.

All grids, per-family selections and predictions are retained in [results/](results/). The [summary directory](summary/) contains all family scores, matched contrasts, selected procedures, subject distributions and class-level DSADS results. [Transform checks](checks.json), [readout checks](readout_checks.json) and the [independent audit](results/verification.json) cover direct signature identities, train-only calibration, geometric invariance, numerical ridge agreement, saved scores, selection and split disjointness.

The completed audit passes **40 evaluation units, 2,124 final model rows and 141,456 recorded validation-grid rows**. Every final metric is independently recomputed from saved predictions, and every choice is independently recomputed from the full saved grid. Four representative selected fits are also solved independently with sklearn LSQR: DSADS and NATOPS labels agree exactly; maximum regression prediction differences are below **9.9 × 10⁻⁷** for NinaPro and **2.2 × 10⁻⁹** for KIMORE. This does not independently refit every grid candidate. The [runtime and hashes](environment.json) and [source snapshot](source_snapshot.zip) preserve the executed implementation and supporting code.

Source provenance retains the initial menu pin and an explicit finalization record: the feature author's final formatting and synthetic-check additions overlapped the master hash capture. All 40 per-unit protocols record the same final feature-source hash. The original pin is preserved in [protocol_initial.json](protocol_initial.json); [protocol.json](protocol.json) records the resolved source and scope of the correction. Inherited `data.protocol` strings describe the source loader's old DSADS split; the active split is the top-level nested `train_indices`, `test_indices` and `folds` metadata.
