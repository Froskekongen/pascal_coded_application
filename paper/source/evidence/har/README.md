# HAR and optional transfer evidence

This directory packages the small sources and saved records used by revision 16, without datasets or generated feature matrices. Paths in three transfer records are relative to the historical application checkout, with `../tensorsig/` referring to the sibling dependency checkout. `source_manifest.json` records original repository paths and the sizes and SHA-256 hashes of the packaged files. For the edited records it also retains the original sizes and hashes. The paper calls saved model `pool803` **hybrid803**.

## Claim-to-evidence map

| Paper claim | Evidence here | Exact identifier or implementation |
|---|---|---|
| Compact368: 368 coordinates, 58/2,947 errors, 98.03% | `retained_results/results.csv`, `retained_results/run.json` | `compact368` |
| Strength, persistence and alignment alone: 57 coordinates, 438/2,947 errors, 85.14% | `geometry57_results/results.csv`, `geometry57_results/run.json`, `geometry57_results/predictions.npz` | `geometry57`; Compact368's 36 invariant moments, 6 cosine moments and 15 trends |
| Hybrid803: 803 coordinates, 56/2,947 errors, 98.10% | Same retained records | `pool803` |
| Spectral Pascal only: 100 coordinates, 504/2,947 errors, 82.90% | `pool_results/results.csv`, `pool_results/run.json` | `spectral_only/log/ridge`; only Compact's spectral logsignature branch |
| Raw Pascal only: 171 coordinates, 531/2,947 errors, 81.98% | Same pool records | `raw_only/log/ridge`; only Compact's raw logsignature branch |
| Original UCI HAR features with ridge: 561 coordinates, 126/2,947 errors, 95.72% | `consolidated_results/results.csv`, `consolidated_results/run.json` | `engineered_561/uci_features/ridge`; fixed alpha 10 |
| Conventional raw replacement: 377 coordinates, 75 errors, 97.46% | `consolidated_results/results.csv`, `consolidated_results/run.json` | `ordinary_raw_compact368`, `total_leadlag_d2_tensor` |
| Conventional spectral replacement: 708 coordinates, 59 errors, 98.00% | Same consolidated records | `ordinary_spectral_compact368`, `spectral_clock_tensor_d4` |
| Compact block deletions | Same consolidated records | `drop_raw_compact368`, `drop_spectral_compact368`, `drop_amplitude40_compact368`, `drop_cosine6_compact368`, `drop_invariant36_compact368`, `drop_trends_compact368` |
| Ordinary spectral depth comparison and exact projection | Same consolidated records; `historical_selected_sources/ordinary_spectral.py` | Nine `ordinary_spectral_compact368` rows, including `spectral_clock_d5_projected_pascal_P3` |
| Explicit ordinary raw-lift agreement | Same consolidated metadata | `raw_lift_max_abs_error` and `paired_vs_retained` for `ordinary_lift_*` |
| Observation order, finite/zero cases, coordinate scales and counts | `retained_features.py` | `build_features`, `spectrum`, `rotation_streams`, `calibrated_graph_features` |
| Split validation, constant mask, final scaling and frozen ridge | `benchmark.py`; both saved `run.json` files | `load_data`, `remove_constants`, `ridge_predict`; `ridge_alpha=10` |
| Exact raw control and scope of conventional comparison | `historical_selected_sources/{features,ordinary_controls,consolidated}.py`; consolidated metadata | 13-channel, 256-point lead–lag path with clock and zero basepoint; 52 raw replacements in each of three contexts |
| Earlier development and selection history | `uci_historical.md`, `uci_har.md`, `historical_sources.zip` | Historical accounts and retired sources, including dependency modules needed by the archived runner |
| Latest transfer selected results | `transfer/transforms/summary/selected_procedures.csv` | Rows `model=validation_selected`, with protocol and selected model named |
| Latest transfer matched/fixed-family contrasts | `transfer/transforms/summary/dataset_overview.csv` | DSADS, NinaPro, KIMORE and other dataset-specific rows; scopes must remain separate |
| DSADS variability and selection failure | `transfer/transforms/summary/dsads_model_distribution.csv`, `dsads_validation_test.csv`, `selected_procedures.csv` | Eight outer people; 91.33% pooled selected accuracy; 3.93 pp sample SD |
| Earlier follow-up comparison values | `transfer/followup/results/{summary.csv,old_selected_comparisons.json}`, associated `README.md` | Prior selected NinaPro/KIMORE values; separate earlier protocol |
| Original NinaPro moments baseline | `transfer/results/summary.csv` | `unit=ninapro_s1_reps10`, `model=moments`, RMSE 23.9713703237 |
| Transfer protocol and existing independent audit | `transfer/transforms/{protocol,protocol_initial,environment}.json`, `source_snapshot.zip`, `results/verification.json` | Forty evaluation units, 2,124 metrics and 141,456 validation-grid rows audited in the original study |

`saved_metrics_verification.json` records revision 16's independent recalculation of accuracy, error counts and macro-F1 from the saved confusion matrices for all 190 retained and consolidated rows. It also checks the declared split sizes, disjoint train/test subject lists and fixed ridge penalty. No classifiers were refitted in this evidence audit. Confusion matrices do not recover individual predictions; paired correction/introduction counts are retained from the original metadata, not independently reconstructed here. The broader transfer audit retains the original study's results.

`table1_metrics_verification.json` independently checks the two standalone logsignature rows and the supplied-feature baseline from saved CSVs and confusion matrices. The added `pool_results/` files are byte-for-byte copies of the original official-split results, and their source hashes are in `source_manifest.json`. The archived implementation is `pool_story.py` inside `historical_sources.zip`. These standalone rows exclude amplitude, geometric and trend summaries. The main table's published SpikeDCL row uses a different random-window protocol; its 98.86 +/- 0.28% result is from the cited article's Table 2, with no comparable fixed feature count or single error count assigned here. No new fitting was required for the table update.

## Requested 57-feature ablation (17 October 2026)

The main table now retains Compact368, the standalone 57-feature geometric
block, supplied561 and the separately grouped published SpikeDCL row.
Hybrid803 and the raw/spectral standalone rows remain documented here and
in the companion rather than in the main table. The earlier table audit
above records the preceding table version.

`geometry57.py` selects the actual geometric columns of Compact368 before
constant removal, then uses the same training-only mask, scaling and
unweighted ridge alpha 10 on the official subject-disjoint split. No tuning
or observation changes were introduced. All 57 columns remain. Checks
confirm identical columns and train/test scaling to Compact368, exact
reproduction of Compact368's saved confusion matrix, and every standalone
prediction against an independent scikit-learn LSQR fit. The new result is
2,509/2,947 correct (85.14%, macro-F1 84.88%). Its runner, metadata and
predictions are packaged separately; all earlier evidence is unchanged.

From this evidence directory, with a matching runtime and local UCI ZIP:

```sh
python geometry57.py --data /path/to/uci_har.zip --output /tmp/geometry57-replay
```

The runner requires a fresh output directory. This is a requested ablation
on previously inspected data, rather than a fresh confirmatory evaluation.

## Reproduce the retained HAR construction

Install NumPy, PyTorch, scikit-learn, threadpoolctl and the matching `tensorsig` implementation. The saved metadata records the original versions (Python 3.13.15, NumPy 2.5.2, PyTorch 2.13.0, scikit-learn 1.9.1); the revision 16 timing evidence records its own environment and source identity. From the source-package root:

```bash
python evidence/har/benchmark.py --self-check
python evidence/har/benchmark.py --data /path/to/uci_har.zip --output /tmp/har-replay
```

The archive is available from the dataset URL recorded in `retained_results/run.json`. `--download` is an explicit alternative. A replay should use a new output directory so the preserved evidence remains untouched. The default runner computes compact368, pool803 and the historical supplied-feature hybrid uci903; their extraction is shared. Its historical extraction timings therefore are not independent per-model deployment timings. Revision 16's separate computational benchmark supplies the new timing boundaries.

The archived conventional runner and all its historical local imports are in `historical_sources.zip`. The four most relevant sources are additionally expanded under `historical_selected_sources/` for direct inspection; that small expanded subset is not a standalone historical executable. The archived documents preserve their original relative links and terminology.

## Scope of the optional transfer package

The transfer files support the removable companion summary. They retain summaries, protocol/source snapshots and the original comprehensive verification report, not the large dataset caches, individual prediction files or complete validation grids. A complete transfer rerun requires the original `examples/pascal_transfer/` data manifests and caches described by the copied study accounts. This is intentional: revision 16 reuses those accuracy findings and does not rerun or tune them. The source package itself builds the paper without those datasets.

The initial, observable-follow-up and transform studies use different representations and selection procedures. In particular, the latest eight-person nested DSADS evaluation must not be presented as the same protocol as the earlier person-8 holdout, and the later NATOPS observation-based maps must not replace the main article's two-copy trajectory experiment.
