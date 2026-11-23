# Reproduce the paper examples

These scripts reproduce **the three retained official-split UCI HAR maps
(compact368, geometry57 and supplied561) and all nine NATOPS maps** from
[*Building Activity Representations with Pascal-Coded Signatures*](../paper/source/paper/main.tex).
Each script follows a single sequence: load the published split, construct the
fixed features, fit the specified ridge classifier, and compare fresh outputs
with frozen evidence.

The canonical paper also reports conventional HAR controls from an archived
diagnostic search. These scripts do not reproduce that whole search. The
companion cites a published SpikeDCL result from a different split; these
scripts do not run that experiment.
Historical feature searches, alternative representations, transfer experiments,
timing studies, and retired paper revisions are outside this reproduction.

## Install and run

Run these commands from the **repository root**:

```sh
uv sync --locked
uv run python -m reproduction.checks
uv run python -m reproduction.uci_har --download
uv run python -m reproduction.natops --download
```

The root `pyproject.toml` and `uv.lock` define the environment; `.python-version`
selects Python 3.14. `uv sync` installs **tensorsig from the package index** into
the root `.venv`, alongside NumPy, PyTorch and scikit-learn. No sibling checkout,
editable install, `PYTHONPATH`, or separate environment is required. The
reproduction scripts call tensorsig for signature computation.

`--download` caches the official data under `reproduction/data/` and checks
SHA-256 hashes recorded with the paper. Existing files are checked and reused.
Omit `--download` to work entirely offline. UCI HAR is about 58 MB compressed;
the two NATOPS files are about 2 MB each. ZIP files are read without extraction.

To use datasets already downloaded elsewhere, pass their locations explicitly:

```sh
uv run python -m reproduction.uci_har --data /path/to/uci_har.zip
uv run python -m reproduction.natops --data /path/to/natops
```

Results go to `reproduction/results/uci_har/` and `reproduction/results/natops/`.
Each run requires an empty output directory. For a subsequent run, give a new
directory with `--output`, for example `--output reproduction/results/natops-rerun`.
Datasets and generated results are ignored by version control. Run either
module with `--help` for its command-line options.

## Read the implementation

| File | Purpose |
| --- | --- |
| [data.py](data.py) | Verified downloads and readers for the two fixed official splits. |
| [uci_har.py](uci_har.py) | The three compact368 branches, the 57-feature subset, supplied-feature baseline, and frozen ridge fit. |
| [natops.py](natops.py) | Clock and origin anchor, Pascal lift, nine coordinate subsets, and training-only ridge cross-validation. |
| [checks.py](checks.py) | Small offline checks of the feature identities and data handling. |
| [reference/](reference/) | Small frozen expected results, with source paths and hashes; used only to check newly computed outputs. |

There are no imports from `examples/` or `paper/`. The reference
files contain evidence, not trained models or feature matrices, and are never
used to choose parameters or generate predictions.

### Paper-to-code map

Equation labels below are the exact LaTeX labels, so they remain searchable if
the manuscript is renumbered. The main equations are in
[main.tex](../paper/source/paper/main.tex).

| Paper definition | Where it appears in the reproduction |
| --- | --- |
| Eq. (1), `eq:signature`: endpoints and second-level iterated integrals | tensorsig signature calls in both feature scripts. |
| Eq. (2), `eq:area`: independent signed areas | HAR extracts `(B_ij-B_ji)/2`; NATOPS requests tensorsig's depth-two logsignature coordinates. |
| Eq. (3), `eq:lift`: position weights multiply **increments** | HAR uses QR Pascal weights on scaled sensor readings and spectral shares; NATOPS `prepare_paths` explicitly constructs `(z_r, (r/50) z_r)`. |
| Eq. (4), `eq:sep`: separation between movements | The offline feature checks verify the two-rise example and the signed-area identity. |
| Eq. (5), `eq:spectrum`: frequency-ordered power shares | HAR's spectral branch normalizes one-sided Fourier power, then treats the shares as increments. |
| Companion `app:qr`, `eq:qr-first-two` | HAR uses `OffsetCode.pascal(..., orthonormal=True)`, with positive QR diagonal convention. |
| Companion `app:har` | The complete HAR feature and readout recipe in `uci_har.py`. |
| Companion `app:natops` | The nine NATOPS coordinate subsets and cross-validation protocol in `natops.py`. |

The companion recipes are
[appendix_har.tex](../paper/source/companion/appendix_har.tex),
[appendix_natops.tex](../paper/source/companion/appendix_natops.tex),
[appendix_construction.tex](../paper/source/companion/appendix_construction.tex), and
[appendix_offsets.tex](../paper/source/companion/appendix_offsets.tex).
Individual function docstrings explain the operations and their paper references.

## UCI HAR: levels, rhythm and geometry

Use the official 7,352/2,947-window subject-disjoint split. Each window contains
128 readings of three body-acceleration axes, three gyroscope axes, and three
total-acceleration axes. Gravity is total minus body acceleration.

Compact368 combines three blocks:

1. **171 sensor features.** Standardize total acceleration and gyroscope using
   training statistics across all readings. Use readings divided by 128 as
   increments, preserving posture levels. Three QR Pascal copies produce 18
   endpoints and 153 areas.
2. **140 spectral features.** Use body acceleration, gravity, gyroscope, and
   the first differences of body acceleration and gyroscope. Append each
   standardized triad's norm before the FFT. The 20 spectra each contribute
   five nonconstant Pascal logsignature coordinates and two amplitude summaries.
3. **57 geometric features.** Training RMS calibration precedes squared norms,
   simultaneous and lagged dot products, and cosine similarities. Their
   means, standard deviations, skewnesses and linear trends describe strength,
   persistence and alignment. The code evaluates the companion's scalar formulas
   directly for these summaries.

The variance mask is fitted on training rows and removes 20 constant spectral
endpoints, leaving 368 columns. Feature standardization also uses training rows
only. The ridge penalty is fixed at **alpha = 10** for all three rows; there is
no model search. Geometry57 is the same geometric block taken alone. The third
row uses the 561 features supplied with UCI HAR.

To preserve the saved experiment's numerical convention, signature arithmetic
uses float64 and its tensor output is stored as float32 **before** extracting
HAR areas. Raw areas use division by 2; spectral areas use division by sqrt(2).
Geometric calculations and the ridge solve use float64. These details affect
the computed coordinates even though final feature standardization removes
fixed scale factors.

| Representation | Features after masking | Errors / 2,947 | Accuracy |
| --- | ---: | ---: | ---: |
| compact368 | 368 | 58 | 98.03% |
| geometry57 | 57 | 438 | 85.14% |
| supplied561 | 561 | 126 | 95.72% |

## NATOPS: landmark displacements and interaction subsets

Use the published 180/180 split without resampling or per-gesture scaling.
Each sequence has 51 observations of eight xyz landmarks. Prepending an origin
after adding the clock creates **51 increments**: one initial-pose increment
with no clock advance, then 50 displacement increments. Coding uses `(1,r/50)`
on those increments, with ordinary channels first and weighted copies second.

The full map includes all pairwise areas. The local map includes clock–spatial
and within-landmark areas; under coding it also includes all same-channel
cross-copy areas. Both maps use the entire gesture. The moment and coded-clock
controls select subsets of these coordinates, as specified in the companion.

Every representation uses the same five stratified training folds, seed
20261003, and penalties `10**k` for `k=-6,...,6`. Each fold fits its own feature
scaler. Exact validation ties choose the smaller penalty, followed by refitting
on all training cases. Constant clock coordinates remain in the nominal counts.

| Representation | Features | Selected alpha | Correct / 180 |
| --- | ---: | ---: | ---: |
| ordinary_full | 325 | 10 | 163 |
| moments_ordinary_full* | 350 | 10 | 166 |
| coded_clock_full* | 423 | 100 | 163 |
| pascal_full | 1,275 | 10 | 173 |
| ordinary_local | 73 | 10 | 167 |
| moments_ordinary_local | 98 | 1 | 165 |
| coded_clock_local | 171 | 10 | 168 |
| pascal_local | 267 | 10 | 165 |
| moments_only | 50 | 0.1 | 142 |

The two starred controls were designed after the first seven outcomes were
known. The archived evidence calls the Pascal maps `offset_*`; the reference
file records the name mapping. This reproduction preserves the paper's fixed
recipe and its retrospective scope.

## Check the outputs

Both runs write a compact `results.csv`, per-case `predictions.npz`, and a
`run.json` containing package versions and comparisons with frozen evidence.
NATOPS also writes every candidate's fold counts to `cv_scores.json`.
If a comparison fails, the run raises an error after saving its diagnostics.

The offline checks require no datasets or network. Full data runs check feature
counts and paper results; NATOPS additionally checks all nine prediction
vectors and all 585 validation fold counts. The HAR reference checks include
the frozen confusion matrices and the available geometry57 prediction vector.
Accuracy agreement alone is not treated as complete verification.

Validated on 2 November 2026 using `uv sync --locked`, the offline checks, and
both full `--download` commands. All checks passed with Python 3.14.6,
tensorsig 0.1.0, PyTorch 2.15.0, NumPy 2.4.4, SciPy 1.17.1, and
scikit-learn 1.8.0. This includes exact agreement with all three HAR confusion
matrices, all 2,947 available geometry57 predictions, all nine NATOPS prediction
vectors, and all NATOPS validation counts and selected penalties.
