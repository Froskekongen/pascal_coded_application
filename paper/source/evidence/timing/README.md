# Frozen HAR pipeline timing, 16 October 2026

This is the single new quantitative experiment authorized by the revision-16 brief. It measures the existing compact368 and hybrid803 (code name `pool803`) on the saved SpikeDCL random-window partition. It does not tune parameters or alter historical accuracy evidence.

Run from the repository root:

```sh
../tensorsig/.venv/bin/python paper_revision_16/source/evidence/timing/benchmark_pipeline.py
```

The default writes only this directory. Use `--output /path/to/new/results` for a separate reproduction. The existing UCI archive is required; no downloads occur. Five repeats follow one complete model warm-up. The original loader also has one unmeasured warm-up before five timed reads. All numerical libraries and PyTorch use one thread, CPU only. No packages were installed. `run.json` records exact hardware, versions, dtypes, phases, source hashes, split hashes and prediction checks. `dependencies.json` records the tensorsig working source hashes because the local package is editable.

In both records, paths beginning with `../tensorsig/` refer to the sibling dependency checkout relative to the historical application checkout.

## Reproduction requirements and package boundaries

The benchmark is runnable **inside the original repository tree**, with this file at `paper_revision_16/source/evidence/timing/`. It resolves the repository root from its own file location (`HERE.parents[3]`), then imports `examples/uci_har/{benchmark,retained_features,spikedcl_benchmark}.py`. Changing the current working directory does not change that resolution. Moving this folder into a standalone extracted paper package does: the extracted package alone does not contain the full runtime, repository layout or dataset, and running the timing script there without restoring those inputs is unsupported. No timing import was missing in the measured original environment.

Required external inputs are the Python environment recorded in `run.json`, the editable `tensorsig` implementation identified by `dependencies.json`, and the original `examples/uci_har/data/uci_har.zip` (SHA-256 `c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031`). Direct third-party imports are NumPy, PyTorch, scikit-learn, threadpoolctl and tensorsig, with their normal installed dependencies. The script reads saved predictions from `examples/uci_har/spikedcl_results/predictions.npz`. The measured hardware collector calls macOS `system_profiler`; its executable must exist, or the metadata collector must be adapted on another OS. Such an adaptation is separate from the archived measured script. Runtime comparisons across hardware or library versions require new measurements.

For a reproduction from this paper package, first restore its `benchmark_pipeline.py` to the path above in a checkout of the original application repository, restore the three `frozen_accuracy/sources/` files to `examples/uci_har/`, and restore the six `frozen_accuracy/` accuracy files to `examples/uci_har/spikedcl_results/`. Supply the data and tensorsig environment separately; then use `--output /path/to/new/results` to preserve archived measurements. In an existing working checkout, use an isolated copy for restoration rather than overwrite local changes. The package includes evidence for auditing the paper without installing the runtime or rerunning any experiment.

## Packaged evidence map

| File | Purpose |
|---|---|
| `frozen_accuracy/summary.csv` and `results.csv` | Historical aggregate and per-fit random-window accuracy, validation choice and deterministic repetitions |
| `frozen_accuracy/run.json` | Unmodified historical protocol, configuration, confusion matrices and original recorded hashes |
| `frozen_accuracy/predictions.npz` | Frozen ordered partitions, labels and predictions checked by the timing adapter |
| `frozen_accuracy/split_verification.json` | Existing independent agreement with the pinned upstream split |
| `frozen_accuracy/numerical_verification.json` | Existing independent numerical checks |
| `frozen_accuracy/sources/` | The three current retained source files actually imported by the timing run |
| `frozen_accuracy/provenance.json` | Original relative paths, packaged paths, byte counts and SHA-256 values for every copied file |

All six accuracy files are byte-for-byte copies of the frozen evidence; original files were not altered. Source snapshots are the versions used by the new timing verification, including later optional variants. Their hashes match this directory's `run.json`; they are not presented as identical to the older source versions named by `frozen_accuracy/run.json`. Exact feature and prediction agreement is recorded separately, and the historical provenance is preserved rather than rewritten.

`timings.csv` retains every repeat. `summary.csv` reports medians and minimum/maximum elapsed time; totals are sums within repeats. `source_review.json` records the pinned upstream neural source audit and fallback rationale. Every feature agrees exactly with the existing joint extractor and all saved test predictions match, including 38 compact errors and 32 hybrid errors on 2,060 windows.

Training includes calibration, preprocessing, single-model extraction, final scaling and one balanced ridge fit on 6,591 windows. Applying the frozen model includes observation preprocessing, extraction, frozen scaling and prediction for 2,060 test windows. The 1,648 validation windows are excluded from these workloads; the historical choice between the two frozen candidates is not repeated. Predictions from the complete old runner are checked independently of the timed stages. Data loading is measured separately, with the original reader including unused supplied engineered features. The OS cache is warm; host scheduling is not isolated.

The adapter deliberately reuses the existing `spectrum` helper, including its pooled-summary calculation even when compact discards that output. It does not compute the other complete model. No kernel optimization is claimed. The richer hybrid has more columns but fewer graph/signature computations and is faster in this implementation.

Median train/apply times: compact 1.491/0.457 seconds; hybrid 0.813/0.222 seconds. Whole-archive loading is 1.165 seconds separately. These are batch workloads, not single-window device latency. No quantitative speedup over SpikeDCL or energy claim is supported: the fallback reports these measured costs and the publication's documented 60-epoch training and learning-rate selection requirements, without treating the unmatched workloads as a runtime comparison.
