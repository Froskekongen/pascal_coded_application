"""Time frozen HAR pipelines; verify against saved random-window predictions.

Run from the repository root with ../tensorsig/.venv/bin/python PATH_TO_THIS_FILE.
No tuning, source edits, downloads, or writes to historical accuracy artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter

for variable in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
HAR = ROOT / "examples/uci_har"
sys.path.insert(0, str(HAR))
import numpy as np
import sklearn
import torch
import retained_features as rf
from benchmark import load_data, ridge_predict
from spikedcl_benchmark import split_indices
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info, threadpool_limits


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observations(signals, calibration=None):
    """Fit on training windows once; apply identical stored constants thereafter."""
    x = torch.as_tensor(signals, dtype=torch.float64)
    assert x.shape[1:] == (128, 9) and torch.isfinite(x).all()
    vectors = [x[:, :, :3], x[:, :, 6:9] - x[:, :, :3], x[:, :, 3:6]]
    axes = [x[:, :, [6, 7, 8, 3, 4, 5]], *vectors,
            vectors[0].diff(dim=1), vectors[2].diff(dim=1)]
    if calibration is None:
        calibration = {
            "axes": [(v.mean((0, 1)), v.std((0, 1), correction=0).clamp_min(1e-8)) for v in axes],
            "rms": [v.square().mean().sqrt().clamp_min(1e-8) for v in vectors],
        }
    normalized = [(v - mean) / sd for v, (mean, sd) in zip(axes, calibration["axes"])]
    scaled = [v / scale for v, scale in zip(vectors, calibration["rms"])]
    energy = torch.stack([v.square().sum(2) for v in scaled], 2)
    dots = torch.stack([(scaled[i] * scaled[j]).sum(2) for i, j in rf.TRIAD_PAIRS], 2)
    streams = [energy, dots]
    for lag in (2, 5):
        streams.append(torch.stack([(v[:, lag:] * v[:, :-lag]).sum(2) for v in scaled], 2))
    norms = energy.sqrt()
    denominator = torch.stack([norms[:, :, i] * norms[:, :, j] for i, j in rf.TRIAD_PAIRS], 2)
    return (normalized[0], normalized[1:], streams, dots / denominator.clamp_min(1e-12)), calibration


def extract(prepared, model):
    """Reuse retained numerical helpers, evaluating only the requested model."""
    raw, spectral_vectors, streams, cosines = prepared
    raw_tensor = rf.pascal_signature(raw, 256)
    spectra = [rf.spectrum(v) for v in spectral_vectors]
    if model == "pool803":
        spectral = torch.cat([s[2] for s in spectra], 1).float().numpy()
        rotation = torch.cat([*(rf.moment_summary(s) for s in streams),
                              cosines.mean(1), cosines.std(1, correction=0)], 1).float().numpy()
        return np.concatenate((spectral, rotation, raw_tensor), 1)
    spectral = np.concatenate([rf.spectral_logsignature(s[0], 256) for s in spectra], 1)
    amplitude = [rf.calibrated_graph_features(s[1], skewness=False, trend=False, batch_size=256)[0]
                 for s in spectra]
    cosine_stats, cosine_trend = rf.calibrated_graph_features(cosines, skewness=False, trend=True, batch_size=256)
    invariant = [rf.calibrated_graph_features(s, skewness=True, trend=True, batch_size=256) for s in streams]
    return np.concatenate((spectral, *amplitude, cosine_stats, *(s[0] for s in invariant),
                           cosine_trend, *(s[1] for s in invariant), rf.raw_logsignature(raw_tensor)), 1)


def scale_train(values):
    values = np.asarray(values, dtype=np.float64)
    keep = values.std(0) > 1e-6 * np.maximum(values.max(0), -values.min(0))
    scaler = StandardScaler()
    return scaler.fit_transform(values[:, keep]), keep, scaler


def fit_ridge(x, labels):
    """The saved balanced +/-1 ridge objective, split from final feature scaling."""
    weights = len(labels) / (6 * np.bincount(labels, minlength=6)[labels])
    offset = np.average(x, axis=0, weights=weights)
    x = x - offset
    classes = np.unique(labels)
    target = 2.0 * (labels[:, None] == classes) - 1.0
    intercept = np.average(target, axis=0, weights=weights)
    target -= intercept
    root_weight = np.sqrt(weights)[:, None]
    x *= root_weight
    target *= root_weight
    gram = x.T @ x
    gram.flat[::len(gram) + 1] += 10.0
    coefficient = np.linalg.solve(gram, x.T @ target)
    return coefficient, intercept - offset @ coefficient, classes


def predict(scaled, fitted):
    coefficient, intercept, classes = fitted
    return classes[(scaled @ coefficient + intercept).argmax(1)]


def hardware():
    # Whitelist public hardware fields; do not save serial numbers or machine IDs.
    result = subprocess.run(["system_profiler", "SPHardwareDataType", "-json"], capture_output=True, text=True)
    if result.returncode:
        return {"machine": platform.machine(), "logical_cpus": os.cpu_count()}
    entry = json.loads(result.stdout)["SPHardwareDataType"][0]
    return {k: v for k, v in entry.items() if k in ("machine_name", "machine_model", "chip_type", "number_processors", "physical_memory")}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    assert args.repeats >= 3
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rows, verification = [], {}
    archive = HAR / "data/uci_har.zip"
    # One complete untimed loader call warms the OS file cache and parsers.
    load_data(archive)
    for repeat in range(args.repeats):
        tick = perf_counter()
        original = load_data(archive)
        rows.append(dict(model="shared", repeat=repeat, phase="archive_loading", seconds=perf_counter() - tick))
    all_data = {k: np.concatenate([original[s][k] for s in ("train", "test")]) for k in original["train"]}
    partitions = split_indices(all_data["subjects"])
    saved = np.load(HAR / "spikedcl_results/predictions.npz")
    for name, indices in partitions.items():
        np.testing.assert_array_equal(indices, saved[name + "_indices"])
    order = np.concatenate(list(partitions.values()))
    n_train, n_val, n_test = map(len, partitions.values())
    train = all_data["signals"][partitions["train"]]
    test = all_data["signals"][partitions["test"]]
    labels = all_data["labels"][partitions["train"]]
    reference, _ = rf.build_features(all_data["signals"][order], n_train, all_data["engineered"][order], 256)

    for model, width in (("compact368", 368), ("pool803", 803)):
        for repeat in range(-1, args.repeats):
            def timed(phase, function):
                tick = perf_counter()
                result = function()
                elapsed = perf_counter() - tick
                if repeat >= 0:
                    rows.append(dict(model=model, repeat=repeat, phase=phase, seconds=elapsed))
                return result

            prepared, calibration = timed("train_observation_preprocessing", lambda: observations(train))
            features = timed("train_feature_extraction", lambda: extract(prepared, model))
            scaled, keep, scaler = timed("train_feature_scaling", lambda: scale_train(features))
            fitted = timed("train_ridge_fit", lambda: fit_ridge(scaled, labels))
            prepared_test, _ = timed("apply_observation_preprocessing", lambda: observations(test, calibration))
            features_test = timed("apply_feature_extraction", lambda: extract(prepared_test, model))
            scaled_test = timed("apply_feature_scaling", lambda: scaler.transform(np.asarray(features_test, dtype=np.float64)[:, keep]))
            prediction = timed("apply_prediction", lambda: predict(scaled_test, fitted))
            if repeat >= 0:
                for prefix in ("train", "apply"):
                    total = sum(r["seconds"] for r in rows if r["model"] == model and r["repeat"] == repeat and r["phase"].startswith(prefix + "_"))
                    rows.append(dict(model=model, repeat=repeat, phase=prefix + "_total", seconds=total))
            assert int(keep.sum()) == width
            expected = saved["1000_balanced_" + model + "_test"]
            np.testing.assert_array_equal(prediction, expected)
            if repeat == -1:
                np.testing.assert_allclose(features, reference[model][:n_train], atol=1e-12, rtol=1e-12)
                np.testing.assert_allclose(features_test, reference[model][n_train + n_val:], atol=1e-12, rtol=1e-12)
                weights = n_train / (6 * np.bincount(labels, minlength=6)[labels])
                legacy, _, _ = ridge_predict(features[:, keep], labels, features_test[:, keep], weights)
                np.testing.assert_array_equal(prediction, legacy)
                verification[model] = {"retained_coordinates": width, "raw_coordinates": features.shape[1],
                    "train_max_absolute_feature_difference": float(np.max(np.abs(features - reference[model][:n_train]))),
                    "test_max_absolute_feature_difference": float(np.max(np.abs(features_test - reference[model][n_train + n_val:]))),
                    "saved_prediction_mismatches": int((prediction != expected).sum()),
                    "test_errors": int((prediction != saved["test_labels"]).sum()),
                    "prediction_sha256": hashlib.sha256(prediction.astype("<i8").tobytes()).hexdigest()}
        print(model, verification[model], flush=True)

    summary = []
    for model, phase in dict.fromkeys((r["model"], r["phase"]) for r in rows):
        values = [r["seconds"] for r in rows if r["model"] == model and r["phase"] == phase]
        summary.append(dict(model=model, phase=phase, median_s=float(np.median(values)),
                            min_s=min(values), max_s=max(values), repeats=len(values)))
    metadata = {
        "date": "2026-10-16", "hardware": hardware(), "platform": platform.platform(), "python": platform.python_version(),
        "numpy": np.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "threadpools": threadpool_info(), "environment_thread_limits": {v: os.environ[v] for v in
            ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "device": "CPU", "cuda_available": torch.cuda.is_available(), "mps_available": torch.backends.mps.is_available(),
        "training_windows": n_train, "validation_windows_not_timed": n_val, "new_windows": n_test,
        "batch_size": 256, "warmup_complete_runs_per_model": 1, "timing_repeats": args.repeats,
        "precision": "float64 observation/signature arithmetic; retained float32 raw/spectral tensor and pool storage; float64 graph features and ridge/scaling",
        "boundaries": {
            "archive_loading": "Warm OS-cache original loader, all 10299 windows plus labels, subjects and supplied561 (unused); excludes splitting and numerical work",
            "observation_preprocessing": "Finite-input check, triad formation/differences, fit or apply axis means/stds and triad RMS, geometric streams/cosines",
            "feature_extraction": "FFT/normalization and spectral summaries, requested signature/graph/pool blocks, QR plans, coordinate assembly; no trained weights",
            "feature_scaling": "Training constant-column mask and unweighted StandardScaler fit/transform, or frozen mask/transform",
            "ridge_fit": "Balanced targets/centering/weights, Gram formation, alpha10 LU solve, intercept; excludes StandardScaler",
            "prediction": "Scaled feature-matrix multiplication, intercept, argmax only",
            "total": "Sum of the four corresponding timed stages; train excludes archive loading and validation selection; apply includes preprocessing through prediction",
        },
        "implementation_scope": "Single requested model; retains rf.spectrum helper's pooled-summary computation even when compact discards that pool. No deployment kernel optimization. Reference generation and correctness checks are outside timing.",
        "timing_scope": "Wall-clock warmed process, five sequential repeats, one numerical thread, no accelerator, no enforced CPU affinity or exclusive host reservation; min/max describes timing variation, not accuracy uncertainty.",
        "selection_budget": "No new selection. Saved comparison chose between two frozen models at alpha10 using validation; five deterministic seeds and companion heads were historical checks, not independent accuracy samples.",
        "dataset_sha256": digest(archive),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
            (Path(__file__), HAR / "retained_features.py", HAR / "benchmark.py", HAR / "spikedcl_benchmark.py")},
        "historical_prediction_sha256": digest(HAR / "spikedcl_results/predictions.npz"),
        "split_indices_sha256": {k: hashlib.sha256(v.astype("<i8").tobytes()).hexdigest() for k, v in partitions.items()},
        "verification": verification,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in (("timings.csv", rows), ("summary.csv", summary)):
        with (args.output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for row in summary:
        print(f"{row['model']:12} {row['phase']:32} {row['median_s']:.6f} [{row['min_s']:.6f}, {row['max_s']:.6f}]", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
