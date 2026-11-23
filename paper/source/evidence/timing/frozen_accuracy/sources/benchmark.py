"""Run the three retained UCI HAR representations with a fixed ridge head."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
import shutil
import urllib.request
import zipfile
from pathlib import Path
from time import perf_counter

# Limit Apple Accelerate too: threadpoolctl alone does not control it.
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import sklearn
import torch
from retained_features import build_features
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
DATA_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
ACTIVITIES = ("walking", "upstairs", "downstairs", "sitting", "standing", "lying")
MODELS = {
    "compact368": "Raw/spectral Pascal logsignatures + calibrated graph moments and trends",
    "pool803": "Reduced spectral/rotation pool + raw Pascal",
    "uci903": "561 + raw Pascal P3/depth 2",
}


def download(destination: Path) -> None:
    """Cache the official ZIP; never replace an existing local archive."""
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".part")
    try:
        with urllib.request.urlopen(DATA_URL, timeout=120) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        if not zipfile.is_zipfile(temporary):
            raise ValueError("UCI download is not a ZIP archive")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_data(source: Path) -> dict:
    """Read the UCI archive directly, including its nested ZIP; no extraction."""
    result = {}
    with zipfile.ZipFile(source) as outer:
        inner_name = next((n for n in outer.namelist() if n.endswith("UCI HAR Dataset.zip")), None)
        inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name))) if inner_name else outer
        try:
            prefix = next(n[:-len("train/y_train.txt")] for n in inner.namelist() if n.endswith("train/y_train.txt"))
            for split, count in (("train", 7352), ("test", 2947)):
                def read(name, dtype=float, split=split):
                    with inner.open(prefix + split + "/" + name) as stream:
                        return np.loadtxt(stream, dtype=dtype)

                signals = np.stack([
                    read(f"Inertial Signals/{sensor}_{axis}_{split}.txt")
                    for sensor in ("body_acc", "body_gyro", "total_acc") for axis in "xyz"
                ], axis=-1)
                engineered = read(f"X_{split}.txt")
                labels = read(f"y_{split}.txt", int) - 1
                subjects = read(f"subject_{split}.txt", int)
                if (signals.shape != (count, 128, 9) or engineered.shape != (count, 561)
                        or labels.shape != (count,) or subjects.shape != (count,)
                        or not np.isfinite(signals).all() or not np.isfinite(engineered).all()
                        or not np.array_equal(np.unique(labels), np.arange(6))):
                    raise ValueError(f"Unexpected UCI HAR data in {split}")
                result[split] = {"signals": signals, "engineered": engineered, "labels": labels, "subjects": subjects}
        finally:
            if inner is not outer:
                inner.close()
    if set(result["train"]["subjects"]) & set(result["test"]["subjects"]):
        raise ValueError("Training and test subjects overlap")
    return result


def remove_constants(values: np.ndarray, n_train: int) -> tuple[np.ndarray, int]:
    """Use relative training variability; compute the mask in float64."""
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Feature matrix contains nonfinite values")
    train = values[:n_train]
    magnitude = np.maximum(train.max(0), -train.min(0))
    keep = train.std(0) > 1e-6 * magnitude
    return values[:, keep], int((~keep).sum())


def ridge_predict(train: np.ndarray, labels: np.ndarray, test: np.ndarray,
                  sample_weight: np.ndarray | None = None, *,
                  alpha: float = 10.0) -> tuple[np.ndarray, float, float]:
    """Standardized multiclass ridge with an unpenalized intercept (default alpha 10).

    This is sklearn's +/-1-target objective, optionally sample-weighted.
    Feature scaling remains unweighted and training-only. A float64 general LU
    solve avoids the Accelerate Cholesky problems encountered in this example.
    The dual form keeps the small smoke run inexpensive too.
    """
    started = perf_counter()
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Ridge alpha must be finite and positive")
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if (sample_weight.shape != (len(train),) or not np.isfinite(sample_weight).all()
                or np.any(sample_weight <= 0)):
            raise ValueError("Expected one finite positive weight per training row")
    scaler = StandardScaler()
    x = scaler.fit_transform(np.asarray(train, dtype=np.float64))
    offset = np.average(x, axis=0, weights=sample_weight)
    x -= offset
    classes = np.unique(labels)
    target = 2.0 * (labels[:, None] == classes) - 1.0
    intercept = np.average(target, axis=0, weights=sample_weight)
    target -= intercept
    if sample_weight is not None:
        root_weight = np.sqrt(sample_weight)[:, None]
        x *= root_weight
        target *= root_weight
    primal = x.shape[1] <= len(x)
    gram = x.T @ x if primal else x @ x.T
    gram.flat[::len(gram) + 1] += alpha
    solution = np.linalg.solve(gram, x.T @ target if primal else target)
    coefficient = solution if primal else x.T @ solution
    intercept -= offset @ coefficient
    fit_seconds = perf_counter() - started
    started = perf_counter()
    prediction = classes[(scaler.transform(test) @ coefficient + intercept).argmax(1)]
    return prediction, fit_seconds, perf_counter() - started


def self_check() -> None:
    """One small mathematical/runner check; no dataset or download required."""
    from retained_features import self_check as check_features
    from sklearn.linear_model import RidgeClassifier
    from sklearn.pipeline import make_pipeline

    check_features()
    rng = np.random.default_rng(43)
    for width in (7, 70):
        values = rng.normal(size=(80, width))
        values[:, -1] = values[:, 0] + values[:, 1]
        with_constant = np.column_stack((values, np.ones(80)))
        actual, removed = remove_constants(with_constant, 50)
        np.testing.assert_array_equal(actual, values)
        assert removed == 1
        labels = np.arange(50) % 3
        for weights in (None, np.linspace(0.2, 2.0, 50)):
            for alpha in (0.03, 10.0, 300.0):
                expected = make_pipeline(StandardScaler(), RidgeClassifier(alpha=alpha, solver="lsqr", tol=1e-12))
                expected.fit(values[:50], labels, ridgeclassifier__sample_weight=weights)
                prediction, _, _ = ridge_predict(actual[:50], labels, actual[50:], weights, alpha=alpha)
                np.testing.assert_array_equal(prediction, expected.predict(values[50:]))
    print("Feature and ridge checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "data/uci_har.zip", help="Local official UCI ZIP")
    parser.add_argument("--download", action="store_true", help="Download only if the local ZIP is absent")
    parser.add_argument("--output", type=Path, help="Default: retained_results/ (smoke_results/ for --smoke)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--smoke", action="store_true", help="240 train / 120 test windows; software check only")
    parser.add_argument("--self-check", action="store_true", help="Check feature identities and ridge without data")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    torch.set_num_threads(1)
    if args.self_check:
        with threadpool_limits(limits=1):
            self_check()
        return
    if args.download:
        download(args.data)
    if not args.data.is_file():
        parser.error(f"Dataset not found: {args.data}; use --download or --data PATH")
    output = args.output or HERE / ("smoke_results" if args.smoke else "retained_results")
    started = perf_counter()
    data = load_data(args.data)
    if args.smoke:
        rng = np.random.default_rng(2025)
        for split, count in (("train", 240), ("test", 120)):
            indices = np.sort(rng.choice(len(data[split]["labels"]), count, replace=False))
            data[split] = {key: value[indices] for key, value in data[split].items()}
    train, test = data["train"], data["test"]
    n_train = len(train["labels"])
    signals = np.concatenate((train["signals"], test["signals"]))
    engineered = np.concatenate((train["engineered"], test["engineered"]))
    metadata = {
        "dataset_url": DATA_URL, "smoke": args.smoke,
        "train_windows": n_train, "test_windows": len(test["labels"]),
        "train_subjects": np.unique(train["subjects"]).tolist(),
        "test_subjects": np.unique(test["subjects"]).tolist(),
        "activities": ACTIVITIES, "python": platform.python_version(),
        "platform": platform.platform(), "numpy": np.__version__,
        "torch": torch.__version__, "sklearn": sklearn.__version__,
        "settings": {"ridge_alpha": 10, "code": "Pascal QR", "P": 3, "depth": 2,
                     "graph_trend_P": 2, "seed": 2025, "numerical_threads": 1,
                     "batch_size": args.batch_size, "feature_scaling": "Training-only StandardScaler",
                     "constant_mask": "Training std > 1e-6 * maximum training absolute value"},
        "models": MODELS, "confusion_matrices": {}, "constant_columns_removed": {},
        "timing_note": "Feature preparation is shared across all three models. Supplied 561 extraction is unmeasured. Wall time includes ZIP loading.",
    }
    rows = []
    with threadpool_limits(limits=1):
        features, preparation = build_features(signals, n_train, engineered, args.batch_size)
        metadata["feature_blocks"] = preparation
        for name, values in features.items():
            values, removed = remove_constants(values, n_train)
            predicted, fit_seconds, predict_seconds = ridge_predict(values[:n_train], train["labels"], values[n_train:])
            row = {"model": name, "head": "ridge", "features": values.shape[1],
                   "accuracy": accuracy_score(test["labels"], predicted),
                   "macro_f1": f1_score(test["labels"], predicted, average="macro", zero_division=0),
                   "errors": int((test["labels"] != predicted).sum()),
                   "fit_s": round(fit_seconds, 6), "predict_s": round(predict_seconds, 6)}
            rows.append(row)
            metadata["confusion_matrices"][name] = confusion_matrix(test["labels"], predicted, labels=np.arange(6)).tolist()
            metadata["constant_columns_removed"][name] = removed
            print(f"{name:12} {values.shape[1]:4} features  {100 * row['accuracy']:.2f}%  "
                  f"{row['errors']:3} errors  fit {fit_seconds:.3f}s / predict {predict_seconds:.3f}s", flush=True)
    metadata["wall_s"] = perf_counter() - started
    output.mkdir(parents=True, exist_ok=True)
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Shared feature preparation: {preparation['shared_preparation_s']:.3f}s; "
          f"total wall time: {metadata['wall_s']:.3f}s. Results: {output}")


if __name__ == "__main__":
    main()
