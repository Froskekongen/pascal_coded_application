"""Reproduce the three official-split HAR rows in revision 16, Table ``tab:har``.

Read ``build_features`` from top to bottom alongside ``appendix_har.tex``.
It constructs only Compact368 and its unchanged geometric block; the third
model uses the 561 features supplied with the dataset. No representation or
classifier search is performed. Labels are zero-based in the saved outputs.

Run ``uv run python -m reproduction.uci_har --help`` from the repository root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path

# Apple Accelerate is not controlled by threadpoolctl on every installation.
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import sklearn
import torch
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorsig.offset_coded_signatures.codes import OffsetCode
from tensorsig.offset_coded_signatures.plans import LocalRealization
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
TRIAD_PAIRS = ((0, 1), (0, 2), (1, 2))


def qr_code(length: int, copies: int) -> OffsetCode:
    """Positive-diagonal QR of Pascal columns: companion Section ``app:qr``."""
    return OffsetCode.pascal(length, copies, orthonormal=True, dtype=torch.float64)


def standardize_axes(values: torch.Tensor, n_train: int) -> torch.Tensor:
    """Fit one mean/std per axis over training windows and samples together.

    Individual windows retain their levels, including static posture. All
    subsequent test windows use these same training statistics (``app:har``).
    """
    train = values[:n_train]
    return (values - train.mean((0, 1))) / train.std((0, 1), correction=0).clamp_min(1e-8)


def coded_signature(increments: torch.Tensor, batch_size: int) -> np.ndarray:
    """P=3, depth=2 signature of QR-weighted increments, via tensorsig.

    This is Eq. ``eq:app-pascal`` with the QR basis of ``app:qr``. LocalRealization
    orders letters by input channel, then copy. Its input has shape
    (batch, windows, samples, channels), hence the singleton window axis.
    No clock or extra basepoint increment is added: the cumulative path starts
    at zero. The output excludes the scalar signature coordinate 1.

    The historical experiment rounded the dense signature to float32 *before*
    extracting areas. Preserve that numerical convention for reproducibility;
    the signature computation itself is float64.
    """
    length, channels = increments.shape[1:]
    plan = LocalRealization(qr_code(length, 3), channels, 2).signature(coordinates="dense")
    return np.concatenate([
        plan.evaluate(batch[:, None])[:, 0].float().numpy().copy()
        for batch in increments.split(batch_size)
    ])


def endpoints_and_areas(tensor: np.ndarray, width: int, *, spectral: bool = False) -> np.ndarray:
    """Read endpoints and upper-triangular signed areas from levels 1 and 2.

    The paper's depth-two logsignature is (s,J), with J_ij=(B_ij-B_ji)/2.
    The spectral branch stores sqrt(2)*J instead, as specified in ``app:har``.
    This fixed scaling is followed by training-only feature standardization.
    """
    second = tensor[:, width:].reshape(-1, width, width)
    left, right = np.triu_indices(width, 1)
    areas = second[:, left, right] - second[:, right, left]
    areas = areas / np.sqrt(2) if spectral else areas / 2
    return np.concatenate((tensor[:, :width], areas), axis=1)


def spectral_features(vector: torch.Tensor, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Four independent frequency-ordered signatures and amplitude summaries.

    Append the triad norm *before* the Fourier transform. One-sided bin energies
    e_k sum to the time-domain mean square; their shares p_k are the signature
    increments. See main Eq. ``eq:spectrum`` and the spectral branch in
    ``app:har``. Each scalar spectrum uses its own three-copy signature, so no
    areas between different spectra are retained.
    """
    signals = torch.cat((vector, vector.norm(dim=2, keepdim=True)), dim=2)
    length = signals.shape[1]
    energy = (torch.fft.rfft(signals, dim=1).abs() / length).square()
    # Interior positive frequencies have a matching negative-frequency bin.
    # For even lengths the Nyquist bin, like DC, is counted just once.
    energy[:, 1:energy.shape[1] - (length % 2 == 0)] *= 2
    shares = energy / energy.sum(1).clamp_min(1e-20)[:, None]
    if not torch.allclose(shares.sum(1), torch.ones_like(shares[:, 0]), atol=1e-8, rtol=1e-7):
        raise ValueError("Every retained spectral channel must have positive energy")

    # Treat each of the four spectra as an independent scalar path. This
    # avoids computing and then discarding cross-spectrum tensor coordinates.
    windows, bins, channels = shares.shape
    scalar_paths = shares.transpose(1, 2).reshape(windows * channels, bins, 1)
    signature = coded_signature(scalar_paths, batch_size)
    shape = endpoints_and_areas(signature, 3, spectral=True).reshape(windows, channels * 6)
    amplitude = energy.sqrt()
    scale = torch.stack((amplitude.mean(1), amplitude.std(1, correction=0)), dim=2)
    return shape, scale.flatten(1).numpy()


def geometric_summary(stream: torch.Tensor, *, skewness: bool) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std/(skewness) and a standardized linear Pascal trend per stream.

    These are the explicit scalar formulas in ``app:har``. Computing the
    moments directly is the algebraic evaluation of the closed value-clock
    signature readouts described there; it avoids building the graph path.
    The trend is T(z)=mean(q_1*(z-mean(z))/std(z)), with q_1 from
    Eq. ``eq:qr-first-two``. Near-constant streams have zero skewness and trend.
    """
    mean = stream.mean(1)
    centered = stream - mean[:, None]
    variance = centered.square().mean(1)
    sd = variance.sqrt()
    normalized = centered / sd.clamp_min(1e-8)[:, None]
    normalized = torch.where((variance > 1e-12)[:, None], normalized, 0.0)
    statistics = [mean, sd]
    if skewness:
        statistics.append(normalized.pow(3).mean(1))
    linear_weight = qr_code(stream.shape[1], 2).matrix[:, 1]
    trend = (normalized * linear_weight[None, :, None]).mean(1)
    return torch.stack(statistics, dim=2).flatten(1).numpy(), trend.numpy()


def geometric_features(vectors: list[torch.Tensor], n_train: int) -> np.ndarray:
    """The unchanged 57-feature strength/persistence/alignment block (``app:har``).

    One uncentered training RMS scale per triad preserves common-rotation
    invariance. The 12 quadratic streams are 3 energies, 3 simultaneous dot
    products and 3 self-dot products at each lag 2 and 5 (40 and 100 ms).
    Three cosine streams add alignment independent of magnitude.
    """
    scaled = [v / v[:n_train].square().mean().sqrt().clamp_min(1e-8) for v in vectors]
    energy = torch.stack([v.square().sum(2) for v in scaled], dim=2)
    dots = torch.stack([(scaled[i] * scaled[j]).sum(2) for i, j in TRIAD_PAIRS], dim=2)
    streams = [energy, dots]
    for lag in (2, 5):
        streams.append(torch.stack([(v[:, lag:] * v[:, :-lag]).sum(2) for v in scaled], dim=2))
    norms = energy.sqrt()
    denominator = torch.stack([norms[:, :, i] * norms[:, :, j] for i, j in TRIAD_PAIRS], dim=2)
    cosines = dots / denominator.clamp_min(1e-12)
    cosine_stats, cosine_trends = geometric_summary(cosines, skewness=False)
    statistics, trends = zip(*(geometric_summary(stream, skewness=True) for stream in streams))
    # Keep the archived column order: cosine6, quadratic36, cosine trends3,
    # quadratic trends12. Geometry57 is this exact submatrix of Compact368.
    return np.concatenate((cosine_stats, *statistics, cosine_trends, *trends), axis=1)


@torch.inference_mode()
def build_features(signals: np.ndarray, n_train: int, engineered: np.ndarray,
                   batch_size: int = 256) -> dict[str, np.ndarray]:
    """Build exactly the three official-split rows of main Table ``tab:har``.

    ``signals`` has shape (N,128,9): body acceleration xyz, gyroscope xyz,
    total acceleration xyz. The first ``n_train`` rows are training windows.
    ``engineered`` contains the dataset's 561 supplied features in the same
    row order. Known constants remain until the training variance mask.
    """
    x = torch.as_tensor(signals, dtype=torch.float64)
    if (x.ndim != 3 or x.shape[1:] != (128, 9) or not 1 <= n_train <= len(x)
            or engineered.shape != (len(x), 561) or batch_size < 1
            or not torch.isfinite(x).all() or not np.isfinite(engineered).all()):
        raise ValueError("Expected finite (N,128,9) signals, (N,561) features and a training prefix")

    # 1. Sensor levels: total acceleration followed by gyroscope. Readings/128
    # are increments, not differences; constant-copy endpoints retain posture.
    raw = standardize_axes(x[:, :, [6, 7, 8, 3, 4, 5]], n_train)
    raw_log = endpoints_and_areas(coded_signature(raw / 128, batch_size), 18)

    # 2. Spectra of body acceleration, gravity and gyroscope, plus differences
    # of body acceleration and gyroscope. Four scalar signals per triad.
    vectors = [x[:, :, :3], x[:, :, 6:9] - x[:, :, :3], x[:, :, 3:6]]
    spectral_vectors = vectors + [vectors[0].diff(dim=1), vectors[2].diff(dim=1)]
    shape, scale = zip(*(spectral_features(standardize_axes(v, n_train), batch_size)
                         for v in spectral_vectors))

    # 3. Physical magnitude, persistence and alignment; do not axis-center.
    geometry = geometric_features(vectors, n_train)
    compact = np.concatenate((*shape, *scale, geometry, raw_log), axis=1)
    # 388 before masking = 120 spectral + 40 scale + 57 geometric + 171 raw.
    # Exactly 20 spectral constant-copy endpoints are removed on real HAR data.
    return {"compact368": compact, "geometry57": geometry, "supplied561": engineered}


def remove_constants(features: np.ndarray, n_train: int) -> tuple[np.ndarray, np.ndarray]:
    """Training-only relative variance mask from the last paragraph of ``app:har``."""
    features = np.asarray(features, dtype=np.float64)
    train = features[:n_train]
    keep = train.std(0) > 1e-6 * np.abs(train).max(0)
    return features[:, keep], keep


def ridge_predict(train: np.ndarray, labels: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Frozen alpha=10 one-versus-all ridge, with an unpenalized intercept.

    Standardize on training windows only. For each class use +1 for its members
    and -1 otherwise; center these targets before the closed-form solve. The
    label is the class with the greatest score (``app:har``, final paragraph).
    A general float64 solve matches the archived numerical convention.
    """
    scaler = StandardScaler()
    x = scaler.fit_transform(np.asarray(train, dtype=np.float64))
    offset = x.mean(0)
    x -= offset
    classes = np.unique(labels)
    target = 2.0 * (labels[:, None] == classes) - 1.0
    intercept = target.mean(0)
    gram = x.T @ x
    gram.flat[::len(gram) + 1] += 10.0
    coefficients = np.linalg.solve(gram, x.T @ (target - intercept))
    scores = (scaler.transform(test) - offset) @ coefficients + intercept
    return classes[scores.argmax(1)]


@torch.inference_mode()
def self_check() -> None:
    """Small offline checks of the lift, train/test boundary and ridge readout."""
    from sklearn.linear_model import RidgeClassifier
    from sklearn.pipeline import make_pipeline
    from tensorsig.classical.signatures import signature

    rng = np.random.default_rng(37)
    increments = torch.as_tensor(rng.normal(size=(3, 13, 2)), dtype=torch.float64)
    code = qr_code(13, 3).matrix
    np.testing.assert_allclose(code.T @ code, np.eye(3), atol=1e-14)
    np.testing.assert_allclose(code[:, 0], np.ones(13) / np.sqrt(13), atol=1e-14)
    lifted = (increments[..., None] * code[None, :, None]).flatten(2)
    path = torch.cat((torch.zeros_like(lifted[:, :1]), lifted.cumsum(1)), dim=1)
    # An ordinary path signature independently checks the coded plan and its
    # channel/copy ordering, including the historical float32 storage boundary.
    np.testing.assert_allclose(coded_signature(increments, 2), signature(path, 2).float().numpy(),
                               atol=1e-6, rtol=1e-6)

    signals = rng.normal(size=(8, 128, 9))
    engineered = rng.normal(size=(8, 561))
    features = build_features(signals, 5, engineered, batch_size=3)
    assert {name: values.shape for name, values in features.items()} == {
        "compact368": (8, 388), "geometry57": (8, 57), "supplied561": (8, 561)}
    np.testing.assert_array_equal(features["compact368"][:, 160:217], features["geometry57"])
    masked, keep = remove_constants(features["compact368"], 5)
    assert masked.shape == (8, 368) and (~keep).sum() == 20
    changed = signals.copy()
    changed[5:] = changed[5:] * 3 + 7
    perturbed = build_features(changed, 5, engineered, batch_size=2)
    for name in features:
        np.testing.assert_array_equal(perturbed[name][:5], features[name][:5])
    statistics, trend = geometric_summary(torch.ones(2, 13, 3, dtype=torch.float64), skewness=True)
    np.testing.assert_array_equal(statistics.reshape(2, 3, 3)[:, :, 1:], 0)
    np.testing.assert_array_equal(trend, 0)

    values = rng.normal(size=(45, 7))
    labels = np.arange(30) % 3
    independent = make_pipeline(StandardScaler(), RidgeClassifier(alpha=10, solver="lsqr", tol=1e-12))
    independent.fit(values[:30], labels)
    np.testing.assert_array_equal(ridge_predict(values[:30], labels, values[30:]),
                                   independent.predict(values[30:]))


def main() -> None:
    """Load official splits, construct the three fixed maps, fit, save and verify."""
    from .data import download_har, load_har

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "data/uci_har.zip")
    parser.add_argument("--download", action="store_true", help="Download the official archive if absent")
    parser.add_argument("--output", type=Path, default=HERE / "results/uci_har")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("Choose an empty --output directory to preserve completed results")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.download:
        download_har(args.data)
    data = load_har(args.data)
    n_train = len(data["train"]["labels"])
    signals = np.concatenate([data[split]["signals"] for split in ("train", "test")])
    engineered = np.concatenate([data[split]["engineered"] for split in ("train", "test")])
    expected = json.loads((HERE / "reference/har.json").read_text())
    torch.set_num_threads(1)
    rows, predictions, matrices, masks = [], {}, {}, {}
    with threadpool_limits(limits=1):
        print("Building the three fixed HAR feature maps...", flush=True)
        models = build_features(signals, n_train, engineered, args.batch_size)
        for name, features in models.items():
            features, keep = remove_constants(features, n_train)
            prediction = ridge_predict(features[:n_train], data["train"]["labels"], features[n_train:])
            matrix = confusion_matrix(data["test"]["labels"], prediction, labels=np.arange(6))
            errors = int((prediction != data["test"]["labels"]).sum())
            reference = expected["models"][name]
            row = {"model": name, "features": features.shape[1], "alpha": 10,
                   "correct": len(prediction) - errors, "errors": errors,
                   "accuracy": 1 - errors / len(prediction),
                   "predictions_match": (bool(np.array_equal(prediction, reference["predictions"]))
                                         if "predictions" in reference else None)}
            row["matches_paper"] = (features.shape[1] == reference["features"]
                                    and matrix.tolist() == reference["confusion_matrix"]
                                    and row["predictions_match"] is not False)
            rows.append(row)
            predictions[name], matrices[name], masks[name] = prediction, matrix.tolist(), keep.tolist()
            print(f"{name}: {features.shape[1]} features, {errors} errors, "
                  f"{100 * row['accuracy']:.2f}% accuracy; paper match: {row['matches_paper']}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.output / "predictions.npz", labels=data["test"]["labels"], **predictions)
    with args.data.open("rb") as stream:
        dataset_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = {"paper": expected["paper"], "results": rows, "confusion_matrices": matrices,
                "training_feature_masks": masks, "train_windows": n_train,
                "test_windows": len(data["test"]["labels"]),
                "train_subjects": np.unique(data["train"]["subjects"]).tolist(),
                "test_subjects": np.unique(data["test"]["subjects"]).tolist(),
                "dataset_sha256": dataset_hash,
                "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                "torch": torch.__version__, "scikit_learn": sklearn.__version__,
                                "tensorsig": version("tensorsig")},
                "settings": {"alpha": 10, "numerical_threads": 1, "batch_size": args.batch_size}}
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if not all(row["matches_paper"] for row in rows):
        raise SystemExit("HAR results differ from the paper; inspect the saved results and environment")


if __name__ == "__main__":
    main()
