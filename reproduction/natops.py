"""Reproduce all nine NATOPS rows of revision 16, Table ``tab:natops``.

Read this file in order: prepare the paths, compute two logsignatures, select
the paper's coordinates, then fit ridge classifiers using training-only CV.
The detailed specification is ``source/companion/appendix_natops.tex`` (``app:natops``).
The two full-map controls were designed after the other seven outcomes were
known; reproducing them does not change that chronology.

Run ``uv run python -m reproduction.natops --help`` from the repository root.
"""

import argparse
import csv
from importlib.metadata import version
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch
import tensorsig
from tensorsig.classical.signatures import log_signature


HERE = Path(__file__).resolve().parent
ALPHAS = [10.0**power for power in range(-6, 7)]
# This is the main table's row order. "Pascal" replaces the archived name
# "offset"; reference/natops.json records the correspondence to saved evidence.
DIMENSIONS = {
    "ordinary_full": 325,
    "moments_ordinary_full": 350,
    "coded_clock_full": 423,
    "pascal_full": 1275,
    "ordinary_local": 73,
    "moments_ordinary_local": 98,
    "coded_clock_local": 171,
    "pascal_local": 267,
    "moments_only": 50,
}


def prepare_paths(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Add the clock and origin anchor, then code *increments*, not positions.

    Appendix ``app:natops``, "Trajectory path construction", fixes 51 landmark
    observations and an initial zero-clock step retaining the first pose. Thus
    the anchored ordinary path has 52 points and 51 increments. Equation
    ``eq:lift`` uses the weights (1, r/50), including that initial-pose step.
    The first 25 coded channels are ordinary; the next 25 are weighted copies.
    """
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[1:] != (51, 24) or not np.isfinite(raw).all():
        raise ValueError("NATOPS observations must be finite with shape (cases, 51, 24)")
    clock = np.broadcast_to(np.linspace(0, 1, 51)[None, :, None], (len(raw), 51, 1))
    observations = np.concatenate((clock, raw), axis=2)
    ordinary = np.concatenate((np.zeros((len(raw), 1, 25)), observations), axis=1)
    increments = np.diff(ordinary, axis=1)
    position = np.arange(51, dtype=np.float64) / 50
    coded_increments = np.concatenate(
        (increments, position[None, :, None] * increments), axis=2
    )
    coded = np.concatenate(
        (np.zeros((len(raw), 1, 50)), np.cumsum(coded_increments, axis=1)), axis=1
    )
    return ordinary, coded


def area_pairs() -> dict[str, list[tuple[int, int]]]:
    """Select the interaction subsets in Appendix ``app:natops``.

    Channel zero is time; channels 1..24 are eight consecutive XYZ landmarks.
    "Local" means within one landmark plus its clock interactions. It does
    not mean a temporal window. Every map still uses the entire trajectory.
    """
    ordinary_full = set(combinations(range(25), 2))
    ordinary_local = {(0, channel) for channel in range(1, 25)}
    for landmark in range(8):
        ordinary_local.update(combinations(range(1 + 3 * landmark, 4 + 3 * landmark), 2))

    # Keep every copy pairing of a local edge, and each channel's self-pair
    # across copies. The self-pairs include the separation statistic (eq:sep).
    coded_local = {(channel, 25 + channel) for channel in range(25)}
    for left, right in ordinary_local:
        for p in range(2):
            for q in range(2):
                coded_local.add(tuple(sorted((25 * p + left, 25 * q + right))))
    coded_full = set(combinations(range(50), 2))
    coded_clock = {pair for pair in coded_full if 0 in pair or 25 in pair}
    within_landmark = {pair for pair in ordinary_local if 0 not in pair}
    spatial_full = set(combinations(range(1, 25), 2))
    return {
        "ordinary_full": sorted(ordinary_full),
        "moments_ordinary_full": sorted(ordinary_full),
        "coded_clock_full": sorted(coded_clock | spatial_full),
        "pascal_full": sorted(coded_full),
        "ordinary_local": sorted(ordinary_local),
        "moments_ordinary_local": sorted(ordinary_local),
        "coded_clock_local": sorted(coded_clock | within_landmark),
        "pascal_local": sorted(coded_local),
        "moments_only": [],
    }


def features(raw: np.ndarray) -> dict[str, np.ndarray]:
    """Compute the nine fixed representations using the installed tensorsig.

    At depth two, tensorsig's ``mode="words"`` returns the endpoints followed
    by J[i,j] for i<j in lexicographic order: exactly equation ``eq:area``.
    Compute ordinary and coded logsignatures once, then gather the relevant
    columns. Constant clock coordinates are retained, as in the paper.
    """
    ordinary_path, coded_path = prepare_paths(raw)
    with torch.no_grad():
        ordinary = log_signature(torch.from_numpy(ordinary_path), depth=2).numpy()
        coded = log_signature(torch.from_numpy(coded_path), depth=2).numpy()
    ordinary_lookup = {pair: 25 + k for k, pair in enumerate(combinations(range(25), 2))}
    coded_lookup = {pair: 50 + k for k, pair in enumerate(combinations(range(50), 2))}
    result = {}
    for name, pairs in area_pairs().items():
        if name.startswith("ordinary_"):
            source, width, lookup = ordinary, 25, ordinary_lookup
        else:
            source, width, lookup = coded, 50, coded_lookup
        indices = list(range(width)) + [lookup[pair] for pair in pairs]
        result[name] = source[:, indices]
        assert result[name].shape == (len(raw), DIMENSIONS[name])
    return result


def fit_ridge(x: np.ndarray, y: np.ndarray, folds: list) -> tuple[object, dict]:
    """Select alpha with fold-local scaling; exact ties keep the smaller alpha.

    This is Appendix ``app:natops``, "Cross-validation protocol and results".
    All five validation folds contain 36 observations, so maximizing their
    total correct count is exactly maximizing the mean validation accuracy.
    The official test set is never used for selection or standardization.
    """
    def pipeline(alpha):
        return make_pipeline(StandardScaler(), RidgeClassifier(alpha=alpha, solver="cholesky"))

    candidates = []
    best_count = -1
    for alpha in ALPHAS:
        counts = []
        for train, valid in folds:
            model = pipeline(alpha).fit(x[train], y[train])
            counts.append(int(np.sum(model.predict(x[valid]) == y[valid])))
        candidates.append({"alpha": alpha, "fold_correct": counts})
        if sum(counts) > best_count:
            best_count, best_alpha = sum(counts), alpha
    model = pipeline(best_alpha).fit(x, y)
    return model, {
        "selected_alpha": best_alpha,
        "selected_cv_accuracy": best_count / len(y),
        "candidates": candidates,
    }


def self_check() -> None:
    """Check coordinates against equation (eq:area), independently of the fit."""
    raw = np.random.default_rng(7).normal(size=(2, 51, 24))
    ordinary_path, coded_path = prepare_paths(raw)
    computed = features(raw)
    for path, name in ((ordinary_path, "ordinary_full"), (coded_path, "pascal_full")):
        increments = np.diff(path, axis=1)
        # Independent, direct ordered-pair formula; only used in this check.
        strict_upper = np.triu(np.ones((51, 51)), k=1)
        ordered = np.einsum("bri,rs,bsj->bij", increments, strict_upper, increments)
        area = (ordered - ordered.transpose(0, 2, 1)) / 2
        i, j = np.triu_indices(increments.shape[-1], k=1)
        expected = np.concatenate((increments.sum(axis=1), area[:, i, j]), axis=1)
        np.testing.assert_allclose(computed[name], expected, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(
        computed["ordinary_full"][:, :25], np.c_[np.ones(2), raw[:, -1]], atol=1e-13
    )
    np.testing.assert_allclose(
        computed["moments_only"][:, 25:],
        np.einsum("r,bri->bi", np.arange(51) / 50, np.diff(ordinary_path, axis=1)),
        atol=1e-13,
    )


def run(data_dir: Path, output_dir: Path) -> None:
    """Fit the fixed table, write new results, and compare with saved evidence."""
    from .data import load_natops

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    reference = json.loads((HERE / "reference/natops.json").read_text())
    train, train_y, test, test_y = load_natops(data_dir)
    # One thread gives repeatable CPU work and avoids overhead on these small fits.
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        train_features, test_features = features(train), features(test)
        folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=20261003).split(train, train_y))
        assert [valid.tolist() for _, valid in folds] == reference["validation_indices"]
        assert all(len(valid) == 36 for _, valid in folds)
        rows, predictions, cv_scores = [], {}, {}
        for name in DIMENSIONS:
            model, cv = fit_ridge(train_features[name], train_y, folds)
            predicted = model.predict(test_features[name])
            correct = int(np.sum(predicted == test_y))
            rows.append({"model": name, "features": DIMENSIONS[name],
                         "alpha": cv["selected_alpha"], "cv_accuracy": cv["selected_cv_accuracy"],
                         "correct": correct, "total": len(test_y), "test_accuracy": correct / len(test_y)})
            predictions[name], cv_scores[name] = predicted, cv
            print(f"{name:25s} {DIMENSIONS[name]:4d} features  alpha={cv['selected_alpha']:g}  "
                  f"CV={100 * cv['selected_cv_accuracy']:.2f}%  test={correct}/{len(test_y)}", flush=True)

    with (output_dir / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output_dir / "predictions.npz", truth=test_y, **predictions)
    (output_dir / "cv_scores.json").write_text(json.dumps(cv_scores, indent=2) + "\n")
    checks = {}
    for row in rows:
        name = row["model"]
        expected = reference["models"][name]
        checks[name] = {
            "predictions_match": predictions[name].tolist() == expected["predictions"],
            "cv_counts_match": [c["fold_correct"] for c in cv_scores[name]["candidates"]] == expected["fold_correct"],
            "alpha_matches": row["alpha"] == expected["selected_alpha"],
            "correct_matches": row["correct"] == expected["correct"],
        }
    passed = all(all(check.values()) for check in checks.values())
    report = {"paper": "paper, Table tab:natops; Appendix app:natops",
              "tensorsig_source": tensorsig.__file__,
              "packages": {name: version(name) for name in ("tensorsig", "torch", "numpy", "scipy", "scikit-learn")},
              "elapsed_seconds": time.monotonic() - started,
              "reference_matches": checks, "all_reference_checks_passed": passed}
    (output_dir / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise RuntimeError(f"Reproduction differs from saved evidence; inspect {output_dir / 'run.json'}")
    print(f"All nine prediction vectors and all 585 CV fold counts match. Results: {output_dir}")


def main() -> None:
    from .data import download_natops

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "data/natops", help="Directory containing NATOPS_TRAIN.ts and NATOPS_TEST.ts")
    parser.add_argument("--download", action="store_true", help="Download and verify the public data into --data first")
    parser.add_argument("--output", type=Path, default=HERE / "results/natops", help="Empty directory for freshly computed results")
    args = parser.parse_args()
    if args.download:
        download_natops(args.data)
    run(args.data, args.output)


if __name__ == "__main__":
    main()
