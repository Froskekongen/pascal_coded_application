"""Evaluate the frozen retained models on the released SpikeDCL UCI HAR split.

Split seed 0 is fixed; seeds 1000..1004 affect neural training, not partitions.
Our ridge heads are deterministic, so their repeated-run SD should be zero.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
from pathlib import Path
from time import perf_counter

# Import benchmark first so numerical thread limits are set before torch/numpy.
from benchmark import (ACTIVITIES, DATA_URL, HERE, MODELS, download, load_data,
                       remove_constants, ridge_predict, self_check as check_retained)
import numpy as np
import sklearn
import torch
from retained_features import build_features
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

UPSTREAM_COMMIT = "ef1969d940fb62d94af6600fcae880f51930b7dd"
UPSTREAM = "https://github.com/Intelligent-Computing-Lab-Panda/SNN_HAR/blob/" + UPSTREAM_COMMIT
SEEDS = tuple(range(1000, 1005))
RAW_MODELS = ("compact368", "pool803")  # First wins a validation tie.
HEADS = ("balanced", "retained_unweighted")
NINE_CHANNEL_MODELS = {
    "compact_raw9": "Retained compact model with nine-channel raw Pascal logsignature",
    "pool_raw9": "Retained spectral/rotation pool with nine-channel raw Pascal tensor",
    "uci_raw9": "Supplied561 with nine-channel raw Pascal tensor",
    "compact_raw9_spectral9_factorized": "Nine-channel raw log + per-axis supplied9 spectral logs + retained 97 moments/trends",
    "compact_raw9_spectral9_joint": "Nine-channel raw log + joint supplied9 spectral log + retained 97 moments/trends",
}


def split_indices(subjects: np.ndarray, *, seed: int = 0) -> dict[str, np.ndarray]:
    """Original train-then-test indices, regrouped exactly as the source loader."""
    pool = np.argsort(subjects, kind="stable")
    train, test = train_test_split(pool, test_size=0.2, random_state=seed)
    train, validation = train_test_split(train, test_size=0.2, random_state=seed)
    return {"train": train, "validation": validation, "test": test}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def self_check() -> None:
    """Verify partition mapping against independent legacy-RNG permutations."""
    check_retained()
    subjects = np.array([3, 1, 2, 1, 3, 2, 1, 2, 3] * 7)
    pool = np.concatenate([np.flatnonzero(subjects == s) for s in np.unique(subjects)])
    order = np.random.RandomState(0).permutation(len(pool))
    cut = int(np.ceil(0.2 * len(pool)))
    remainder, test = pool[order[cut:]], pool[order[:cut]]
    order = np.random.RandomState(0).permutation(len(remainder))
    cut = int(np.ceil(0.2 * len(remainder)))
    expected = {"train": remainder[order[cut:]], "validation": remainder[order[:cut]], "test": test}
    actual = split_indices(subjects)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
    np.testing.assert_array_equal(np.sort(np.concatenate(list(actual.values()))), np.arange(len(pool)))
    print("SpikeDCL partition checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "data/uci_har.zip")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, help="Default: a separate results directory for the selected experiment")
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--all-channels", action="store_true", help="Also compare nine-channel raw and spectral Pascal blocks")
    variants.add_argument("--visibility", action="store_true", help="Fixed theory-guided feature round on Compact and pool")
    parser.add_argument("--batch-size", type=int, default=256, help="Feature extraction batch size only")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    torch.set_num_threads(1)
    if args.self_check:
        with threadpool_limits(limits=1):
            self_check()
            if args.visibility:
                from visibility_features import self_check as check_visibility
                check_visibility()
        return
    if args.download:
        download(args.data)
    if not args.data.is_file():
        parser.error(f"Dataset not found: {args.data}; use --download or --data PATH")
    output = args.output or HERE / ("visibility_results" if args.visibility else
                                   "spikedcl_nine_channel_results" if args.all_channels else "spikedcl_results")
    models = MODELS | (NINE_CHANNEL_MODELS if args.all_channels else {})
    raw_models = RAW_MODELS
    if args.all_channels:
        raw_models += tuple(name for name in NINE_CHANNEL_MODELS if name != "uci_raw9")
    if args.visibility:
        from visibility_features import MODELS as visibility_models, RESEARCH_ARTIFACT, build_features as visibility_features
        models = visibility_models
        raw_models = tuple(models)
    started = perf_counter()
    original = load_data(args.data)
    all_data = {key: np.concatenate([original[s][key] for s in ("train", "test")])
                for key in original["train"]}
    partitions = split_indices(all_data["subjects"])
    order = np.concatenate(list(partitions.values()))
    np.testing.assert_array_equal(np.sort(order), np.arange(len(order)))
    assert [len(i) for i in partitions.values()] == [6591, 1648, 2060]
    n_train, n_val = len(partitions["train"]), len(partitions["validation"])
    data = {key: values[order] for key, values in all_data.items()}
    y_train = data["labels"][:n_train]
    y_val = data["labels"][n_train:n_train + n_val]
    y_test = data["labels"][n_train + n_val:]
    counts = np.bincount(y_train, minlength=len(ACTIVITIES))
    balanced_weights = n_train / (len(ACTIVITIES) * counts[y_train])
    metadata = {
        "dataset_url": DATA_URL, "dataset_sha256": sha256(args.data),
        "upstream_commit": UPSTREAM_COMMIT,
        "sources": {"paper": "https://doi.org/10.3389/fnins.2023.1233037",
                    "earlier_manuscript": "https://openreview.net/references/pdf?id=KuhFFuu6SK",
                    "split": UPSTREAM + "/data_preprocess/data_preprocess_utils.py#L13-L24",
                    "loader": UPSTREAM + "/data_preprocess/data_preprocess_ucihar.py#L218-L260",
                    "training": UPSTREAM + "/main.py"},
        "published_comparators_percent": {"journal": {"mean": 98.86, "std": 0.28},
                                           "earlier_manuscript": {"mean": 98.63, "std": 0.24}},
        "settings": {"split_seed": 0, "training_seeds": SEEDS, "stratify": None,
                     "split": "Stable ascending subject order, test_size=.2 twice, random_state=0 twice",
                     "index_convention": "Zero-based original train rows followed by original test rows",
                     "input_channels": [s + "_" + a for s in ("body_acc", "body_gyro", "total_acc") for a in "xyz"],
                     "ridge_alpha": 10, "feature_definitions": "retained_features.py",
                     "all_channels_experiment": args.all_channels,
                     "visibility_experiment": args.visibility,
                     "feature_scaling": "Training-only, unweighted axis statistics/RMS and StandardScaler",
                     "constant_mask": "Training std > 1e-6 * maximum training absolute value",
                     "primary_head": "balanced", "heads": HEADS,
                     "balanced_weight": "n_train / (6 * training_count_of_class), mean weight one",
                     "selection": "Highest validation accuracy among selection_candidates; first listed wins ties",
                     "selection_candidates": raw_models,
                     "refit_on_validation": False, "new_hyperparameter_search": False,
                     "std_ddof": 0, "numerical_threads": 1, "feature_batch_size": args.batch_size},
        "adaptation_notes": [
            "Same fixed data partition for every seed; deterministic ridge gives identical repetitions.",
            "Balanced ridge matches the expected class proportions of inverse-frequency sampling, not sampled SGD trajectories.",
            "Ridge uses a closed-form solve; Adam, learning rates, 60 epochs, neural batch size and cosine schedule do not apply.",
            "Paper neural batch size is 128; released main.py defaults to 64 and has no cosine scheduler.",
            "uci903 and uci_raw9 use the additional supplied 561 engineered features and are separate hybrid comparators.",
            "Feature designs were previously explored on the original UCI test set; this is not untouched holdout evidence.",
            "Random windows share subjects and may overlap in underlying observations across partitions.",
        ],
        "activities": ACTIVITIES, "models": models,
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "torch": torch.__version__, "sklearn": sklearn.__version__,
        "source_sha256": {p: sha256(HERE / p) for p in ("spikedcl_benchmark.py", "benchmark.py", "retained_features.py")},
        "partitions": {name: {"windows": len(indices),
                              "indices_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
                              "subjects": np.unique(all_data["subjects"][indices]).tolist(),
                              "class_counts": np.bincount(all_data["labels"][indices], minlength=6).tolist()}
                       for name, indices in partitions.items()},
        "confusion_matrices": {}, "constant_columns_removed": {}, "validation_choices": {},
    }
    if args.visibility:
        metadata["research_artifact"] = RESEARCH_ARTIFACT
        metadata["settings"]["feature_definitions"] = "visibility_features.py + unchanged retained baselines"
        metadata["source_sha256"]["visibility_features.py"] = sha256(HERE / "visibility_features.py")
        metadata["adaptation_notes"].append("A prespecified exploratory round informed by previously inspected baseline residuals; all fourteen candidates retained.")
    predictions = {f"{name}_indices": indices for name, indices in partitions.items()}
    predictions.update({"validation_labels": y_val, "test_labels": y_test,
                        "original_subjects": all_data["subjects"], "original_labels": all_data["labels"]})
    rows = []
    with threadpool_limits(limits=1):
        if args.visibility:
            features, preparation = visibility_features(data["signals"], n_train, data["engineered"], args.batch_size)
        else:
            features, preparation = build_features(data["signals"], n_train, data["engineered"], args.batch_size,
                                                   all_channels=args.all_channels)
        assert tuple(features) == tuple(models)
        metadata["feature_blocks"] = preparation
        for name, values in features.items():
            features[name], removed = remove_constants(values, n_train)
            metadata["constant_columns_removed"][name] = removed
        # Selection reads only validation metrics; all test scores are computed afterwards.
        for seed in SEEDS:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            for head in HEADS:
                candidates = {}
                for name, values in features.items():
                    pred, fit_s, predict_s = ridge_predict(values[:n_train], y_train, values[n_train:],
                                                         balanced_weights if head == "balanced" else None)
                    candidates[name] = {"prediction": pred, "fit_s": fit_s, "predict_s": predict_s,
                                        "validation_accuracy": accuracy_score(y_val, pred[:n_val])}
                selected = max(raw_models, key=lambda name: candidates[name]["validation_accuracy"])
                metadata["validation_choices"][f"{seed}/{head}"] = selected
                for name, candidate in candidates.items():
                    pred = candidate["prediction"]
                    test_pred = pred[n_val:]
                    key = f"{seed}_{head}_{name}"
                    predictions[key + "_validation"] = pred[:n_val]
                    predictions[key + "_test"] = test_pred
                    rows.append({"seed": seed, "head": head, "model": name,
                                 "features": features[name].shape[1],
                                 "validation_accuracy": candidate["validation_accuracy"],
                                 "test_accuracy": accuracy_score(y_test, test_pred),
                                 "test_macro_f1": f1_score(y_test, test_pred, average="macro", zero_division=0),
                                 "test_errors": int((y_test != test_pred).sum()),
                                 "selected_raw_model": name == selected,
                                 "fit_s": candidate["fit_s"], "validation_and_test_predict_s": candidate["predict_s"]})
                    metadata["confusion_matrices"][key] = confusion_matrix(y_test, test_pred, labels=np.arange(6)).tolist()
    summary = []
    for head in HEADS:
        for model in (*models, "validation_selected_raw"):
            selected_rows = [r for r in rows if r["head"] == head and
                             (r["selected_raw_model"] if model == "validation_selected_raw" else r["model"] == model)]
            assert len(selected_rows) == len(SEEDS)
            summary.append({"head": head, "model": model, "repetitions": len(selected_rows),
                            "selected_model": selected_rows[0]["model"] if model == "validation_selected_raw" else "",
                            "features": selected_rows[0]["features"],
                            **{metric + suffix: float(getattr(np, method)([r[metric] for r in selected_rows]))
                               for metric in ("validation_accuracy", "test_accuracy", "test_macro_f1", "test_errors")
                               for suffix, method in (("_mean", "mean"), ("_std", "std"))}})
    metadata["wall_s"] = perf_counter() - started
    metadata["timing_note"] = f"Feature extraction shared across all models/seeds/heads; wall time includes ZIP loading and all {len(rows)} fits."
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "results.csv", rows)
    write_csv(output / "summary.csv", summary)
    np.savez_compressed(output / "predictions.npz", **predictions)
    metadata["evidence_sha256"] = {p: sha256(output / p) for p in ("results.csv", "summary.csv", "predictions.npz")}
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for row in summary:
        print(f"{row['head']:20} {row['model']:24} "
              f"val {100 * row['validation_accuracy_mean']:.3f}%  "
              f"test {100 * row['test_accuracy_mean']:.3f} +/- {100 * row['test_accuracy_std']:.3f}%  "
              f"errors {row['test_errors_mean']:.0f}")
    print(f"Fixed split: 6591/1648/2060; all {len(SEEDS)} repetitions use the same windows.")
    print(f"Wall time: {metadata['wall_s']:.3f}s. Results: {output}")


if __name__ == "__main__":
    main()
