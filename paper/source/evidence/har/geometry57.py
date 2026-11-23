"""Evaluate Compact368's unchanged 57 strength/persistence/alignment features.

Official subject-disjoint split, training-only preprocessing, frozen ridge
alpha10. Run with the sibling tensorsig Python environment; no tuning.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from benchmark import (ACTIVITIES, HERE, build_features, load_data, np,
                       remove_constants, ridge_predict, threadpool_limits, torch)
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "data/uci_har.zip")
    parser.add_argument("--output", type=Path, default=HERE / "geometry57_results")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory to preserve completed evidence")
    torch.set_num_threads(1)
    data = load_data(args.data)
    n = len(data["train"]["labels"])
    signals, engineered = (np.concatenate([data[s][key] for s in ("train", "test")])
                           for key in ("signals", "engineered"))
    with threadpool_limits(limits=1):
        features, preparation = build_features(signals, n, engineered)
        compact = features["compact368"]
        start = preparation["spectral_log_before_constant_removal"] + preparation["amplitude_moments"]
        stop = compact.shape[1] - preparation["raw_log"]
        geometry = compact[:, start:stop]
        assert (start, stop, geometry.shape[1]) == (160, 217, 57)
        compact, _ = remove_constants(compact, n)
        geometry, removed = remove_constants(geometry, n)
        assert geometry.shape[1] == 57 and removed == 0
        # Verify training and held-out preprocessing against the full model.
        np.testing.assert_array_equal(geometry, compact[:, 140:197])
        np.testing.assert_array_equal(
            StandardScaler().fit(geometry[:n]).transform(geometry),
            StandardScaler().fit(compact[:n]).transform(compact)[:, 140:197])
        control, _, _ = ridge_predict(compact[:n], data["train"]["labels"], compact[n:])
        saved = json.loads((HERE / "retained_results/run.json").read_text())
        np.testing.assert_array_equal(confusion_matrix(data["test"]["labels"], control),
                                      saved["confusion_matrices"]["compact368"])
        prediction, _, _ = ridge_predict(geometry[:n], data["train"]["labels"], geometry[n:])
        independent = make_pipeline(StandardScaler(), RidgeClassifier(
            alpha=10, solver="lsqr", tol=1e-12, max_iter=20000))
        independent.fit(geometry[:n], data["train"]["labels"])
        np.testing.assert_array_equal(prediction, independent.predict(geometry[n:]))
    row = {"model": "geometry57", "features": 57, "head": "ridge", "ridge_alpha": 10,
           "accuracy": accuracy_score(data["test"]["labels"], prediction),
           "macro_f1": f1_score(data["test"]["labels"], prediction, average="macro"),
           "errors": int((prediction != data["test"]["labels"]).sum())}
    metadata = {"date": "2026-10-17", "train_windows": n, "test_windows": len(prediction),
                "train_subjects": np.unique(data["train"]["subjects"]).tolist(),
                "test_subjects": np.unique(data["test"]["subjects"]).tolist(),
                "definition": "Compact368 geometry block: 36 invariant moments, 6 cosine moments, 15 standardized P2 trends; no raw/spectral branch or spectral amplitude summaries",
                "settings": {"ridge_alpha": 10, "feature_scaling": "Training-only StandardScaler",
                             "class_weighting": "None", "hyperparameter_search": False},
                "checks": {"compact368_saved_confusion_matrix_matches": True,
                           "columns_and_train_test_scaling_match_compact368": True,
                           "independent_lsqr_predictions_match": True},
                "dataset_sha256": sha256(args.data),
                "source_sha256": {name: sha256(HERE / name) for name in
                                  ("geometry57.py", "retained_features.py", "benchmark.py")},
                "activities": ACTIVITIES,
                "confusion_matrix": confusion_matrix(data["test"]["labels"], prediction).tolist(),
                "result": row, "scope": "Requested standalone ablation on an already inspected benchmark; no representation changes or tuning"}
    args.output.mkdir(parents=True)
    with (args.output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    np.savez_compressed(args.output / "predictions.npz", labels=data["test"]["labels"], predictions=prediction)
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
