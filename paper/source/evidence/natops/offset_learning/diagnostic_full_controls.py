#!/usr/bin/env python3
"""Two post-outcome full-space controls; no additional hypothesis tests.

The primary seven-model experiment is immutable. These controls were designed
after observing its outcomes to investigate the full offset representation.
"""
import argparse
import csv
from fractions import Fraction
from itertools import combinations
import json
from pathlib import Path
import time

import common
import run_experiment as primary

ROOT = Path(__file__).resolve().parent


def definitions():
    original_pairs = list(combinations(range(25), 2))
    clock_pairs = {pair for pair in combinations(range(50), 2) if 0 in pair or 25 in pair}
    assert len(clock_pairs) == 97
    clock_pairs.update(combinations(range(1, 25), 2))
    assert len(clock_pairs) == 373
    return [
        {"id": "full_moments_plus_ordinary", "source": "offset", "first_levels": 50,
         "pairs": original_pairs, "dimension": 350},
        {"id": "full_coded_clock_control", "source": "offset", "first_levels": 50,
         "pairs": sorted(clock_pairs), "dimension": 423},
    ]


def freeze():
    if (ROOT / "diagnostic_full_protocol.json").exists():
        raise RuntimeError("Diagnostic protocol already exists")
    outcomes = json.loads((ROOT / "results.json").read_text())
    assert json.loads((ROOT / "reproduction_verification.json").read_text())["all_checks_passed"]
    names = ["diagnostic_full_controls.py", "run_experiment.py", "common.py", "folds.json",
             "protocol.json", "models.json", "results.json", "test_predictions.csv", "requirements.txt",
             "environment.json"]
    common.save("diagnostic_full_protocol.json", {
        "frozen_utc": common.timestamp(),
        "status": "post-outcome mechanism diagnostic; all seven offset-experiment outcomes already known",
        "rationale": "Full offset performed well while local offset did not. Test whether added first moments or coded clock terms with all ordinary spatial areas can account for full offset performance.",
        "known_primary_correct_counts": {name: record["correct"] for name, record in outcomes["models"].items()},
        "scope": "Exactly two specified controls, both fixed before any diagnostic fitting; no new coding order, augmentations, alpha grid or selected test winner",
        "models": definitions(),
        "path_and_lift": "Identical to primary frozen protocol:51 prepared increments; v_r=(z_r,(r/50)z_r)",
        "training": "Reuse exact primary folds, fold-local StandardScaler, RidgeClassifier parameters and alpha grid; maximum exact mean accuracy and lowest-alpha ties",
        "test_access": "All prior outcomes known; save BOTH diagnostic CV choices before fetching test observations in this diagnostic run",
        "inference": "descriptive paired prediction counts against saved offset_full only; no new p-values or significance tests",
        "raw_data": "Public URLs and verification hashes in original protocol.json; no redistributed or cached raw data",
        "frozen_files": {name: common.digest((ROOT / name).read_bytes()) for name in names},
    })
    print(json.dumps({"frozen": True, "sha256": common.digest((ROOT / "diagnostic_full_protocol.json").read_bytes()),
                      "dimensions": {m["id"]: m["dimension"] for m in definitions()}}, indent=2), flush=True)


def run():
    if (ROOT / "diagnostic_full_results.json").exists():
        raise RuntimeError("Completed diagnostic already exists")
    protocol = json.loads((ROOT / "diagnostic_full_protocol.json").read_text())
    for name, expected in protocol["frozen_files"].items():
        assert common.digest((ROOT / name).read_bytes()) == expected, name
    import numpy as np
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    started = time.monotonic()
    events = [{"event": "diagnostic_started", "utc": common.timestamp()}]
    original = json.loads((ROOT / "protocol.json").read_text())
    raw = common.fetch_data(original["data"]["NATOPS_TRAIN.ts"])
    train, labels = common.parse_data(raw)
    events.append({"event": "training_fetched_and_hash_verified", "utc": common.timestamp()})
    z = common.increments(train)
    v = primary.lift(z)
    f = primary.full_features(v)
    verification = primary.verify_features(z, v, primary.full_features(z), f)
    verification["diagnostic_dimensions"] = {m["id"]: m["dimension"] for m in protocol["models"]}
    verification["both_controls_literal_subsets_of_offset_full"] = True
    common.save("diagnostic_full_feature_verification.json", verification)
    folds = json.loads((ROOT / "folds.json").read_text())
    choices, estimators = {}, {}
    for model in protocol["models"]:
        name = model["id"]
        idx = primary.feature_indices(model)
        assert len(idx) == len(set(idx)) == model["dimension"] and max(idx) < f.shape[1]
        x = f[:, idx]
        candidates, best_score, best_alpha = [], Fraction(-1), None
        for alpha in primary.ALPHAS:
            counts, sizes = [], []
            for fold in folds:
                a, b = fold["training_indices"], fold["validation_indices"]
                estimator = common.make_pipeline(alpha)
                estimator.fit(x[a], labels[a])
                counts.append(int(np.sum(estimator.predict(x[b]) == labels[b])))
                sizes.append(len(b))
            score = sum((Fraction(c, n) for c, n in zip(counts, sizes)), Fraction(0)) / len(folds)
            candidates.append({"alpha": alpha, "fold_correct": counts, "fold_sizes": sizes,
                               "mean_accuracy": float(score), "exact_mean_accuracy": str(score)})
            if score > best_score:
                best_score, best_alpha = score, alpha
        estimator = common.make_pipeline(best_alpha)
        estimator.fit(x, labels)
        estimators[name] = estimator
        choices[name] = {"dimension": x.shape[1], "selected_alpha": best_alpha,
                         "selected_cv_accuracy": float(best_score), "candidates": candidates}
        common.save("diagnostic_full_cv_scores.json", choices)
        print(f"Diagnostic CV: {name}, alpha={best_alpha:g}, accuracy={float(best_score):.6f}", flush=True)
    events.append({"event": "both_diagnostic_cv_choices_saved_and_refit", "utc": common.timestamp(),
                   "cv_scores_sha256": common.digest((ROOT / "diagnostic_full_cv_scores.json").read_bytes())})
    common.save("diagnostic_full_events.json", events)
    raw = common.fetch_data(original["data"]["NATOPS_TEST.ts"])
    test, truth = common.parse_data(raw)
    events.append({"event": "test_fetched_and_hash_verified_this_diagnostic", "utc": common.timestamp()})
    test_f = primary.full_features(primary.lift(common.increments(test)))
    old_rows = list(csv.DictReader((ROOT / "test_predictions.csv").open()))
    assert [int(row["true_label"]) for row in old_rows] == truth.tolist()
    old_correct = np.array([row["offset_full"] == row["true_label"] for row in old_rows])
    predictions, results, pairs = {}, {}, {}
    for model in protocol["models"]:
        name = model["id"]
        pred = estimators[name].predict(test_f[:, primary.feature_indices(model)])
        predictions[name] = pred
        ok = pred == truth
        count = int(np.sum(ok))
        results[name] = {"dimension": model["dimension"], "correct": count, "total": len(truth),
                         "accuracy": count / len(truth), "selected_alpha": choices[name]["selected_alpha"],
                         "selected_cv_accuracy": choices[name]["selected_cv_accuracy"]}
        pairs[name] = {"offset_full_correct_control_wrong": int(np.sum(old_correct & ~ok)),
                       "offset_full_wrong_control_correct": int(np.sum(~old_correct & ok)),
                       "both_correct": int(np.sum(old_correct & ok)), "both_wrong": int(np.sum(~old_correct & ~ok))}
    names = [m["id"] for m in protocol["models"]]
    with (ROOT / "diagnostic_full_predictions.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["official_test_row_zero_based", "true_label"] + names)
        writer.writerows([i, int(label)] + [int(predictions[name][i]) for name in names]
                         for i, label in enumerate(truth))
    result = {"diagnostic_protocol_sha256": common.digest((ROOT / "diagnostic_full_protocol.json").read_bytes()),
              "completed_utc": common.timestamp(), "elapsed_seconds": time.monotonic() - started,
              "models": results, "descriptive_paired_counts_reference_offset_full": pairs,
              "p_values_computed": False, "post_offset_outcomes_diagnostic": True}
    common.save("diagnostic_full_results.json", result)
    events.append({"event": "diagnostic_complete", "utc": common.timestamp()})
    common.save("diagnostic_full_events.json", events)
    files = ["diagnostic_full_controls.py", "diagnostic_full_protocol.json", "diagnostic_full_cv_scores.json",
             "diagnostic_full_predictions.csv", "diagnostic_full_results.json", "diagnostic_full_events.json",
             "diagnostic_full_feature_verification.json"]
    common.save("diagnostic_full_checksums.json", {name: common.digest((ROOT / name).read_bytes()) for name in files})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run"])
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.command]()
