#!/usr/bin/env python3
"""Fixed two-copy offset-coded NATOPS experiment, designed after ordinary results.

Freeze before data: python run_experiment.py freeze
Run once in a fresh output directory: python run_experiment.py run
No raw dataset is redistributed or cached. See protocol.json for exact choices.
"""
import argparse
import csv
from fractions import Fraction
from importlib.metadata import version
from itertools import combinations
import json
from pathlib import Path
import platform
import sys
import time

import common

ROOT = Path(__file__).resolve().parent
ALPHAS = [10.0 ** k for k in range(-6, 7)]
NAMES = ["ordinary_full", "ordinary_local", "offset_full", "offset_local",
         "moments_only", "moments_plus_ordinary_local", "offset_clock_control"]


def definitions():
    ordinary_local = {(0, j) for j in range(1, 25)}
    for landmark in range(8):
        ordinary_local.update(combinations(range(1 + 3 * landmark, 4 + 3 * landmark), 2))
    ordinary_local = sorted(ordinary_local)
    coded_local = {(i, 25 + i) for i in range(25)}
    for i, j in ordinary_local:
        for p in range(2):
            for q in range(2):
                coded_local.add(tuple(sorted((25 * p + i, 25 * q + j))))
    assert len(ordinary_local) == 48 and len(coded_local) == 217
    clock_control = {pair for pair in coded_local if 0 in pair or 25 in pair}
    clock_control.update(pair for pair in ordinary_local if 0 not in pair)
    assert len(clock_control) == 121 and clock_control.issubset(coded_local)
    models = [
        {"id": "ordinary_full", "source": "ordinary", "first_levels": 25,
         "pairs": list(combinations(range(25), 2)), "dimension": 325},
        {"id": "ordinary_local", "source": "ordinary", "first_levels": 25,
         "pairs": ordinary_local, "dimension": 73},
        {"id": "offset_full", "source": "offset", "first_levels": 50,
         "pairs": list(combinations(range(50), 2)), "dimension": 1275},
        {"id": "offset_local", "source": "offset", "first_levels": 50,
         "pairs": sorted(coded_local), "dimension": 267},
        {"id": "moments_only", "source": "offset", "first_levels": 50,
         "pairs": [], "dimension": 50},
        {"id": "moments_plus_ordinary_local", "source": "offset", "first_levels": 50,
         "pairs": ordinary_local, "dimension": 98},
        {"id": "offset_clock_control", "source": "offset", "first_levels": 50,
         "pairs": sorted(clock_control), "dimension": 171},
    ]
    for model in models:
        assert model["first_levels"] + len(model["pairs"]) == model["dimension"]
    return models


def freeze():
    if (ROOT / "protocol.json").exists():
        raise RuntimeError("Protocol already exists; refusing replacement")
    original = ROOT.parent / "learning"
    primary = json.loads((original / "results.json").read_text())
    clock = json.loads((original / "clock_control_result.json").read_text())
    previous_protocol = json.loads((original / "protocol.json").read_text())
    assert (ROOT / "folds.json").read_bytes() == (original / "folds.json").read_bytes()
    common.save("models.json", definitions())
    frozen = ["run_experiment.py", "common.py", "folds.json", "models.json", "requirements.txt"]
    protocol = {
        "name": "Two-copy position-coded increment signature diagnostic on NATOPS",
        "frozen_utc": common.timestamp(),
        "design_status": "post-primary retrospective diagnostic; all previous ordinary and clock test outcomes known",
        "known_prior_test_correct": {**{k: primary["models"][k]["correct"]
                                           for k in ["first_level", "full", "local", "skeleton", "bilateral"]},
                                     "clock_only": clock["correct"]},
        "scope": "Exactly seven specified representations; fixed P=2; no data-dependent choice of coding order or additional models",
        "pre_outcome_design_amendment": {
            "recorded_utc": common.timestamp(),
            "status": "seventh control added before initial protocol freeze, data access, fitting or any new offset test outcomes",
            "reason": "Theory audit: coded clock areas encode higher spatial moments; first coded moments alone do not isolate weighted spatial interactions",
            "added_model": "offset_clock_control, 171 coordinates:50 first levels,97 coded areas involving either clock copy,24 original within-landmark spatial areas",
            "added_comparison": "offset_local versus offset_clock_control; fixed Holm family increased from3 to4"},
        "data": previous_protocol["data"], "channel_order": previous_protocol["channel_order"],
        "path": previous_protocol["path"],
        "offset": {"P": 2, "R": 51,
                   "z": "51 increments after normalized-time augmentation and prepended all-zero basepoint",
                   "index": "r=0,...,50, including basepoint-to-first-observation increment at r=0",
                   "position": "u_r=r/(R-1)=r/50",
                   "lift": "v_r=(z_r,u_r*z_r)",
                   "order": "code block then original channel; first25 P0, next25 P1",
                   "interpretation": "depth-two piecewise-linear logsignature of the cumulative lifted increments, not multiplication of path coordinates by position"},
        "features": {"all_first_levels_retained": True,
                     "area": "0.5*sum_{r<s}(v_r[i]*v_s[j]-v_r[j]*v_s[i]) for i<j",
                     "ordinary_local_pairs": "24 time-spatial pairs and24 within-landmark XYZ pairs",
                     "offset_local_pairs": "all4 code-copy pairs for every ordinary_local edge plus25 same-original-channel cross-code pairs",
                     "offset_clock_control_pairs": "all97 coded pairs involving either clock copy plus24 code0 within-landmark spatial pairs; subset of offset_local",
                     "constant_or_redundant_coordinates": "Retained identically; dimensions nominal; StandardScaler uses its standard constant-column handling"},
        "learner": previous_protocol["learner"],
        "cv": {"folds": "reuse exactly the previous saved five stratified training folds",
               "seed_of_original_split": 20261003,
               "alpha_grid": ALPHAS,
               "selection": "maximum exact mean validation accuracy; lowest alpha on ties"},
        "test_access_policy": "Within this run, save all seven training-only alpha choices before first test fetch; earlier test scores already known",
        "paired_comparisons": {"reference": "offset_local",
                               "others": ["ordinary_local", "ordinary_full", "moments_plus_ordinary_local", "offset_clock_control"],
                               "test": "two-sided exact McNemar; Holm adjustment across four comparisons",
                               "status": "descriptive retrospective diagnostic; not proof of general superiority"},
        "provenance": {"prior_primary_protocol_sha256": common.digest((original / "protocol.json").read_bytes()),
                       "prior_primary_results_sha256": common.digest((original / "results.json").read_bytes()),
                       "prior_clock_results_sha256": common.digest((original / "clock_control_result.json").read_bytes()),
                       "original_helper_source_sha256": common.digest((original / "run_experiment.py").read_bytes()),
                       "common_py": "Shared loading, path preparation and learner functions copied verbatim from original helper source"},
        "frozen_files": {name: common.digest((ROOT / name).read_bytes()) for name in frozen},
        "freeze_python": sys.version,
    }
    common.save("protocol.json", protocol)
    print(json.dumps({"frozen": True, "protocol_sha256": common.digest((ROOT / "protocol.json").read_bytes()),
                      "dimensions": {m["id"]: m["dimension"] for m in definitions()}}, indent=2), flush=True)


def full_features(z):
    import numpy as np
    prefix = np.cumsum(z, axis=1) - z
    areas = 0.5 * (np.einsum("nti,ntj->nij", prefix, z) - np.einsum("nti,ntj->nij", z, prefix))
    i, j = np.triu_indices(z.shape[-1], 1)
    return np.concatenate([z.sum(axis=1), areas[:, i, j]], axis=1)


def lift(z):
    import numpy as np
    assert z.shape[1:] == (51, 25)
    u = np.arange(51, dtype=np.float64) / 50.0
    return np.concatenate([z, u[None, :, None] * z], axis=2)


def feature_indices(model):
    d = 25 if model["source"] == "ordinary" else 50
    lookup = {pair: d + k for k, pair in enumerate(combinations(range(d), 2))}
    idx = list(range(model["first_levels"])) + [lookup[tuple(pair)] for pair in model["pairs"]]
    assert len(idx) == model["dimension"]
    return idx


def verify_features(z, v, ordinary, offset):
    import numpy as np
    errors = {}
    for name, increments, features in [("ordinary", z, ordinary), ("offset", v, offset)]:
        maximum = 0.0
        d = increments.shape[-1]
        for k in range(8):
            inc = increments[k]
            s = inc.sum(axis=0)
            tensor2 = inc.T @ np.triu(np.ones((len(inc), len(inc))), 1) @ inc + 0.5 * inc.T @ inc
            log2 = tensor2 - 0.5 * np.outer(s, s)
            expected = np.concatenate([s, log2[np.triu_indices(d, 1)]])
            maximum = max(maximum, float(np.max(np.abs(features[k] - expected))))
            np.testing.assert_allclose(features[k], expected, rtol=1e-10, atol=1e-10)
        errors[name] = maximum
    p0 = feature_indices({"source": "offset", "first_levels": 25,
                          "pairs": list(combinations(range(25), 2)), "dimension": 325})
    projection_error = float(np.max(np.abs(ordinary - offset[:, p0])))
    np.testing.assert_allclose(ordinary, offset[:, p0], rtol=1e-12, atol=1e-12)
    moments = np.einsum("r,nri->ni", np.arange(51) / 50., z)
    moment_error = float(np.max(np.abs(offset[:, 25:50] - moments)))
    np.testing.assert_allclose(offset[:, 25:50], moments, rtol=1e-12, atol=1e-12)
    # Coded local extraction projects to ordinary local at code P0.
    models = {m["id"]: m for m in definitions()}
    coded_pairs = {tuple(pair) for pair in models["offset_local"]["pairs"]}
    assert all(tuple(pair) in coded_pairs for pair in models["ordinary_local"]["pairs"])
    assert all((i, 25 + i) in coded_pairs for i in range(25))
    return {"passed": True, "independent_tensor_checks_per_representation": 8,
            "tensor_formula": "sum_{r<s} inc_r tensor inc_s +0.5 sum_r inc_r tensor inc_r, minus0.5*s tensor s",
            "maximum_absolute_tensor_errors": errors,
            "p0_projection_checks": len(z), "p0_projection_maximum_absolute_error": projection_error,
            "coded_first_moment_maximum_absolute_error": moment_error,
            "dimensions": {m["id"]: m["dimension"] for m in definitions()},
            "test_observations_used": False}


def run():
    if (ROOT / "results.json").exists():
        raise RuntimeError("Completed results exist; refusing replacement")
    started = time.monotonic()
    protocol = json.loads((ROOT / "protocol.json").read_text())
    for name, expected in protocol["frozen_files"].items():
        assert common.digest((ROOT / name).read_bytes()) == expected, name
    import numpy as np
    from scipy.stats import binomtest
    from threadpoolctl import threadpool_info, threadpool_limits
    threadpool_limits(limits=1)
    common.save("environment.json", {"python": sys.version, "executable": sys.executable,
                "platform": platform.platform(), "packages": {name: version(name) for name in
                ["numpy", "scipy", "scikit-learn", "threadpoolctl", "joblib", "certifi"]},
                "threadpool_info": threadpool_info()})
    events = [{"event": "run_started", "utc": common.timestamp()}]
    raw = common.fetch_data(protocol["data"]["NATOPS_TRAIN.ts"])
    train, labels = common.parse_data(raw)
    events.append({"event": "training_fetched_and_hash_verified", "utc": common.timestamp()})
    z = common.increments(train)
    v = lift(z)
    features = {"ordinary": full_features(z), "offset": full_features(v)}
    common.save("feature_verification.json", verify_features(z, v, features["ordinary"], features["offset"]))
    models = json.loads((ROOT / "models.json").read_text())
    folds = json.loads((ROOT / "folds.json").read_text())
    assert len(folds) == 5
    diagnostics, choices, estimators = {}, {}, {}
    for model in models:
        name = model["id"]
        x = features[model["source"]][:, feature_indices(model)]
        width = np.ptp(x, axis=0)
        exact_constant = np.flatnonzero(width == 0).tolist()
        near_constant = np.flatnonzero(width <= 1e-12 * np.maximum(1., np.max(np.abs(x), axis=0))).tolist()
        diagnostics[name] = {"nominal_dimension": x.shape[1],
                             "exact_constant_positions": exact_constant,
                             "exact_constant_values": x[0, exact_constant].tolist(),
                             "near_constant_positions_relative_tolerance_1e_12": near_constant,
                             "coordinates_removed": 0}
        candidates, best_score, best_alpha = [], Fraction(-1), None
        for alpha in ALPHAS:
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
        common.save("cv_scores.json", choices)
        print(f"CV complete: {name}, alpha={best_alpha:g}, accuracy={float(best_score):.6f}", flush=True)
    common.save("training_covariate_diagnostics.json", diagnostics)
    events.append({"event": "all_seven_cv_choices_saved_and_refit", "utc": common.timestamp(),
                   "cv_scores_sha256": common.digest((ROOT / "cv_scores.json").read_bytes())})
    common.save("execution_events.json", events)
    # First test access in THIS run; previous ordinary scores were already known.
    raw = common.fetch_data(protocol["data"]["NATOPS_TEST.ts"])
    test, truth = common.parse_data(raw)
    events.append({"event": "test_first_fetched_and_hash_verified_this_run", "utc": common.timestamp()})
    z = common.increments(test)
    test_features = {"ordinary": full_features(z), "offset": full_features(lift(z))}
    predictions, scores = {}, {}
    for model in models:
        name = model["id"]
        pred = estimators[name].predict(test_features[model["source"]][:, feature_indices(model)])
        predictions[name] = pred
        count = int(np.sum(pred == truth))
        scores[name] = {"dimension": model["dimension"], "correct": count, "total": len(truth),
                        "accuracy": count / len(truth), "selected_alpha": choices[name]["selected_alpha"],
                        "selected_cv_accuracy": choices[name]["selected_cv_accuracy"]}
    with (ROOT / "test_predictions.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["official_test_row_zero_based", "true_label"] + NAMES)
        writer.writerows([i, int(label)] + [int(predictions[name][i]) for name in NAMES]
                         for i, label in enumerate(truth))
    paired = {}
    ref = predictions["offset_local"] == truth
    for other in protocol["paired_comparisons"]["others"]:
        baseline = predictions[other] == truth
        b, c = int(np.sum(ref & ~baseline)), int(np.sum(~ref & baseline))
        paired[other] = {"offset_local_correct_other_wrong": b, "offset_local_wrong_other_correct": c,
                         "both_correct": int(np.sum(ref & baseline)), "both_wrong": int(np.sum(~ref & ~baseline)),
                         "two_sided_exact_mcnemar_p": float(binomtest(b, b+c, 0.5).pvalue) if b+c else 1.0}
    running = 0.0
    for i, name in enumerate(sorted(paired, key=lambda n: paired[n]["two_sided_exact_mcnemar_p"])):
        running = max(running, min(1., (len(paired)-i) * paired[name]["two_sided_exact_mcnemar_p"]))
        paired[name]["holm_adjusted_p_four_comparisons"] = running
    events.append({"event": "evaluation_complete", "utc": common.timestamp()})
    common.save("execution_events.json", events)
    result = {"protocol_sha256": common.digest((ROOT / "protocol.json").read_bytes()),
              "completed_utc": common.timestamp(), "elapsed_seconds": time.monotonic()-started,
              "models": scores, "paired_tests": paired,
              "training_cases": len(labels), "test_cases": len(truth),
              "post_primary_retrospective_diagnostic": True,
              "source_study_reproduction": False, "adaptive_models_or_coding_orders": False}
    common.save("results.json", result)
    files = [*protocol["frozen_files"], "protocol.json", "environment.json", "feature_verification.json",
             "cv_scores.json", "training_covariate_diagnostics.json", "execution_events.json",
             "test_predictions.csv", "results.json"]
    common.save("artifact_checksums.json", {name: common.digest((ROOT / name).read_bytes()) for name in files})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run"])
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.command]()
