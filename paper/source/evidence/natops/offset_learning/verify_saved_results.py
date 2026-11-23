#!/usr/bin/env python3
"""Offline, standard-library checks of hashes, CV selection, predictions and tests."""
import csv
from fractions import Fraction
import hashlib
import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    checked = {}
    manifest = json.loads((ROOT / "artifact_checksums.json").read_text())
    for name, expected in manifest.items():
        checked["hash." + name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    protocol = json.loads((ROOT / "protocol.json").read_text())
    results = json.loads((ROOT / "results.json").read_text())
    checked["protocol_hash"] = hashlib.sha256((ROOT / "protocol.json").read_bytes()).hexdigest() == results["protocol_sha256"]
    models = json.loads((ROOT / "models.json").read_text())
    for model in models:
        pairs = [tuple(pair) for pair in model["pairs"]]
        checked["model." + model["id"]] = (len(pairs) == len(set(pairs))
            and len(pairs) + model["first_levels"] == model["dimension"]
            and all(0 <= i < j < (25 if model["source"] == "ordinary" else 50) for i, j in pairs))
    by_name = {m["id"]: m for m in models}
    checked["clock_control_subset"] = set(map(tuple, by_name["offset_clock_control"]["pairs"])).issubset(
        set(map(tuple, by_name["offset_local"]["pairs"])))
    folds = json.loads((ROOT / "folds.json").read_text())
    checked["fold_partition"] = sorted(i for f in folds for i in f["validation_indices"]) == list(range(180))
    checked["fold_complements"] = all(set(f["training_indices"]) == set(range(180)) - set(f["validation_indices"])
                                       and len(f["training_indices"]) == 144 and len(f["validation_indices"]) == 36 for f in folds)
    cv = json.loads((ROOT / "cv_scores.json").read_text())
    for name, record in cv.items():
        scored = []
        for candidate in record["candidates"]:
            score = sum((Fraction(c, n) for c, n in zip(candidate["fold_correct"], candidate["fold_sizes"])), Fraction(0)) / 5
            assert str(score) == candidate["exact_mean_accuracy"] and float(score) == candidate["mean_accuracy"]
            scored.append((score, candidate["alpha"]))
        score, alpha = max(scored, key=lambda item: (item[0], -item[1]))
        checked["cv." + name] = (alpha == record["selected_alpha"] == results["models"][name]["selected_alpha"]
                                  and float(score) == record["selected_cv_accuracy"])
    rows = list(csv.DictReader((ROOT / "test_predictions.csv").open()))
    checked["row_count_and_order"] = len(rows) == 180 and [int(r["official_test_row_zero_based"]) for r in rows] == list(range(180))
    for name, score in results["models"].items():
        count = sum(r[name] == r["true_label"] for r in rows)
        checked["predictions." + name] = count == score["correct"] and count / 180 == score["accuracy"]
    raw = {}
    for other, expected in results["paired_tests"].items():
        b = sum(r["offset_local"] == r["true_label"] and r[other] != r["true_label"] for r in rows)
        c = sum(r["offset_local"] != r["true_label"] and r[other] == r["true_label"] for r in rows)
        p = min(Fraction(1), 2 * Fraction(sum(comb(b+c, k) for k in range(min(b, c)+1)), 2 ** (b+c)))
        raw[other] = p
        checked["paired." + other] = (b == expected["offset_local_correct_other_wrong"]
                                      and c == expected["offset_local_wrong_other_correct"]
                                      and float(p) == expected["two_sided_exact_mcnemar_p"])
    running = Fraction(0)
    for i, name in enumerate(sorted(raw, key=raw.get)):
        running = max(running, min(Fraction(1), (len(raw)-i) * raw[name]))
        checked["holm." + name] = float(running) == results["paired_tests"][name]["holm_adjusted_p_four_comparisons"]
    if (ROOT / "diagnostic_full_results.json").exists():
        for name, expected in json.loads((ROOT / "diagnostic_full_checksums.json").read_text()).items():
            checked["diagnostic_hash." + name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
        diagnostic = json.loads((ROOT / "diagnostic_full_results.json").read_text())
        diagnostic_rows = list(csv.DictReader((ROOT / "diagnostic_full_predictions.csv").open()))
        checked["diagnostic_row_alignment"] = len(diagnostic_rows) == len(rows) and all(
            a["true_label"] == b["true_label"] and a["official_test_row_zero_based"] == b["official_test_row_zero_based"]
            for a, b in zip(diagnostic_rows, rows))
        for name, record in json.loads((ROOT / "diagnostic_full_cv_scores.json").read_text()).items():
            scored = []
            for candidate in record["candidates"]:
                score = sum((Fraction(c, n) for c, n in zip(candidate["fold_correct"], candidate["fold_sizes"])), Fraction(0)) / 5
                assert str(score) == candidate["exact_mean_accuracy"] and float(score) == candidate["mean_accuracy"]
                scored.append((score, candidate["alpha"]))
            score, alpha = max(scored, key=lambda item: (item[0], -item[1]))
            result = diagnostic["models"][name]
            checked["diagnostic_cv." + name] = (alpha == record["selected_alpha"] == result["selected_alpha"]
                                                and float(score) == record["selected_cv_accuracy"])
            count = sum(r[name] == r["true_label"] for r in diagnostic_rows)
            checked["diagnostic_predictions." + name] = count == result["correct"] and count / 180 == result["accuracy"]
            b = sum(a["offset_full"] == a["true_label"] and d[name] != d["true_label"] for a, d in zip(rows, diagnostic_rows))
            c = sum(a["offset_full"] != a["true_label"] and d[name] == d["true_label"] for a, d in zip(rows, diagnostic_rows))
            pair = diagnostic["descriptive_paired_counts_reference_offset_full"][name]
            checked["diagnostic_pairs." + name] = (b == pair["offset_full_correct_control_wrong"]
                                                   and c == pair["offset_full_wrong_control_correct"])
    report = {"all_checks_passed": all(checked.values()), "checks": checked,
              "scope": "offline saved-output verification; does not refit models"}
    (ROOT / "saved_output_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["all_checks_passed"]:
        raise SystemExit("Saved-output verification failed")


if __name__ == "__main__":
    main()
