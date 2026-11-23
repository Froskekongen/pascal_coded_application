#!/usr/bin/env python3
"""Rerun the frozen offset experiment in a fresh directory and compare outcomes."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("offset_natops_reproduced"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--primary-only", action="store_true")
    mode.add_argument("--diagnostics-only", action="store_true")
    args = parser.parse_args()
    target = args.output.resolve()
    if target.exists():
        raise SystemExit("Choose a fresh --output directory; original results are never overwritten.")
    target.mkdir(parents=True)
    checked = {}
    if not args.diagnostics_only:
        protocol = json.loads((ROOT / "protocol.json").read_text())
        for name in [*protocol["frozen_files"], "protocol.json"]:
            shutil.copy2(ROOT / name, target / name)
        subprocess.run([sys.executable, str(target / "run_experiment.py"), "run"], check=True)
        checked.update({name: (ROOT / name).read_bytes() == (target / name).read_bytes()
                        for name in ["models.json", "folds.json", "cv_scores.json", "test_predictions.csv"]})
        before = json.loads((ROOT / "results.json").read_text())
        after = json.loads((target / "results.json").read_text())
        for key in ["models", "paired_tests", "protocol_sha256"]:
            checked["results." + key] = before[key] == after[key]
    if not args.primary_only:
        diagnostic_target = target / "full_controls"
        diagnostic_target.mkdir()
        protocol = json.loads((ROOT / "diagnostic_full_protocol.json").read_text())
        # Use original immutable primary provenance; newly reproduced results
        # have different timestamps and must not replace these pinned inputs.
        for name in [*protocol["frozen_files"], "diagnostic_full_protocol.json"]:
            shutil.copy2(ROOT / name, diagnostic_target / name)
        subprocess.run([sys.executable, str(diagnostic_target / "diagnostic_full_controls.py"), "run"], check=True)
        for name in ["diagnostic_full_cv_scores.json", "diagnostic_full_predictions.csv"]:
            checked[name] = (ROOT / name).read_bytes() == (diagnostic_target / name).read_bytes()
        before = json.loads((ROOT / "diagnostic_full_results.json").read_text())
        after = json.loads((diagnostic_target / "diagnostic_full_results.json").read_text())
        for key in ["models", "descriptive_paired_counts_reference_offset_full", "diagnostic_protocol_sha256"]:
            checked["diagnostic_full_results." + key] = before[key] == after[key]
    report = {"all_checks_passed": all(checked.values()), "checks": checked,
              "note": "Timestamps and elapsed runtime are deliberately excluded from equality checks"}
    (target / "reproduction_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["all_checks_passed"]:
        raise SystemExit("Reproduction differs; inspect saved comparisons and runtime versions")


if __name__ == "__main__":
    main()
