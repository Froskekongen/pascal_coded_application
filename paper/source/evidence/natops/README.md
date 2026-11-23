# Focused construction and NATOPS evidence

This directory accompanies revision 16. It retains the two-copy NATOPS experiment and the movement illustration from revision 15; it does not copy the earlier graph-only experiment or introduce a new accuracy study.

From this directory, run the offline standard-library checks:

```sh
python3 verify_pascal.py
python3 verify_offset.py
python3 verify_spectral.py
python3 offset_learning/verify_saved_results.py
```

`verify_pascal.py` checks exact polynomial-basis, shift, projection and concatenation identities using rational arithmetic. `verify_offset.py` checks the complete movement-example signatures, the within-segment correction, a depth-three collision, and coordinate counts. These two scripts are unchanged copies from revision 15. `verify_spectral.py` independently checks the QR scale and spectral identities added to the revision 16 companion. The generated JSON files record their scopes and outcomes.

`movement_schedules.pdf` is the revision 15 figure used in the paper. To regenerate it, run `make_movement_figure.py` with NumPy and Matplotlib.

`offset_learning/` contains the frozen seven-model and two-follow-up protocols, their implementation, exact folds, all validation scores, predictions, results, environment and hash manifests. The scripts and frozen evidence were copied byte for byte. Internal `offset` model identifiers refer to the normalized two-copy Pascal lift. The offline verifier recomputes the nine saved scores and their CV selections without downloading data or fitting models.

To rerun the frozen experiment, install its pinned requirements in an appropriate Python environment and use a fresh destination:

```sh
python3 -m pip install -r offset_learning/requirements.txt
python3 offset_learning/reproduce.py --output /absolute/path/to/fresh/natops-reproduction
```

The historical tested environment was Python 3.12.5 with NumPy 2.4.4, SciPy 1.17.1 and scikit-learn 1.8.0. Reproduction downloads the two public NATOPS files from the URLs in `protocol.json`, verifies bytes and hashes, and compares settings, scores and predictions with the frozen records. Raw data are not bundled. Do not invoke the scripts' `freeze` commands: the original protocols are already frozen, and the old design history remains in revision 15. This rebuild ran the offline checks only; it did not rerun training.
