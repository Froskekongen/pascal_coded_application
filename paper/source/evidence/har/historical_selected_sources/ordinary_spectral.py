"""Ordinary frequency-clock signatures and an exact Pascal QR projection.

Each scalar spectrum gives a two-dimensional staircase: advance the frequency
clock, then add that bin's probability mass. Both endpoints are fixed at one.
The five variable P3/depth-two Pascal log coordinates are linear readouts of
this ordinary signature through depth five, not information exclusive to coding.
"""

from itertools import product
from time import perf_counter

import numpy as np
import torch
from generators import temporal_code
from tensorsig.classical.log_signature import signature_to_logsignature
from tensorsig.classical.signatures import signature


def _paths(mass: torch.Tensor) -> torch.Tensor:
    """Mass has shape (scalar windows, bins); letters are t=0 and x=1."""
    bins = mass.shape[1]
    path = torch.zeros(len(mass), 2 * bins, 2, dtype=torch.float64)
    path[:, :, 0] = torch.linspace(0, 1, bins, dtype=torch.float64).repeat_interleave(2)
    cumulative = mass.cumsum(1)
    path[:, 1::2, 1] = cumulative
    path[:, 2::2, 1] = cumulative[:, :-1]
    return path


def _word(values: np.ndarray, word: str) -> np.ndarray:
    index = int("".join("0" if letter == "t" else "1" for letter in word), 2)
    return values[:, 2 ** len(word) - 2 + index]


def _pascal_projection(values: np.ndarray, bins: int) -> np.ndarray:
    """Read P3 QR endpoints and orthonormally scaled areas from depth five."""
    location = np.linspace(0, 1, bins)
    monomials = np.stack((np.ones(bins), location, location**2), axis=1)
    basis = temporal_code(bins, 3).matrix.numpy()
    change, *_ = np.linalg.lstsq(monomials, basis, rcond=None)
    moments = np.stack((np.ones(len(values)), _word(values, "tx"),
                        2 * _word(values, "ttx")), axis=1)
    areas = np.stack((_word(values, "xtx"),
                      2 * (_word(values, "txtx") + _word(values, "xttx")),
                      2 * (_word(values, "ttxtx") + _word(values, "txttx"))), axis=1)
    antisymmetric = np.zeros((len(values), 3, 3))
    i, j = np.triu_indices(3, 1)
    antisymmetric[:, i, j] = areas
    antisymmetric[:, j, i] = -areas
    coded_areas = np.einsum("ia,nij,jb->nab", change, antisymmetric, change)
    return np.concatenate((moments @ change, coded_areas[:, i, j] / np.sqrt(2)), axis=1)


def ordinary_spectral_controls(observations: list[torch.Tensor], batch_size: int = 256):
    """Yield ``(label, values, extraction_seconds, metadata)`` for nine controls.

    Input groups follow ``families``: nonnegative bin masses times bin count,
    with unit integrated mass in every scalar channel. Groups and channels
    remain separate. Actual float64 tensorsig signatures are computed once at
    depth five; lower depths share prefixes, so reported extraction times are
    shared, not additive. Tensor outputs discard pure-clock and pure-mass
    words; Lyndon log outputs discard their two constant first levels.

    The final projection contains six columns per scalar spectrum, including
    one known constant, in precisely the order of
    ``spectral_readouts.within_channel_first_areas``. Its 120 columns become
    the existing 100 spectral features after the benchmark's constant mask.
    """
    if not observations or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Expected spectral groups and a positive integer batch size")
    windows = len(observations[0])
    if windows < 1:
        raise ValueError("Expected at least one spectral window")
    start = perf_counter()
    outputs: dict[str, list[np.ndarray]] = {}
    channels_total = 0
    for observation in observations:
        observation = torch.as_tensor(observation, dtype=torch.float64)
        if (observation.ndim != 3 or len(observation) != windows
                or observation.shape[1] < 3 or observation.shape[2] < 1
                or observation.device.type != "cpu" or not torch.isfinite(observation).all()
                or (observation < 0).any()):
            raise ValueError("Expected finite nonnegative CPU spectra with matching windows")
        if not torch.allclose(observation.mean(1), torch.ones_like(observation.mean(1)),
                              atol=1e-8, rtol=1e-7):
            raise ValueError("Spectral groups must have unit integrated mass in every channel")
        _, bins, channels = observation.shape
        channels_total += channels
        mass = observation.permute(0, 2, 1).reshape(-1, bins) / bins
        tensors = []
        scalar_batch = min(batch_size, max(1, 4_000_000 // (2 * bins * 32)))
        with torch.inference_mode():
            for batch in mass.split(scalar_batch):
                tensors.append(signature(_paths(batch), depth=5).numpy().copy())
        tensor = np.concatenate(tensors)

        def add(label: str, values: np.ndarray, channels: int = channels) -> None:
            values = values.reshape(windows, channels * values.shape[1])
            outputs.setdefault(label, []).append(values.astype(np.float32))

        for depth in (2, 3, 4, 5):
            prefix = tensor[:, :2 ** (depth + 1) - 2]
            keep = np.array([len(set(word)) > 1 for level in range(1, depth + 1)
                             for word in product(range(2), repeat=level)])
            add(f"spectral_clock_tensor_d{depth}", prefix[:, keep])
            with torch.inference_mode():
                logs = signature_to_logsignature(torch.from_numpy(prefix), depth).numpy()
            add(f"spectral_clock_log_d{depth}", logs[:, 2:])
        add("spectral_clock_d5_projected_pascal_P3", _pascal_projection(tensor, bins))
    elapsed = perf_counter() - start
    for label, groups in outputs.items():
        values = np.concatenate(groups, axis=1)
        projected = "projected" in label
        metadata = {
            "embedding": "frequency-clock / cumulative-mass staircase",
            "channels": channels_total, "depth": 5 if projected else int(label[-1]),
            "representation": "Pascal QR projection" if projected else label.split("_")[2],
            "features_before_training_mask": values.shape[1],
            "shared_extraction": "one depth-five extraction for all nine spectral controls",
            "known_constants_removed": "none; one per scalar channel remains" if projected
            else "pure clock/mass words (tensor); first-level endpoints (log)",
        }
        yield label, values, elapsed, metadata


def self_check() -> None:
    """Check the ordinary-word identity against the existing coded extractor."""
    from spectral_readouts import signature_readouts

    generator = torch.Generator().manual_seed(713)
    groups = [torch.rand(5, bins, 2, generator=generator, dtype=torch.float64)
              for bins in (9, 13)]
    groups = [values / values.mean(1, keepdim=True) for values in groups]
    outputs = {label: values for label, values, _, _ in ordinary_spectral_controls(groups, 3)}
    coded, _ = signature_readouts(groups)
    np.testing.assert_allclose(outputs["spectral_clock_d5_projected_pascal_P3"],
                               coded["within_channel_first_areas"], atol=1e-7, rtol=1e-6)
    for depth, tensor_width, log_width in ((2, 2, 1), (3, 8, 3), (4, 22, 6), (5, 52, 12)):
        assert outputs[f"spectral_clock_tensor_d{depth}"].shape == (5, 4 * tensor_width)
        assert outputs[f"spectral_clock_log_d{depth}"].shape == (5, 4 * log_width)
    assert all(np.isfinite(values).all() for values in outputs.values())
    for invalid in (torch.zeros_like(groups[0]), torch.full_like(groups[0], float("nan"))):
        try:
            list(ordinary_spectral_controls([invalid]))
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid spectral mass must be rejected")


if __name__ == "__main__":
    torch.set_num_threads(1)
    self_check()
    print("Ordinary spectral checks passed.")
