"""Small CPU feature helpers for the exploratory UCI HAR comparison.

Tensorsig computes every signature. At depth two, log coordinates and graph
quotients are exact inexpensive projections of those tensor coordinates.
"""

from itertools import combinations
from math import sqrt
from time import perf_counter

import numpy as np
import torch
from tensorsig.classical.signatures import signature
from tensorsig.offset_coded_signatures.codes import OffsetCode
from tensorsig.offset_coded_signatures.plans import LocalRealization

AUGMENTATIONS = (
    "time", "basepoint", "integrated", "leadlag", "integrated_leadlag", "magnitude",
    "delay2", "delay5",
)


def _standardize(x: torch.Tensor, train_size: int | None) -> torch.Tensor:
    if train_size is None:
        return x
    if not 1 <= train_size <= len(x):
        raise ValueError("train_size must select a nonempty prefix of signals")
    reference = x[:train_size]
    return (x - reference.mean((0, 1))) / reference.std(
        (0, 1), correction=0
    ).clamp_min(1e-8)


def augment(
    signals: np.ndarray, mode: str, train_size: int | None = None,
) -> tuple[torch.Tensor, tuple]:
    """Augment six channels ordered as acceleration xyz, gyroscope xyz.

    All modes append one clock in [0, 1]. ``basepoint`` prepends the origin;
    ``integrated`` uses signal/L as increments (first level = sample means);
    ``leadlag`` doubles sensor channels and prepends the origin;
    ``integrated_leadlag`` integrates first, then doubles the sensor channels;
    ``delay2``/``delay5`` append sensor copies delayed by 2/5 samples, using the
    first sample for the missing past, and prepend the origin;
    ``magnitude`` appends each triad's norm and prepends the origin. Magnitudes supplement
    the axes, so this option is not rotation invariant. If ``train_size`` is
    given, fit channel means/std on the first train_size windows, then apply
    that global affine map to every window. Compute norms before this map.
    This preserves relative posture levels; no windowwise centering is used.

    The returned modality labels define the graph: all pairs within a modality
    and all pairs involving the clock are retained, including lead/lag copies.
    """
    x = torch.as_tensor(signals, dtype=torch.float64)
    if x.ndim != 3 or x.shape[1] < 2 or x.shape[2] != 6:
        raise ValueError("signals must have shape (samples, >=2 points, 6)")
    if not torch.isfinite(x).all() or mode not in AUGMENTATIONS:
        raise ValueError(f"Expected finite signals and mode in {AUGMENTATIONS}")
    groups = ("accel",) * 3 + ("gyro",) * 3
    if mode == "magnitude":
        x = torch.cat((x, x[:, :, :3].norm(dim=2, keepdim=True),
                       x[:, :, 3:].norm(dim=2, keepdim=True)), 2)
        groups += ("accel", "gyro")
    x = _standardize(x, train_size)
    if mode in ("delay2", "delay5"):
        indices = (torch.arange(x.shape[1]) - int(mode[-1])).clamp_min(0)
        x = torch.cat((x, x[:, indices]), 2)
        groups = groups * 2
    if mode in ("integrated", "integrated_leadlag"):
        x = torch.cat((torch.zeros_like(x[:, :1]), x.cumsum(1) / x.shape[1]), 1)
    if mode in ("leadlag", "integrated_leadlag"):
        repeated = x.repeat_interleave(2, dim=1)
        x = torch.cat((repeated[:, 1:], repeated[:, :-1]), 2)
        groups = groups * 2
    clock = torch.linspace(0, 1, x.shape[1], dtype=x.dtype)
    paths = torch.cat((x, clock[None, :, None].expand(len(x), -1, 1)), 2)
    if mode in ("basepoint", "leadlag", "magnitude", "delay2", "delay5"):
        paths = torch.cat((torch.zeros_like(paths[:, :1]), paths), 1)
    return paths, groups + ("time",)


def direct_covariation(
    signals: np.ndarray, train_size: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Cheap moment/roughness controls and a shared 42-coordinate append block.

    Fit one affine scaling per channel on training windows only. For the scaled
    samples x, M = mean(x x^T), Cov = M - mean(x) mean(x)^T, and
    Q = mean(diff(x) diff(x)^T). Use each symmetric matrix's upper triangle,
    including its diagonal. Q is normalized sampled roughness, not a claim
    about a continuous-time stochastic quadratic variation.

    Return three readouts (27, 48, and 54 coordinates), plus [upper(M), upper(Q)]
    for supplementing ordinary/offset signatures without duplicating means.
    """
    x = torch.as_tensor(signals, dtype=torch.float64)
    if x.ndim != 3 or x.shape[1] < 2 or x.shape[2] != 6 or not torch.isfinite(x).all():
        raise ValueError("signals must be finite with shape (samples, >=2 points, 6)")
    x = _standardize(x, train_size).numpy()
    mean = x.mean(1)
    moment = np.einsum("nti,ntj->nij", x, x) / x.shape[1]
    centered = x - mean[:, None, :]
    covariance = np.einsum("nti,ntj->nij", centered, centered) / x.shape[1]
    std = np.sqrt(covariance.diagonal(axis1=1, axis2=2))
    increments = np.diff(x, axis=1)
    roughness = np.einsum("nti,ntj->nij", increments, increments) / increments.shape[1]
    upper = np.triu_indices(x.shape[2])
    moment, covariance, roughness = (
        matrix[:, upper[0], upper[1]] for matrix in (moment, covariance, roughness)
    )
    features = {
        "mean_second_moment": np.concatenate((mean, moment), 1).astype(np.float32),
        "mean_second_moment_roughness": np.concatenate(
            (mean, moment, roughness), 1
        ).astype(np.float32),
        "mean_std_covariation": np.concatenate(
            (mean, std, covariance, roughness), 1
        ).astype(np.float32),
    }
    return features, np.concatenate((moment, roughness), 1).astype(np.float32)


def _coordinates(values: torch.Tensor, groups: tuple) -> dict[str, torch.Tensor]:
    """Project depth-two tensors; nonedge words commute, they are not deleted."""
    width = len(groups)
    first, second = values[:, :width], values[:, width:].reshape(-1, width, width)
    pairs = tuple(combinations(range(width), 2))
    edges = tuple((i, j) for i, j in pairs
                  if groups[i] == groups[j] or "time" in (groups[i], groups[j]))
    nonedges = tuple(pair for pair in pairs if pair not in edges)
    result = {"full_signature": values}
    for name, selected in (("full", pairs), ("modality", edges)):
        left, right = zip(*selected)
        areas = (second[:, left, right] - second[:, right, left]) / 2
        result[f"{name}_logsignature"] = torch.cat((first, areas), 1)
    left, right = zip(*edges)
    nl, nr = zip(*nonedges) if nonedges else ((), ())
    # Orthonormal tensor quotient: [i,j]=0 identifies ij and ji off the graph.
    result["modality_signature"] = torch.cat(
        (first, second.diagonal(dim1=1, dim2=2), second[:, left, right],
         second[:, right, left], (second[:, nl, nr] + second[:, nr, nl]) / sqrt(2)),
        1,
    )
    return result


@torch.inference_mode()
def extract_features(
    paths: torch.Tensor, modalities: tuple, batch_size: int = 256,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Return eight ordinary/offset, full/modality, tensor/log feature matrices.

    Offset increments are dx[r,c] * (1, r/(R-1)), channel-major. R includes an
    added basepoint segment, when present. Every channel, including the clock,
    is coded. This explicit matrix code makes no temporal-ideal claim.

    Extraction times include the shared full signature and all four projections;
    related variants share this timing, so these costs are not additive. Graph
    quotients reduce readout size here; extraction still computes full tensors.
    """
    if paths.ndim != 3 or paths.shape[1] < 3 or paths.shape[2] != len(modalities):
        raise ValueError("Expected paths with >=3 points and one modality per channel")
    if paths.device.type != "cpu" or batch_size < 1 or not torch.isfinite(paths).all():
        raise ValueError("Expected finite CPU paths and positive batch_size")
    steps, channels = paths.shape[1] - 1, paths.shape[2]
    phase = torch.linspace(0, 1, steps, dtype=paths.dtype)
    code = OffsetCode(torch.stack((torch.ones_like(phase), phase), 1))
    plan = LocalRealization(code, channels=channels, max_depth=2).signature(
        coordinates="dense"
    )
    lifted_groups = tuple(group for group in modalities for _ in range(2))
    chunks, timings = {}, {}
    for batch in paths.split(batch_size):
        for kind, groups in (("ordinary", modalities), ("offset", lifted_groups)):
            start = perf_counter()
            values = (signature(batch, depth=2) if kind == "ordinary" else
                      plan.evaluate(batch.diff(dim=1)[:, None])[:, 0])
            features = _coordinates(values, groups)
            elapsed = perf_counter() - start
            for name, values in features.items():
                key = f"{kind}_{name}"
                chunks.setdefault(key, []).append(values.float().numpy().copy())
                timings[key] = timings.get(key, 0.0) + elapsed
    return {key: np.concatenate(parts) for key, parts in chunks.items()}, timings


@torch.inference_mode()
def extract_coded_features(
    increments: torch.Tensor, code: OffsetCode, batch_size: int = 256,
    *, depth: int = 2,
) -> tuple[np.ndarray, float]:
    """Compute dense coded tensors, with float64 internal arithmetic.

    Increments have shape (windows, steps, channels); output is float32 in
    channel-major code order. The elapsed time includes plan construction.
    """
    start = perf_counter()
    increments = torch.as_tensor(increments, dtype=torch.float64)
    if (increments.ndim != 3 or increments.device.type != "cpu"
            or not torch.isfinite(increments).all() or batch_size < 1):
        raise ValueError("Expected finite CPU increment windows and positive batch_size")
    plan = LocalRealization(code, channels=increments.shape[2], max_depth=depth).signature(
        coordinates="dense"
    )
    values = np.concatenate([
        plan.evaluate(batch[:, None])[:, 0].float().numpy().copy()
        for batch in increments.split(batch_size)
    ])
    return values, perf_counter() - start


def extract_fourier_features(
    signals: np.ndarray, train_size: int | None, batch_size: int = 256,
    frequencies: tuple[int, ...] = (2, 4),
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Fixed Fourier codes on integrated samples, by default at DFT bins 2 and 4.

    The increment is x[r]/N, coded by (1, cos(2*pi*2*r/N), sin(...),
    cos(2*pi*4*r/N), sin(...)). First levels therefore give means and normalized
    DFT coefficients (imaginary part = minus the sine coordinate). For UCI's
    128 samples at 50 Hz the frequencies are 0.78125 and 1.5625 Hz.

    Keep first levels, their phase-invariant cross-spectra, and full tensor/log
    readouts. Cross-spectra contain every channel power plus real and imaginary
    cross-channel products: 6 means + 2 * (21 real + 15 imaginary) = 78 features.
    All readouts share extraction timing; no clock is coded or appended.
    """
    start = perf_counter()
    x = torch.as_tensor(signals, dtype=torch.float64)
    if x.ndim != 3 or x.shape[1] < 3 or x.shape[2] != 6 or not torch.isfinite(x).all():
        raise ValueError("signals must be finite with shape (samples, >=3 points, 6)")
    if (not frequencies or len(set(frequencies)) != len(frequencies)
            or any(not isinstance(k, int) or k <= 0 or 2 * k >= x.shape[1]
                   for k in frequencies)):
        raise ValueError("frequencies must be distinct positive DFT bins below Nyquist")
    x = _standardize(x, train_size)
    phase = torch.arange(x.shape[1], dtype=x.dtype) / x.shape[1]
    angle = 2 * torch.pi * phase[:, None] * torch.tensor(frequencies, dtype=x.dtype)
    matrix = torch.cat((torch.ones_like(phase[:, None]),
                        torch.stack((angle.cos(), angle.sin()), 2).flatten(1)), 1)
    values, _ = extract_coded_features(x / x.shape[1], OffsetCode(matrix), batch_size)
    width = 6 * (1 + 2 * len(frequencies))
    first = values[:, :width].reshape(-1, 6, 1 + 2 * len(frequencies))
    coefficients = first[:, :, 1::2] - 1j * first[:, :, 2::2]
    upper, strict = np.triu_indices(6), np.triu_indices(6, 1)
    spectra = [first[:, :, 0]]
    for frequency in range(len(frequencies)):
        z = coefficients[:, :, frequency]
        cross = z[:, :, None] * z[:, None, :].conj()
        spectra.extend((cross.real[:, upper[0], upper[1]],
                        cross.imag[:, strict[0], strict[1]]))
    second = values[:, width:].reshape(-1, width, width)
    left, right = np.triu_indices(width, 1)
    features = {
        "fourier_first_level": values[:, :width].copy(),
        "fourier_cross_spectrum": np.concatenate(spectra, 1),
        "fourier_full_signature": values,
        "fourier_full_logsignature": np.concatenate(
            (values[:, :width], (second[:, left, right] - second[:, right, left]) / 2), 1
        ),
    }
    elapsed = perf_counter() - start
    return features, dict.fromkeys(features, elapsed)


def self_check() -> None:
    """Runnable algebra/augmentation check, small enough to run before a benchmark."""
    generator = torch.Generator().manual_seed(42)
    signals = torch.randn(3, 7, 6, generator=generator, dtype=torch.float64).numpy()
    paths, groups = augment(signals, "time")
    features, _ = extract_features(paths, groups, batch_size=2)
    shifted_paths, _ = augment(signals + 4, "time")
    shifted, _ = extract_features(shifted_paths, groups)
    for key in features:
        np.testing.assert_allclose(features[key], shifted[key], atol=1e-6)
    based, _ = augment(signals, "basepoint")
    based_shifted, _ = augment(signals + 4, "basepoint")
    assert not torch.allclose(signature(based, 2), signature(based_shifted, 2))
    # Anchored offset tensors recover means/M linearly via shuffle products.
    anchored, _ = extract_features(based, groups)
    tensor = anchored["offset_full_signature"]
    first, second = tensor[:, :14], tensor[:, 14:].reshape(-1, 14, 14)
    products = second + second.transpose(0, 2, 1)
    areas = (second - second.transpose(0, 2, 1)) / 2
    cross_products = products[:, :12:2, 1:12:2]
    cross_areas = areas[:, :12:2, 1:12:2]
    factor = (signals.shape[1] - 1) / signals.shape[1]
    recovered_mean = first[:, :12:2] - factor * first[:, 1:12:2]
    recovered_moment = products[:, :12:2, :12:2] - factor * (
        (cross_products + cross_products.transpose(0, 2, 1)) / 2
        + cross_areas + cross_areas.transpose(0, 2, 1)
    )
    np.testing.assert_allclose(recovered_mean, signals.mean(1), atol=1e-6)
    np.testing.assert_allclose(
        recovered_moment, np.einsum("nti,ntj->nij", signals, signals) / signals.shape[1],
        atol=1e-6,
    )
    integrated, _ = augment(signals, "integrated")
    np.testing.assert_allclose(integrated[:, -1, :6].numpy(), signals.mean(1))
    for delay in (2, 5):
        delayed, _ = augment(signals, f"delay{delay}")
        np.testing.assert_array_equal(delayed[:, 1:, :6].numpy(), signals)
        indices = np.maximum(np.arange(signals.shape[1]) - delay, 0)
        np.testing.assert_array_equal(delayed[:, 1:, 6:12].numpy(), signals[:, indices])
        torch.testing.assert_close(delayed[:, 0], torch.zeros_like(delayed[:, 0]))
    dx = paths.diff(dim=1)
    phase = torch.linspace(0, 1, dx.shape[1], dtype=dx.dtype)
    lifted = (dx[..., None] * torch.stack((torch.ones_like(phase), phase), 1)
              [None, :, None, :]).flatten(-2)
    lifted_path = torch.cat((torch.zeros_like(lifted[:, :1]), lifted.cumsum(1)), 1)
    np.testing.assert_allclose(features["offset_full_signature"],
                               signature(lifted_path, 2).numpy(), rtol=1e-6, atol=1e-6)
    assert features["ordinary_full_logsignature"].shape[1] == 28
    assert features["ordinary_modality_logsignature"].shape[1] == 19
    assert features["offset_full_logsignature"].shape[1] == 105
    assert features["offset_modality_logsignature"].shape[1] == 69
    # Swapping cross-modality events changes full signatures but not the quotient.
    events = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [1., 1., 0.]],
                           [[0., 0., 0.], [0., 1., 0.], [1., 1., 0.]]])
    projected = _coordinates(signature(events, 2), ("accel", "gyro", "time"))
    assert not torch.equal(projected["full_signature"][0], projected["full_signature"][1])
    for name in ("modality_signature", "modality_logsignature"):
        torch.testing.assert_close(projected[name][0], projected[name][1])
    for mode in AUGMENTATIONS:
        augmented, labels = augment(signals, mode)
        assert len(labels) == augmented.shape[-1] and torch.isfinite(augmented).all()
    leadlag, _ = augment(signals, "leadlag")
    second = signature(leadlag, 2)[:, 13:].reshape(-1, 13, 13)
    cross_areas = (second - second.transpose(1, 2))[:, :6, 6:12]
    increments = np.diff(signals, axis=1)
    np.testing.assert_allclose(
        ((cross_areas + cross_areas.transpose(1, 2)) / 2).numpy(),
        np.einsum("nti,ntj->nij", increments, increments), atol=1e-12,
    )
    integrated_leadlag, _ = augment(signals, "integrated_leadlag", train_size=2)
    scaled = ((signals - signals[:2].mean((0, 1))) /
              signals[:2].std((0, 1)))
    assert integrated_leadlag.shape[1] == 2 * signals.shape[1] + 1
    np.testing.assert_allclose(integrated_leadlag[:, -1, :6].numpy(), scaled.mean(1))
    np.testing.assert_allclose(integrated_leadlag[:, -1, 6:12].numpy(), scaled.mean(1))
    second = signature(integrated_leadlag, 2)[:, 13:].reshape(-1, 13, 13)
    cross_areas = (second - second.transpose(1, 2))[:, :6, 6:12]
    moments = ((cross_areas + cross_areas.transpose(1, 2)) / 2).numpy() * signals.shape[1]
    direct, block = direct_covariation(signals, train_size=2)
    assert [values.shape[1] for values in direct.values()] == [27, 48, 54]
    assert block.shape == (len(signals), 42)
    upper = np.triu_indices(6)
    for row, window in enumerate(scaled):
        moment = window.T @ window / len(window)
        centered = window - window.mean(0)
        covariance = centered.T @ centered / len(window)
        delta = np.diff(window, axis=0)
        roughness = delta.T @ delta / len(delta)
        np.testing.assert_allclose(moments[row], moment, atol=1e-12)
        np.testing.assert_allclose(block[row], np.r_[moment[upper], roughness[upper]],
                                   atol=1e-6)
        np.testing.assert_allclose(
            direct["mean_second_moment"][row], np.r_[window.mean(0), moment[upper]],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            direct["mean_second_moment_roughness"][row], np.r_[window.mean(0), block[row]],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            direct["mean_std_covariation"][row],
            np.r_[window.mean(0), window.std(0), covariance[upper], roughness[upper]],
            atol=1e-6,
        )
    constant = np.broadcast_to(np.arange(18).reshape(3, 1, 6), signals.shape).copy()
    constant_direct, constant_block = direct_covariation(constant, train_size=2)
    np.testing.assert_allclose(constant_block[:, 21:], 0, atol=1e-12)
    np.testing.assert_allclose(constant_direct["mean_std_covariation"][:, 6:], 0,
                               atol=1e-12)
    constant_path, _ = augment(constant, "integrated_leadlag")
    np.testing.assert_allclose(constant_path[:, -1, :6].numpy(), constant[:, 0])
    changed = signals.copy()
    changed[2] += 100
    for mode in AUGMENTATIONS:
        before, _ = augment(signals, mode, train_size=2)
        after, _ = augment(changed, mode, train_size=2)
        torch.testing.assert_close(before[:2], after[:2])
    changed_direct, changed_block = direct_covariation(changed, train_size=2)
    np.testing.assert_allclose(block[:2], changed_block[:2])
    for name in direct:
        np.testing.assert_allclose(direct[name][:2], changed_direct[name][:2])
    # Fourier first levels are normalized DFT coefficients of integrated samples.
    samples = torch.randn(3, 16, 6, generator=generator, dtype=torch.float64).numpy()
    fourier, _ = extract_fourier_features(samples, train_size=2, batch_size=2)
    assert [value.shape[1] for value in fourier.values()] == [30, 78, 930, 465]
    scaled = (samples - samples[:2].mean((0, 1))) / samples[:2].std((0, 1))
    first = fourier["fourier_first_level"].reshape(-1, 6, 5)
    expected = np.fft.rfft(scaled, axis=1)[:, [2, 4]].transpose(0, 2, 1) / len(samples[0])
    np.testing.assert_allclose(first[:, :, 0], scaled.mean(1), atol=1e-7)
    np.testing.assert_allclose(first[:, :, 1::2] - 1j * first[:, :, 2::2],
                               expected, atol=1e-7)
    shifted, _ = extract_fourier_features(np.roll(samples, 3, axis=1), train_size=2)
    np.testing.assert_allclose(fourier["fourier_cross_spectrum"],
                               shifted["fourier_cross_spectrum"], atol=1e-6)


if __name__ == "__main__":
    self_check()
    print("UCI HAR feature checks passed.")
