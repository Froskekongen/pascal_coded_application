"""The three retained UCI HAR representations; see uci_har.md for the mathematics.

Inputs are windows of body acceleration, gyroscope and total acceleration,
each in xyz order. Every fitted scaling uses the training prefix only.
Tensorsig evaluates the Pascal signatures and selected value–clock graph words.
"""

from time import perf_counter

import numpy as np
import torch
from tensorsig.offset_coded_signatures.codes import OffsetCode
from tensorsig.offset_coded_signatures.plans import LocalRealization

TRIAD_PAIRS = ((0, 1), (0, 2), (1, 2))


def standardize_axes(x: torch.Tensor, n_train: int) -> torch.Tensor:
    """Preserve window levels using one training mean/std per sensor axis."""
    train = x[:n_train]
    return (x - train.mean((0, 1))) / train.std((0, 1), correction=0).clamp_min(1e-8)


def pascal_code(samples: int, columns: int) -> OffsetCode:
    return OffsetCode.pascal(samples, columns, orthonormal=True, dtype=torch.float64)


def pascal_signature(observations: torch.Tensor, batch_size: int) -> np.ndarray:
    """P3 QR, depth 2, integrating observations/N without an extra clock.

    Letter order is channel, then code column. Signature arithmetic is float64;
    storage is float32, as in the retained experiments.
    """
    samples, channels = observations.shape[1:]
    plan = LocalRealization(pascal_code(samples, 3), channels=channels, max_depth=2).signature(
        coordinates="dense"
    )
    return np.concatenate([
        plan.evaluate((batch / samples)[:, None])[:, 0].float().numpy().copy()
        for batch in observations.split(batch_size)
    ])


def raw_logsignature(tensor: np.ndarray, width: int = 18) -> np.ndarray:
    """Endpoints and independent antisymmetric depth-two coordinates.

    Six channels give 18 letters; nine channels give 27. Areas use the raw
    block's /2 convention, also used by the optional joint spectral block.
    """
    values = torch.as_tensor(tensor)
    second = values[:, width:].reshape(-1, width, width)
    left, right = np.triu_indices(width, 1)
    return torch.cat((values[:, :width], (second[:, left, right] - second[:, right, left]) / 2), 1).numpy()


def spectral_logsignature(mass: torch.Tensor, batch_size: int) -> np.ndarray:
    """Six coordinates per scalar spectrum; its constant endpoint is removed later.

    Compute each four-channel group's tensor, then retain only within-channel
    endpoints and areas. The fixed sqrt(2) area convention matches the experiment.
    """
    values = pascal_signature(mass * mass.shape[1], batch_size)
    channels = mass.shape[2]
    width = channels * 3
    first, second = values[:, :width], values[:, width:].reshape(-1, width, width)
    left, right = np.triu_indices(3, 1)
    parts = []
    for channel in range(channels):
        indices = slice(3 * channel, 3 * (channel + 1))
        block = second[:, indices, indices]
        parts.append(np.concatenate((first[:, indices],
                                     (block[:, left, right] - block[:, right, left]) / np.sqrt(2)), 1))
    return np.concatenate(parts, 1)


def graph_moments(z: torch.Tensor, columns: int, power: int, batch_size: int) -> tuple[np.ndarray, float]:
    """Selected closed value–clock log words, evaluated by a polynomial lift.

    The graph moves to value z[k], advances clock a by q[k,a]/N, and closes
    the value axis. Its log word x^r t_a is mean(z^r q_a)/r!. Return the
    factorial-unscaled moments and the constant first QR column's value.
    Only powers 1..3 and one/two clock columns are needed by the retained model.
    """
    windows, samples, channels = z.shape
    code = pascal_code(samples, columns)
    plan = LocalRealization(code, channels=channels * power, max_depth=1).signature(coordinates="dense")
    parts = []
    for batch in z.split(batch_size):
        lifted = torch.stack([batch**r for r in range(1, power + 1)], -1).flatten(2)
        parts.append(plan.evaluate((lifted / samples)[:, None])[:, 0].numpy().copy())
    moments = np.concatenate(parts).reshape(windows, channels, power, columns)
    return moments, float(code.matrix[0, 0])


def calibrated_graph_features(z: torch.Tensor, *, skewness: bool, trend: bool,
                              batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std/(skewness), and optionally one standardized P2 location moment.

    Calibration is explicit: square roots and variance normalization are not
    linear signature readouts. Near-constant streams have zero shape/trend.
    """
    first, scale = graph_moments(z, 1, 1, batch_size)
    mean = first[:, :, 0, 0] / scale
    centered = z - torch.as_tensor(mean)[:, None]
    moments, scale = graph_moments(centered, 2, 3, batch_size)
    variance = np.maximum(moments[:, :, 1, 0] / scale, 0)
    sd = np.sqrt(variance)
    normalized = moments / np.maximum(sd, 1e-8)[:, :, None, None] ** np.arange(1, 4)[None, None, :, None]
    normalized = np.where((variance > 1e-12)[:, :, None, None], normalized, 0)
    skew = normalized[:, :, 2, 0] / scale
    statistics = np.stack((mean, sd, skew), -1)[:, :, :3 if skewness else 2].reshape(len(z), -1)
    trends = normalized[:, :, 0, 1] if trend else np.empty((len(z), 0))
    return statistics, trends


def moment_summary(x: torch.Tensor) -> torch.Tensor:
    """Mean, std, absolute mean, RMS, skewness, kurtosis, min and max per channel."""
    mean = x.mean(1)
    centered = x - mean[:, None]
    variance = centered.square().mean(1)
    z = centered / variance.sqrt().clamp_min(1e-8)[:, None]
    skew = torch.where(variance > 1e-12, z.pow(3).mean(1), 0.)
    kurtosis = torch.where(variance > 1e-12, z.pow(4).mean(1), 0.)
    return torch.cat((mean, variance.sqrt(), x.abs().mean(1), x.square().mean(1).sqrt(),
                      skew, kurtosis, x.amin(1), x.amax(1)), 1)


def spectrum(vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized power, bin amplitudes and the retained 72-feature pool.

    Append the triad norm before the FFT. One-sided bin energies sum to the
    time-domain mean square (Parseval). Frequencies are fractions of Nyquist.
    """
    x = torch.cat((vector, vector.norm(dim=2, keepdim=True)), 2)
    power = (torch.fft.rfft(x, dim=1).abs() / x.shape[1]).square()
    power[:, 1:power.shape[1] - (x.shape[1] % 2 == 0)] *= 2
    energy = power.sum(1)
    mass = power / energy.clamp_min(1e-20)[:, None]
    if not torch.allclose(mass.sum(1), torch.ones_like(energy), atol=1e-8, rtol=1e-7):
        raise ValueError("Every retained spectral channel must have positive energy")
    frequency = 2 * torch.fft.rfftfreq(x.shape[1], dtype=x.dtype)
    mean = (mass * frequency[None, :, None]).sum(1)
    spread = (mass * (frequency[None, :, None] - mean[:, None]).square()).sum(1).sqrt()
    entropy = -(mass * mass.clamp_min(1e-20).log()).sum(1)
    peak = frequency[power.argmax(1)]
    bands = torch.cat([
        mass[:, (frequency >= band / 8) &
             (frequency < (band + 1) / 8 if band < 7 else frequency <= 1)].sum(1)
        for band in range(8)
    ], 1)
    amplitude = power.sqrt()
    summary = moment_summary(amplitude)
    amplitude_statistics = torch.cat((summary[:, :8], summary[:, 16:24]), 1)
    pooled = torch.cat((amplitude_statistics, energy, energy.log1p(), mean, spread, entropy, peak, bands), 1)
    return mass, amplitude, pooled


def rotation_streams(vectors: list[torch.Tensor], n_train: int) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Energies, simultaneous/lagged dot products, and cross-triad cosines.

    One uncentered training RMS scalar per triad preserves invariance under a
    common orthogonal rotation. Lags 2 and 5 use only valid sample pairs.
    """
    scaled = [v / v[:n_train].square().mean().sqrt().clamp_min(1e-8) for v in vectors]
    energy = torch.stack([v.square().sum(2) for v in scaled], 2)
    dots = torch.stack([(scaled[i] * scaled[j]).sum(2) for i, j in TRIAD_PAIRS], 2)
    streams = [energy, dots]
    for lag in (2, 5):
        streams.append(torch.stack([(v[:, lag:] * v[:, :-lag]).sum(2) for v in scaled], 2))
    norms = energy.sqrt()
    denominator = torch.stack([norms[:, :, i] * norms[:, :, j] for i, j in TRIAD_PAIRS], 2)
    return streams, dots / denominator.clamp_min(1e-12)


@torch.inference_mode()
def build_features(signals: np.ndarray, n_train: int, engineered: np.ndarray,
                   batch_size: int = 256, *, all_channels: bool = False) -> tuple[dict[str, np.ndarray], dict]:
    """Build the retained models and optionally five nine-channel variants.

    Known constant coordinates remain here. The runner removes columns using
    training rows only, after concatenation, exactly as in the retained runs.
    The default reproduces compact368, pool803 and uci903 unchanged.
    """
    started = perf_counter()
    x = torch.as_tensor(signals, dtype=torch.float64)
    if (x.ndim != 3 or x.shape[1:] != (128, 9) or not 1 <= n_train <= len(x)
            or not torch.isfinite(x).all() or engineered.shape != (len(x), 561)
            or not np.isfinite(engineered).all() or batch_size < 1):
        raise ValueError("Expected finite UCI windows (N,128,9), features (N,561), training prefix and positive batch size")

    tick = perf_counter()
    raw = standardize_axes(x[:, :, [6, 7, 8, 3, 4, 5]], n_train)
    raw_tensor = pascal_signature(raw, batch_size)
    raw_log = raw_logsignature(raw_tensor)
    raw_seconds = perf_counter() - tick

    # Body acceleration, inferred gravity, and angular velocity. Differences
    # supplement these with two dynamic streams; norms give 20 scalar spectra.
    vectors = [x[:, :, :3], x[:, :, 6:9] - x[:, :, :3], x[:, :, 3:6]]
    spectral_vectors = [standardize_axes(v, n_train) for v in vectors]
    spectral_vectors += [standardize_axes(v.diff(dim=1), n_train) for v in (vectors[0], vectors[2])]
    spectral_logs, spectral_pools, amplitude_stats = [], [], []
    for vector in spectral_vectors:
        mass, amplitude, pool = spectrum(vector)
        spectral_logs.append(spectral_logsignature(mass, batch_size))
        spectral_pools.append(pool)
        statistics, _ = calibrated_graph_features(amplitude, skewness=False, trend=False, batch_size=batch_size)
        amplitude_stats.append(statistics)

    streams, cosines = rotation_streams(vectors, n_train)
    cosine_stats, cosine_trends = calibrated_graph_features(cosines, skewness=False, trend=True, batch_size=batch_size)
    invariant_stats, invariant_trends = [], []
    for stream in streams:
        statistics, trends = calibrated_graph_features(stream, skewness=True, trend=True, batch_size=batch_size)
        invariant_stats.append(statistics)
        invariant_trends.append(trends)

    spectral_log = np.concatenate(spectral_logs, 1)
    calibrated82 = np.concatenate([*amplitude_stats, cosine_stats, *invariant_stats], 1)
    trends15 = np.concatenate([cosine_trends, *invariant_trends], 1)
    spectral_pool = torch.cat(spectral_pools, 1).float().numpy()
    rotation_pool = torch.cat([*(moment_summary(stream) for stream in streams),
                               cosines.mean(1), cosines.std(1, correction=0)], 1).float().numpy()
    models = {
        "compact368": np.concatenate((spectral_log, calibrated82, trends15, raw_log), 1),
        "pool803": np.concatenate((spectral_pool, rotation_pool, raw_tensor), 1),
        "uci903": np.concatenate((engineered, raw_tensor), 1),
    }
    preparation = {"raw_preparation_s": raw_seconds, "raw_tensor": 342, "raw_log": 171,
                   "spectral_log_before_constant_removal": 120, "amplitude_moments": 40,
                   "cosine_moments": 6, "invariant_moments": 36, "graph_trends": 15,
                   "spectral_pool_before_constant_removal": 360, "rotation_pool": 102}
    if all_channels:
        tick = perf_counter()
        raw9_tensor = pascal_signature(standardize_axes(x, n_train), batch_size)
        raw9_log = raw_logsignature(raw9_tensor, width=27)
        # Supplied axes only, ordered body acceleration, gyroscope, total
        # acceleration. spectrum appends a norm; [:, :, :3] excludes it.
        mass9 = torch.cat([spectrum(standardize_axes(x[:, :, i:i + 3], n_train))[0][:, :, :3]
                           for i in (0, 3, 6)], 2)
        spectral9_factorized = spectral_logsignature(mass9, batch_size)
        spectral9_joint = raw_logsignature(pascal_signature(mass9 * mass9.shape[1], batch_size), width=27)
        models.update({
            "compact_raw9": np.concatenate((spectral_log, calibrated82, trends15, raw9_log), 1),
            "pool_raw9": np.concatenate((spectral_pool, rotation_pool, raw9_tensor), 1),
            "uci_raw9": np.concatenate((engineered, raw9_tensor), 1),
            "compact_raw9_spectral9_factorized": np.concatenate((spectral9_factorized, calibrated82, trends15, raw9_log), 1),
            "compact_raw9_spectral9_joint": np.concatenate((spectral9_joint, calibrated82, trends15, raw9_log), 1),
        })
        preparation["all_channels"] = {
            "preparation_s": perf_counter() - tick,
            "channels": [sensor + "_" + axis for sensor in ("body_acc", "body_gyro", "total_acc") for axis in "xyz"],
            "raw_definition": "Training-standardized supplied nine axes; P3 QR, depth 2, observations/128, no extra clock",
            "raw_tensor": raw9_tensor.shape[1], "raw_log": raw9_log.shape[1],
            "spectral_definition": "Normalized one-sided power of the training-standardized supplied nine axes; no norms or differences in this replacement block; P3 QR, depth 2, frequency-bin order",
            "spectral_factorized_definition": "Endpoints and within-axis areas only, using the retained /sqrt(2) convention",
            "spectral_joint_definition": "All 27 endpoints and 351 areas, including cross-axis areas, using /2; feature standardization removes the fixed convention difference",
            "spectral_factorized_before_constant_removal": spectral9_factorized.shape[1],
            "spectral_joint_before_constant_removal": spectral9_joint.shape[1],
            "unchanged_blocks": "Original spectral/rotation pools for pool_raw9; original calibrated82 and trends15 for every compact variant, including their norm/difference information",
            "model_widths_before_constant_removal": {name: values.shape[1] for name, values in models.items() if "raw9" in name},
        }
    preparation["shared_preparation_s"] = perf_counter() - started
    return models, preparation


def self_check() -> None:
    """Check the two signature identities used by the feature generator."""
    from math import factorial

    from tensorsig.algebra.lyndon_words import lyndon_words
    from tensorsig.classical.signatures import log_signature, signature

    rng = torch.Generator().manual_seed(37)
    z = torch.randn(3, 13, 2, generator=rng, dtype=torch.float64)
    code = pascal_code(13, 3).matrix
    lifted = ((z / 13)[..., None] * code[None, :, None]).flatten(2)
    path = torch.cat((torch.zeros_like(lifted[:, :1]), lifted.cumsum(1)), 1)
    np.testing.assert_allclose(pascal_signature(z, 2), signature(path, 2).numpy(), atol=1e-7, rtol=1e-6)

    # An independent full graph verifies x^r t_a = mean(z^r q_a)/r!.
    moments, _ = graph_moments(z, 2, 3, 2)
    q = pascal_code(13, 2).matrix
    clock = torch.cat((q.new_zeros((1, 2)), (q / 13).cumsum(0)))
    scalar = z.transpose(1, 2).reshape(-1, 13)
    graph = z.new_zeros((len(scalar), 28, 3))
    graph[:, 1:-1, 0] = scalar.repeat_interleave(2, 1)
    graph[:, 1:-1, 1:] = torch.stack((clock[:-1], clock[1:]), 1).flatten(0, 1)
    graph[:, -1, 1:] = clock[-1]
    words = lyndon_words(3, 4)
    indices = [words.index((1,) * r + (a + 2,)) for r in range(1, 4) for a in range(2)]
    full = log_signature(graph, depth=4, mode="words").numpy()[:, indices].reshape(3, 2, 3, 2)
    np.testing.assert_allclose(full, moments / np.array([factorial(r) for r in range(1, 4)])[None, None, :, None],
                               atol=2e-13, rtol=2e-11)

    signals = torch.randn(8, 128, 9, generator=rng, dtype=torch.float64).numpy()
    models, _ = build_features(signals, 5, np.ones((8, 561)), batch_size=3)
    assert {name: values.shape for name, values in models.items()} == {
        "compact368": (8, 388), "pool803": (8, 804), "uci903": (8, 903)}
    assert all(np.isfinite(values).all() for values in models.values())

    extended, metadata = build_features(signals, 5, np.ones((8, 561)), batch_size=3, all_channels=True)
    expected_widths = {"compact_raw9": 595, "pool_raw9": 1218, "uci_raw9": 1317,
                       "compact_raw9_spectral9_factorized": 529, "compact_raw9_spectral9_joint": 853}
    assert metadata["all_channels"]["model_widths_before_constant_removal"] == expected_widths
    assert all(extended[name].shape == (8, width) for name, width in expected_widths.items())
    assert all(np.isfinite(values).all() for values in extended.values())
    for name, values in models.items():
        np.testing.assert_array_equal(extended[name], values)

    # Selecting total acceleration and gyroscope from raw9 recovers raw6,
    # including both orders of every pair of code letters.
    raw9 = extended["uci_raw9"][:, 561:]
    indices = (3 * np.array([6, 7, 8, 3, 4, 5])[:, None] + np.arange(3)).ravel()
    second = raw9[:, 27:].reshape(-1, 27, 27)[:, indices][:, :, indices]
    recovered_raw6 = np.concatenate((raw9[:, indices], second.reshape(8, -1)), 1)
    np.testing.assert_allclose(recovered_raw6, models["uci903"][:, 561:], atol=1e-7, rtol=1e-6)

    # Factorized spectral features are contained in the joint logsignature.
    # Match its /2 areas to the retained spectral /sqrt(2) convention.
    joint = extended["compact_raw9_spectral9_joint"][:, :378]
    left, right = np.triu_indices(27, 1)
    recovered = []
    for channel in range(9):
        own = (left // 3 == channel) & (right // 3 == channel)
        recovered.extend((joint[:, 3 * channel:3 * (channel + 1)], joint[:, 27:][:, own] * np.sqrt(2)))
    np.testing.assert_allclose(np.concatenate(recovered, 1),
                               extended["compact_raw9_spectral9_factorized"][:, :54], atol=1e-7, rtol=1e-6)

    changed = signals.copy()
    changed[5:] = changed[5:] * 3 + 7
    perturbed, _ = build_features(changed, 5, np.ones((8, 561)), batch_size=3, all_channels=True)
    for name, values in extended.items():
        np.testing.assert_array_equal(perturbed[name][:5], values[:5])
