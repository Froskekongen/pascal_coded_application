"""Ordinary signature controls for the consolidated, ridge-only HAR comparison.

Every embedding uses the same training-only global channel scaling. The
paper-derived embeddings are adaptations: the reference does not specify that
input scaling, and describes both a simultaneous shift and a staircase lead-lag.
"""

from itertools import product
from time import perf_counter

import numpy as np
import torch
from features import _standardize, augment
from tensorsig.algebra.lyndon_words import lyndon_words
from tensorsig.classical.log_signature import signature_to_logsignature
from tensorsig.classical.signatures import signature


def _coordinate_masks(width: int, depth: int, clock_channels: tuple[int, ...]):
    """Remove known deterministic words, including mixed words of two clocks."""
    clocks = set(clock_channels)
    if any(channel < 0 or channel >= width for channel in clocks):
        raise ValueError("Clock channel outside the path width")
    tensor = np.array([
        not all(letter in clocks for letter in word)
        for level in range(1, depth + 1)
        for word in product(range(width), repeat=level)
    ])
    log = np.array([
        not all(letter - 1 in clocks for letter in word)
        for word in lyndon_words(width, depth)
    ])
    return {"tensor": tensor, "log": log}


@torch.inference_mode()
def ordinary_signature_features(
    paths: torch.Tensor, depth: int, batch_size: int = 256, *,
    clock_channels: tuple[int, ...] = (), max_features: int = 1600,
) -> tuple[dict[str, np.ndarray], float]:
    """Compute tensorsig tensors/Lyndon log coordinates, dropping pure clocks.

    ``clock_channels`` must contain only channels with the same complete path
    in every window. Their words are known constants, removed before float32
    conversion rather than detected by a numerical variance threshold. All
    other coordinates are kept, even when small. Outputs wider than the fixed
    feature cap are skipped; the dense intermediate is capped at 3,000 columns.
    Tensor and log outputs share extraction, so the returned time is not additive.
    """
    start = perf_counter()
    paths = torch.as_tensor(paths, dtype=torch.float64)
    if (paths.ndim != 3 or paths.shape[0] < 1 or paths.shape[1] < 2
            or paths.device.type != "cpu" or not torch.isfinite(paths).all()
            or depth not in (1, 2, 3, 4) or batch_size < 1 or max_features < 1):
        raise ValueError("Expected finite CPU paths, depth 1--4 and positive sizes")
    width = paths.shape[2]
    dense_size = sum(width**level for level in range(1, depth + 1))
    if dense_size > 3000:
        raise ValueError("Ordinary controls are limited to 3,000 dense coordinates")
    if clock_channels and not torch.equal(
        paths[:, :, clock_channels],
        paths[:1, :, clock_channels].expand(len(paths), -1, -1),
    ):
        raise ValueError("Clock channels must have identical paths in every window")
    masks = {
        name: mask for name, mask in _coordinate_masks(width, depth, clock_channels).items()
        if 0 < int(mask.sum()) <= max_features
    }
    if not masks:
        raise ValueError("No ordinary readout fits the feature cap")
    # The native kernel materializes time-indexed tensors. Bound a batch's
    # highest-level tensor to roughly 32 MB, independently of the embedding.
    batch_size = min(batch_size, max(1, 4_000_000 // (paths.shape[1] * width**depth)))
    outputs = {name: [] for name in masks}
    for batch in paths.split(batch_size):
        tensor = signature(batch, depth)
        for name, mask in masks.items():
            values = tensor if name == "tensor" else signature_to_logsignature(tensor, depth)
            outputs[name].append(values[:, mask].float().numpy().copy())
    return {name: np.concatenate(parts) for name, parts in outputs.items()}, perf_counter() - start


def _paper_paths(signals: np.ndarray, n_train: int, mode: str):
    """Time first, then either simultaneous one-step shift or staircase lift."""
    paths, _ = augment(signals, "time", n_train)
    if mode == "shift1":
        paths = torch.cat((paths[:, :-1], paths[:, 1:]), 2)
    elif mode == "staircase_leadlag":
        repeated = paths.repeat_interleave(2, dim=1)
        paths = torch.cat((repeated[:, 1:], repeated[:, :-1]), 2)
    else:
        raise ValueError("Unknown paper embedding")
    return paths, (6, 13)


def ordinary_controls(signals: np.ndarray, n_train: int, batch_size: int = 256):
    """Yield ``(label, features, extraction_seconds, metadata)`` lazily.

    Input is the nine UCI channels in their usual body-acc/gyro/total-acc order.
    Matched controls use total acceleration and gyro; the separately named
    paper adaptations use body acceleration and gyro. At most one embedding's
    feature matrices are retained by this generator. Tensor/log depth variants
    have independent measured extraction times; each pair shares one evaluation.
    """
    signals = np.asarray(signals)
    if (signals.ndim != 3 or signals.shape[2] != 9 or signals.shape[1] < 6
            or not np.isfinite(signals).all() or not 1 <= n_train <= len(signals)):
        raise ValueError("Expected finite nine-channel UCI windows and a training prefix")
    sources = {"total": signals[..., [6, 7, 8, 3, 4, 5]], "body": signals[..., :6]}
    specs = [
        ("total", "integrated_no_clock", (2, 3, 4)),
        ("total", "time", (2, 3, 4)),
        ("total", "basepoint", (2, 3, 4)),
        ("total", "integrated", (2, 3, 4)),
        *(("total", mode, (2, 3)) for mode in (
            "delay1", "delay2", "delay5", "leadlag", "integrated_leadlag",
            "paper_shift1", "paper_staircase_leadlag",
        )),
        ("body", "time", (2, 3)),
        ("body", "paper_shift1", (2, 3)),
        ("body", "paper_staircase_leadlag", (2, 3)),
    ]
    for source, embedding, depths in specs:
        start = perf_counter()
        raw = sources[source]
        if embedding == "integrated_no_clock":
            x = _standardize(torch.as_tensor(raw, dtype=torch.float64), n_train)
            paths = torch.cat((torch.zeros_like(x[:, :1]), x.cumsum(1) / x.shape[1]), 1)
            clocks = ()
        elif embedding.startswith("paper_"):
            paths, clocks = _paper_paths(raw, n_train, embedding.removeprefix("paper_"))
        elif embedding == "delay1":
            x = _standardize(torch.as_tensor(raw, dtype=torch.float64), n_train)
            delayed = x[:, (torch.arange(x.shape[1]) - 1).clamp_min(0)]
            time = torch.linspace(0, 1, x.shape[1], dtype=x.dtype)
            paths = torch.cat((x, delayed, time[None, :, None].expand(len(x), -1, 1)), 2)
            paths = torch.cat((torch.zeros_like(paths[:, :1]), paths), 1)
            clocks = (12,)
        else:
            paths, _ = augment(raw, embedding, n_train)
            clocks = (paths.shape[2] - 1,)
        prepare_s = perf_counter() - start
        for depth in depths:
            outputs, elapsed = ordinary_signature_features(
                paths, depth, batch_size, clock_channels=clocks,
            )
            masks = _coordinate_masks(paths.shape[2], depth, clocks)
            for representation, values in outputs.items():
                metadata = {
                    "source": source, "embedding": embedding, "depth": depth,
                    "representation": representation, "path_width": paths.shape[2],
                    "path_points": paths.shape[1], "clock_channels": list(clocks),
                    "known_constant_coordinates_removed": int((~masks[representation]).sum()),
                    "preparation_s": prepare_s,
                    "scaling": "training-only global axis mean/std",
                    "paper_adaptation": source == "body" or embedding.startswith("paper_"),
                    "readout": "tensor" if representation == "tensor" else "Lyndon tensor-log words",
                }
                yield f"{source}_{embedding}_d{depth}_{representation}", values, prepare_s + elapsed, metadata


def self_check() -> None:
    """Check log coordinates, deterministic-clock removal and paper lifts."""
    rng = np.random.default_rng(17)
    signals = rng.normal(size=(3, 8, 6))
    time_path, _ = augment(signals, "time", 2)
    for depth in (2, 3, 4):
        outputs, _ = ordinary_signature_features(time_path, depth, 2, clock_channels=(6,))
        expected = signature(time_path, depth)
        masks = _coordinate_masks(7, depth, (6,))
        for name, values in outputs.items():
            full = expected if name == "tensor" else signature_to_logsignature(expected, depth)
            np.testing.assert_allclose(values, full[:, masks[name]].numpy(), rtol=1e-6, atol=1e-7)
        assert masks["tensor"].sum() == sum(7**level - 1 for level in range(1, depth + 1))
    shifted, clocks = _paper_paths(signals, 2, "shift1")
    torch.testing.assert_close(shifted[:, :, :7], time_path[:, :-1])
    torch.testing.assert_close(shifted[:, :, 7:], time_path[:, 1:])
    staircase, _ = _paper_paths(signals, 2, "staircase_leadlag")
    assert staircase.shape == (3, 15, 14)
    increments = staircase.diff(dim=1)
    torch.testing.assert_close(increments[:, ::2, 7:], torch.zeros_like(increments[:, ::2, 7:]))
    torch.testing.assert_close(increments[:, 1::2, :7], torch.zeros_like(increments[:, 1::2, :7]))
    outputs, _ = ordinary_signature_features(shifted, 2, 2, clock_channels=clocks)
    assert outputs["tensor"].shape == (3, 204)
    assert outputs["log"].shape == (3, 102)


if __name__ == "__main__":
    self_check()
    print("Ordinary-control self-check passed")
