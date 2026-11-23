"""Three retained models and matched ordinary-signature ablations."""

from time import perf_counter

import numpy as np
import torch
from coded_occupation import coded_occupation_readouts
from families import _power, _with_norm, build_families
from features import _coordinates, _standardize
from generators import coded_integration, temporal_code
from moment_lifts import _cosines
from occupation_graph import occupation_readouts
from ordinary_controls import ordinary_controls
from ordinary_spectral import ordinary_spectral_controls
from spectral_readouts import signature_readouts
from tensorsig.classical.signatures import signature


@torch.inference_mode()
def build_retained(signals, n_train, engineered, batch_size=256):
    """Build fixed blocks, preserving the preceding experiments' definitions."""
    start = perf_counter()
    raw = _standardize(torch.as_tensor(signals[..., [6, 7, 8, 3, 4, 5]], dtype=torch.float64), n_train)
    raw_tensor, raw_s = coded_integration(raw, 3, 2, batch_size=batch_size)
    raw_log = _coordinates(torch.as_tensor(raw_tensor), ("accel",) * 9 + ("gyro",) * 9)["full_logsignature"].numpy()
    families = build_families(signals, n_train)
    spectral, spectral_s = signature_readouts(families["spectral"]["observations"], 3, 2, batch_size)
    spectral_log = spectral["within_channel_first_areas"]
    vectors = [obs[:, :, :3] for obs in families["moments"]["observations"]]
    vectors += [obs[:, :, :3] for obs in families["filters_delays"]["observations"][:2]]
    observations = {"amplitude": [_power(_with_norm(v)).sqrt() for v in vectors],
                    "cosine": [_cosines(*families["rotation"]["observations"][:2])],
                    "invariant": families["rotation"]["observations"]}
    cat = lambda values: np.concatenate(list(values), 1)
    calibration, trends, ordinary_trends = {}, {}, {}
    for family, groups in observations.items():
        stats, coded, ordinary = [], [], []
        for obs in groups:
            first, _ = coded_occupation_readouts(obs, 1, 2, batch_size)
            mean = first["moments"][:, :, 0, 0] / first["code_scale"]
            center = obs - torch.as_tensor(mean)[:, None]
            moments, _ = coded_occupation_readouts(center, 2, 4, batch_size)
            variance = np.maximum(moments["moments"][:, :, 1, 0] / moments["code_scale"], 0)
            sd = np.sqrt(variance)
            normalized = moments["moments"] / np.maximum(sd, 1e-8)[:, :, None, None] ** np.arange(1, 4)[None, None, :, None]
            normalized = np.where((variance > 1e-12)[:, :, None, None], normalized, 0)
            skew = normalized[:, :, 2, 0] / moments["code_scale"]
            count = 3 if family == "invariant" else 2
            stats.append(np.stack((mean, sd, skew), -1)[:, :, :count].reshape(len(signals), -1))
            if family != "amplitude":
                coded.append(normalized[:, :, 0, 1])
                z = center / torch.as_tensor(np.maximum(sd, 1e-8))[:, None]
                z = torch.where(torch.as_tensor(variance > 1e-12)[:, None], z, 0.)
                # This control uses an actual ordinary graph, not the coded evaluator.
                graph, _ = occupation_readouts(z, batch_size)
                ordinary.append(graph["order_d3"])
        calibration[family] = cat(stats)
        if coded:
            trends[family], ordinary_trends[family] = cat(coded), cat(ordinary)
    calibration82 = cat(calibration.values())
    trend15, ordinary15 = cat(trends.values()), cat(ordinary_trends.values())
    pool = cat((families["spectral"]["pooled"], families["rotation"]["pooled"][:, :102]))
    blocks = {"raw_tensor": raw_tensor, "raw_log": raw_log, "spectral_log": spectral_log,
              "calibration82": calibration82, "trend15": trend15, "ordinary_trend15": ordinary15,
              "pool": pool, "engineered": engineered}
    return blocks, raw, families["spectral"]["observations"], {
        "shared_preparation_s": perf_counter() - start,
        "raw_signature_s": raw_s, "spectral_signature_s": spectral_s,
        "calibration_widths": {name: value.shape[1] for name, value in calibration.items()},
        "trend_widths": {name: value.shape[1] for name, value in trends.items()},
    }


@torch.inference_mode()
def run(evaluate, signals, n_train, engineered, args, metadata):
    blocks, raw, spectra, timings = build_retained(signals, n_train, engineered, args.batch_size)
    cat = lambda values: np.concatenate(list(values), 1)
    preparation = timings["shared_preparation_s"]
    compact_fixed = cat((blocks["calibration82"], blocks["trend15"]))
    raw_contexts = {
        "compact368": cat((blocks["spectral_log"], compact_fixed)),
        "pool803": blocks["pool"], "uci903": engineered,
    }
    settings = {
        "retained": {
            "compact368": "raw Pascal P3 QR log171 + spectral Pascal P3 QR log100 + calibrated82 + standardized P2 graph trends15",
            "pool803": "spectral pooled359 + invariant moment summaries96 + cross-cosines6 + raw Pascal P3 QR tensor342",
            "uci903": "supplied561 + raw Pascal P3 QR tensor342",
        },
        "head": "Fixed alpha10 ridge, training-only StandardScaler, existing float64 LU solver",
        "raw_control": "Replace raw branch only; all other blocks fixed. Total-acc source matched; body-acc separately paper-inspired.",
        "spectral_control": "Replace spectral branch only in compact368; mass/frequency staircase tensor/log depths2--5",
        "paper_scope": "One local PDF; time and both lead-lag interpretations. Adapted input scaling and ridge head, not paper reproduction.",
        "bounds": "Raw outputs <=1600, dense intermediates <=3000; time/integration depths2--4, delays/leadlag depths2--3",
        "timings": "Shared preparation includes all retained blocks and ordinary trend control; extractions shared across readouts, not additive",
        "ordinary_raw": {}, "ordinary_spectral": {}, **timings,
    }
    metadata["consolidated_settings"] = settings
    for model, raw_name in (("compact368", "raw_log"), ("pool803", "raw_tensor"), ("uci903", "raw_tensor")):
        evaluate("retained_" + model, "pascal_qr", cat((raw_contexts[model], blocks[raw_name])), preparation)
        evaluate("drop_raw_" + model, "none", raw_contexts[model], preparation)

    # Exact lifted-path equivalence, computed independently by classical tensorsig.
    tick = perf_counter()
    q = temporal_code(raw.shape[1], 3).matrix
    parts = []
    for batch in raw.split(args.batch_size):
        increments = ((batch / raw.shape[1])[..., None] * q[None, :, None]).flatten(2)
        path = torch.cat((torch.zeros_like(increments[:, :1]), increments.cumsum(1)), 1)
        parts.append(signature(path, 2).float().numpy())
    tensor = np.concatenate(parts)
    projected = _coordinates(torch.as_tensor(tensor), ("accel",) * 9 + ("gyro",) * 9)["full_logsignature"].numpy()
    settings["raw_lift_max_abs_error"] = float(np.max(np.abs(tensor - blocks["raw_tensor"])))
    np.testing.assert_allclose(tensor, blocks["raw_tensor"], atol=2e-7, rtol=2e-6)
    lift_seconds = perf_counter() - tick
    for model, values in (("compact368", projected), ("pool803", tensor), ("uci903", tensor)):
        evaluate("ordinary_lift_" + model, "explicit_polynomial_path_d2",
                 cat((raw_contexts[model], values)), preparation + lift_seconds)
    del tensor, projected, parts

    # Two predetermined raw controls are retained for joint raw/spectral swaps.
    joint_raw = {}
    joint_labels = {"total_integrated_d3_log", "total_delay2_d2_log"}
    for label, values, seconds, detail in ordinary_controls(signals, n_train, args.batch_size):
        settings["ordinary_raw"][label] = detail
        for model, context in raw_contexts.items():
            evaluate("ordinary_raw_" + model, label, cat((context, values)), preparation + seconds)
        if label in joint_labels:
            joint_raw[label] = (values, seconds)

    # With raw Pascal held fixed, isolate the spectral contribution.
    spec_context = cat((blocks["raw_log"], compact_fixed))
    evaluate("drop_spectral_compact368", "none", spec_context, preparation)
    for label, values, seconds, detail in ordinary_spectral_controls(spectra, args.batch_size):
        settings["ordinary_spectral"][label] = detail
        evaluate("ordinary_spectral_compact368", label, cat((spec_context, values)), preparation + seconds)
        # The trends have an ordinary graph equivalent; use it in all-ordinary rows.
        if label in {"spectral_clock_log_d4", "spectral_clock_log_d5", "spectral_clock_d5_projected_pascal_P3"}:
            for raw_label, (raw_values, raw_seconds) in joint_raw.items():
                evaluate("ordinary_joint_compact368", raw_label + "__" + label,
                         cat((raw_values, values, blocks["calibration82"], blocks["ordinary_trend15"])),
                         preparation + seconds + raw_seconds)

    no_trends = cat((blocks["raw_log"], blocks["spectral_log"], blocks["calibration82"]))
    evaluate("drop_trends_compact368", "none", no_trends, preparation)
    evaluate("ordinary_trends_compact368", "closed_value_graph_xtt",
             cat((no_trends, blocks["ordinary_trend15"])), preparation)
    # These frozen ablations interpret components of the chosen compact model.
    for name, start, end in (("amplitude40", 0, 40), ("cosine6", 40, 46), ("invariant36", 46, 82)):
        remaining = np.delete(blocks["calibration82"], np.s_[start:end], axis=1)
        evaluate("drop_" + name + "_compact368", "none",
                 cat((blocks["raw_log"], blocks["spectral_log"], remaining, blocks["trend15"])), preparation)
