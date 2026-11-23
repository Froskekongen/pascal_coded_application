"""Portable helpers copied verbatim from the original frozen NATOPS experiment.

Only shared loading, path preparation and learner functions are included.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import ssl
import urllib.request
ROOT = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(name, obj):
    (ROOT / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def fetch_data(entry):
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    with urllib.request.urlopen(entry["url"], context=context, timeout=60) as response:
        raw = response.read(5_000_001)
    assert len(raw) == entry["bytes"]
    assert digest(raw) == entry["sha256"]
    assert hashlib.md5(raw).hexdigest() == entry["md5"]
    return raw


def parse_data(raw):
    import numpy as np
    header, body = raw.decode("utf-8-sig").split("@data", 1)
    rows, labels = [], []
    for line in body.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.strip().split(":")
        rows.append([[float(v) for v in channel.split(",")] for channel in parts[:-1]])
        labels.append(int(float(parts[-1])))
    x = np.asarray(rows, dtype=np.float64).transpose(0, 2, 1)
    y = np.asarray(labels, dtype=int)
    assert x.shape == (180, 51, 24) and np.isfinite(x).all()
    assert np.array_equal(np.bincount(y)[1:], [30] * 6)
    return x, y


def increments(raw):
    import numpy as np
    t = np.broadcast_to(np.linspace(0.0, 1.0, raw.shape[1])[None, :, None],
                        (len(raw), raw.shape[1], 1))
    path = np.concatenate([t, raw], axis=2)
    return np.diff(np.concatenate([np.zeros((len(raw), 1, 25)), path], axis=1), axis=1)


def make_pipeline(alpha):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeClassifier
    return make_pipeline(StandardScaler(), RidgeClassifier(alpha=alpha, solver="cholesky",
                                                          fit_intercept=True, class_weight=None))
