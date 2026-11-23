"""Read the two official datasets without changing their published splits.

The checksums come from revision 16's evidence/har/geometry57_results/run.json
and evidence/natops/offset_learning/protocol.json. Downloads are opt-in, cached,
and checked before parsing. ZIP contents are read in memory, never extracted.
"""

from contextlib import ExitStack
import hashlib
import io
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

import numpy as np

HAR_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
HAR_SHA256 = "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031"
NATOPS_SHA256 = {
    "TRAIN": "3567f0aa297e89327215ee1147537e06f7bdc51f48ac8d2c39aaa34fb1eab1b7",
    "TEST": "3a9a41a948c30d19db6b9e0ec44d946d42199c4f8724c2aa6e8b71b28612404e",
}


def check_hash(path: Path, expected: str) -> None:
    """Reject changed archives, partial downloads, and HTML error responses."""
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")


def download(url: str, destination: Path, expected: str) -> None:
    """Cache one verified file atomically; never replace an existing dataset."""
    if destination.exists():
        check_hash(destination, expected)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                shutil.copyfileobj(response, stream)
            stream.close()
            check_hash(temporary, expected)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def download_har(path: Path) -> None:
    """Fetch the UCI archive only if absent (about 58 MB compressed)."""
    download(HAR_URL, path, HAR_SHA256)


def load_har(path: Path) -> dict:
    """Return official train/test signals, supplied features, labels and subjects.

    Signals have shape (windows, 128, 9), in body-acceleration xyz,
    gyroscope xyz, total-acceleration xyz order. Labels are zero-based (0..5).
    Appendix app:har requires 7,352/2,947 windows and disjoint subjects.
    """
    check_hash(path, HAR_SHA256)
    data = {}
    with ExitStack() as stack:
        archive = stack.enter_context(zipfile.ZipFile(path))
        # The current official download wraps the original dataset ZIP.
        nested = next((n for n in archive.namelist() if n.endswith("UCI HAR Dataset.zip")), None)
        if nested:
            archive = stack.enter_context(zipfile.ZipFile(io.BytesIO(archive.read(nested))))
        prefix = next(n.removesuffix("train/y_train.txt") for n in archive.namelist()
                      if n.endswith("train/y_train.txt"))
        for split, count in (("train", 7352), ("test", 2947)):
            def read(name, dtype=float):
                with archive.open(f"{prefix}{split}/{name}") as stream:
                    return np.loadtxt(stream, dtype=dtype)

            signals = np.stack([
                read(f"Inertial Signals/{sensor}_{axis}_{split}.txt")
                for sensor in ("body_acc", "body_gyro", "total_acc") for axis in "xyz"
            ], axis=-1)
            engineered = read(f"X_{split}.txt")
            labels = read(f"y_{split}.txt", int) - 1
            subjects = read(f"subject_{split}.txt", int)
            if (signals.shape != (count, 128, 9) or engineered.shape != (count, 561)
                    or labels.shape != (count,) or subjects.shape != (count,)
                    or not np.isfinite(signals).all() or not np.isfinite(engineered).all()
                    or not np.array_equal(np.unique(labels), np.arange(6))):
                raise ValueError(f"Unexpected UCI HAR {split} data")
            data[split] = dict(signals=signals, engineered=engineered, labels=labels, subjects=subjects)
    train_subjects, test_subjects = (set(data[s]["subjects"]) for s in ("train", "test"))
    if len(train_subjects) != 21 or len(test_subjects) != 9 or train_subjects & test_subjects:
        raise ValueError("Expected the official subject-disjoint UCI HAR split")
    return data


def download_natops(directory: Path) -> None:
    """Fetch the two official UEA files (about 2 MB each)."""
    for split, expected in NATOPS_SHA256.items():
        name = f"NATOPS_{split}.ts"
        download(f"https://zenodo.org/records/11206248/files/{name}?download=1",
                 directory / name, expected)


def load_natops(directory: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return train_x, train_y, test_x, test_y; keep original class labels 1..6.

    Each split has 180 trajectories of 51 observations and 24 coordinates.
    The eight xyz landmark blocks follow the order in Appendix app:natops.
    No centering, resampling, or per-trajectory scaling is performed here.
    """
    result = []
    for split, expected in NATOPS_SHA256.items():
        path = directory / f"NATOPS_{split}.ts"
        check_hash(path, expected)
        body = path.read_text(encoding="utf-8-sig").split("@data", 1)[1]
        rows, labels = [], []
        for line in body.splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            *channels, label = line.strip().split(":")
            rows.append([[float(value) for value in channel.split(",")] for channel in channels])
            labels.append(int(float(label)))
        x = np.asarray(rows, dtype=np.float64).transpose(0, 2, 1)
        y = np.asarray(labels, dtype=int)
        if (x.shape != (180, 51, 24) or not np.isfinite(x).all()
                or not np.array_equal(np.unique(y, return_counts=True),
                                      (np.arange(1, 7), np.full(6, 30)))):
            raise ValueError(f"Unexpected NATOPS {split} data")
        result.extend((x, y))
    return tuple(result)
