"""Small offline checks for the reproduction, without downloading datasets.

Run ``uv run python -m reproduction.checks`` from the repository root.
These checks exercise the feature identities and data boundary; the full
dataset commands separately compare fitted results with the frozen paper.
"""

import hashlib
import io
from pathlib import Path
import tempfile
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch
from tensorsig.classical.signatures import log_signature
from threadpoolctl import threadpool_limits

from . import data, natops, uci_har


def check_two_rises() -> None:
    """Verify the paper's eq:sep: equal moments, distinct separation areas."""
    rises = torch.tensor([[1, 0, 0, 1], [0, 1, 1, 0]], dtype=torch.float64)
    position = torch.arange(4, dtype=torch.float64) / 3
    increments = torch.stack((rises, rises * position), dim=-1)
    path = torch.cat((torch.zeros(2, 1, 2, dtype=torch.float64), increments.cumsum(1)), dim=1)
    actual = log_signature(path, depth=2)
    expected = torch.tensor([[2, 1, 1 / 2], [2, 1, 1 / 6]], dtype=torch.float64)
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)


def check_data() -> None:
    """Check verified caching, failure cleanup, and trajectory axis order."""
    assertions = TestCase()
    payload = b"A complete, verified download."
    expected = hashlib.sha256(payload).hexdigest()
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        cached = directory / "download.bin"
        with patch.object(data.urllib.request, "urlopen", return_value=io.BytesIO(payload)):
            data.download("https://example.invalid/dataset", cached, expected)
        assert cached.read_bytes() == payload

        # A verified cache works offline. A changed cache is rejected, rather
        # than silently overwritten or passed to a dataset parser.
        with patch.object(data.urllib.request, "urlopen") as fetch:
            data.download("https://example.invalid/dataset", cached, expected)
            fetch.assert_not_called()
        cached.write_bytes(b"corrupt cache")
        with assertions.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            data.download("https://example.invalid/dataset", cached, expected)
        assert cached.read_bytes() == b"corrupt cache"
        with assertions.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            data.load_har(cached)

        # Neither bad content nor a network failure may leave a completed
        # dataset or an abandoned temporary download behind.
        missing = directory / "missing.bin"
        with patch.object(data.urllib.request, "urlopen", return_value=io.BytesIO(b"wrong bytes")):
            with assertions.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                data.download("https://example.invalid/dataset", missing, expected)
        with patch.object(data.urllib.request, "urlopen", side_effect=TimeoutError):
            with assertions.assertRaises(TimeoutError):
                data.download("https://example.invalid/dataset", missing, expected)
        assert not missing.exists() and not list(directory.glob("*.part"))

        # A tiny-in-memory .ts fixture uses the actual official dimensions.
        # Distinct frame/channel values catch an accidental transpose; the
        # frozen checksum is replaced by this fixture's checksum, not bypassed.
        channels = ":".join(
            ",".join(str(100 * channel + frame) for frame in range(51))
            for channel in range(24)
        )
        text = "\ufeff@problemName fixture\n@data\n# comment\n\n" + "\n".join(
            f"{channels}:{label}" for label in range(1, 7) for _ in range(30)
        )
        fixture_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        for split in ("TRAIN", "TEST"):
            (directory / f"NATOPS_{split}.ts").write_text(text, encoding="utf-8")
        with patch.dict(data.NATOPS_SHA256, TRAIN=fixture_hash, TEST=fixture_hash):
            train_x, train_y, test_x, test_y = data.load_natops(directory)
        expected_path = np.arange(51)[:, None] + 100 * np.arange(24)[None, :]
        np.testing.assert_array_equal(train_x[0], expected_path)
        np.testing.assert_array_equal(train_x, test_x)
        np.testing.assert_array_equal(train_y, np.repeat(np.arange(1, 7), 30))
        np.testing.assert_array_equal(train_y, test_y)


def main() -> None:
    """Run the dataset boundary and each example's mathematical checks."""
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        check_two_rises()
        check_data()
        natops.self_check()
        uci_har.self_check()
    print("Offline data, NATOPS, and UCI HAR checks passed.")


if __name__ == "__main__":
    main()
