"""Read the pinned tmQMg-Corrected release used by the offline algorithm fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPOSITORY = "https://github.com/gentle1999/tmQMg-Corrected"
REVISION = "aac372de0e859a6daf451084abe84a9ae58ad8e6"
VERSION = "2026-11-23"
MANIFEST_SHA256 = "ea04563079dfe4344601f6c896e36c24a847a69e4f320ec4f906a69203c3faf3"
DATASET = Path(__file__).resolve().parents[2] / ".local" / "datasets" / "tmqmg-corrected"


def load_manifest(directory: Path) -> dict:
    path = Path(directory) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"tmQMg-Corrected {VERSION} is not installed at {directory}. "
            f"Check out {REPOSITORY} at {REVISION} or pass its path to the fixture generator."
        )
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != MANIFEST_SHA256:
        raise ValueError(
            f"Not the pinned tmQMg-Corrected {VERSION} manifest: {path}. "
            f"Use {REPOSITORY} at {REVISION}, not the old draft or upstream labels."
        )
    return json.loads(data)


def provenance(directory: Path) -> dict:
    manifest = load_manifest(directory)
    return {
        "dataset": manifest["dataset"],
        "version": manifest["version"],
        "repository": REPOSITORY,
        "revision": REVISION,
        "manifest_sha256": MANIFEST_SHA256,
        "inputs": manifest["inputs"],
        "targets": manifest["targets"],
        "exclusions": manifest.get("exclusions", {}),
        "benchmark": manifest["benchmark"],
    }


def _sdf_records(stream):
    record = []
    for line in stream:
        if line.rstrip(b"\r\n") == b"$$$$":
            yield b"".join(record)
            record = []
        else:
            record.append(line)
    if record:
        raise ValueError("Truncated corrected SDF record")
