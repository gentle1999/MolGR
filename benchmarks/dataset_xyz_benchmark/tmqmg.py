"""Read the pinned, independently published tmQMg-Corrected standard.

The old in-tree draft and upstream CSV labels are not benchmark standards.
See benchmarks/dataset_xyz_benchmark/README.md#tmqmg-corrected for checkout instructions; no implicit network access.
"""

from __future__ import annotations

import base64
import hashlib
import json
import zlib
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
            f"Check out {REPOSITORY} at {REVISION}; see benchmarks/dataset_xyz_benchmark/README.md#tmqmg-corrected. "
            "Use prepare --dataset tmqmg-corrected --source to select that checkout. The old bundled dataset is retired."
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


def rows(directory: Path, *, include_excluded=False):
    """Yield released inputs/targets; optionally retain excluded inputs for coverage.

    Validate the manifest and each consumed part before yielding any of its rows.
    Hash keys are repository-relative paths (not basenames), as in layout v4.
    """
    directory = Path(directory)
    manifest = load_manifest(directory)
    total_inputs = total_targets = 0
    for part in manifest["parts"]:
        for key in ("sdf", "xyz"):
            path = directory / part[key]
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != manifest["sha256"][part[key]]:
                raise ValueError(f"Incorrect corrected dataset file: {path}")
        with (directory / part["sdf"]).open("rb") as sdf, (directory / part["xyz"]).open(
            "rb"
        ) as xyz:
            records = _sdf_records(sdf)
            target = next(records, None)
            targets = inputs = 0
            for count in xyz:
                inputs += 1
                comment = dict(item.split("=", 1) for item in xyz.readline().decode().split())
                case_id = comment["id"]
                atom_lines = [xyz.readline() for _ in range(int(count))]
                if not atom_lines or any(not line.strip() for line in atom_lines):
                    raise ValueError(f"Truncated corrected XYZ record: {case_id}")
                atoms = b"".join(atom_lines)
                original_xyz = count + b"\n" + atoms
                if target is None or target.partition(b"\n")[0] != case_id.encode():
                    if include_excluded:
                        yield {
                            "id": case_id,
                            "charge": comment["charge"],
                            "multiplicity": comment["multiplicity"],
                            "xyz": original_xyz.decode(),
                            "native_graph": None,
                        }
                    continue
                props = {}
                for field in target.split(b"\n>  <")[1:]:
                    key, value = field.split(b">\n", 1)
                    if key in {
                        b"CASE_ID",
                        b"INPUT_CHARGE",
                        b"INPUT_MULTIPLICITY",
                        b"XYZ_SHA256",
                        b"NATIVE_GRAPH_ZLIB_BASE64",
                        b"GRAPH_SHA256",
                    }:
                        props[key] = value.partition(b"\n")[0].decode("ascii")
                if (
                    props[b"CASE_ID"] != case_id
                    or props[b"INPUT_CHARGE"] != comment["charge"]
                    or props[b"INPUT_MULTIPLICITY"] != comment["multiplicity"]
                    or hashlib.sha256(original_xyz).hexdigest() != props[b"XYZ_SHA256"]
                ):
                    raise ValueError(f"Incorrect XYZ for selected graph {case_id}")
                graph = zlib.decompress(base64.b64decode(props[b"NATIVE_GRAPH_ZLIB_BASE64"]))
                if hashlib.sha256(graph).hexdigest() != props[b"GRAPH_SHA256"]:
                    raise ValueError(f"Incorrect native graph for {case_id}")
                yield {
                    "id": case_id,
                    "charge": comment["charge"],
                    "multiplicity": comment["multiplicity"],
                    "xyz": original_xyz.decode(),
                    "native_graph": props[b"NATIVE_GRAPH_ZLIB_BASE64"],
                }
                targets += 1
                target = next(records, None)
            if target is not None or inputs != part["inputs"] or targets != part["targets"]:
                raise ValueError(f"Incomplete corrected dataset part {part['sdf']}")
            total_inputs += inputs
            total_targets += targets
    if total_inputs != manifest["inputs"] or total_targets != manifest["targets"]:
        raise ValueError("Incomplete tmQMg-Corrected dataset")
