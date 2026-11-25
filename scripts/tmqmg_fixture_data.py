"""Offline tmQMg-Corrected fixture access; SMILES are display-only, never targets."""

from __future__ import annotations

import base64
import hashlib
import json
import zlib
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from benchmarks.dataset_xyz_benchmark import tmqmg as corrected


if TYPE_CHECKING:
    from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1] / "tests/data/tmqmg/published"


@lru_cache(maxsize=1)
def fixtures() -> dict:
    payload = json.loads((ROOT / "cases.json").read_text(encoding="utf-8"))
    source = payload["source_dataset"]
    if (
        payload.get("schema_version") != 1
        or source.get("dataset") != "tmqmg-corrected"
        or source.get("version") != corrected.VERSION
        or source.get("repository") != corrected.REPOSITORY
        or source["revision"] != corrected.REVISION
        or source["manifest_sha256"] != corrected.MANIFEST_SHA256
    ):
        raise ValueError("tmQMg fixtures must be regenerated for the pinned published standard")
    if (
        hashlib.sha256((ROOT / "release-manifest.json").read_bytes()).hexdigest()
        != corrected.MANIFEST_SHA256
    ):
        raise ValueError("Fixture release manifest checksum mismatch")
    if any(not c["benchmark"] for c in payload["cases"].values()):
        raise ValueError("Only benchmark-eligible tmQMg fixtures may be retained")
    return payload["cases"]


def fixture(case_id: str) -> dict:
    case = fixtures()[case_id]
    if hashlib.sha256(case["xyz"].encode()).hexdigest() != case["xyz_sha256"]:
        raise ValueError(f"Fixture XYZ checksum mismatch: {case_id}")
    return case


def reference_graph(case_id: str) -> Chem.Mol:
    """Read the native reference exactly as supplied by the pinned release."""
    from rdkit import Chem

    case = fixture(case_id)
    if not case["benchmark"]:
        raise ValueError(f"{case_id} is excluded by the release: {case['exclusion']}")
    data = zlib.decompress(base64.b64decode(case["native_graph"]))
    if hashlib.sha256(data).hexdigest() != case["graph_sha256"]:
        raise ValueError(f"Fixture native graph checksum mismatch: {case_id}")
    return Chem.Mol(data)


@lru_cache(maxsize=1)
def _reviewed_graphs() -> dict[str, Chem.Mol]:
    """Index local Lewis assignments by ID and verify their parent graph and geometry provenance."""
    from rdkit import Chem

    graphs = {}
    path = ROOT.parent / "reviewed-references.sdf"
    for graph in Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False):
        if graph is None:
            raise ValueError(f"Invalid reviewed graph in {path}")
        case_id = graph.GetProp("_Name")
        case = fixture(case_id)
        if (
            case_id in graphs
            or graph.GetProp("PARENT_GRAPH_SHA256") != case["graph_sha256"]
            or graph.GetProp("XYZ_SHA256") != case["xyz_sha256"]
        ):
            raise ValueError(f"Reviewed reference provenance mismatch: {case_id}")
        graph.UpdatePropertyCache()
        graphs[case_id] = graph
    return graphs


def regression_reference_graph(case_id: str) -> Chem.Mol:
    """Return a fresh graph with the reviewed assignment when the case has one."""
    from rdkit import Chem

    reviewed = _reviewed_graphs().get(case_id)
    if reviewed is not None:
        return Chem.Mol(reviewed)
    return reference_graph(case_id)


def regression_case_ids() -> list[str]:
    """Read the stable regression selection from the eligible published input IDs."""
    ids = json.loads((ROOT / "regression_case_ids.json").read_text(encoding="utf-8"))
    if not isinstance(ids, list) or any(not isinstance(key, str) for key in ids):
        raise ValueError("Regression selection must be a list of case IDs")
    if ids != sorted(set(ids)) or any(key not in fixtures() for key in ids):
        raise ValueError("Regression selection must contain unique, sorted, published fixture IDs")
    return ids
