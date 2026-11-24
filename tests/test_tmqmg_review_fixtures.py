from __future__ import annotations

import base64
import hashlib
import json
import zlib
from pathlib import Path

import pytest
from rdkit import Chem

from benchmarks.dataset_xyz_benchmark import tmqmg as corrected
from molgr import MolGRConfig
from molgr.interface import xyz_to_rdmol
from molgr.utils.equivalence import check_equivalence
from scripts.reconstruction_trace import load_review_fixture_cases
from scripts.tmqmg_fixture_data import fixture, reference_graph


ROOT = Path(__file__).parent / "data/reviewed/tmqmg"
MANIFEST = ROOT / "manifest.json"
PAYLOAD = json.loads(MANIFEST.read_text())
RECORDS = PAYLOAD["fixtures"]


def test_review_fixture_manifest_pins_published_dataset():
    assert PAYLOAD["schema_version"] == 2
    assert PAYLOAD["source_dataset"]["revision"] == corrected.REVISION
    assert PAYLOAD["source_dataset"]["manifest_sha256"] == corrected.MANIFEST_SHA256
    assert len({record["case_id"] for record in RECORDS}) == len(RECORDS)


@pytest.mark.parametrize("record", RECORDS, ids=lambda r: r["case_id"])
def test_review_sdf_retains_exact_published_native_graph(record):
    case = fixture(record["case_id"])
    assert record["kind"] == "published_reference"
    assert {"accepted_smiles", "approved_smiles", "review_status"}.isdisjoint(record)
    mol = next(
        iter(
            Chem.SDMolSupplier(str(ROOT / record["structure_file"]), sanitize=False, removeHs=False)
        )
    )
    native = zlib.decompress(base64.b64decode(mol.GetProp("NATIVE_GRAPH_ZLIB_BASE64")))
    assert hashlib.sha256(native).hexdigest() == record["graph_sha256"] == case["graph_sha256"]
    assert mol.GetProp("XYZ_SHA256") == case["xyz_sha256"]
    assert int(mol.GetProp("INPUT_CHARGE")) == record["total_charge"] == case["charge"]
    assert (
        int(mol.GetProp("INPUT_MULTIPLICITY"))
        == record["spin_multiplicity"]
        == case["multiplicity"]
    )


@pytest.mark.parametrize("backend", ["cpp", "python"])
@pytest.mark.parametrize("record", RECORDS, ids=lambda r: r["case_id"])
def test_reviewed_graphs_match_released_target(record, backend):
    case = fixture(record["case_id"])
    config = MolGRConfig()
    config.cpp_backend.max_threads = 1
    config.cpp_backend.target_bucket_parallel_max_threads = 1
    rebuilt = xyz_to_rdmol(
        case["xyz"], case["charge"], case["multiplicity"], backend=backend, config=config
    )
    equivalent, info = check_equivalence(
        reference_graph(record["case_id"]),
        rebuilt,
        use_chirality=False,
        ignore_metal_coordination_bonds=True,
        max_resonance=100000,
    )
    assert equivalent, info.reason


def test_review_manifest_is_a_live_trace_input_source():
    cases = load_review_fixture_cases(MANIFEST)
    assert {case.id for case in cases} == {r["case_id"] for r in RECORDS}
    for case in cases:
        published = fixture(case.id)
        assert case.xyz_source == "review_fixture"
        assert case.fixture_kind == "published_reference"
        assert (
            case.xyz_block == published["xyz"]
        )  # Never regenerate input from rounded SDF coordinates.
        assert case.expected_native_graph == published["native_graph"]
        assert (
            not case.expected_smiles_options
        )  # One native release target, not old accepted alternatives.


@pytest.mark.parametrize(
    "field,value",
    [
        ("total_charge", 999),
        ("spin_multiplicity", 999),
        ("kind", "published_excluded"),
        ("graph_sha256", "wrong"),
    ],
)
def test_trace_rejects_stale_published_review_metadata(tmp_path, field, value):
    payload = json.loads(MANIFEST.read_text())
    record = payload["fixtures"][0]
    payload["fixtures"] = [record]
    record["structure_file"] = str(ROOT / record["structure_file"])
    record[field] = value
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Published review"):
        load_review_fixture_cases(path)
