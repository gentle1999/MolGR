"""Offline integrity and generator tests; no network or full dataset required."""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import zlib
from pathlib import Path

import pytest
from rdkit import Chem, rdBase

from benchmarks.dataset_xyz_benchmark import tmqmg as corrected
from scripts import freeze_tmqmg_fixtures as freezer
from scripts import tmqmg_fixture_data as data


ROOT = Path(__file__).parent / "data"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def test_all_offline_inputs_and_native_targets_are_consistent():
    cases = data.fixtures()
    assert len(cases) == 941
    assert sum(c["benchmark"] for c in cases.values()) == 941
    assert digest((data.ROOT / "release-manifest.json").read_bytes()) == corrected.MANIFEST_SHA256
    for case_id in cases:
        case = data.fixture(case_id)
        assert case["id"] == case_id
        assert case["benchmark"] and not case["exclusion"]
        native = data.reference_graph(case_id)
        xyz = Chem.MolFromXYZBlock(case["xyz"])
        assert [a.GetAtomicNum() for a in native.GetAtoms()] == [
            a.GetAtomicNum() for a in xyz.GetAtoms()
        ]
        assert Chem.GetFormalCharge(native) == case["charge"]
        assert (native.GetConformer().GetPositions() == xyz.GetConformer().GetPositions()).all()
        assert case["reconstruction_scope"] == (
            native.GetProp("STANDARD_RECONSTRUCTION_SCOPE")
            if native.HasProp("STANDARD_RECONSTRUCTION_SCOPE")
            else ""
        )
    for path in ROOT.rglob("*.xyz"):
        if path.stem.isalpha() and path.stem.isupper():
            assert path.stem in cases, str(path)
            assert path.read_bytes() == cases[path.stem]["xyz"].encode(), str(path)


@pytest.mark.parametrize("case_id", sorted(freezer.DIRECT_CASE_IDS))
def test_direct_regression_inputs_have_published_native_targets(case_id):
    case = data.fixture(case_id)
    assert case["benchmark"] and not case["exclusion"]
    assert data.reference_graph(case_id).GetNumAtoms() == int(case["xyz"].splitlines()[0])


@pytest.mark.parametrize(
    "case_id,metal_charge",
    [("BARSET", 4), ("BIMLUF", 1), ("KIWCEY", 6), ("KIWCIC", 6), ("SOVZOT", 0), ("YICXUE", 2)],
)
def test_reviewed_regression_graphs_preserve_the_published_inputs(case_id, metal_charge):
    case = data.fixture(case_id)
    graph = data.regression_reference_graph(case_id)
    published = data.reference_graph(case_id)
    assert graph.GetProp("PARENT_GRAPH_SHA256") == case["graph_sha256"]
    assert graph.GetProp("XYZ_SHA256") == case["xyz_sha256"]
    assert graph.GetProp("CSD_TRANSCRIPTION") and graph.GetProp("RATIONALE")
    assert [a.GetAtomicNum() for a in graph.GetAtoms()] == [
        a.GetAtomicNum() for a in published.GetAtoms()
    ]
    assert (graph.GetConformer().GetPositions() == published.GetConformer().GetPositions()).all()
    assert Chem.GetFormalCharge(graph) == case["charge"]
    assert sum(a.GetNumRadicalElectrons() for a in graph.GetAtoms()) == case["multiplicity"] - 1
    metal = next(a for a in graph.GetAtoms() if a.GetSymbol() == case["metal_center"])
    assert metal.GetFormalCharge() == metal_charge


def test_historical_candidate_selection_keeps_stable_ids_and_original_ordinals():
    expected = {
        113: "ABISEL",
        232: "ACIFEA",
        430: "ADIXUI",
        431: "ADIYAP",
        522: "AFAKAW",
        627: "AFUGEQ",
        684: "AGAZOA",
        727: "AGICOJ",
        1132: "AKONEW",
    }
    assert set(expected) == freezer.SEARCH_INPUT_INDICES
    for ordinal, case_id in expected.items():
        assert data.fixture(case_id)["input_index"] == ordinal


def test_group_manifests_have_only_published_reference_metadata():
    spec = json.loads((ROOT / "tmqmg/fixture_sources.json").read_text(encoding="utf-8"))
    for group, selections in spec.items():
        with (ROOT / "tmqmg" / group / "manifest.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert [r["case_id"] for r in rows] == [s["id"] for s in selections]
        for row, selection in zip(rows, selections):
            assert {
                "approved_graph",
                "accepted_as_alternative",
                "approved_answer_database",
                "classification",
                "reason",
            }.isdisjoint(selection)
            case = data.fixture(row["case_id"])
            for key in ("charge", "multiplicity", "metal_center", "xyz_sha256"):
                assert row[key] == str(case[key])
            assert row["graph_sha256"] == (case["graph_sha256"] or "")
            assert row["exclusion"] == (case["exclusion"] or "")
            assert row["benchmark"] == str(case["benchmark"]).lower()
            assert not (ROOT / "tmqmg" / group / "approved_graph").exists()


def test_regression_selection_preserves_coverage_without_database():
    ids = data.regression_case_ids()
    assert len(ids) == 706
    cases = [data.fixture(case_id) for case_id in ids]
    assert sum(c["reconstruction_scope"] == "spin_coupling_out_of_scope" for c in cases) == 36
    assert all(c["benchmark"] and c["native_graph"] for c in cases)
    assert not list(ROOT.rglob("*.sqlite*"))


@pytest.mark.parametrize("selection", [["ZZ"], ["ABELOK", "ABELOK"], ["ZZ", "AA"], [1], {}])
def test_regression_selection_rejects_invalid_ids(tmp_path, monkeypatch, selection):
    # Exercise the selection reader without duplicating the complete published store.
    cases = data.fixtures()
    monkeypatch.setattr(data, "fixtures", lambda: cases)
    monkeypatch.setattr(data, "ROOT", tmp_path)
    (tmp_path / "regression_case_ids.json").write_text(json.dumps(selection))
    with pytest.raises(ValueError):
        data.regression_case_ids()


def test_auxiliary_manifests_only_select_retained_eligible_inputs():
    manifests = list((ROOT / "tmqmg").rglob("manifest.json")) + [
        ROOT / "reviewed/tmqmg/manifest.json",
        ROOT / "reviewed/reconstruction/manifest.json",
    ]
    for path in manifests:
        for row in json.loads(path.read_text(encoding="utf-8"))["fixtures"]:
            if row.get("source") == "molgr_reconstruction":
                continue  # Independent manual controls, not tmQMg inputs.
            assert data.fixture(row["case_id"])["benchmark"]
            assert row.get("kind") != "published_excluded"


@pytest.fixture
def tiny_release(tmp_path, monkeypatch):
    root = tmp_path / "release"
    root.mkdir()
    xyz = b"2\n\nCu 0 0 0\nH 0 0 1.5\n"
    mol = Chem.MolFromXYZBlock(xyz.decode())
    for atom in mol.GetAtoms():
        atom.SetNoImplicit(True)
    mol.UpdatePropertyCache(strict=False)
    graph = mol.ToBinary(
        Chem.PropertyPickleOptions.AllProps | Chem.PropertyPickleOptions.CoordsAsDouble
    )
    props = {
        "CASE_ID": "AA",
        "INPUT_CHARGE": "0",
        "INPUT_MULTIPLICITY": "1",
        "XYZ_SHA256": digest(xyz),
        "GRAPH_SHA256": digest(graph),
        "NATIVE_GRAPH_ZLIB_BASE64": base64.b64encode(zlib.compress(graph)).decode(),
        "NATIVE_RDKIT_VERSION": rdBase.rdkitVersion,
    }
    sdf = b"AA\n" + "".join(f"\n>  <{k}>\n{v}\n\n" for k, v in props.items()).encode() + b"$$$$\n"
    metadata = [
        {
            "id": case_id,
            "charge": 0,
            "multiplicity": 1,
            "benchmark": case_id == "AA",
            "exclusion": None if case_id == "AA" else "Missing input hydrogens",
            "xyz_sha256": digest(xyz),
            "graph_sha256": digest(graph) if case_id == "AA" else None,
        }
        for case_id in ("AA", "AB")
    ]
    part = {
        "first": "AA",
        "last": "AB",
        "inputs": 2,
        "targets": 1,
        "xyz": "inputs.xyz",
        "sdf": "targets.sdf",
        "metadata": "inputs.jsonl",
    }
    contents = {
        "xyz": b"".join(
            xyz.replace(b"2\n\n", f"2\nid={case_id} charge=0 multiplicity=1\n".encode(), 1)
            for case_id in ("AA", "AB")
        ),
        "sdf": sdf,
        "metadata": "".join(json.dumps(m) + "\n" for m in metadata).encode(),
    }
    manifest = {"parts": [part], "sha256": {part[k]: digest(v) for k, v in contents.items()}}
    for key, content in contents.items():
        (root / part[key]).write_bytes(content)
    (root / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(corrected, "load_manifest", lambda directory: manifest)
    source = {
        "repository": corrected.REPOSITORY,
        "revision": corrected.REVISION,
        "manifest_sha256": corrected.MANIFEST_SHA256,
    }
    monkeypatch.setattr(corrected, "provenance", lambda directory: source)
    for name in ("DATA-LICENSE.md", "CC-BY-4.0.txt", "tmQMg-LICENSE.txt", "xyz2mol_tm-LICENSE.txt"):
        path = root / "sources" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("Synthetic test notice\n")
    return root, manifest, xyz, graph


def test_extract_includes_exclusions_and_uses_original_input_ordinals(tiny_release):
    root, _, xyz, graph = tiny_release
    records, sdfs = freezer.read_release(root, {"AA"}, {2})
    assert set(records) == {"AA", "AB"}
    assert records["AB"]["input_index"] == 2
    assert records["AA"]["xyz"].encode() == xyz
    assert records["AA"]["metal_center"] == "Cu"
    assert zlib.decompress(base64.b64decode(records["AA"]["native_graph"])) == graph
    assert not records["AB"]["benchmark"] and "native_graph" not in records["AB"]
    assert set(sdfs) == {"AA"}


@pytest.mark.parametrize("name", ["inputs.xyz", "targets.sdf", "inputs.jsonl"])
def test_extraction_rejects_corrupted_parts(tiny_release, name):
    root, _, _, _ = tiny_release
    with (root / name).open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum mismatch"):
        freezer.read_release(root, {"AA"})


@pytest.mark.parametrize(
    "corruption, message",
    [
        ("state", "electronic state"),
        ("duplicate", "duplicate ID"),
        ("count", "metadata count"),
        ("graph", "Graph identity"),
        ("eligibility", "eligibility"),
    ],
)
def test_extraction_validates_internal_metadata(tiny_release, corruption, message):
    root, manifest, _, _ = tiny_release
    path = root / "inputs.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if corruption == "state":
        rows[0]["charge"] = 1
    elif corruption == "duplicate":
        rows[1]["id"] = "AA"
    elif corruption == "count":
        rows.pop()
    elif corruption == "graph":
        rows[0]["graph_sha256"] = "0" * 64
    else:
        rows[0]["exclusion"] = "Invalid"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest["sha256"][path.name] = digest(path.read_bytes())
    with pytest.raises(ValueError, match=message):
        freezer.read_release(root, {"AA", "AB"})


def test_absent_ids_and_input_ordinals_fail(tiny_release):
    root, _, _, _ = tiny_release
    with pytest.raises(ValueError, match="IDs absent"):
        freezer.read_release(root, {"ZZ"})
    with pytest.raises(ValueError, match="indices absent"):
        freezer.read_release(root, set(), {3})


def test_regeneration_is_idempotent_and_removes_retired_approval(
    tiny_release, tmp_path, monkeypatch
):
    root, _, xyz, _ = tiny_release
    target = tmp_path / "data"
    monkeypatch.setattr(freezer, "SEARCH_INPUT_INDICES", set())
    monkeypatch.setattr(freezer, "DIRECT_CASE_IDS", set())
    freezer._write_json(
        target / "tmqmg/fixture_sources.json",
        {
            "sulfur_electronic_assignment": [
                {
                    "id": "AA",
                    "classification": "old",
                    "reason": "local control",
                    "approved_graph": "approved_graph/AA.sdf",
                }
            ]
        },
    )
    xyz_path = target / "tmqmg/sulfur_electronic_assignment/AA.xyz"
    xyz_path.parent.mkdir(parents=True)
    xyz_path.write_bytes(xyz)
    (xyz_path.parent / "manifest.csv").write_text("case_id,classification,reason\n")
    excluded_xyz = target / "tmqmg/excluded/AB.xyz"
    excluded_xyz.parent.mkdir()
    excluded_xyz.write_bytes(xyz)
    excluded_sdf = target / "reviewed/reconstruction/reference_graph/AB.sdf"
    excluded_sdf.parent.mkdir(parents=True)
    excluded_sdf.write_text("Retired excluded target")
    freezer._write_json(
        target / "tmqmg/excluded/manifest.json",
        {"fixtures": [{"case_id": "AB", "xyz_file": "AB.xyz"}]},
    )
    freezer._write_json(
        target / "tmqmg/fixture_sources.json",
        {
            **json.loads((target / "tmqmg/fixture_sources.json").read_text(encoding="utf-8")),
            "excluded": [
                {"id": "AB", "selection_classification": "old", "selection_reason": "old"}
            ],
        },
    )
    (excluded_xyz.parent / "manifest.csv").write_text("case_id,classification,reason\nAB,old,old\n")
    old = xyz_path.parent / "approved_graph/AA.sdf"
    old.parent.mkdir()
    old.write_text("Retired target")
    freezer._write_json(
        target / "reviewed/tmqmg/manifest.json",
        {"fixtures": [{"case_id": "AA"}, {"case_id": "AB"}]},
    )
    freezer._write_json(
        target / "reviewed/reconstruction/manifest.json",
        {
            "fixtures": [
                {"case_id": "AB", "source": "tmQMg-Corrected", "kind": "published_excluded"}
            ]
        },
    )
    freezer._write_json(target / "tmqmg/published/regression_case_ids.json", ["AA", "AB"])
    (target / "xyz_stress_manifest.csv").write_text(
        "case_id,charge,spin_multiplicity,xyz_file,reference_file,reference_source,reference_status\n"
        "AB,0,1,tmqmg/excluded/AB.xyz,,,source_issue\n"
    )
    freezer.freeze_fixtures(dataset=root, data_dir=target)
    before = {str(p.relative_to(target)): p.read_bytes() for p in target.rglob("*") if p.is_file()}
    freezer.freeze_fixtures(dataset=root, data_dir=target)
    after = {str(p.relative_to(target)): p.read_bytes() for p in target.rglob("*") if p.is_file()}
    assert before == after
    assert not old.exists()
    assert [
        r["case_id"]
        for r in json.loads((target / "reviewed/tmqmg/manifest.json").read_text(encoding="utf-8"))[
            "fixtures"
        ]
    ] == ["AA"]
    assert "AB" not in (target / "xyz_stress_manifest.csv").read_text(encoding="utf-8")
    assert not excluded_xyz.exists() and not excluded_sdf.exists()
    assert set(
        json.loads((target / "tmqmg/published/cases.json").read_text(encoding="utf-8"))["cases"]
    ) == {"AA"}
    assert (
        json.loads((target / "tmqmg/fixture_sources.json").read_text(encoding="utf-8"))["excluded"]
        == []
    )
    assert (
        json.loads((target / "tmqmg/excluded/manifest.json").read_text(encoding="utf-8"))[
            "fixtures"
        ]
        == []
    )
    assert (
        json.loads((target / "reviewed/reconstruction/manifest.json").read_text(encoding="utf-8"))[
            "fixtures"
        ]
        == []
    )
    with (target / "tmqmg/excluded/manifest.csv").open(encoding="utf-8", newline="") as stream:
        assert list(csv.DictReader(stream)) == []
    assert json.loads(
        (target / "tmqmg/published/regression_case_ids.json").read_text(encoding="utf-8")
    ) == ["AA"]
    assert not list(target.rglob("*.sqlite*"))


@pytest.mark.parametrize("corruption", ["source", "manifest", "xyz", "graph", "excluded"])
def test_offline_loader_rejects_corruption(tmp_path, monkeypatch, corruption):
    payload = json.loads((data.ROOT / "cases.json").read_text(encoding="utf-8"))
    payload["cases"] = {"ABELOK": payload["cases"]["ABELOK"]}
    manifest = (data.ROOT / "release-manifest.json").read_bytes()
    if corruption == "source":
        payload["source_dataset"]["revision"] = "wrong"
    elif corruption == "manifest":
        manifest += b"\n"
    elif corruption == "excluded":
        payload["cases"]["ABELOK"]["benchmark"] = False
    elif corruption == "xyz":
        payload["cases"]["ABELOK"]["xyz"] += "\n"
    else:
        payload["cases"]["ABELOK"]["graph_sha256"] = "0" * 64
    (tmp_path / "cases.json").write_text(json.dumps(payload))
    (tmp_path / "release-manifest.json").write_bytes(manifest)
    monkeypatch.setattr(data, "ROOT", tmp_path)
    data.fixtures.cache_clear()
    try:
        with pytest.raises(ValueError):
            data.reference_graph("ABELOK")
    finally:
        data.fixtures.cache_clear()
