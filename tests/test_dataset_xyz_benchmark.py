from __future__ import annotations

import gzip
import io
import json
import os
import pickle
import sqlite3
import tarfile
import zlib
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from benchmarks.dataset_xyz_benchmark.prepare import (
    GeomUnpickler,
    bde_cases,
    file_hash,
    make_case,
    prepare,
)
from benchmarks.dataset_xyz_benchmark.report import summarize


def water(x=0.0):
    mol = Chem.MolFromSmiles("[H]O[H]", sanitize=False)
    Chem.SanitizeMol(mol)
    conf = Chem.Conformer(3)
    conf.Set3D(True)
    for i, pos in enumerate(((x + 0.96, 0, 0), (x, 0, 0), (x - 0.24, 0.93, 0))):
        conf.SetAtomPosition(i, pos)
    mol.AddConformer(conf)
    return mol


def geom_archive(path):
    source = {
        "smiles": "O",
        "charge": 0,
        "totalconfs": 100,
        "conformers": [
            {"rd_mol": water(i), "relativeenergy": energy, "geom_id": i, "degeneracy": 30 + i}
            for i, energy in enumerate((2.0, 0.0, 1.0))
        ],
    }
    data = pickle.dumps(source)
    with tarfile.open(path, "w") as archive:
        info = tarfile.TarInfo("rdkit_folder/drugs/O.pickle")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))


def test_all_released_conformers_preserve_coordinates_and_groups(tmp_path):
    archive, out = tmp_path / "geom.tar", tmp_path / "cases.sqlite"
    geom_archive(archive)
    provenance = prepare(archive, out, "geom-drugs", limit_molecules=1)
    assert provenance["cases"] == 3
    assert provenance["molecules"] == 1
    assert provenance["pilot"] is True
    with sqlite3.connect(out) as db:
        rows = db.execute(
            "SELECT molecule,xyz,reference_status,evidence FROM cases ORDER BY id"
        ).fetchall()
    assert [r[0] for r in rows] == ["O"] * 3
    assert [r[2] for r in rows] == ["valid"] * 3
    assert [json.loads(r[3])["relativeenergy"] for r in rows] == [2, 0, 1]
    for i, row in enumerate(rows):
        xyz = Chem.MolFromXYZBlock(zlib.decompress(row[1]).decode())
        assert xyz.GetNumBonds() == 0
        assert xyz.GetConformer().GetPositions() == pytest.approx(
            water(i).GetConformer().GetPositions()
        )


def test_geom_benchmark_selection(tmp_path, monkeypatch):
    from benchmarks.dataset_xyz_benchmark import prepare as importer

    def add(archive, name, data):
        member = tarfile.TarInfo("rdkit_folder/" + name)
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))

    path = tmp_path / "geom.tar"
    with tarfile.open(path, "w") as archive:
        inventory = {key: {"pickle_path": f"drugs/{key}.pickle"} for key in "ABCDEFG"}
        inventory["alias"] = inventory["B"]
        inventory["no_path"] = {}
        add(archive, "summary_drugs.json", json.dumps(inventory).encode())
        for subset in ("qm9", "drugs"):
            for key in "ABCDEF":
                source = {
                    "smiles": "C" if key == "F" else key,
                    "charge": int(key == "E"),
                    "conformers": [{"rd_mol": water(i), "geom_id": i} for i in range(3)],
                }
                if subset == "qm9":
                    source["smiles"] = key
                add(archive, f"{subset}/{key}.pickle", pickle.dumps(source))

    # Drugs rank order is G (missing), E (invalid), C, F (same parent as C), D, B, A.
    monkeypatch.setattr(importer, "GEOM_DRUGS_PARENTS", 3)
    drugs = list(importer.geom_cases(path, "geom-drugs"))
    assert [(r[1], r[2], json.loads(r[9])["sample_rank"]) for r in drugs] == [
        ("C", "0", 3),
        ("D", "2", 5),
        ("B", "1", 6),
    ]
    qm9 = list(importer.geom_cases(path, "geom-qm9"))
    assert [(r[1], r[2]) for r in qm9] == list(zip("ABCDEF", "222210"))
    assert [r[1] for r in qm9 if r[7] != "valid"] == ["E"]
    for row in drugs + qm9:
        evidence = json.loads(row[9])
        assert evidence["seed"] == 20261125
        assert evidence["source_geom_id"] == int(row[2])
        assert Chem.Mol(row[10]).GetConformer().GetPositions() == pytest.approx(
            water(int(row[2])).GetConformer().GetPositions()
        )
    monkeypatch.setattr(importer, "GEOM_DRUGS_PARENTS", 5)
    with pytest.raises(ValueError, match="exhausted"):
        list(importer.geom_cases(path, "geom-drugs"))


def test_geom_scoring_concretizes_only_exact_element_queries():
    from benchmarks.dataset_xyz_benchmark.prepare import _geom_scoring_reference

    exact = Chem.MolFromSmarts("[#6]").ToBinary()
    concrete, count = _geom_scoring_reference(exact)
    atom = Chem.Mol(concrete).GetAtomWithIdx(0)
    assert count == 1 and atom.GetAtomicNum() == 6 and not atom.HasQuery()
    for smarts in ("[#6,#7]", "[#6]~[#7]"):
        ambiguous = Chem.MolFromSmarts(smarts).ToBinary()
        assert _geom_scoring_reference(ambiguous) == (ambiguous, -1)


def test_reference_cannot_add_unobserved_hydrogen():
    mol = water()
    case = make_case("test", "water", 0, mol, Chem.MolFromSmiles("CO"), 0, 1, {})
    assert case[7] == "excluded"
    assert "composition" in case[8]


def test_geometries_are_not_silently_replaced(tmp_path):
    archive, out = tmp_path / "geom.tar", tmp_path / "cases.sqlite"
    geom_archive(archive)
    with pytest.raises(ValueError, match="size/MD5"):
        prepare(archive, out, "geom-drugs")
    with pytest.raises(pickle.UnpicklingError):
        GeomUnpickler(io.BytesIO(pickle.dumps(Path("/tmp")))).load()


def test_macro_matching_weights_parents_equally_and_keeps_failures(tmp_path):
    source = tmp_path / "cases.sqlite"
    results = tmp_path / "results.sqlite"
    with sqlite3.connect(source) as db:
        db.executescript(
            "CREATE TABLE metadata(key TEXT,value TEXT); CREATE TABLE cases(id INTEGER, molecule TEXT, reference_status TEXT);"
        )
        db.execute(
            "INSERT INTO metadata VALUES (?,?)",
            (
                "provenance",
                json.dumps(
                    {
                        "complete": True,
                        "dataset": "test",
                        "molecules": 2,
                        "cases": 4,
                        "reference_statuses": {"valid": 4},
                    }
                ),
            ),
        )
        db.executemany(
            "INSERT INTO cases VALUES (?,?,?)",
            [(1, "a", "valid"), (2, "b", "valid"), (3, "b", "valid"), (4, "b", "valid")],
        )
    with sqlite3.connect(results) as db:
        db.execute("CREATE TABLE results(case_id INTEGER,decision TEXT)")
        db.executemany(
            "INSERT INTO results VALUES (?,?)",
            [(1, "equivalent"), (2, "not_equivalent"), (3, None)],
        )
    results.with_name("results-provenance.json").write_text(
        json.dumps({"case_store_sha256": file_hash(source)})
    )
    with pytest.raises(ValueError, match="Incomplete run"):
        summarize(source, results)
    partial = summarize(source, results, allow_partial=True)
    assert partial["complete"] is False
    with sqlite3.connect(results) as db:
        db.execute("INSERT INTO results VALUES (4,?)", ("inconclusive",))
    summary = summarize(source, results)
    assert summary["molecule_macro_match_rate"] == 0.5
    assert summary["conformer_micro_match_rate"] == 0.25
    assert summary["all_eligible_conformers_match_molecule_rate"] == 0.5
    assert summary["decision_counts"]["unassessed"] == 1


def test_native_references_are_not_replaced_by_display_smiles(monkeypatch):
    from benchmarks.dataset_xyz_benchmark.comparison import native_reference
    from benchmarks.dataset_xyz_benchmark.prepare import make_case

    reference = Chem.MolFromSmiles("[13CH4]")
    geometry = Chem.AddHs(reference)
    AllChem.EmbedMolecule(geometry, randomSeed=42)
    native = reference.ToBinary()

    def fail(*args, **kwargs):
        raise ValueError("cannot serialize display SMILES")

    monkeypatch.setattr(Chem, "MolToSmiles", fail)
    row = make_case("geom-qm9", "parent", 0, geometry, reference, 0, 1, {})
    assert row[4] is None and row[7] == "valid"
    assert row[10] == native
    assert native_reference(row[10], row[11]).ToBinary() == native
    with pytest.raises(ValueError, match="Native reference hash mismatch"):
        native_reference(native + b"changed", row[11])


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_invalid_comparison_timeouts_are_rejected(timeout):
    from benchmarks.dataset_xyz_benchmark.comparison import DEFAULT_POLICY, comparison_policy

    with pytest.raises(ValueError, match="comparison_timeout_seconds"):
        comparison_policy({"comparison_policy": DEFAULT_POLICY}, timeout)


def test_default_policy_can_be_overridden_without_mutating_source():
    from benchmarks.dataset_xyz_benchmark.comparison import DEFAULT_POLICY, comparison_policy

    source = {"comparison_policy": dict(DEFAULT_POLICY)}
    assert comparison_policy(source)["max_resonance"] == 5000
    assert comparison_policy(source, 8)["comparison_timeout_seconds"] == 8
    assert source["comparison_policy"] == DEFAULT_POLICY


@pytest.mark.parametrize(
    ("reference_smiles", "candidate_smiles", "decision"),
    [
        ("C", "C", "equivalent"),
        ("C", "N", "not_equivalent"),
        ("[O]N=O", "[O-][N+]=O", "inconclusive"),
    ],
)
def test_unified_comparison_preserves_real_decisions(reference_smiles, candidate_smiles, decision):
    from benchmarks.dataset_xyz_benchmark.comparison import compare_output
    from benchmarks.smiles_xyz_benchmark.methods.base import MethodRunOutput

    reference = {
        "reference_status": "valid",
        "ground_truth_rdmol": Chem.MolFromSmiles(reference_smiles),
    }
    output = MethodRunOutput(
        status="error",
        predicted_smiles=reference_smiles,
        equivalent=True,
        rdkit_mol=Chem.MolFromSmiles(candidate_smiles),
    )
    assert compare_output(reference, output)["decision"] == decision


def test_unified_comparison_does_not_accept_local_alternative_answers():
    from benchmarks.dataset_xyz_benchmark.comparison import compare_output
    from benchmarks.smiles_xyz_benchmark.methods.base import MethodRunOutput

    reference = {
        "reference_status": "valid",
        "ground_truth_rdmol": Chem.MolFromSmiles("C"),
        "alternative_rdmols": [Chem.MolFromSmiles("N")],
    }
    output = MethodRunOutput(status="ok", rdkit_mol=Chem.MolFromSmiles("N"))
    assert compare_output(reference, output)["decision"] == "not_equivalent"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("max_resonance", 0),
        ("max_resonance", True),
        ("max_resonance", 1.5),
        ("use_chirality", 0),
        ("ignore_metal_coordination_bonds", "true"),
        ("unexpected", 1),
    ],
)
def test_malformed_source_comparison_policy_is_rejected(key, value):
    from benchmarks.dataset_xyz_benchmark.comparison import DEFAULT_POLICY, comparison_policy

    with pytest.raises(ValueError, match="comparison policy"):
        comparison_policy({"comparison_policy": {**DEFAULT_POLICY, key: value}})


def test_unified_comparison_failures_and_ineligible_references(monkeypatch):
    from benchmarks.dataset_xyz_benchmark.comparison import compare_output
    from benchmarks.smiles_xyz_benchmark.methods.base import MethodRunOutput
    from molgr.utils import equivalence

    reference = {"reference_status": "valid", "ground_truth_rdmol": Chem.MolFromSmiles("C")}
    output = MethodRunOutput(status="ok", rdkit_mol=reference["ground_truth_rdmol"])

    def fail(*args, **kwargs):
        raise RuntimeError("evaluator failed")

    monkeypatch.setattr(equivalence, "evaluate_equivalence", fail)
    result = compare_output(reference, output)
    assert result["decision"] == "inconclusive"
    assert "evaluator failed" in result["comparison_reason"]
    assert compare_output({**reference, "reference_status": "excluded"}, output) == {}
    assert compare_output(reference, None) == {}
    assert compare_output(reference, MethodRunOutput(status="error")) == {}


@pytest.mark.skipif(os.name != "posix", reason="benchmark uses POSIX comparison timers")
def test_native_batch_scores_duplicate_geometries_and_resumes(tmp_path, monkeypatch):
    from benchmarks.dataset_xyz_benchmark import native_batch

    archive, cases, out = tmp_path / "geom.tar", tmp_path / "cases.sqlite", tmp_path / "out"
    geom_archive(archive)
    prepare(archive, cases, "geom-drugs", limit_molecules=1)
    with sqlite3.connect(cases) as db:
        db.execute("UPDATE cases SET xyz=(SELECT xyz FROM cases WHERE id=1) WHERE id=2")
        db.execute("UPDATE cases SET multiplicity=2 WHERE id=3")
    native_batch.run(cases, out, workers=2, checkers=1)
    with sqlite3.connect(out / "molgr_cpp.sqlite") as db:
        rows = db.execute("SELECT case_id,status,decision FROM results ORDER BY case_id").fetchall()
    assert rows == [(1, "ok", "equivalent"), (2, "ok", "equivalent"), (3, "error", None)]
    assert summarize(cases, out / "molgr_cpp.sqlite")["complete"]

    def unexpected_batch(*args):
        pytest.fail("Reconstructed an already scored case")

    monkeypatch.setattr(native_batch, "reconstruction_groups", unexpected_batch)
    native_batch.run(cases, out, workers=2, checkers=1)
    with pytest.raises(ValueError, match="provenance differs"):
        native_batch.run(cases, out, workers=1, checkers=1)


def test_bde_selects_only_carbon_containing_doublets(tmp_path):
    source = tmp_path / "test.sdf"
    with Chem.SDWriter(str(source)) as writer:
        for smiles in ("C", "[OH]", "[CH3]"):
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            assert AllChem.EmbedMolecule(mol, randomSeed=11) == 0
            mol.SetProp("_Name", smiles)
            mol.SetProp("SMILES", smiles)
            if smiles != "C":
                mol.SetProp("AtomSpins", json.dumps([1] + [0] * (mol.GetNumAtoms() - 1)))
            writer.write(mol)
    compressed = source.with_suffix(".sdf.gz")
    compressed.write_bytes(gzip.compress(source.read_bytes()))
    (case,) = bde_cases(compressed)
    assert case[:3] == ("bde", "[CH3]", "0")
    assert case[5:8] == (0, 2, "valid")
    assert json.loads(case[9])["sdf_record"] == 2
    assert Chem.Mol(case[10]).GetNumAtoms() == 4
    source.write_text(source.read_text().replace("[1, 0, 0, 0]", "[0, 0, 0, 0]"))
    with pytest.raises(ValueError, match="Inconsistent doublet"):
        list(bde_cases(source))
    with pytest.raises(ValueError, match="checksum"):
        prepare(source, tmp_path / "cases.sqlite", "bde")


@pytest.mark.parametrize(
    ("element", "entry"),
    [
        ("Fe", "xyz2mol_tm"),
        ("La", "xyz2mol_tm"),
        ("Lu", "xyz2mol_tm"),
        ("B", "xyz2mol_tm_local"),
        ("C", "xyz2mol_tm_local"),
        ("Cl", "xyz2mol_tm_local"),
    ],
)
def test_xyz2mol_routing_uses_xyz_atoms_only(element, entry):
    from benchmarks.dataset_xyz_benchmark.run import xyz2mol_entry

    xyz = f"2\nFe in comment\nH 0 0 0\n{element} 1 0 0\n"
    assert xyz2mol_entry(xyz) == entry
