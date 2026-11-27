import hashlib
import json
import sqlite3
import struct
import zlib

import pytest
from rdkit import Chem

from benchmarks.dataset_xyz_benchmark import borylxat
from benchmarks.dataset_xyz_benchmark.comparison import comparison_policy
from benchmarks.dataset_xyz_benchmark.prepare import prepare


AAM = "[H:3][B:1][H:4].[Cl:2][C:5]([H:6])([H:7])[H:8]>>[Cl:2][B:1]([H:3])[H:4].[C:5]([H:8])([H:7])[H:6]"
COMPONENTS = [
    ("br", "complex_r", [1, 1, 5]),
    ("clr", "Cl", [6, 17, 1, 1, 1]),
    ("bp", "complex_p", [1, 17, 5, 1]),
    ("cp", "c_radical", [6, 1, 1, 1]),
]


def test_optimized_structures_are_scored_once_against_mapped_species(tmp_path, monkeypatch):
    source = tmp_path / "BorylXAT-DB.db"
    with sqlite3.connect(source) as db:
        db.execute(
            "CREATE TABLE systems(id INTEGER PRIMARY KEY,key_value_pairs TEXT,natoms INTEGER,charge REAL,numbers BLOB,positions BLOB)"
        )
        for key, category, nums in [*COMPONENTS, ("no_aam", "complex_r", [5, 1, 1])]:
            n = len(nums)
            meta = {"key": key, "category": category, "smiles": "C"}
            db.execute(
                "INSERT INTO systems VALUES(NULL,?,?,?,?,?)",
                (
                    json.dumps(meta),
                    n,
                    0,
                    struct.pack(f"<{n}i", *nums),
                    struct.pack(f"<{3 * n}d", 1e-6, *range(1, 3 * n)),
                ),
            )
        for key in ("ts1", "ts2"):
            label = (
                AAM
                if key == "ts1"
                else ">>".join(".".join(reversed(side.split("."))) for side in AAM.split(">>"))
            )
            meta = {
                "key": key,
                "category": "ts",
                "smiles": label,
                **{role: comp[0] for (role, _, _), comp in zip(borylxat.ROLES, COMPONENTS)},
            }
            db.execute("INSERT INTO systems VALUES(NULL,?,0,0,NULL,NULL)", (json.dumps(meta),))
    monkeypatch.setitem(borylxat.SOURCE, "size_bytes", source.stat().st_size)
    monkeypatch.setitem(
        borylxat.SOURCE, "published_md5", hashlib.md5(source.read_bytes()).hexdigest()
    )
    dest = tmp_path / "cases.sqlite"
    result = prepare(source, dest, "borylxat")
    assert result["reference_statuses"] == {"unresolved": 1, "valid": 4}
    with sqlite3.connect(dest) as db:
        for key, _, numbers in COMPONENTS:
            xyz, native, multiplicity, evidence = db.execute(
                "SELECT xyz,reference_native,multiplicity,evidence FROM cases WHERE molecule=?",
                (key,),
            ).fetchone()
            geometry = Chem.MolFromXYZBlock(zlib.decompress(xyz).decode())
            assert [a.GetAtomicNum() for a in geometry.GetAtoms()] == numbers
            assert geometry.GetConformer().GetAtomPosition(0).x == 1e-6
            assert tuple(geometry.GetConformer().GetAtomPosition(1)) == (3.0, 4.0, 5.0)
            graph = Chem.Mol(native)
            assert graph.GetNumAtoms() == len(numbers)
            assert sorted(a.GetAtomicNum() for a in graph.GetAtoms()) == sorted(numbers)
            assert multiplicity == (2 if key in ("br", "cp") else 1)
            assert json.loads(evidence)["reaction_keys"] == ["ts1", "ts2"]
    for release in ({}, {"reference_basis": "AAM"}):
        with pytest.raises(ValueError, match="Reprepare"):
            comparison_policy({"dataset": "borylxat", "release": release})


@pytest.mark.parametrize(
    "bad",
    [
        AAM.replace("[H:3]", "[H:4]", 1),
        AAM.replace("[H:3]", "[2H:3]", 1),
        AAM.replace("[B:1][H:4]", "[BH:1]"),
    ],
)
def test_explicit_mapped_hydrogens_are_conserved(bad):
    with pytest.raises(ValueError):
        borylxat.reaction_fragments(bad)
