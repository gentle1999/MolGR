"""Check the optional baseline's input boundary and preservation of upstream graphs."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from rdkit import Chem

from benchmarks.smiles_xyz_benchmark.methods import xyz2mol_tm as adapter


def test_charge_is_forwarded_without_reference_or_multiplicity(monkeypatch):
    molecule = Chem.MolFromSmiles("[Cl-]->[Cu+2]<-[Cl-]")
    paths = []
    xyz = "3\ninput\nCl -2 0 0\nCu 0 0 0\nCl 2 0 0\n"

    def reconstruct(path, charge, *, with_stereo):
        assert Path(path).read_text() == xyz
        assert charge == -1
        assert with_stereo is False
        paths.append(Path(path))
        return molecule

    monkeypatch.setattr(adapter, "_load_reconstructor", lambda: reconstruct)
    output = adapter.XYZ2MolTMMethod().run(
        {
            "xyz_block": xyz,
            "total_charge": -1,
            "total_radical_electrons": 3,
            "ground_truth_smiles": "must not enter reconstruction",
        }
    )
    assert output.status == "ok"
    assert output.rdkit_mol is molecule
    assert molecule.GetNumConformers() == 0
    assert output.predicted_smiles == "[Cl-]->[Cu+2]<-[Cl-]"
    assert all(b.GetBondType() == Chem.BondType.DATIVE for b in molecule.GetBonds())
    assert paths and not paths[0].exists()


def test_serialization_failure_retains_returned_graph(monkeypatch):
    molecule = Chem.MolFromSmiles("[Fe+2]")
    monkeypatch.setattr(adapter, "_load_reconstructor", lambda: lambda *a, **k: molecule)

    def fail(*args, **kwargs):
        raise ValueError("serialization failed")

    monkeypatch.setattr(adapter.Chem, "MolToSmiles", fail)
    output = adapter.XYZ2MolTMMethod().run({"xyz_block": "1\n\nFe 0 0 0\n"})
    assert output.status == "error"
    assert output.rdkit_mol is molecule
    assert "serialization failed" in output.error


@pytest.mark.parametrize("failure", [None, ValueError("upstream failed")])
def test_failed_reconstruction_does_not_fabricate_a_graph(monkeypatch, failure):
    def reconstruct(*args, **kwargs):
        if failure is not None:
            raise failure
        return None

    monkeypatch.setattr(adapter, "_load_reconstructor", lambda: reconstruct)
    output = adapter.XYZ2MolTMMethod().run({"xyz_block": "1\n\nFe 0 0 0\n"})
    assert output.status == "error"
    assert output.rdkit_mol is None
    assert output.predicted_smiles is None


def test_missing_optional_dependency_is_skipped(monkeypatch):
    def missing():
        raise ModuleNotFoundError("xyz2mol_tm")

    monkeypatch.setattr(adapter, "_load_reconstructor", missing)
    output = adapter.XYZ2MolTMMethod().run({"xyz_block": "1\n\nFe 0 0 0\n"})
    assert output.status == "skipped"


def test_upstream_reconstructs_metal_fixture():
    if importlib.util.find_spec("xyz2mol_tm") is None:
        pytest.skip("optional upstream package not installed")
    root = Path(__file__).resolve().parents[1]
    code = """
from pathlib import Path
from rdkit import Chem
from benchmarks.smiles_xyz_benchmark.methods.xyz2mol_tm import XYZ2MolTMMethod
xyz = Path('tests/data/tmqmg/reconstruction/ABACAL.xyz').read_text()
out = XYZ2MolTMMethod().run({'xyz_block': xyz, 'total_charge': 0})
assert out.status == 'ok', out.error
assert Chem.GetFormalCharge(out.rdkit_mol) == 0
assert Chem.AddHs(out.rdkit_mol).GetNumAtoms() == int(xyz.splitlines()[0])
assert any(b.GetBondType() == Chem.BondType.DATIVE for b in out.rdkit_mol.GetBonds())
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


def test_late_registry_import_does_not_crash_native_runtime():
    code = """
import molgr
from benchmarks.smiles_xyz_benchmark.methods.xyz2mol_tm import XYZ2MolTMMethod
out = XYZ2MolTMMethod().run({'xyz_block': chr(10).join(['1', '', 'Fe 0 0 0', ''])})
assert out.status == 'skipped', out
assert 'fresh process' in out.error
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
