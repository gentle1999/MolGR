# pyright: reportMissingImports=false

from __future__ import annotations

from pathlib import Path

import pytest


pytest.importorskip("openbabel")
pytest.importorskip("rdkit")

from rdkit import Chem

from molgr.interface import xyz_to_rdmol


_EXPECTED_SMILES = "CC1=C(C)CCC1.COc1n[s+]([O-])nc1OC"


def _xyz_block() -> str:
    return (Path(__file__).parent / "data" / "xyz" / "aromatic_sulfur_oxide.xyz").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_aromatic_sulfur_oxide_uses_charge_separated_s_o_state(backend: str) -> None:
    reconstructed = xyz_to_rdmol(
        _xyz_block(),
        total_charge=0,
        spin_multiplicity=1,
        backend=backend,
    )
    heavy = Chem.RemoveHs(reconstructed)
    # This test compares a hydrogen-folded reference. Drop the explicit-H
    # writer's bracket requests on this copy before comparing canonical text.
    for atom in heavy.GetAtoms():
        if atom.HasProp("smilesSymbol"):
            atom.ClearProp("smilesSymbol")
    sulfur = next(atom for atom in heavy.GetAtoms() if atom.GetAtomicNum() == 16)
    negative_oxygen = next(
        neighbor
        for neighbor in sulfur.GetNeighbors()
        if neighbor.GetAtomicNum() == 8 and neighbor.GetFormalCharge() == -1
    )

    assert Chem.MolToSmiles(heavy, canonical=True, isomericSmiles=True) == _EXPECTED_SMILES
    assert sulfur.GetFormalCharge() == 1
    assert sulfur.GetIsAromatic()
    assert all(
        neighbor.GetFormalCharge() == 0
        for neighbor in sulfur.GetNeighbors()
        if neighbor.GetAtomicNum() == 7
    )
    assert heavy.GetBondBetweenAtoms(sulfur.GetIdx(), negative_oxygen.GetIdx()).GetBondType() == (
        Chem.BondType.SINGLE
    )
    assert heavy.GetBondBetweenAtoms(1, 2).GetBondType() == Chem.BondType.DOUBLE
    assert sum(atom.GetFormalCharge() for atom in heavy.GetAtoms()) == 0
    assert sum(atom.GetNumRadicalElectrons() for atom in heavy.GetAtoms()) == 0
