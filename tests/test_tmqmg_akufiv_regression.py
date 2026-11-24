from __future__ import annotations

from pathlib import Path

import pytest


pytest.importorskip("openbabel")

from molgr.interface import xyz_to_rdmol
from molgr.utils.equivalence import check_equivalence
from scripts.tmqmg_fixture_data import reference_graph


def _akufiv_xyz_block() -> str:
    return (Path(__file__).parent / "data" / "xyz" / "AKUFIV.xyz").read_text(encoding="utf-8")


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_akufiv_selects_platinum_ii_without_pentavalent_carbon(backend: str) -> None:
    reconstructed = xyz_to_rdmol(
        _akufiv_xyz_block(),
        total_charge=0,
        spin_multiplicity=1,
        backend=backend,
        make_dative_bonds=True,
    )
    reference = reference_graph("AKUFIV")
    assert reference is not None
    platinum = next(atom for atom in reconstructed.GetAtoms() if atom.GetAtomicNum() == 78)

    assert platinum.GetFormalCharge() == 2
    assert all(
        sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds()) <= 4
        for atom in reconstructed.GetAtoms()
        if atom.GetAtomicNum() == 6
    )
    equivalent, info = check_equivalence(
        reference,
        reconstructed,
        use_chirality=False,
        ignore_metal_coordination_bonds=True,
        max_resonance=100000,
    )
    assert equivalent is True, info.reason
