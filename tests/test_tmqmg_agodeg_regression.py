from __future__ import annotations

from pathlib import Path

import pytest


pytest.importorskip("openbabel")

from molgr.interface import xyz_to_rdmol
from molgr.utils.equivalence import check_equivalence
from scripts.tmqmg_fixture_data import reference_graph


def _agodeg_xyz_block() -> str:
    return (Path(__file__).parent / "data" / "xyz" / "AGODEG.xyz").read_text(encoding="utf-8")


def test_agodeg_cpp_preserves_aromatic_ligand_and_selects_cobalt_iii() -> None:
    reconstructed = xyz_to_rdmol(
        _agodeg_xyz_block(),
        total_charge=0,
        spin_multiplicity=1,
        backend="cpp",
        make_dative_bonds=True,
    )
    cobalt = next(atom for atom in reconstructed.GetAtoms() if atom.GetAtomicNum() == 27)
    reference = reference_graph("AGODEG")
    assert reference is not None

    equivalent, info = check_equivalence(
        reference,
        reconstructed,
        use_chirality=False,
        ignore_metal_coordination_bonds=True,
        max_resonance=100000,
    )

    assert cobalt.GetFormalCharge() == 3
    assert equivalent is True, info.reason
