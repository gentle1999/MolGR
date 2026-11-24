"""A bent N-N-C ligand must remain reachable after triple-bond filtering."""

from pathlib import Path

import pytest
from rdkit import Chem

from molgr import MolGRConfig
from molgr.interface import xyz_to_rdmol
from molgr.utils.converter import get_atom_unpaired_electrons
from molgr.utils.equivalence import evaluate_equivalence
from scripts.tmqmg_fixture_data import reference_graph


@pytest.mark.parametrize("rotate", [False, True])
def test_pubked_cpp_preserves_input_and_reaches_hydrazido_candidate(rotate):
    xyz = (Path(__file__).parent / "data/tmqmg/geometry_inputs/PUBKED.xyz").read_text()
    if rotate:
        lines = xyz.splitlines()
        for i in range(2, len(lines)):
            symbol, x, y, z = lines[i].split()
            lines[i] = f"{symbol} {float(y) + 20:.8f} {float(z) - 30:.8f} {float(x) + 40:.8f}"
        xyz = "\n".join(lines) + "\n"
    original = Chem.MolFromXYZBlock(xyz)
    config = MolGRConfig()
    config.cpp_backend.max_threads = 1
    config.cpp_backend.target_bucket_parallel_max_threads = 1
    mol = xyz_to_rdmol(xyz, 0, 1, backend="cpp", config=config)
    assert [a.GetAtomicNum() for a in mol.GetAtoms()] == [
        a.GetAtomicNum() for a in original.GetAtoms()
    ]
    assert mol.GetConformer().GetPositions() == pytest.approx(
        original.GetConformer().GetPositions(), abs=1e-8
    )
    assert Chem.GetFormalCharge(mol) == 0
    assert sum(get_atom_unpaired_electrons(a) for a in mol.GetAtoms()) == 0
    assert not any(a.GetNumImplicitHs() or a.GetNumExplicitHs() for a in mol.GetAtoms())
    # The original input has a bent N-N-C angle (~128 degrees), incompatible
    # with the previous N#N+-C- candidate. The reduced ligand is reported for
    # the same composition in DOI 10.1021/ja808542j, compound 5. This local
    # regression also checks the complete graph approved on 2026-11-17 below.
    assert mol.GetAtomWithIdx(0).GetFormalCharge() == 5
    assert mol.GetAtomWithIdx(6).GetFormalCharge() == -2
    assert mol.GetAtomWithIdx(7).GetFormalCharge() == 0
    assert mol.GetAtomWithIdx(56).GetFormalCharge() == 0
    assert mol.GetBondBetweenAtoms(6, 7).GetBondType() == Chem.BondType.SINGLE
    assert mol.GetBondBetweenAtoms(7, 56).GetBondType() == Chem.BondType.DOUBLE
    comparison = evaluate_equivalence(
        reference_graph("PUBKED"),
        mol,
        use_chirality=False,
        ignore_metal_coordination_bonds=True,
        max_resonance=100000,
    )
    assert comparison.decision.value == "equivalent"
