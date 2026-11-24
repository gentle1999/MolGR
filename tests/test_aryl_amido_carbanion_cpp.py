"""A long exocyclic C-C must not force loss of a geometry-supported aryl ring."""

import numpy as np
import pytest
from rdkit import Chem

from molgr import MolGRConfig, _core
from molgr.interface import xyz_to_rdmol
from scripts.tmqmg_fixture_data import fixture


@pytest.fixture(scope="module")
def palsif():
    case = dict(fixture("PALSIF"))
    config = MolGRConfig()
    config.cpp_backend.max_threads = 1
    config.cpp_backend.target_bucket_parallel_max_threads = 1
    mol = xyz_to_rdmol(
        case["xyz"],
        case["charge"],
        case["multiplicity"],
        backend="cpp",
        kekulize=False,
        config=config,
    )
    return case, config, mol


def test_cpp_palsif_preserves_aryl_ring_and_long_exocyclic_single_bonds(palsif):
    case, _, mol = palsif
    xyz = Chem.MolFromXYZBlock(case["xyz"])
    assert [a.GetAtomicNum() for a in mol.GetAtoms()] == [a.GetAtomicNum() for a in xyz.GetAtoms()]
    assert mol.GetConformer().GetPositions() == pytest.approx(xyz.GetConformer().GetPositions())
    assert not any(a.GetNumImplicitHs() or a.GetNumExplicitHs() for a in mol.GetAtoms())
    assert Chem.GetFormalCharge(mol) == case["charge"]
    assert sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()) == case["multiplicity"] - 1
    assert mol.GetAtomWithIdx(0).GetFormalCharge() == 4
    assert all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in range(6, 12))
    for i, j in [(1, 6), (11, 12)]:
        assert mol.GetBondBetweenAtoms(i, j).GetBondType() == Chem.BondType.SINGLE
    assert mol.GetAtomWithIdx(1).GetFormalCharge() == -1
    assert mol.GetAtomWithIdx(12).GetFormalCharge() == -1


@pytest.mark.parametrize("length", [1.43, 1.61])
def test_cpp_aryl_preference_abstains_outside_local_cc_window(palsif, length):
    case, config, source = palsif
    # Translate the complete dimethyl-carbon branch, preserving every internal
    # bond and explicit H, to isolate the exocyclic C-C length evidence.
    moving, pending = {12}, [12]
    while pending:
        atom = source.GetAtomWithIdx(pending.pop())
        for neighbor in atom.GetNeighbors():
            index = neighbor.GetIdx()
            if index not in moving and index not in (0, 11):
                moving.add(index)
                pending.append(index)
    assert len(moving) == 9
    xyz = Chem.MolFromXYZBlock(case["xyz"])
    positions = xyz.GetConformer().GetPositions()
    vector = positions[12] - positions[11]
    delta = vector * (length / np.linalg.norm(vector) - 1)
    for index in moving:
        xyz.GetConformer().SetAtomPosition(index, positions[index] + delta)
    candidates = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        Chem.MolToXYZBlock(xyz),
        case["charge"],
        case["multiplicity"] - 1,
        config=config,
    )["candidates"]
    states = {c["metal_states"][0]["valence"]: c for c in candidates}
    assert {2, 4}.issubset(states)
    assert (
        states[2]["metal_discordance_structural_count"]
        == states[4]["metal_discordance_structural_count"]
    )
    assert states[2]["selected"]
