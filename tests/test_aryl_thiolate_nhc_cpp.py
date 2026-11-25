"""Thiolate/NHC geometry distinguishes aromatic and oxidized ligand forms."""

import numpy as np
import pytest
from rdkit import Chem

from molgr import MolGRConfig, _core
from molgr.interface import xyz_to_rdmol
from molgr.utils.converter import get_atom_unpaired_electrons
from molgr.utils.equivalence import evaluate_equivalence
from scripts.tmqmg_fixture_data import fixture, reference_graph


def case_and_config(case_id):
    case = dict(fixture(case_id))
    config = MolGRConfig()
    config.cpp_backend.max_threads = 1
    config.cpp_backend.target_bucket_parallel_max_threads = 1
    return case, config


@pytest.mark.parametrize("case_id", ["NUDRUZ", "NUDSEK", "UBISAZ"])
@pytest.mark.parametrize("rigid_motion", [False, True])
def test_cpp_thiolate_nhc_matches_unchanged_standard(case_id, rigid_motion):
    case, config = case_and_config(case_id)
    xyz = Chem.MolFromXYZBlock(case["xyz"])
    if rigid_motion:
        for i, position in enumerate(xyz.GetConformer().GetPositions()):
            xyz.GetConformer().SetAtomPosition(i, position[[1, 2, 0]] + (20, -30, 40))
    text = Chem.MolToXYZBlock(xyz)
    mol = xyz_to_rdmol(
        text, case["charge"], case["multiplicity"], backend="cpp", kekulize=False, config=config
    )
    assert [a.GetAtomicNum() for a in mol.GetAtoms()] == [a.GetAtomicNum() for a in xyz.GetAtoms()]
    assert mol.GetConformer().GetPositions() == pytest.approx(xyz.GetConformer().GetPositions())
    assert not any(a.GetNumImplicitHs() or a.GetNumExplicitHs() for a in mol.GetAtoms())
    assert Chem.GetFormalCharge(mol) == case["charge"]
    assert sum(get_atom_unpaired_electrons(a) for a in mol.GetAtoms()) == case["multiplicity"] - 1
    assert mol.GetAtomWithIdx(0).GetFormalCharge() == 2
    for i in (1, 2):
        sulfur = mol.GetAtomWithIdx(i)
        assert sulfur.GetAtomicNum() == 16 and sulfur.GetFormalCharge() == -1
        carbon = next(a for a in sulfur.GetNeighbors() if a.GetAtomicNum() == 6)
        assert mol.GetBondBetweenAtoms(i, carbon.GetIdx()).GetBondType() == Chem.BondType.SINGLE
        assert carbon.GetIsAromatic()
    assert (
        evaluate_equivalence(
            reference_graph(case_id),
            mol,
            use_chirality=False,
            ignore_metal_coordination_bonds=True,
            max_resonance=100000,
        ).decision.value
        == "equivalent"
    )


@pytest.mark.parametrize("short_cs", [False, True])
def test_cpp_thiolate_nhc_preference_requires_long_cs(short_cs):
    case, config = case_and_config("NUDRUZ")
    xyz = Chem.MolFromXYZBlock(case["xyz"])
    if short_cs:
        positions = xyz.GetConformer().GetPositions()
        # Both ortho-thiolate arms must be changed, since either can supply
        # the oxidized redox partner. No input atom or hydrogen is removed.
        for sulfur, carbon in ((1, 11), (2, 17)):
            vector = positions[sulfur] - positions[carbon]
            xyz.GetConformer().SetAtomPosition(
                sulfur, positions[carbon] + vector * (1.65 / np.linalg.norm(vector))
            )
    pool = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        Chem.MolToXYZBlock(xyz),
        case["charge"],
        case["multiplicity"] - 1,
        config=config,
    )["candidates"]
    by_valence = {c["metal_states"][0]["valence"]: c for c in pool}
    assert {0, 2}.issubset(by_valence), "Both redox partners remain searchable"
    assert (
        by_valence[0]["metal_discordance_structural_count"]
        - by_valence[2]["metal_discordance_structural_count"]
    ) == (0 if short_cs else 1)
    assert by_valence[0 if short_cs else 2]["selected"]
