"""Mono- and bis-phenoxy formazan geometry must use the same guarded selection."""

import numpy as np
import pytest
from rdkit import Chem

from molgr import MolGRConfig, _core
from molgr.interface import xyz_to_rdmol
from molgr.utils.converter import get_atom_unpaired_electrons
from molgr.utils.equivalence import evaluate_equivalence
from scripts.tmqmg_fixture_data import fixture, reference_graph


KEY = "metal_formazan_geometry_conflict_count"


def case_and_config(case_id):
    case = dict(fixture(case_id))
    config = MolGRConfig()
    config.cpp_backend.max_threads = 1
    config.cpp_backend.target_bucket_parallel_max_threads = 1
    return case, config


@pytest.mark.parametrize("case_id", ["CARVEW", "NIFORN", "BIXBUG"])
@pytest.mark.parametrize("rigid_motion", [False, True])
def test_cpp_formazan_matches_unchanged_standard(case_id, rigid_motion):
    case, config = case_and_config(case_id)
    xyz = Chem.MolFromXYZBlock(case["xyz"])
    if rigid_motion:
        for i, point in enumerate(xyz.GetConformer().GetPositions()):
            xyz.GetConformer().SetAtomPosition(i, point[[1, 2, 0]] + (30, -40, 50))
    mol = xyz_to_rdmol(
        Chem.MolToXYZBlock(xyz),
        case["charge"],
        case["multiplicity"],
        backend="cpp",
        config=config,
    )
    assert [a.GetAtomicNum() for a in mol.GetAtoms()] == [a.GetAtomicNum() for a in xyz.GetAtoms()]
    assert mol.GetConformer().GetPositions() == pytest.approx(xyz.GetConformer().GetPositions())
    assert not any(a.GetNumImplicitHs() or a.GetNumExplicitHs() for a in mol.GetAtoms())
    assert Chem.GetFormalCharge(mol) == case["charge"]
    assert sum(get_atom_unpaired_electrons(a) for a in mol.GetAtoms()) == case["multiplicity"] - 1
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


@pytest.mark.parametrize("case_id", ["CARVEW", "NIFORN", "BIXBUG"])
@pytest.mark.parametrize("long_co", [False, True])
def test_cpp_formazan_preference_requires_complete_phenoxy_geometry(case_id, long_co):
    case, config = case_and_config(case_id)
    xyz = Chem.MolFromXYZBlock(case["xyz"])
    if long_co:
        positions = xyz.GetConformer().GetPositions()
        # Only one O is moved. In the bis-phenoxy control the other remains
        # valid: an invalid arm must not be reclassified as a plain phenyl arm.
        oxygen = next(a.GetIdx() for a in xyz.GetAtoms() if a.GetAtomicNum() == 8)
        carbon = min(
            (a.GetIdx() for a in xyz.GetAtoms() if a.GetAtomicNum() == 6),
            key=lambda i: np.linalg.norm(positions[i] - positions[oxygen]),
        )
        vector = positions[oxygen] - positions[carbon]
        xyz.GetConformer().SetAtomPosition(
            oxygen, positions[carbon] + vector * (1.45 / np.linalg.norm(vector))
        )
    pool = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        Chem.MolToXYZBlock(xyz),
        case["charge"],
        case["multiplicity"] - 1,
        config=config,
    )["candidates"]
    by_valence = {c["metal_states"][0]["valence"]: c for c in pool}
    reduced_valence = 3 if case_id == "BIXBUG" else 2
    oxidized_valence = reduced_valence - 2
    assert {reduced_valence, oxidized_valence}.issubset(by_valence)
    assert by_valence[reduced_valence][KEY] == 0
    assert by_valence[oxidized_valence][KEY] == (0 if long_co else 1)
    if not long_co:
        assert by_valence[reduced_valence]["selected"]
