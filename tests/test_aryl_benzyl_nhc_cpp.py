"""Pyramidal benzyl/NHC chelates retain aromaticity and the sigma donor."""

import numpy as np
import pytest
from rdkit import Chem

from molgr import MolGRConfig, _core
from molgr.interface import xyz_to_rdmol
from molgr.utils.equivalence import evaluate_equivalence
from scripts.tmqmg_fixture_data import fixture, reference_graph


@pytest.fixture(scope="module")
def wedvaf():

    case = dict(fixture("WEDVAF"))
    cfg = MolGRConfig()
    cfg.cpp_backend.max_threads = 1
    cfg.cpp_backend.target_bucket_parallel_max_threads = 1
    return case, cfg


def test_cpp_benzyl_nhc_matches_unchanged_standard(wedvaf):
    case, cfg = wedvaf
    xyz = Chem.MolFromXYZBlock(case["xyz"])
    mol = xyz_to_rdmol(case["xyz"], 0, 1, backend="cpp", kekulize=False, config=cfg)
    assert [a.GetAtomicNum() for a in mol.GetAtoms()] == [a.GetAtomicNum() for a in xyz.GetAtoms()]
    assert mol.GetConformer().GetPositions() == pytest.approx(xyz.GetConformer().GetPositions())
    assert not any(a.GetNumImplicitHs() or a.GetNumExplicitHs() for a in mol.GetAtoms())
    assert Chem.GetFormalCharge(mol) == 0
    assert sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()) == 0
    assert mol.GetAtomWithIdx(0).GetFormalCharge() == 3
    assert mol.GetAtomWithIdx(16).GetFormalCharge() == -1
    assert mol.GetBondBetweenAtoms(16, 9).GetBondType() == Chem.BondType.SINGLE
    assert mol.GetBondBetweenAtoms(16, 0) is not None
    assert all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in (8, 9, 10, 12, 13, 15))
    assert (
        evaluate_equivalence(
            reference_graph("WEDVAF"),
            mol,
            ignore_metal_coordination_bonds=True,
            max_resonance=100000,
            use_chirality=False,
        ).decision.value
        == "equivalent"
    )


@pytest.mark.parametrize(
    "variant", ["original", "rigid_motion", "planar", "opposite_face", "short_cc"]
)
def test_cpp_benzyl_nhc_preference_requires_local_geometry(wedvaf, variant):
    case, cfg = wedvaf
    xyz = Chem.MolFromXYZBlock(case["xyz"])
    p = xyz.GetConformer().GetPositions()
    conf = xyz.GetConformer()
    center, aryl, hydrogens = 16, 9, (17, 18)
    axis = p[aryl] - p[center]
    axis /= np.linalg.norm(axis)
    if variant == "planar":
        transverse = p[hydrogens[0]] - p[center]
        transverse -= np.dot(transverse, axis) * axis
        transverse /= np.linalg.norm(transverse)
        for index, sign in zip(hydrogens, (1, -1)):
            conf.SetAtomPosition(
                index,
                p[center]
                + np.linalg.norm(p[index] - p[center])
                * (-0.5 * axis + sign * np.sqrt(0.75) * transverse),
            )
    elif variant == "opposite_face":
        normal = p[0] - p[center]
        normal -= np.dot(normal, axis) * axis
        normal /= np.linalg.norm(normal)
        for index in hydrogens:
            conf.SetAtomPosition(
                index, p[index] - 2 * np.dot(p[index] - p[center], normal) * normal
            )
    elif variant == "short_cc":
        delta = (p[center] - p[aryl]) * (1.40 / np.linalg.norm(p[center] - p[aryl]) - 1)
        for index in (center, *hydrogens):
            conf.SetAtomPosition(index, p[index] + delta)
    elif variant == "rigid_motion":
        for index, pos in enumerate(p):
            conf.SetAtomPosition(index, pos[[1, 2, 0]] + (30, -50, 10))
    pool = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        Chem.MolToXYZBlock(xyz), 0, 0, config=cfg
    )["candidates"]
    candidates = {c["metal_states"][0]["valence"]: c for c in pool}
    assert {1, 3}.issubset(candidates), "Both redox partners must remain available"
    if variant == "rigid_motion":
        # Search may choose a different rejected Y(I) contributor after ties
        # change order; the reconstructed graph must still match the standard.
        mol = xyz_to_rdmol(Chem.MolToXYZBlock(xyz), 0, 1, backend="cpp", config=cfg)
        assert mol.GetConformer().GetPositions() == pytest.approx(conf.GetPositions())
        assert (
            evaluate_equivalence(
                reference_graph("WEDVAF"),
                mol,
                ignore_metal_coordination_bonds=True,
                max_resonance=100000,
                use_chirality=False,
            ).decision.value
            == "equivalent"
        )
        assert candidates[3]["selected"]
        return
    conflict = (
        candidates[1]["metal_discordance_structural_count"]
        - candidates[3]["metal_discordance_structural_count"]
    )
    assert conflict == (1 if variant == "original" else 0)
