"""Pyramidal sigma-bound C- pairs must not be treated as planar conjugated ions."""

from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from molgr import MolGRConfig, _core
from molgr.interface import xyz_to_rdmol
from molgr.utils.converter import get_atom_unpaired_electrons


XYZ = Path(__file__).parent / "data/tmqmg/geometry_inputs/OXYLDW.xyz"
# Input atom indices, including every explicit H vertex.
CENTERS = (
    (4, 3, 11, 12),
    (8, 7, 15, 16),
    (17, 21, 22, 23),
    (18, 24, 25, 26),
    (19, 27, 28, 29),
    (20, 30, 31, 32),
)


def config():
    cfg = MolGRConfig()
    cfg.cpp_backend.max_threads = 1
    cfg.cpp_backend.target_bucket_parallel_max_threads = 1
    return cfg


def test_cpp_xylyl_sigma_chelate_preserves_three_equivalent_ligands():
    xyz = XYZ.read_text()
    mol = xyz_to_rdmol(xyz, 0, 1, backend="cpp", config=config())
    original = Chem.MolFromXYZBlock(xyz)
    assert [a.GetAtomicNum() for a in mol.GetAtoms()] == [
        a.GetAtomicNum() for a in original.GetAtoms()
    ]
    assert mol.GetConformer().GetPositions() == pytest.approx(
        original.GetConformer().GetPositions()
    )
    assert Chem.GetFormalCharge(mol) == 0
    assert sum(get_atom_unpaired_electrons(a) for a in mol.GetAtoms()) == 0
    assert not any(a.GetNumImplicitHs() or a.GetNumExplicitHs() for a in mol.GetAtoms())
    assert mol.GetAtomWithIdx(0).GetFormalCharge() == 6
    assert len(Chem.GetMolFrags(mol)) == 1
    for center, carbon, _, _ in CENTERS:
        assert mol.GetAtomWithIdx(center).GetFormalCharge() == -1
        assert mol.GetBondBetweenAtoms(center, carbon).GetBondType() == Chem.BondType.SINGLE
        assert mol.GetBondBetweenAtoms(center, 0) is not None


@pytest.mark.parametrize("variant", ["original", "planar", "opposite_face"])
def test_cpp_sigma_pair_exemption_requires_pyramid_and_metal_approach(variant):
    mol = Chem.MolFromXYZBlock(XYZ.read_text())
    p = mol.GetConformer().GetPositions()
    conf = mol.GetConformer()
    for center, carbon, h1, h2 in CENTERS:
        axis = p[carbon] - p[center]
        axis /= np.linalg.norm(axis)
        if variant == "planar":
            transverse = p[h1] - p[center]
            transverse -= np.dot(transverse, axis) * axis
            transverse /= np.linalg.norm(transverse)
            for h, sign in ((h1, 1), (h2, -1)):
                vector = -0.5 * axis + sign * np.sqrt(0.75) * transverse
                conf.SetAtomPosition(h, p[center] + np.linalg.norm(p[h] - p[center]) * vector)
        elif variant == "opposite_face":
            normal = p[0] - p[center]
            normal -= np.dot(normal, axis) * axis
            normal /= np.linalg.norm(normal)
            for h in (h1, h2):
                vector = p[h] - p[center]
                conf.SetAtomPosition(h, p[h] - 2 * np.dot(vector, normal) * normal)
    pool = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        Chem.MolToXYZBlock(mol), 0, 0, config=config()
    )["candidates"]
    w6 = [c for c in pool if c["metal_states"][0]["valence"] == 6]
    assert w6, "The boundary probe must still explore the W(+6) candidate"
    assert w6[0]["metal_discordance_inner_visible_conjugated_carbanion_pair_count"] == (
        0 if variant == "original" else 3
    )
