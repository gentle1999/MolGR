"""Geometry must separate ligand redox assignments before metal ranking."""

from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from molgr import MolGRConfig, _core
from molgr.interface import xyz_to_rdmol
from molgr.utils.converter import get_atom_unpaired_electrons


INPUTS = Path(__file__).parent / "data/tmqmg/geometry_inputs"


def config():
    cfg = MolGRConfig()
    cfg.cpp_backend.max_threads = 1
    cfg.cpp_backend.target_bucket_parallel_max_threads = 1
    return cfg


@pytest.mark.parametrize("case_id", ["GOSHIF", "ZEGTUC"])
@pytest.mark.parametrize("backend", ["cpp", "python"])
def test_mixed_ligand_redox_assignment(backend, case_id):
    xyz = (INPUTS / f"{case_id}.xyz").read_text()
    result = xyz_to_rdmol(xyz, 0, 1, backend=backend, config=config())
    original = Chem.MolFromXYZBlock(xyz)
    assert [a.GetAtomicNum() for a in result.GetAtoms()] == [
        a.GetAtomicNum() for a in original.GetAtoms()
    ]
    assert result.GetConformer().GetPositions() == pytest.approx(
        original.GetConformer().GetPositions()
    )
    assert not any(a.GetNumImplicitHs() or a.GetNumExplicitHs() for a in result.GetAtoms())
    assert Chem.GetFormalCharge(result) == 0
    assert sum(get_atom_unpaired_electrons(a) for a in result.GetAtoms()) == 0
    metal = next(a for a in result.GetAtoms() if a.GetAtomicNum() == 28)
    assert metal.GetFormalCharge() == 2
    # Check the ligand allocation, not merely the metal oxidation state.
    if case_id == "GOSHIF":
        for nitrogen, carbon in ((28, 7), (29, 8)):
            assert result.GetAtomWithIdx(nitrogen).GetFormalCharge() == 0
            assert (
                result.GetBondBetweenAtoms(nitrogen, carbon).GetBondType() == Chem.BondType.DOUBLE
            )
        for sulfur in (1, 2):
            assert result.GetAtomWithIdx(sulfur).GetFormalCharge() == -1
        assert result.GetBondBetweenAtoms(3, 4).GetBondType() == Chem.BondType.DOUBLE
    else:
        for selenium in (4, 5):
            assert result.GetAtomWithIdx(selenium).GetFormalCharge() == -1
        for sulfur, carbon in ((7, 16), (10, 13)):
            assert result.GetAtomWithIdx(sulfur).GetFormalCharge() == 0
            assert result.GetBondBetweenAtoms(sulfur, carbon).GetBondType() == Chem.BondType.DOUBLE
        assert result.GetBondBetweenAtoms(13, 16).GetBondType() == Chem.BondType.SINGLE


@pytest.mark.parametrize("backend", ["cpp", "python"])
def test_long_cn_control_keeps_competing_reduced_diimine(backend):
    mol = Chem.MolFromXYZBlock((INPUTS / "GOSHIF.xyz").read_text())
    pos = mol.GetConformer().GetPositions()
    for nitrogen, carbon, hydrogen in ((28, 7, 37), (29, 8, 38)):
        vector = pos[nitrogen] - pos[carbon]
        delta = vector * (1.40 / np.linalg.norm(vector) - 1)
        for atom in (nitrogen, hydrogen):
            mol.GetConformer().SetAtomPosition(atom, pos[atom] + delta)
    result = xyz_to_rdmol(Chem.MolToXYZBlock(mol), 0, 1, backend=backend, config=config())
    assert result.GetAtomWithIdx(0).GetFormalCharge() == 4
    assert all(result.GetAtomWithIdx(i).GetFormalCharge() == -1 for i in (28, 29))


@pytest.mark.parametrize("elongated", [False, True])
def test_cpp_thioamide_exemption_requires_short_cn_bonds(elongated):
    mol = Chem.MolFromXYZBlock((INPUTS / "ZEGTUC.xyz").read_text())
    if elongated:
        pos = mol.GetConformer().GetPositions()
        vector = pos[12] - pos[16]
        mol.GetConformer().SetAtomPosition(12, pos[16] + vector * (1.46 / np.linalg.norm(vector)))
    pool = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        Chem.MolToXYZBlock(mol), 0, 0, config=config()
    )["candidates"]
    ni2 = [c for c in pool if c["metal_states"][0]["valence"] == 2]
    assert ni2, "The geometry control must retain the competing Ni(+2) candidate"
    assert ni2[0]["metal_discordance_structural_count"] == (1 if elongated else 0)


@pytest.mark.parametrize("backend", ["cpp", "python"])
def test_diimine_double_bonds_follow_the_shorter_cn_edges(backend):
    xyz = (INPUTS / "WOLSOF.xyz").read_text()
    result = xyz_to_rdmol(xyz, 0, 1, backend=backend, config=config())
    assert result.GetAtomWithIdx(0).GetFormalCharge() == 2
    # Both imines belong to the central quinoid ring. Placing them on
    # the longer outer C-N edges incorrectly oxidizes both phenolates.
    for nitrogen, inner, outer in ((8, 9, 7), (15, 14, 16)):
        assert result.GetBondBetweenAtoms(nitrogen, inner).GetBondType() == Chem.BondType.DOUBLE
        assert result.GetBondBetweenAtoms(nitrogen, outer).GetBondType() == Chem.BondType.SINGLE
    assert all(result.GetAtomWithIdx(i).GetFormalCharge() == -1 for i in (2, 18))
    original = Chem.MolFromXYZBlock(xyz)
    assert result.GetConformer().GetPositions() == pytest.approx(
        original.GetConformer().GetPositions()
    )
    assert [a.GetAtomicNum() for a in result.GetAtoms()] == [
        a.GetAtomicNum() for a in original.GetAtoms()
    ]
    assert not any(a.GetNumImplicitHs() or a.GetNumExplicitHs() for a in result.GetAtoms())
    assert Chem.GetFormalCharge(result) == 0
    assert sum(get_atom_unpaired_electrons(a) for a in result.GetAtoms()) == 0


@pytest.mark.parametrize("backend", ["cpp", "python"])
def test_neutral_dithiooxamide_branch_preserves_ancillary_reduction(backend):
    xyz = (INPUTS / "HOHMUP.xyz").read_text()
    result = xyz_to_rdmol(xyz, 0, 1, backend=backend, config=config())
    assert result.GetAtomWithIdx(0).GetFormalCharge() == 2
    for sulfur, carbon in ((2, 24), (6, 30)):
        assert result.GetAtomWithIdx(sulfur).GetFormalCharge() == 0
        assert result.GetBondBetweenAtoms(sulfur, carbon).GetBondType() == Chem.BondType.DOUBLE
    assert result.GetBondBetweenAtoms(24, 30).GetBondType() == Chem.BondType.SINGLE
    assert all(result.GetAtomWithIdx(i).GetFormalCharge() == -1 for i in (1, 3))
    original = Chem.MolFromXYZBlock(xyz)
    assert [a.GetAtomicNum() for a in result.GetAtoms()] == [
        a.GetAtomicNum() for a in original.GetAtoms()
    ]
    assert result.GetConformer().GetPositions() == pytest.approx(
        original.GetConformer().GetPositions()
    )
    assert not any(a.GetNumImplicitHs() or a.GetNumExplicitHs() for a in result.GetAtoms())
    assert Chem.GetFormalCharge(result) == 0
    assert sum(get_atom_unpaired_electrons(a) for a in result.GetAtoms()) == 0


def test_native_dithiooxamide_pool_keeps_competing_metal_state():
    xyz = (INPUTS / "HOHMUP.xyz").read_text()
    pool = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        xyz, 0, 0, config=config()
    )["candidates"]
    assert any(c["metal_states"][0]["valence"] == 4 for c in pool)
