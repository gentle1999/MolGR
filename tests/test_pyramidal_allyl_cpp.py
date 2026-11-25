"""Geometry selects a localized allyl contributor without changing its electrons."""

from pathlib import Path

import numpy as np
import pytest
from openbabel import openbabel as ob
from openbabel import pybel
from rdkit import Chem

from molgr import MolGRConfig, _core
from molgr.fallback.stages.clean import clean_resonances
from molgr.interface import xyz_to_rdmol
from molgr.utils.converter import get_atom_unpaired_electrons


def allyl(long_cc=1.48, short_cc=1.34, planar=False, rotate=False):
    positions = np.array(
        [
            (0, 0, 0),
            (long_cc, 0, 0),
            (long_cc + short_cc / 2, short_cc * np.sqrt(3) / 2, 0),
            (-0.35, 0.90, -0.45),
            (-0.35, -0.90, -0.45),
            (long_cc + 0.5, -0.95, 0),
            (long_cc + short_cc / 2 + 1, short_cc * np.sqrt(3) / 2, 0),
            (long_cc + short_cc / 2 - 0.5, short_cc * np.sqrt(3) / 2 + 0.9, 0),
        ],
        dtype=float,
    )
    if planar:
        positions[3:5] = [(-0.55, 0.95, 0), (-0.55, -0.95, 0)]
    if rotate:
        positions = positions[:, [1, 2, 0]] + (30, -20, 10)
    mol = ob.OBMol()
    for z, pos in zip([6, 6, 6, 1, 1, 1, 1, 1], positions):
        atom = mol.NewAtom()
        atom.SetAtomicNum(z)
        atom.SetImplicitHCount(0)
        atom.SetVector(*pos)
    for i, j, order in [
        (1, 2, 2),
        (2, 3, 1),
        (1, 4, 1),
        (1, 5, 1),
        (2, 6, 1),
        (3, 7, 1),
        (3, 8, 1),
    ]:
        mol.AddBond(i, j, order)
    mol.GetAtom(3).SetFormalCharge(-1)
    return mol


@pytest.mark.parametrize("rotate", [False, True])
@pytest.mark.parametrize(
    "options,localized",
    [
        ({}, True),
        ({"planar": True}, False),
        ({"long_cc": 1.36}, False),
        ({"short_cc": 1.46}, False),
        ({"long_cc": 1.65}, False),
    ],
)
@pytest.mark.parametrize("backend", ["cpp", "python"])
def test_cleanup_requires_pyramid_and_alternating_lengths(options, localized, rotate, backend):
    mol = allyl(**options, rotate=rotate)
    original = [(a.GetAtomicNum(), a.GetX(), a.GetY(), a.GetZ()) for a in ob.OBMolAtomIter(mol)]
    for _ in range(2):  # A second cleanup must preserve the selected contributor.
        if backend == "cpp":
            _core.dev.stages.clean.clean_resonances_ptr(int(mol.this))
        else:
            clean_resonances(pybel.Molecule(mol))
        assert original == [
            (a.GetAtomicNum(), a.GetX(), a.GetY(), a.GetZ()) for a in ob.OBMolAtomIter(mol)
        ]
        assert [mol.GetAtom(i).GetFormalCharge() for i in (1, 2, 3)] == (
            [-1, 0, 0] if localized else [0, 0, -1]
        )
        assert [mol.GetBond(i, j).GetBondOrder() for i, j in [(1, 2), (2, 3)]] == (
            [1, 2] if localized else [2, 1]
        )
        assert not any(a.GetImplicitHCount() for a in ob.OBMolAtomIter(mol))


def test_cpp_pijnus_recovers_three_carbon_donors_without_changing_input():
    xyz = (Path(__file__).parent / "data/tmqmg/geometry_inputs/PIJNUS.xyz").read_text()
    original = Chem.MolFromXYZBlock(xyz)
    config = MolGRConfig()
    config.cpp_backend.max_threads = 1
    config.cpp_backend.target_bucket_parallel_max_threads = 1
    mol = xyz_to_rdmol(xyz, 0, 1, backend="cpp", config=config)
    assert [a.GetAtomicNum() for a in mol.GetAtoms()] == [
        a.GetAtomicNum() for a in original.GetAtoms()
    ]
    assert mol.GetConformer().GetPositions() == pytest.approx(
        original.GetConformer().GetPositions()
    )
    assert Chem.GetFormalCharge(mol) == 0
    assert sum(get_atom_unpaired_electrons(a) for a in mol.GetAtoms()) == 0
    assert not any(a.GetNumImplicitHs() or a.GetNumExplicitHs() for a in mol.GetAtoms())
    assert mol.GetAtomWithIdx(0).GetFormalCharge() == 4
    # CIF compound 12 has three sigma C-Pd bonds; this is a local structural
    # regression, not approval of a replacement for the complete standard graph.
    for idx in (10, 19, 29):
        assert mol.GetAtomWithIdx(idx).GetFormalCharge() == -1
        assert mol.GetBondBetweenAtoms(idx, 0).GetBondType() == Chem.BondType.DATIVE
    assert mol.GetAtomWithIdx(34).GetFormalCharge() == 0
    assert mol.GetBondBetweenAtoms(29, 32).GetBondType() == Chem.BondType.SINGLE
    assert mol.GetBondBetweenAtoms(32, 34).GetBondType() == Chem.BondType.DOUBLE
    assert len(Chem.GetMolFrags(mol)) == 1
