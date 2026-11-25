"""Regression tests for charge-conserving B–B seeds and hydride connectivity."""

import pytest
from rdkit import Chem

from molgr.interface import xyz_to_rdmol


@pytest.mark.parametrize("backend", ["cpp", "python"])
@pytest.mark.parametrize("charge,order", [(0, Chem.BondType.SINGLE), (-2, Chem.BondType.DOUBLE)])
def test_diboron_search_respects_available_electrons(backend, charge, order):
    xyz = """6
Planar B2F4
B -0.85 0 0
B  0.85 0 0
F -1.525  1.1691 0
F -1.525 -1.1691 0
F  1.525  1.1691 0
F  1.525 -1.1691 0
"""
    mol = xyz_to_rdmol(xyz, charge, 1, backend=backend)
    assert mol.GetBondBetweenAtoms(0, 1).GetBondType() == order
    assert [a.GetFormalCharge() for a in mol.GetAtoms()] == [charge // 2] * 2 + [0] * 4


@pytest.mark.parametrize("backend", ["cpp", "python"])
@pytest.mark.parametrize("geometry", ["hydride", "short_ch", "bent_carbonyl", "remote_metal"])
def test_formyl_and_adjacent_hydride_connectivity(backend, geometry):
    # Mo–C and C–O are collinear. Moving one atom at a time removes
    # one necessary piece of evidence for detaching the perceived C–H.
    points = {"Mo": (0, 0, 0), "C": (1.95, 0, 0), "O": (3.13, 0, 0), "H": (1.12, 1.27, 0)}
    if geometry == "short_ch":
        points["H"] = (1.35, 0.91, 0)
    elif geometry == "bent_carbonyl":
        points["O"] = (1.95, -1.18, 0)
    elif geometry == "remote_metal":
        points["Mo"] = (-10, 0, 0)
    xyz = "4\n\n" + "".join(f"{symbol} {x} {y} {z}\n" for symbol, (x, y, z) in points.items())
    mol = xyz_to_rdmol(xyz, -1, 1, backend=backend)
    assert (mol.GetBondBetweenAtoms(1, 3) is None) == (geometry == "hydride")
    if geometry == "hydride":
        assert mol.GetBondBetweenAtoms(0, 3) is not None
        assert mol.GetBondBetweenAtoms(1, 2).GetBondType() == Chem.BondType.TRIPLE
    assert Chem.GetFormalCharge(mol) == -1
    assert mol.GetConformer().GetPositions() == pytest.approx(
        Chem.MolFromXYZBlock(xyz).GetConformer().GetPositions(), abs=1e-8
    )
