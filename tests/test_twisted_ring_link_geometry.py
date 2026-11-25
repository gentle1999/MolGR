"""A twisted biaryl single bond must not become a closed-shell C=C link."""

import math

import pytest
from openbabel import openbabel as ob
from openbabel import pybel

from molgr import _core
from molgr.fallback.stages.preprocess import validate_omol
from molgr.fallback.utils.bond_geometry import twisted_ring_link_order_allowed
from molgr.fallback.utils.electrons import (
    set_unpaired_electron_count,
    set_unresolved_two_electron_center,
)


def _rings(*, length=1.50, twist=60, order=2, ring_length=1.40):
    mol = ob.OBMol()
    phi = math.radians(twist)
    for side in range(2):
        for i in range(6):
            theta = i * math.pi / 3
            x = ring_length * (math.cos(theta) - 1)
            y = ring_length * math.sin(theta)
            atom = mol.NewAtom()
            atom.SetAtomicNum(6)
            atom.SetImplicitHCount(0)
            atom.SetVector(
                x if side == 0 else length - x,
                y if side == 0 else y * math.cos(phi),
                0 if side == 0 else y * math.sin(phi),
            )
        for i in range(6):
            mol.AddBond(side * 6 + i + 1, side * 6 + (i + 1) % 6 + 1, 2 if i in (1, 3) else 1)
    mol.AddBond(1, 7, order)
    for idx in range(1, 13):
        if idx in (1, 7):
            continue
        carbon = mol.GetAtom(idx)
        hydrogen = mol.NewAtom()
        hydrogen.SetAtomicNum(1)
        hydrogen.SetImplicitHCount(0)
        hydrogen.SetVector(carbon.GetX(), carbon.GetY(), carbon.GetZ() + 1)
        mol.AddBond(idx, hydrogen.GetIdx(), 1)
    mol.FindRingAtomsAndBonds()
    return mol


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize(
    "order,twist,length,expected",
    [
        (2, 60, 1.50, False),
        (2, 90, 1.50, False),
        (1, 60, 1.50, True),
        (2, 0, 1.50, True),
        (2, 30, 1.50, True),
        (2, 60, 1.34, True),
        (2, 60, 1.70, True),
    ],
)
def test_final_validation_checks_joint_length_and_torsion(backend, order, twist, length, expected):
    mol = _rings(order=order, twist=twist, length=length)
    if backend == "python":
        result = validate_omol(pybel.Molecule(mol))
    else:
        result = _core.dev.stages.preprocess.validate_omol_ptr(int(mol.this), 0, 0)
    assert result is expected


def test_twisted_link_screen_is_symmetric_and_rigid_motion_invariant():
    mol = _rings()
    first, second = mol.GetAtom(1), mol.GetAtom(7)
    assert not twisted_ring_link_order_allowed(first, second, 2)
    assert not twisted_ring_link_order_allowed(second, first, 2)
    for atom in ob.OBMolAtomIter(mol):
        atom.SetVector(atom.GetY() + 30, atom.GetZ() - 20, atom.GetX() + 50)
    assert not twisted_ring_link_order_allowed(first, second, 2)


@pytest.mark.parametrize(
    "defect",
    [
        "implicit_h",
        "charged",
        "radical",
        "unresolved",
        "heteroatom",
        "long_ring",
        "no_ring",
        "missing",
        "nan",
    ],
)
def test_twisted_link_abstains_outside_complete_geometry(defect):
    mol = _rings(ring_length=1.48 if defect == "long_ring" else 1.40)
    first, second = mol.GetAtom(1), mol.GetAtom(7)
    if defect == "implicit_h":
        first.SetImplicitHCount(1)
    elif defect == "charged":
        first.SetFormalCharge(-1)
    elif defect == "radical":
        set_unpaired_electron_count(first, 1)
    elif defect == "unresolved":
        set_unresolved_two_electron_center(first, True)
    elif defect == "heteroatom":
        mol.GetAtom(2).SetAtomicNum(7)
    elif defect == "no_ring":
        mol.DeleteBond(mol.GetBond(2, 3))
        mol.FindRingAtomsAndBonds()
    elif defect in ("missing", "nan"):
        for atom in ob.OBMolAtomIter(mol):
            atom.SetVector(0, 0, 0 if defect == "missing" else float("nan"))
    assert twisted_ring_link_order_allowed(first, second, 2)
