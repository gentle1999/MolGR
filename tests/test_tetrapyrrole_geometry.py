"""Boundaries and branch isolation for geometry-guided macrocycle recovery."""

import math

import pytest
from openbabel import openbabel as ob
from openbabel import pybel

from molgr import _core
from molgr.fallback.stages.preprocess import validate_omol
from molgr.fallback.utils.bond_geometry import planar_tetrapyrrole_circuits
from molgr.fallback.utils.electrons import (
    has_unresolved_two_electron_center,
    set_unpaired_electron_count,
    set_unresolved_two_electron_center,
)
from molgr.fallback.utils.search_constraints import require_nitrosonium


def _macrocycle(size):
    """Four fused C4N rings; the inner circuit has explicit substituents."""
    mol = ob.OBMol()
    radius = 1.40 / (2 * math.sin(math.pi / size))
    nitrogens = {0, 4, 8, 12}
    for i in range(size):
        atom = mol.NewAtom()
        atom.SetAtomicNum(7 if i in nitrogens else 6)
        atom.SetImplicitHCount(0)
        angle = 2 * math.pi * i / size
        atom.SetVector(radius * math.cos(angle), radius * math.sin(angle), 0)
    for i in range(size):
        mol.AddBond(i + 1, (i + 1) % size + 1, 1)
    for i in nitrogens:
        ends = [(i - 1) % size + 1, (i + 1) % size + 1]
        outer = []
        for end in ends:
            atom = mol.NewAtom()
            atom.SetAtomicNum(6)
            atom.SetImplicitHCount(0)
            neighbor = mol.GetAtom(end)
            atom.SetVector(neighbor.GetX() * 1.3, neighbor.GetY() * 1.3, 0)
            outer.append(atom.GetIdx())
            mol.AddBond(end, atom.GetIdx(), 1)
        mol.AddBond(*outer, 1)
    for i in range(1, size + 1):
        atom = mol.GetAtom(i)
        if atom.GetAtomicNum() == 6 and atom.GetExplicitDegree() == 2:
            hydrogen = mol.NewAtom()
            hydrogen.SetAtomicNum(1)
            hydrogen.SetImplicitHCount(0)
            hydrogen.SetVector(atom.GetX() * 1.3, atom.GetY() * 1.3, 0)
            mol.AddBond(i, hydrogen.GetIdx(), 1)
    return mol


@pytest.mark.parametrize("size", [15, 16])
def test_planar_tetrapyrrole_recognition_ignores_electronic_labels_and_rigid_motion(size):
    mol = _macrocycle(size)
    sites = list(planar_tetrapyrrole_circuits(mol))
    assert len(sites) == 1 and set(sites[0]) == set(range(1, size + 1))
    for atom in ob.OBMolAtomIter(mol):
        atom.SetVector(atom.GetY() + 100, atom.GetZ() - 50, atom.GetX() + 20)
        atom.SetFormalCharge(-1)
    assert list(planar_tetrapyrrole_circuits(mol)) == sites


@pytest.mark.parametrize("size", [15, 16])
@pytest.mark.parametrize(
    "defect", ["implicit_h", "element", "length", "nonplanar", "missing", "nan", "fused_ring"]
)
def test_tetrapyrrole_geometry_abstains_outside_bounded_motif(size, defect):
    mol = _macrocycle(size)
    if defect == "implicit_h":
        mol.GetAtom(1).SetImplicitHCount(1)
    elif defect == "element":
        mol.GetAtom(1).SetAtomicNum(15)
    elif defect == "length":
        for atom in ob.OBMolAtomIter(mol):
            atom.SetVector(atom.GetX() * 1.2, atom.GetY() * 1.2, 0)
    elif defect == "nonplanar":
        mol.GetAtom(1).SetVector(mol.GetAtom(1).GetX(), 0, 0.45)
    elif defect in ("missing", "nan"):
        for atom in ob.OBMolAtomIter(mol):
            atom.SetVector(0, 0, 0 if defect == "missing" else float("nan"))
    else:
        mol.GetAtom(size + 1).SetAtomicNum(8)
    assert list(planar_tetrapyrrole_circuits(mol)) == []


def _nitrosonium():
    mol = ob.OBMol()
    for z, charge in [(7, 0), (8, 1)]:
        atom = mol.NewAtom()
        atom.SetAtomicNum(z)
        atom.SetFormalCharge(charge)
        atom.SetImplicitHCount(0)
    mol.AddBond(1, 2, 3)
    return mol


def _validate(mol, backend, charge=1):
    if backend == "python":
        return validate_omol(pybel.Molecule(mol), charge, 0)
    return _core.dev.stages.preprocess.validate_omol_ptr(int(mol.this), charge, 0)


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_recovery_constraint_survives_copy_and_does_not_leak_between_branches(backend):
    plain = _nitrosonium()
    protected = ob.OBMol(plain)
    require_nitrosonium(protected.GetAtom(1), protected.GetAtom(2))
    copied = ob.OBMol(protected)
    assert _validate(copied, backend)
    # Both molecules still meet charge/spin accounting, but only the recovery
    # branch requires NO+. Run them interleaved to catch process-global state.
    for mol in (plain, copied):
        mol.GetBond(1, 2).SetBondOrder(2)
        mol.GetAtom(1).SetFormalCharge(1)
        mol.GetAtom(2).SetFormalCharge(0)
    assert not _validate(copied, backend)
    assert _validate(plain, backend)
    assert _validate(protected, backend)
    assert not _validate(copied, backend)


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize(
    "defect", ["element", "charge", "order", "radical", "deferred", "implicit_h", "missing_o"]
)
def test_invalid_recovery_endpoint_does_not_resolve_unrelated_deferred_labels(backend, defect):
    mol = _nitrosonium()
    nitrogen, oxygen = mol.GetAtom(1), mol.GetAtom(2)
    require_nitrosonium(nitrogen, oxygen)
    if defect == "element":
        oxygen.SetAtomicNum(16)
    elif defect == "charge":
        nitrogen.SetFormalCharge(1)
        oxygen.SetFormalCharge(0)
    elif defect == "order":
        mol.GetBond(1, 2).SetBondOrder(2)
    elif defect == "radical":
        set_unpaired_electron_count(nitrogen, 1)
    elif defect == "deferred":
        set_unresolved_two_electron_center(nitrogen, True)
    elif defect == "implicit_h":
        oxygen.SetImplicitHCount(1)
    else:
        mol.DeleteAtom(oxygen)
    carbon = mol.NewAtom()
    carbon.SetAtomicNum(6)
    set_unresolved_two_electron_center(carbon, True)
    assert not _validate(mol, backend)
    assert has_unresolved_two_electron_center(carbon)
