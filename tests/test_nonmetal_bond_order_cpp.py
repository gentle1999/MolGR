"""Nonmetal radical pairing must preserve electrons beyond triple-bond capacity."""

import pytest
from openbabel import openbabel as ob
from openbabel import pybel
from rdkit import Chem

from molgr import _core
from molgr.fallback.stages.clean import clean_neighbor_radicals, clean_resonances_9
from molgr.fallback.stages.preprocess import validate_omol
from molgr.fallback.utils.electrons import (
    get_unpaired_electron_count,
    set_unpaired_electron_count,
)
from molgr.interface import xyz_to_rdmol


def carbon_pair(order, radicals=0, charges=(0, 0)):
    mol = ob.OBMol()
    for i, charge in enumerate(charges):
        atom = mol.NewAtom()
        atom.SetAtomicNum(6)
        atom.SetFormalCharge(charge)
        atom.SetImplicitHCount(0)
        atom.SetVector(1.28 * i, 0, 0)
        set_unpaired_electron_count(atom, radicals)
    mol.AddBond(1, 2, order)
    return mol


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("target_radicals", [0, 2])
def test_radical_cleanup_retains_electrons_beyond_triple_bond(order, target_radicals):
    mol = carbon_pair(order, radicals=4 - order)
    python_mol = pybel.Molecule(ob.OBMol(mol))
    _, python_hit = clean_neighbor_radicals(python_mol, 0, target_radicals)
    assert python_hit is (order < 3)
    assert python_mol.OBMol.GetBond(1, 2).GetBondOrder() == 3
    assert [get_unpaired_electron_count(a) for a in ob.OBMolAtomIter(python_mol.OBMol)] == [1, 1]
    hit = _core.dev.stages.clean.clean_neighbor_radicals_ptr(int(mol.this), 0, target_radicals)
    assert hit is (order < 3)
    assert mol.GetBond(1, 2).GetBondOrder() == 3
    assert [get_unpaired_electron_count(a) for a in ob.OBMolAtomIter(mol)] == [1, 1]
    assert all(a.GetFormalCharge() == a.GetImplicitHCount() == 0 for a in ob.OBMolAtomIter(mol))
    assert not _core.dev.stages.clean.clean_neighbor_radicals_ptr(int(mol.this), 0, 0)


def test_charge_cleanup_cannot_cancel_charge_into_quadruple_bond():
    mol = carbon_pair(2, charges=(2, -2))
    python_mol = pybel.Molecule(ob.OBMol(mol))
    assert clean_resonances_9(python_mol)[1]
    assert python_mol.OBMol.GetBond(1, 2).GetBondOrder() == 3
    assert [a.GetFormalCharge() for a in ob.OBMolAtomIter(python_mol.OBMol)] == [1, -1]
    assert _core.dev.stages.clean.clean_resonances_ptr(int(mol.this))
    assert mol.GetBond(1, 2).GetBondOrder() == 3
    assert [a.GetFormalCharge() for a in ob.OBMolAtomIter(mol)] == [1, -1]


@pytest.mark.parametrize("order,radicals,valid", [(4, 0, False), (3, 1, True)])
def test_validation_checks_bond_order_even_when_budgets_match(order, radicals, valid):
    mol = carbon_pair(order, radicals=radicals)
    assert validate_omol(pybel.Molecule(ob.OBMol(mol)), 0, radicals * 2) is valid
    assert _core.dev.stages.preprocess.validate_omol_ptr(int(mol.this), 0, radicals * 2) is valid


def test_validation_does_not_reject_metal_metal_quadruple_bonds():
    mol = carbon_pair(4)
    for atom in ob.OBMolAtomIter(mol):
        atom.SetAtomicNum(75)
    assert _core.dev.stages.preprocess.validate_omol_ptr(int(mol.this), 0, 0)


def test_dicarbon_search_respects_bond_capacity_and_singlet_input():
    result = xyz_to_rdmol("2\n\nC 0 0 0\nC 1.28 0 0\n", 0, 1, backend="cpp")
    assert result.GetNumAtoms() == 2
    assert result.GetBondBetweenAtoms(0, 1).GetBondType() == Chem.BondType.TRIPLE
    assert Chem.GetFormalCharge(result) == 0
    assert sum(a.GetNumRadicalElectrons() for a in result.GetAtoms()) == 0
    assert all(a.GetNumImplicitHs() == a.GetNumExplicitHs() == 0 for a in result.GetAtoms())

    # This asserts algorithm invariants, not recovery of the missing six H or
    # approval of a chemically correct answer for the defective input.
