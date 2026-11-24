"""Closed resonance witnesses must make the zero-metal cation guard consistent."""

import pytest
from openbabel import pybel
from rdkit import Chem

from molgr.fallback.utils.dataclasses import MetalAtomPosition
from molgr.fallback.utils.electrons import (
    set_unpaired_electron_count,
    set_unresolved_two_electron_center,
)
from molgr.fallback.utils.metals import scoring
from molgr.utils.equivalence import evaluate_equivalence


SOURCE = "[NH2+]=CC=CO[B-](F)(F)F"
LOCAL = "NC=CC=[O+][B-](F)(F)F"
METALS = (MetalAtomPosition(1, "Cr", 24, 0, 0, 0, 0, 0),)


def explicit(smiles=SOURCE):
    mol = pybel.readstring("smi", smiles)
    mol.addh()
    assert all(a.OBAtom.GetImplicitHCount() == 0 for a in mol.atoms)
    return mol


def score(mol):
    return scoring._zero_valent_metals_with_organic_cation_count(mol.OBMol, METALS, total_charge=0)


@pytest.mark.parametrize("smiles", [SOURCE, "F[B-](F)(F)OC=CC=[NH2+]"])
def test_equivalent_cation_positions_have_the_same_guard_without_mutation(smiles):
    source, local = explicit(smiles), explicit(LOCAL)
    before = source.write("sdf"), local.write("sdf")
    assert score(source) == score(local) == 0
    assert (source.write("sdf"), local.write("sdf")) == before
    assert scoring._uncompensated_organic_cation_indices(source.OBMol)
    assert not scoring._uncompensated_organic_cation_indices(local.OBMol)


def test_simplified_contributors_have_an_independent_native_resonance_witness():
    source, local = [Chem.AddHs(Chem.MolFromSmiles(s)) for s in (SOURCE, LOCAL)]
    result = evaluate_equivalence(source, local, use_chirality=False, max_resonance=100000)
    assert result.decision.value == "equivalent", result.reason


@pytest.mark.parametrize("index", [1, 3, 5])
@pytest.mark.parametrize("defect", ["radical", "unresolved", "implicit_h"])
def test_every_path_atom_requires_resolved_explicit_electrons(index, defect):
    mol = explicit()
    atom = mol.OBMol.GetAtom(index)
    if defect == "radical":
        set_unpaired_electron_count(atom, 1)
    elif defect == "unresolved":
        set_unresolved_two_electron_center(atom, True)
    else:
        atom.SetImplicitHCount(1)
    before = mol.write("sdf")
    assert score(mol) == 1
    assert mol.write("sdf") == before


@pytest.mark.parametrize(
    "smiles",
    [
        "[NH2+]=CCCO[B-](F)(F)F",  # Saturated bridge interrupts the path.
        "[NH2+]=CC=CO.[Cl-]",  # A remote counterion does not compensate the endpoint.
        SOURCE + ".[NH4+].[Cl-]",  # Fixing one cation must not excuse another one.
    ],
)
def test_incomplete_or_partial_compensation_keeps_the_guard(smiles):
    mol = explicit(smiles)
    before = mol.write("sdf")
    assert score(mol) == 1
    assert mol.write("sdf") == before


@pytest.mark.parametrize(
    "name,value",
    [
        ("_CATION_COMPENSATION_MAX_STATES", 0),
        ("_CATION_COMPENSATION_MAX_ATOMS", 2),
        ("_CATION_COMPENSATION_MAX_PATHS", 0),
    ],
)
def test_search_bound_exhaustion_cannot_grant_partial_credit(monkeypatch, name, value):
    mol = explicit()
    assert score(mol) == 0
    monkeypatch.setattr(scoring, name, value)
    before = mol.write("sdf")
    assert score(mol) == 1
    assert mol.write("sdf") == before


def test_electron_deficient_source_cannot_borrow_this_closed_octet_exception():
    mol = explicit()
    mol.OBMol.GetBond(1, 2).SetBondOrder(1)
    assert score(mol) == 1
