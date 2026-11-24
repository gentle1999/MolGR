"""Local charge compensation must not require pi overlap in closed-shell ylides."""

import pytest
from openbabel import openbabel as ob
from openbabel import pybel

from molgr.fallback.utils.electrons import (
    set_unpaired_electron_count,
    set_unresolved_two_electron_center,
)
from molgr.fallback.utils.metals import scoring


def ylide(reverse=False):
    mol = ob.OBMol()
    atoms = [
        (6, -1, (0, 0, 0)),
        (7, 1, (1.45, 0, 0)),
        (1, 0, (-0.5, 0.86, 0)),
        (1, 0, (-0.5, -0.86, 0)),
        (1, 0, (1.8, 0, 1)),
        (1, 0, (1.8, 0.866, -0.5)),
        (1, 0, (1.8, -0.866, -0.5)),
    ]
    for z, charge, xyz in atoms:
        atom = mol.NewAtom()
        atom.SetAtomicNum(z)
        atom.SetFormalCharge(charge)
        atom.SetImplicitHCount(0)
        atom.SetVector(*xyz)
    for i, j in [(1, 2), (1, 3), (1, 4), (2, 5), (2, 6), (2, 7)]:
        mol.AddBond(j if reverse else i, i if reverse else j, 1)
    return mol


def signature(mol):
    return (
        [
            (
                a.GetAtomicNum(),
                a.GetFormalCharge(),
                a.GetImplicitHCount(),
                a.GetX(),
                a.GetY(),
                a.GetZ(),
            )
            for a in ob.OBMolAtomIter(mol)
        ],
        [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondOrder()) for b in ob.OBMolBondIter(mol)],
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_nonconjugated_octet_ylide_retains_local_compensation(reverse):
    mol = ylide(reverse)
    before = signature(mol)
    metrics = scoring._compute_organic_electronic_state_metrics(pybel.Molecule(mol))
    # The quaternary N has no pi channel. Removing the inversion penalty must
    # not invent conjugation or zero the residual individual atom penalties.
    assert metrics.conjugated_atom_count == metrics.conjugated_bond_count == 0
    penalties = [
        scoring._charge_localization_penalty_for_atom(mol.GetAtom(i), is_conjugated=False)
        for i in (1, 2)
    ]
    assert metrics.charge_localization_polarity_inversion_penalty == 0
    assert metrics.charge_localization_penalty == pytest.approx(abs(penalties[0] - penalties[1]))
    assert metrics.charge_localization_penalty > 0
    assert metrics.charge_localization_component_cancellation == pytest.approx(2 * min(penalties))
    assert signature(mol) == before


@pytest.mark.parametrize("index", [1, 2])
@pytest.mark.parametrize(
    "defect", ["radical", "unresolved", "implicit_h", "missing_bond", "charge"]
)
def test_ylide_exemption_requires_two_resolved_explicit_octets(index, defect):
    mol = ylide()
    atom = mol.GetAtom(index)
    if defect == "radical":
        set_unpaired_electron_count(atom, 1)
    elif defect == "unresolved":
        set_unresolved_two_electron_center(atom, True)
    elif defect == "implicit_h":
        atom.SetImplicitHCount(1)
    elif defect == "missing_bond":
        mol.DeleteBond(mol.GetBond(index, 3 if index == 1 else 5))
    else:
        atom.SetFormalCharge(0)
    assert not scoring._is_closed_shell_cn_ylide_bond(mol.GetBond(1, 2))


def test_ylide_exception_does_not_cover_borate_oxonium_polarity():
    mol = pybel.readstring("smi", "[B-](F)(F)(F)[O+]=N")
    mol.addh()
    metrics = scoring._compute_organic_electronic_state_metrics(mol)
    assert metrics.charge_localization_polarity_inversion_penalty > 0
    assert metrics.charge_localization_component_cancellation == 0
