"""Terminal NO+ bookkeeping must not force reduction of neutral ligands."""

import pytest
from openbabel import openbabel as ob
from openbabel import pybel

from molgr.fallback.state import ReconstructionState, make_metal_candidate_state
from molgr.fallback.utils.dataclasses import MetalAtomPosition
from molgr.fallback.utils.electrons import (
    set_unpaired_electron_count,
    set_unresolved_two_electron_center,
)
from molgr.fallback.utils.metals import scoring


KEY = "metal_nitrosonium_formal_charge_exemption"


def _candidate():
    mol = ob.OBMol()
    for z, charge, x, y in [
        (7, 0, 2, 0),
        (8, 1, 3.15, 0),
        (7, 0, -2, 0),
        (8, 1, -3.15, 0),
        (6, -1, 0, 2),
        (8, 1, 0, 3.15),
    ]:
        atom = mol.NewAtom()
        atom.SetAtomicNum(z)
        atom.SetFormalCharge(charge)
        atom.SetImplicitHCount(0)
        atom.SetVector(x, y, 0)
    for i, j in [(1, 2), (3, 4), (5, 6)]:
        mol.AddBond(i, j, 3)
    candidate = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Fe", 26, -2, 0, 0, 0, 0),),
        2,
        0,
        combination_index=0,
    )
    candidate.no_metal_state = ReconstructionState(
        omol=pybel.Molecule(mol),
        given_charge=2,
        total_charge=2,
        total_radical_electrons=0,
    )
    return candidate


def test_no_charge_compensation_preserves_raw_features_and_is_idempotent():
    candidate = _candidate()
    result = scoring._annotate_candidate_discordance_features(candidate)
    assert candidate.metadata["metal_discordance_negative_metal_penalty"] == 1
    assert candidate.metadata["metal_discordance_inner_visible_same_sign_charge_penalty"] == 1
    assert candidate.metadata[KEY] == 2
    assert result == candidate.metadata["metal_discordance_structural_count"] == 0
    assert scoring._annotate_candidate_discordance_features(candidate) == result
    assert candidate.metadata[KEY] == 2


@pytest.mark.parametrize(
    "condition",
    ["bent", "long", "NO_minus", "radical", "unresolved", "implicit_h", "ionic_fragments"],
)
def test_no_charge_compensation_abstains_outside_resolved_neutral_ligands(condition):
    candidate = _candidate()
    mol = candidate.no_metal_state.omol.OBMol
    if condition == "bent":
        mol.GetAtom(2).SetVector(2, 1.15, 0)
    elif condition == "long":
        mol.GetAtom(2).SetVector(3.30, 0, 0)
    elif condition == "NO_minus":
        mol.GetAtom(1).SetFormalCharge(-1)
        mol.GetAtom(2).SetFormalCharge(0)
        mol.GetBond(1, 2).SetBondOrder(2)
    elif condition == "radical":
        set_unpaired_electron_count(mol.GetAtom(5), 1)
    elif condition == "unresolved":
        set_unresolved_two_electron_center(mol.GetAtom(5), True)
    elif condition == "implicit_h":
        mol.GetAtom(5).SetImplicitHCount(1)
    elif condition == "ionic_fragments":
        # Globally neutral counterions must not pass the per-ligand check.
        for charge in (-1, 1):
            atom = mol.NewAtom()
            atom.SetAtomicNum(7)
            atom.SetFormalCharge(charge)
            atom.SetImplicitHCount(0)
            atom.SetVector(10 * charge, 0, 0)
    scoring._annotate_candidate_discordance_features(candidate)
    assert candidate.metadata[KEY] == 0


@pytest.mark.parametrize(
    "states,credit",
    [
        ((), 0),
        ((("Fe", -1),), 2),
        ((("Fe", 0),), 0),
        ((("Fe", -3),), 0),
        ((("Fe", -2), ("Fe", 2)), 0),
    ],
)
def test_no_charge_compensation_for_resolved_neutral_ancillary_ligands(states, credit):
    candidate = _candidate()
    metals = tuple(
        MetalAtomPosition(i + 1, name, 26, charge, 0, 0, 0, 0)
        for i, (name, charge) in enumerate(states)
    )
    assert (
        scoring._nitrosonium_formal_charge_exemption(
            candidate.no_metal_state.omol.OBMol, metals, {1, 2, 3, 4}, {5}, 1.0
        )
        == credit
    )


def test_exemption_does_not_credit_donors_without_a_same_sign_penalty():
    candidate = _candidate()
    assert (
        scoring._nitrosonium_formal_charge_exemption(
            candidate.no_metal_state.omol.OBMol, candidate.metal_states, {1, 2, 3, 4}, set(), 1.0
        )
        == 1
    )
