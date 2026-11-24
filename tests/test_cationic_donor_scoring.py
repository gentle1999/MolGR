"""A formal cation can retain a Lewis donor pair after covalent bonding."""

import math

import pytest
from openbabel import openbabel as ob
from openbabel import pybel
from rdkit import Chem

from molgr.fallback.state import ReconstructionState, make_metal_candidate_state
from molgr.fallback.utils.dataclasses import MetalAtomPosition
from molgr.fallback.utils.electrons import (
    UNRESOLVED_TWO_ELECTRON_CENTER_PROP,
    set_lone_pair_count,
    set_unpaired_electron_count,
    set_unresolved_two_electron_center,
)
from molgr.fallback.utils.metals import scoring
from molgr.utils.post_process import _has_sigma_donor_lone_pair


def _explicit_hydride_cation(z, valence, radicals, unresolved, charge=1):
    mol = ob.OBMol()
    center = mol.NewAtom()
    center.SetAtomicNum(z)
    center.SetFormalCharge(charge)
    center.SetImplicitHCount(0)
    center.SetVector(2.0, 0.0, 0.0)
    set_unpaired_electron_count(center, radicals)
    set_unresolved_two_electron_center(center, unresolved)
    rd = Chem.RWMol()
    atom = Chem.Atom(z)
    atom.SetFormalCharge(charge)
    atom.SetNoImplicit(True)
    atom.SetNumRadicalElectrons(radicals)
    atom.SetBoolProp(UNRESOLVED_TWO_ELECTRON_CENTER_PROP, unresolved)
    rd.AddAtom(atom)
    for i in range(valence):
        hydrogen = mol.NewAtom()
        hydrogen.SetAtomicNum(1)
        hydrogen.SetImplicitHCount(0)
        angle = 2 * math.pi * i / valence
        hydrogen.SetVector(3.0, math.cos(angle), math.sin(angle))
        mol.AddBond(1, hydrogen.GetIdx(), 1)
        hydrogen_rd = Chem.Atom(1)
        hydrogen_rd.SetNoImplicit(True)
        idx = rd.AddAtom(hydrogen_rd)
        rd.AddBond(0, idx, Chem.BondType.SINGLE)
    rd.UpdatePropertyCache(strict=False)
    return mol, rd.GetMol()


@pytest.mark.parametrize(
    "z,valence,radicals,unresolved,donor",
    [
        (15, 2, 0, False, True),  # Divalent P+ retains a pair.
        (15, 4, 0, False, False),  # Tetravalent P+ has none, including P+=C.
        (16, 3, 0, False, True),  # Trivalent S+ retains a pair.
        (16, 5, 0, False, False),
        (7, 4, 0, False, False),  # Ammonium has no donor pair.
        (8, 3, 0, False, True),  # Trivalent oxonium retains one pair.
        (15, 2, 2, False, False),  # Two unpaired electrons are not a pair.
        (16, 3, 2, False, False),
        (15, 2, 0, True, False),  # Deferred occupancy cannot justify reduced repulsion.
        (16, 3, 0, True, False),
    ],
)
@pytest.mark.parametrize("active_pairs", [0, 1])
def test_positive_donor_scoring_uses_lewis_pairs(
    z, valence, radicals, unresolved, donor, active_pairs
):
    mol, rd = _explicit_hydride_cation(z, valence, radicals, unresolved)
    # Reconstruction-active pairs are not the full Lewis inventory.
    set_lone_pair_count(mol.GetAtom(1), active_pairs)
    assert scoring._is_cationic_paired_electron_donor(mol.GetAtom(1)) is donor
    assert _has_sigma_donor_lone_pair(rd.GetAtomWithIdx(0)) is donor
    candidate = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Cd", 48, 2, 0, 0.0, 0.0, 0.0),),
        1,
        radicals,
        combination_index=0,
    )
    candidate.no_metal_state = ReconstructionState(
        omol=pybel.Molecule(mol),
        given_charge=1,
        total_charge=1,
        total_radical_electrons=radicals,
    )
    scoring._annotate_candidate_discordance_features(candidate)
    assert candidate.metadata["metal_discordance_inner_visible_same_sign_charge_count"] == 1
    assert candidate.metadata["metal_discordance_inner_visible_cationic_donor_count"] == int(donor)
    assert candidate.metadata["metal_discordance_inner_visible_same_sign_charge_penalty"] == (
        0.5 if donor else 1.0
    )


def test_existing_dative_contact_does_not_spend_another_lewis_pair():
    _, rd = _explicit_hydride_cation(16, 3, 0, False)
    mol = Chem.RWMol(rd)
    metal = Chem.Atom(30)
    metal.SetFormalCharge(2)
    metal.SetNoImplicit(True)
    idx = mol.AddAtom(metal)
    mol.AddBond(0, idx, Chem.BondType.DATIVE)
    mol.UpdatePropertyCache(strict=False)
    assert _has_sigma_donor_lone_pair(mol.GetAtomWithIdx(0))


@pytest.mark.parametrize("z,valence", [(15, 3), (16, 2)])
def test_neutral_donors_have_no_cation_repulsion(z, valence):
    mol, _ = _explicit_hydride_cation(z, valence, 0, False, charge=0)
    candidate = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Cd", 48, 2, 0, 0.0, 0.0, 0.0),),
        0,
        0,
        combination_index=0,
    )
    candidate.no_metal_state = ReconstructionState(
        omol=pybel.Molecule(mol),
        given_charge=0,
        total_charge=0,
        total_radical_electrons=0,
    )
    scoring._annotate_candidate_discordance_features(candidate)
    assert candidate.metadata["metal_discordance_inner_visible_same_sign_charge_penalty"] == 0
    assert candidate.metadata["metal_discordance_inner_visible_cationic_donor_count"] == 0


@pytest.mark.parametrize(
    "anion_z,anion_valence,unpaired,unresolved,expected",
    [
        (5, 4, 0, False, False),  # Tetrahedral borate has no transferable pair.
        (6, 3, 0, False, True),
        (7, 2, 0, False, True),
        (6, 3, 2, False, False),
        (6, 3, 1, False, False),
        (6, 3, 0, True, False),
    ],
)
def test_electron_deficient_cation_requires_an_actual_adjacent_pair(
    anion_z, anion_valence, unpaired, unresolved, expected
):
    mol = ob.OBMol()
    oxygen = mol.NewAtom()
    oxygen.SetAtomicNum(8)
    oxygen.SetFormalCharge(1)
    oxygen.SetImplicitHCount(0)
    anion = mol.NewAtom()
    anion.SetAtomicNum(anion_z)
    anion.SetFormalCharge(-1)
    anion.SetImplicitHCount(0)
    mol.AddBond(1, 2, 1)
    for _ in range(anion_valence - 1):
        hydrogen = mol.NewAtom()
        hydrogen.SetAtomicNum(1)
        hydrogen.SetImplicitHCount(0)
        mol.AddBond(2, hydrogen.GetIdx(), 1)
    set_unpaired_electron_count(anion, unpaired)
    set_unresolved_two_electron_center(anion, unresolved)
    assert scoring._is_unsaturated_organic_cation(oxygen)
    assert scoring._has_adjacent_formal_charge_cancellation(oxygen)
    assert scoring._is_locally_charge_compensated_nonmetal_cation(oxygen) is expected
