# pyright: reportMissingImports=false

from __future__ import annotations

import io
import math
from dataclasses import replace

import pytest


pytest.importorskip("rdkit")

from rdkit import Chem, RDLogger

from molgr import make_dative_bond as public_make_dative_bond
from molgr.config import MolGRConfig
from molgr.fallback.utils.electrons import (
    LONE_PAIR_COUNT_PROP,
    UNRESOLVED_TWO_ELECTRON_CENTER_PROP,
)
from molgr.utils.converter import get_atom_lone_pair_count, get_atom_unpaired_electrons
from molgr.utils.post_process import make_dative_bond


RDLogger.DisableLog("rdApp.*")  # type: ignore


def test_make_dative_bond_is_available_from_public_api() -> None:
    assert public_make_dative_bond is make_dative_bond


def _zinc_nitrogen_pair(distance: float) -> Chem.Mol:
    mol = Chem.RWMol()
    zinc_idx = mol.AddAtom(Chem.Atom(30))
    nitrogen_idx = mol.AddAtom(Chem.Atom(7))
    conf = Chem.Conformer(2)
    conf.SetAtomPosition(zinc_idx, (0.0, 0.0, 0.0))
    conf.SetAtomPosition(nitrogen_idx, (distance, 0.0, 0.0))
    result = mol.GetMol()
    result.AddConformer(conf)
    return result


def _blocked_zinc_nitrogen_pair() -> Chem.Mol:
    mol = Chem.RWMol()
    zinc_idx = mol.AddAtom(Chem.Atom(30))
    blocker_idx = mol.AddAtom(Chem.Atom(6))
    nitrogen_idx = mol.AddAtom(Chem.Atom(7))
    conf = Chem.Conformer(3)
    conf.SetAtomPosition(zinc_idx, (0.0, 0.0, 0.0))
    conf.SetAtomPosition(blocker_idx, (1.05, 0.0, 0.0))
    conf.SetAtomPosition(nitrogen_idx, (2.10, 0.0, 0.0))
    result = mol.GetMol()
    result.AddConformer(conf)
    return result


def _nearly_blocked_zinc_nitrogen_pair() -> Chem.Mol:
    mol = Chem.RWMol()
    zinc_idx = mol.AddAtom(Chem.Atom(30))
    blocker_idx = mol.AddAtom(Chem.Atom(6))
    nitrogen_idx = mol.AddAtom(Chem.Atom(7))
    conf = Chem.Conformer(3)
    conf.SetAtomPosition(zinc_idx, (0.0, 0.0, 0.0))
    conf.SetAtomPosition(blocker_idx, (1.00, 1.00, 0.0))
    conf.SetAtomPosition(nitrogen_idx, (2.00, 0.0, 0.0))
    result = mol.GetMol()
    result.AddConformer(conf)
    return result


def _iron_alkene_pair(distance_1: float, distance_2: float) -> Chem.Mol:
    mol = Chem.RWMol()
    iron_idx = mol.AddAtom(Chem.Atom(26))
    carbon_1_idx = mol.AddAtom(Chem.Atom(6))
    carbon_2_idx = mol.AddAtom(Chem.Atom(6))
    mol.AddBond(carbon_1_idx, carbon_2_idx, Chem.BondType.DOUBLE)
    conf = Chem.Conformer(3)
    conf.SetAtomPosition(iron_idx, (0.0, 0.0, 0.0))
    conf.SetAtomPosition(carbon_1_idx, (distance_1, 0.0, 0.0))
    conf.SetAtomPosition(carbon_2_idx, (0.0, distance_2, 0.0))
    result = mol.GetMol()
    result.AddConformer(conf)
    return result


def _blocked_iron_alkene_pair() -> Chem.Mol:
    mol = Chem.RWMol()
    iron_idx = mol.AddAtom(Chem.Atom(26))
    carbon_1_idx = mol.AddAtom(Chem.Atom(6))
    carbon_2_idx = mol.AddAtom(Chem.Atom(6))
    blocker_idx = mol.AddAtom(Chem.Atom(6))
    mol.AddBond(carbon_1_idx, carbon_2_idx, Chem.BondType.DOUBLE)
    conf = Chem.Conformer(4)
    conf.SetAtomPosition(iron_idx, (0.0, 0.0, 0.0))
    conf.SetAtomPosition(carbon_1_idx, (1.80, 0.0, 0.0))
    conf.SetAtomPosition(carbon_2_idx, (0.0, 1.80, 0.0))
    conf.SetAtomPosition(blocker_idx, (0.90, 0.0, 0.0))
    result = mol.GetMol()
    result.AddConformer(conf)
    return result


def _metal_singlet_two_electron_center(atomic_num: int) -> tuple[Chem.Mol, int, int]:
    mol = Chem.RWMol()
    metal_idx = mol.AddAtom(Chem.Atom(26))
    center_idx = mol.AddAtom(Chem.Atom(atomic_num))
    center = mol.GetAtomWithIdx(center_idx)
    center.SetNoImplicit(True)
    center.SetFormalCharge(0)
    center.SetNumRadicalElectrons(0)
    center.SetIntProp(LONE_PAIR_COUNT_PROP, 1)

    substituent_count = 2 if atomic_num == 6 else 1
    substituent_indices = [mol.AddAtom(Chem.Atom(6)) for _ in range(substituent_count)]
    for substituent_idx in substituent_indices:
        mol.AddBond(center_idx, substituent_idx, Chem.BondType.SINGLE)

    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.SetAtomPosition(metal_idx, (0.0, 0.0, 0.0))
    conf.SetAtomPosition(center_idx, (1.80, 0.0, 0.0))
    for offset, substituent_idx in enumerate(substituent_indices):
        y = -0.65 if offset == 0 else 0.65
        conf.SetAtomPosition(substituent_idx, (2.80, y, 0.0))
    result = mol.GetMol()
    result.AddConformer(conf)
    result.UpdatePropertyCache(strict=False)
    return result, metal_idx, center_idx


def _metal_carbyne_center(*, unpaired_electrons: int = 3) -> tuple[Chem.Mol, int, int]:
    mol = Chem.RWMol()
    metal_idx = mol.AddAtom(Chem.Atom(26))
    center_idx = mol.AddAtom(Chem.Atom(6))
    substituent_idx = mol.AddAtom(Chem.Atom(6))
    center = mol.GetAtomWithIdx(center_idx)
    center.SetNoImplicit(True)
    center.SetFormalCharge(0)
    center.SetNumRadicalElectrons(unpaired_electrons)
    mol.AddBond(center_idx, substituent_idx, Chem.BondType.SINGLE)

    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.SetAtomPosition(metal_idx, (0.0, 0.0, 0.0))
    conf.SetAtomPosition(center_idx, (1.80, 0.0, 0.0))
    conf.SetAtomPosition(substituent_idx, (2.80, 0.0, 0.0))
    result = mol.GetMol()
    result.AddConformer(conf)
    result.UpdatePropertyCache(strict=False)
    return result, metal_idx, center_idx


def test_make_dative_bond_uses_metal_coordination_tolerance_config() -> None:
    base_config = MolGRConfig()
    tight_config = replace(
        base_config,
        metal_scoring=replace(
            base_config.metal_scoring,
            metal_coordination_extra_tolerance_angstrom=0.10,
        ),
    )
    loose_config = replace(
        base_config,
        metal_scoring=replace(
            base_config.metal_scoring,
            metal_coordination_extra_tolerance_angstrom=0.35,
        ),
    )

    tight_mol = make_dative_bond(_zinc_nitrogen_pair(2.10), config=tight_config)
    loose_mol = make_dative_bond(_zinc_nitrogen_pair(2.10), config=loose_config)

    assert tight_mol.GetBondBetweenAtoms(0, 1) is None
    loose_bond = loose_mol.GetBondBetweenAtoms(0, 1)
    assert loose_bond is not None
    assert loose_bond.GetBondType() == Chem.BondType.DATIVE


@pytest.mark.parametrize("atomic_num", [6, 7, 15])
def test_resolved_singlet_two_electron_center_forms_metal_double_bond(
    atomic_num: int,
) -> None:
    mol, metal_idx, center_idx = _metal_singlet_two_electron_center(atomic_num)

    result = make_dative_bond(mol)

    bond = result.GetBondBetweenAtoms(center_idx, metal_idx)
    assert bond is not None
    assert bond.GetBondType() == Chem.BondType.DOUBLE
    assert get_atom_lone_pair_count(result.GetAtomWithIdx(center_idx)) == 0


@pytest.mark.parametrize("atomic_num", [6, 7, 15])
@pytest.mark.parametrize("keep_other_lone_pair", [False, True])
def test_consumed_lone_pair_stays_consumed_after_sdf_round_trip(
    atomic_num: int,
    keep_other_lone_pair: bool,
) -> None:
    mol, metal_idx, center_idx = _metal_singlet_two_electron_center(atomic_num)
    other_idx = None
    if keep_other_lone_pair:
        editable = Chem.RWMol(mol)
        other = Chem.Atom(8)
        other.SetNoImplicit(True)
        other.SetIntProp(LONE_PAIR_COUNT_PROP, 1)
        other_idx = editable.AddAtom(other)
        editable.GetConformer().SetAtomPosition(other_idx, (10.0, 0.0, 0.0))
        mol = editable.GetMol()
    # Both converters create this snapshot before coordination post-processing.
    Chem.CreateAtomIntPropertyList(mol, LONE_PAIR_COUNT_PROP)

    result = make_dative_bond(mol)
    assert result.GetBondBetweenAtoms(metal_idx, center_idx).GetBondType() == Chem.BondType.DOUBLE
    assert get_atom_lone_pair_count(result.GetAtomWithIdx(center_idx)) == 0

    buffer = io.StringIO()
    with Chem.SDWriter(buffer) as writer:
        writer.write(result)
    restored = next(
        Chem.ForwardSDMolSupplier(
            io.BytesIO(buffer.getvalue().encode()), sanitize=False, removeHs=False
        )
    )
    assert restored is not None
    assert get_atom_lone_pair_count(restored.GetAtomWithIdx(center_idx)) == 0
    # Known consumption is zero, not an absent/unknown annotation.
    assert restored.GetAtomWithIdx(center_idx).HasProp(LONE_PAIR_COUNT_PROP)
    if other_idx is not None:
        assert get_atom_lone_pair_count(restored.GetAtomWithIdx(other_idx)) == 1
    # Post-processing returns a copy and must not change the source annotation.
    assert get_atom_lone_pair_count(mol.GetAtomWithIdx(center_idx)) == 1


def test_resolved_carbyne_center_forms_metal_triple_bond() -> None:
    mol, metal_idx, center_idx = _metal_carbyne_center()

    result = make_dative_bond(mol)

    bond = result.GetBondBetweenAtoms(center_idx, metal_idx)
    assert bond is not None
    assert bond.GetBondType() == Chem.BondType.TRIPLE
    assert get_atom_unpaired_electrons(result.GetAtomWithIdx(center_idx)) == 0


def test_non_carbyne_carbon_radical_does_not_form_metal_triple_bond() -> None:
    mol, metal_idx, center_idx = _metal_carbyne_center(unpaired_electrons=1)

    result = make_dative_bond(mol)

    assert result.GetBondBetweenAtoms(center_idx, metal_idx) is None


@pytest.mark.parametrize("state", ["unresolved", "triplet"])
def test_non_singlet_two_electron_center_does_not_donate_to_metal(state: str) -> None:
    mol, metal_idx, center_idx = _metal_singlet_two_electron_center(7)
    center = mol.GetAtomWithIdx(center_idx)
    center.ClearProp(LONE_PAIR_COUNT_PROP)
    if state == "unresolved":
        center.SetBoolProp(UNRESOLVED_TWO_ELECTRON_CENTER_PROP, True)
    else:
        center.SetNumRadicalElectrons(2)

    result = make_dative_bond(mol)

    assert result.GetBondBetweenAtoms(center_idx, metal_idx) is None


@pytest.mark.parametrize("atomic_num", [7, 8, 9])
@pytest.mark.parametrize("charge", [-1, 0])
def test_radical_atom_is_not_used_as_a_dative_donor(atomic_num: int, charge: int) -> None:
    mol = _zinc_nitrogen_pair(1.8)
    donor = mol.GetAtomWithIdx(1)
    donor.SetAtomicNum(atomic_num)
    donor.SetFormalCharge(charge)
    donor.SetNumRadicalElectrons(1)
    donor.SetIntProp(LONE_PAIR_COUNT_PROP, 1)

    result = make_dative_bond(mol)

    assert result.GetBondBetweenAtoms(0, 1) is None
    assert result.GetAtomWithIdx(1).GetNumRadicalElectrons() == 1
    assert result.GetAtomWithIdx(1).GetFormalCharge() == charge


def test_pi_coordination_does_not_bypass_radical_donor_exclusion() -> None:
    mol = _iron_alkene_pair(1.8, 1.8)
    mol.GetAtomWithIdx(1).SetNumRadicalElectrons(1)

    result = make_dative_bond(mol)

    assert result.GetBondBetweenAtoms(0, 1) is None
    assert result.GetBondBetweenAtoms(0, 2) is None


def test_make_dative_bond_requires_visible_coordination_atom() -> None:
    mol = make_dative_bond(_blocked_zinc_nitrogen_pair())

    assert mol.GetBondBetweenAtoms(0, 2) is None


def test_metal_access_radius_scale_expands_post_process_distance_cutoff() -> None:
    base_config = MolGRConfig()
    tight_config = replace(
        base_config,
        metal_scoring=replace(
            base_config.metal_scoring,
            metal_access_radius_scale=1.0,
            metal_coordination_extra_tolerance_angstrom=0.10,
        ),
    )
    loose_config = replace(
        base_config,
        metal_scoring=replace(
            base_config.metal_scoring,
            metal_access_radius_scale=1.50,
            metal_coordination_extra_tolerance_angstrom=0.10,
        ),
    )

    tight_mol = make_dative_bond(_zinc_nitrogen_pair(2.22), config=tight_config)
    loose_mol = make_dative_bond(_zinc_nitrogen_pair(2.22), config=loose_config)

    assert tight_mol.GetBondBetweenAtoms(0, 1) is None
    loose_bond = loose_mol.GetBondBetweenAtoms(0, 1)
    assert loose_bond is not None
    assert loose_bond.GetBondType() == Chem.BondType.DATIVE


def test_metal_access_radius_scale_expands_blocker_radius() -> None:
    base_config = MolGRConfig()
    unscaled_config = replace(
        base_config,
        metal_scoring=replace(
            base_config.metal_scoring,
            metal_access_radius_scale=1.0,
            metal_coordination_extra_tolerance_angstrom=0.35,
            metal_access_clearance_angstrom=0.0,
        ),
    )
    scaled_config = replace(
        base_config,
        metal_scoring=replace(
            base_config.metal_scoring,
            metal_access_radius_scale=1.50,
            metal_coordination_extra_tolerance_angstrom=0.35,
            metal_access_clearance_angstrom=0.0,
        ),
    )

    unscaled_mol = make_dative_bond(_nearly_blocked_zinc_nitrogen_pair(), config=unscaled_config)
    scaled_mol = make_dative_bond(_nearly_blocked_zinc_nitrogen_pair(), config=scaled_config)

    assert unscaled_mol.GetBondBetweenAtoms(0, 2).GetBondType() == Chem.BondType.DATIVE
    assert scaled_mol.GetBondBetweenAtoms(0, 2) is None


def test_make_dative_bond_uses_pi_dative_distance_difference_tolerance_config() -> None:
    base_config = MolGRConfig()
    tight_config = replace(
        base_config,
        metal_scoring=replace(
            base_config.metal_scoring,
            pi_dative_distance_difference_tolerance_angstrom=0.05,
        ),
    )
    loose_config = replace(
        base_config,
        metal_scoring=replace(
            base_config.metal_scoring,
            pi_dative_distance_difference_tolerance_angstrom=0.15,
        ),
    )

    tight_mol = make_dative_bond(_iron_alkene_pair(1.80, 1.90), config=tight_config)
    loose_mol = make_dative_bond(_iron_alkene_pair(1.80, 1.90), config=loose_config)

    assert tight_mol.GetBondBetweenAtoms(0, 1) is None
    assert tight_mol.GetBondBetweenAtoms(0, 2) is None
    assert loose_mol.GetBondBetweenAtoms(0, 1).GetBondType() == Chem.BondType.DATIVE
    assert loose_mol.GetBondBetweenAtoms(0, 2).GetBondType() == Chem.BondType.DATIVE


def test_make_pi_dative_bond_requires_both_atoms_visible_to_metal() -> None:
    mol = make_dative_bond(_blocked_iron_alkene_pair())

    assert mol.GetBondBetweenAtoms(0, 1) is None
    assert mol.GetBondBetweenAtoms(0, 2) is None


def test_haptic_coordination_completes_carbon_between_two_anionic_donors() -> None:
    ring = Chem.MolFromSmiles("[c-]1c[cH-]cc1")
    mol = Chem.RWMol(ring)
    iron = Chem.Atom(26)
    iron.SetFormalCharge(2)
    iron_idx = mol.AddAtom(iron)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(5):
        angle = 2 * math.pi * i / 5
        conf.SetAtomPosition(i, (1.2 * math.cos(angle), 1.2 * math.sin(angle), 0))
    conf.SetAtomPosition(iron_idx, (0, 0, 1.6))
    mol.AddConformer(conf)

    result = make_dative_bond(mol.GetMol())

    # Atom 1 lies between anions 0 and 2; their early dative contacts must not
    # prevent the aromatic pi pass from completing the fifth Fe-C contact.
    assert result.GetAtomWithIdx(iron_idx).GetDegree() == 5
    assert all(
        result.GetBondBetweenAtoms(i, iron_idx).GetBondType() == Chem.BondType.DATIVE
        for i in range(5)
    )
    assert [a.GetFormalCharge() for a in result.GetAtoms()] == [-1, 0, -1, 0, 0, 2]


def test_nonaromatic_anionic_alkene_keeps_only_its_localized_donor_contact() -> None:
    mol = _iron_alkene_pair(1.8, 1.8)
    mol.GetAtomWithIdx(1).SetFormalCharge(-1)

    result = make_dative_bond(mol)

    assert result.GetBondBetweenAtoms(1, 0).GetBondType() == Chem.BondType.DATIVE
    assert result.GetBondBetweenAtoms(2, 0) is None
