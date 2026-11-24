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

    substituent_count = 2 if atomic_num in (6, 14) else 1
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


@pytest.mark.parametrize("atomic_num", [6, 7, 14, 15])
def test_resolved_singlet_two_electron_center_forms_metal_double_bond(
    atomic_num: int,
) -> None:
    mol, metal_idx, center_idx = _metal_singlet_two_electron_center(atomic_num)

    result = make_dative_bond(mol)

    bond = result.GetBondBetweenAtoms(center_idx, metal_idx)
    assert bond is not None
    assert bond.GetBondType() == Chem.BondType.DOUBLE
    assert get_atom_lone_pair_count(result.GetAtomWithIdx(center_idx)) == 0


@pytest.mark.parametrize("atomic_num", [6, 7, 14, 15])
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


@pytest.mark.parametrize("face_on", [False, True])
@pytest.mark.parametrize("prebonded", [False, True])
def test_chelate_linker_pi_contact_depends_on_approach(face_on: bool, prebonded: bool) -> None:
    # Keep the Fe–C distances fixed while moving Fe out of the S–C=C–S plane.
    mol = Chem.RWMol(Chem.MolFromSmiles("[S-]C=C[S-].[Fe+2]"))
    conf = Chem.Conformer(5)
    metal_y = 1.2 if face_on else 2.6
    metal_z = math.sqrt(2.6**2 - metal_y**2)
    for i, point in enumerate(
        [(-1.7, 1.4, 0), (-0.67, 0, 0), (0.67, 0, 0), (1.7, 1.4, 0), (0, metal_y, metal_z)]
    ):
        conf.SetAtomPosition(i, point)
    mol.AddConformer(conf)
    if prebonded:
        for donor in (0, 3):
            mol.AddBond(donor, 4, Chem.BondType.DATIVE)

    result = make_dative_bond(mol)
    contacts = {a.GetIdx() for a in result.GetAtomWithIdx(4).GetNeighbors()}
    assert contacts == ({0, 1, 2, 3} if face_on else {0, 3})
    assert result.GetBondBetweenAtoms(1, 2).GetBondType() == Chem.BondType.DOUBLE


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


@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("C[S+](C)[O-]", True),
        ("[O-][S+]([O-])[O-]", True),
        ("C[S+](C)C", True),
        ("C[P+](C)(C)C", False),
        ("C[NH2+]C", False),
        ("C[S+](C)(C)(C)C", False),
    ],
)
def test_cationic_donation_requires_a_remaining_lewis_pair(smiles, expected):
    from molgr.utils.post_process import _has_sigma_donor_lone_pair

    atom = next(a for a in Chem.MolFromSmiles(smiles).GetAtoms() if a.GetAtomicNum() in (7, 15, 16))
    assert _has_sigma_donor_lone_pair(atom) == expected


@pytest.mark.parametrize("smiles,donates", [("CP(C)C", True), ("CP(C)(C)=O", False)])
def test_phosphine_and_phosphine_oxide_sigma_donors(smiles, donates):
    mol = Chem.MolFromSmiles(smiles + ".[Zn+2]")
    metal = mol.GetNumAtoms() - 1
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, pos in enumerate([(-1, 0, -1), (0, 0, 0), (1, 0, -1), (0, -1, -1)]):
        conf.SetAtomPosition(i, pos)
    if not donates:
        conf.SetAtomPosition(4, (0, 1.48, 0))
    conf.SetAtomPosition(metal, (0, 1, 2.3))
    mol.AddConformer(conf)
    result = make_dative_bond(mol)
    assert (result.GetBondBetweenAtoms(1, metal) is not None) == donates
    if not donates:
        assert result.GetBondBetweenAtoms(4, metal).GetBondType() == Chem.BondType.DATIVE


@pytest.mark.parametrize("side_on", [True, False])
def test_nitrile_coordination_uses_approach_to_triple_bond(side_on):
    mol = Chem.MolFromSmiles("CC#N.[Ni]")
    conf = Chem.Conformer(4)
    for i, pos in enumerate([(-2.05, 0, 0), (-0.58, 0, 0), (0.58, 0, 0)]):
        conf.SetAtomPosition(i, pos)
    conf.SetAtomPosition(3, (0, 2, 0) if side_on else (2.58, 0, 0))
    mol.GetAtomWithIdx(2).SetIntProp(LONE_PAIR_COUNT_PROP, 1)
    mol.AddConformer(conf)
    result = make_dative_bond(mol)
    assert {a.GetIdx() for a in result.GetAtomWithIdx(3).GetNeighbors()} == (
        {1, 2} if side_on else {2}
    )
    assert result.GetBondBetweenAtoms(1, 2).GetBondType() == Chem.BondType.TRIPLE


def test_partially_supplied_alkene_pi_contact_is_completed():
    mol = Chem.RWMol(_iron_alkene_pair(1.8, 1.8))
    mol.AddBond(1, 0, Chem.BondType.DATIVE)
    result = make_dative_bond(mol)
    assert result.GetBondBetweenAtoms(2, 0).GetBondType() == Chem.BondType.DATIVE


def test_repeated_fill_preserves_localized_bent_cumulene_contact():
    mol = Chem.MolFromSmiles("C=C=C.[Fe]")
    conf = Chem.Conformer(4)
    # Fe is equidistant from the central C and one end of the bent allene.
    for i, pos in enumerate([(-1.3, 0, 0), (0, 0, 0), (0.9, 0.94, 0), (-0.65, 0, 1.7)]):
        conf.SetAtomPosition(i, pos)
    mol.AddConformer(conf)
    for _ in range(2):
        mol = make_dative_bond(mol)
        assert {a.GetIdx() for a in mol.GetAtomWithIdx(3).GetNeighbors()} == {1}


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("bridged", [False, True])
def test_hydride_contacts_follow_geometry_across_metals(reverse, bridged):
    from molgr.utils.smiles import validate_smiles_roundtrip

    mol = Chem.MolFromSmiles("[Fe+2].[H-].[Ru+2]")
    conf = Chem.Conformer(3)
    conf.SetAtomPosition(0, (-1.55, 0, 0))
    conf.SetAtomPosition(1, (0, 0.42, 0))
    conf.SetAtomPosition(2, (1.57 if bridged else 2.1, 0, 0))
    mol.AddConformer(conf)
    if reverse:
        mol = Chem.RenumberAtoms(mol, [2, 1, 0])
    result = make_dative_bond(mol)
    h = result.GetAtomWithIdx(1)
    assert {a.GetSymbol() for a in h.GetNeighbors()} == ({"Fe", "Ru"} if bridged else {"Fe"})
    assert h.GetFormalCharge() == (-1 if bridged else 0)
    assert all(
        b.GetBondType() == (Chem.BondType.DATIVE if bridged else Chem.BondType.SINGLE)
        for b in h.GetBonds()
    )
    assert Chem.GetFormalCharge(result) == Chem.GetFormalCharge(mol)
    assert validate_smiles_roundtrip(result) == validate_smiles_roundtrip(make_dative_bond(result))


def test_element_specific_cutoff_does_not_stop_at_out_of_range_hydrogen():
    from molgr.utils.coordination import coordination_distance_cutoff

    config = MolGRConfig()
    kwargs = {
        "radius_scale": config.metal_scoring.metal_access_radius_scale,
        "extra_tolerance_angstrom": config.metal_scoring.metal_coordination_extra_tolerance_angstrom,
    }
    h_distance = coordination_distance_cutoff(30, 1, **kwargs) + 0.05
    s_distance = coordination_distance_cutoff(30, 16, **kwargs) - 0.05
    assert h_distance < s_distance
    mol = Chem.MolFromSmiles("[Zn+2].[H-].S")
    conf = Chem.Conformer(3)
    conf.SetAtomPosition(0, (0, 0, 0))
    conf.SetAtomPosition(1, (h_distance, 0, 0))
    conf.SetAtomPosition(2, (0, s_distance, 0))
    mol.AddConformer(conf)
    result = make_dative_bond(mol, config=config)
    assert result.GetBondBetweenAtoms(0, 1) is None
    assert result.GetBondBetweenAtoms(2, 0).GetBondType() == Chem.BondType.DATIVE


@pytest.mark.parametrize("kekule", [False, True])
@pytest.mark.parametrize("smiles,donates", [("Cn1cccc1", False), ("n1ccccc1", True)])
def test_aromatic_nitrogen_sigma_donation_requires_available_lone_pair(smiles, donates, kekule):
    mol = Chem.MolFromSmiles(smiles + ".[Zn+2]")
    if kekule:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    nitrogen = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
    metal = mol.GetNumAtoms() - 1
    conf = Chem.Conformer(mol.GetNumAtoms())
    # Only N lies in the coordination shell; this tests atom donation, not hapticity.
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (8.0 + i, 5.0, 0.0))
    conf.SetAtomPosition(nitrogen, (2.1, 0, 0))
    conf.SetAtomPosition(metal, (0, 0, 0))
    mol.AddConformer(conf)
    result = make_dative_bond(mol)
    assert (result.GetBondBetweenAtoms(nitrogen, metal) is not None) == donates


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "fe,zn,expected",
    [
        ((2.05, 0), (2.40, 55), {"Fe"}),
        ((2.05, 55), (2.40, 0), {"Zn"}),
        ((2.20, 90), (2.20, -90), set()),
        ((2.20, 35), (2.20, -35), set()),
        ((2.10, 35), (2.40, -35), {"Fe"}),
    ],
)
def test_pyridine_sigma_contact_follows_lone_pair(fe, zn, expected, reverse):
    mol = Chem.AddHs(Chem.MolFromSmiles("n1ccccc1.[Fe+2].[Zn+2]"))
    conf = Chem.Conformer(mol.GetNumAtoms())
    for a in mol.GetAtoms():
        i = a.GetIdx()
        if a.GetSymbol() in ("Fe", "Zn"):
            distance, degrees = fe if a.GetSymbol() == "Fe" else zn
            tilt = math.radians(degrees)
            point = (1.4 + distance * math.cos(tilt), 0, distance * math.sin(tilt))
        else:
            is_h = a.GetAtomicNum() == 1
            ring_index = a.GetNeighbors()[0].GetIdx() if is_h else i
            angle = ring_index * math.pi / 3
            radius = 2.48 if is_h else 1.4
            point = (radius * math.cos(angle), radius * math.sin(angle), 0)
        conf.SetAtomPosition(i, point)
    mol.AddConformer(conf)
    if reverse:
        order = list(range(mol.GetNumAtoms()))
        order[6], order[7] = order[7], order[6]
        mol = Chem.RenumberAtoms(mol, order)
    result = make_dative_bond(mol)
    assert {
        a.GetSymbol() for a in result.GetAtomWithIdx(0).GetNeighbors() if a.GetAtomicNum() > 20
    } == expected


@pytest.mark.parametrize(
    "ligand,ligand_positions",
    [
        # In each ligand the unoccupied sigma direction points along +x.
        ("C#N", [(-1.16, 0, 0), (0, 0, 0), (-2.22, 0, 0)]),
        (
            "C=N",
            [(-0.65, 1.12, 0), (0, 0, 0), (-1.72, 1.12, 0), (-0.12, 2.05, 0), (-0.5, -0.87, 0)],
        ),
        ("N", [(0, 0, 0), (-0.34, 0.94, 0), (-0.34, -0.47, 0.81), (-0.34, -0.47, -0.81)]),
    ],
    ids=["sp-nitrile", "sp2-imine", "sp3-amine"],
)
@pytest.mark.parametrize("aligned", ["Zn", "Fe"])
def test_hybridization_directs_one_nitrogen_sigma_pair(ligand, ligand_positions, aligned):
    mol = Chem.CombineMols(
        Chem.AddHs(Chem.MolFromSmiles(ligand)), Chem.MolFromSmiles("[Fe+2].[Zn+2]")
    )
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, pos in enumerate(ligand_positions):
        conf.SetAtomPosition(i, pos)
    for i in range(len(ligand_positions), mol.GetNumAtoms()):
        # Both metals are visible; the better aligned one is farther away.
        pos = (2.4, 0, 0) if mol.GetAtomWithIdx(i).GetSymbol() == aligned else (1.2, 0, 1.65)
        conf.SetAtomPosition(i, pos)
    mol.AddConformer(conf)
    result = make_dative_bond(mol)
    nitrogen = next(a for a in result.GetAtoms() if a.GetAtomicNum() == 7)
    assert {a.GetSymbol() for a in nitrogen.GetNeighbors() if a.GetAtomicNum() > 20} == {aligned}


def _donor_with_metals(ligand, ligand_positions, metal_positions):
    """Build an explicit-H ligand and visible metal competitors in the same frame."""
    mol = Chem.AddHs(Chem.MolFromSmiles(ligand))
    assert mol.GetNumAtoms() == len(ligand_positions)
    mol = Chem.CombineMols(mol, Chem.MolFromSmiles(".".join(metal_positions)))
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, pos in enumerate([*ligand_positions, *metal_positions.values()]):
        conf.SetAtomPosition(i, pos)
    mol.AddConformer(conf)
    return mol


def _bound_metals(mol, donor):
    return {
        a.GetSymbol()
        for a in mol.GetAtomWithIdx(donor).GetNeighbors()
        if a.GetSymbol() in ("Fe", "Zn", "Ru")
    }


@pytest.mark.parametrize("ligand", ["P", "[AsH3]", "[SbH3]"])
def test_pyramidal_pnictogen_selects_the_aligned_metal(ligand):
    mol = _donor_with_metals(
        ligand,
        [(0, 0, 0), (-0.5, 1, 0), (-0.5, -0.5, 0.87), (-0.5, -0.5, -0.87)],
        {"[Fe+2]": (1.2, 0, 1.65), "[Zn+2]": (2.4, 0, 0)},
    )
    assert _bound_metals(make_dative_bond(mol), 0) == {"Zn"}


@pytest.mark.parametrize("ligand", ["O", "S", "[SeH2]", "[TeH2]"])
@pytest.mark.parametrize("prebound", [True, False])
def test_bent_chalcogen_can_donate_in_two_directions(ligand, prebound):
    mol = Chem.RWMol(
        _donor_with_metals(
            ligand,
            [(0, 0, 0), (-0.65, 0.86, 0), (-0.65, -0.86, 0)],
            {"[Fe+2]": (1.2, 0, 1.7), "[Zn+2]": (1.2, 0, -1.7), "[Ru+2]": (0.2, 1.5, 1.5)},
        )
    )
    if prebound:
        mol.AddBond(0, 3, Chem.BondType.DATIVE)
    for _ in range(2):
        mol = make_dative_bond(mol)
        assert _bound_metals(mol, 0) == {"Fe", "Zn"}


@pytest.mark.parametrize("reverse", [False, True])
def test_carbonyl_oxygen_uses_trigonal_ligand_plane(reverse):
    mol = _donor_with_metals(
        "C=O",
        [(-1.21, 0, 0), (0, 0, 0), (-1.78, 0.94, 0), (-1.78, -0.94, 0)],
        {"[Fe+2]": (1.05, 1.82, 0), "[Zn+2]": (0, 0, 2.05)},
    )
    if reverse:
        mol = Chem.RenumberAtoms(mol, [0, 1, 3, 2, 5, 4])
    assert _bound_metals(make_dative_bond(mol), 1) == {"Fe"}


@pytest.mark.parametrize("ligand", ["[Cl-]", "[Br-]", "[I-]"])
def test_free_halide_accepts_bridging_contacts_without_ligand_axis(ligand):
    mol = _donor_with_metals(ligand, [(0, 0, 0)], {"[Fe+2]": (-2.2, 0, 0), "[Zn+2]": (2.2, 0, 0)})
    assert _bound_metals(make_dative_bond(mol), 0) == {"Fe", "Zn"}


def test_carbon_end_of_bridging_co_accepts_two_outward_metals():
    mol = _donor_with_metals(
        "[C-]#[O+]",
        [(0, 0, 0), (-1.15, 0, 0)],
        {"[Fe+2]": (1.4, 1.45, 0), "[Ru+2]": (1.4, -1.45, 0)},
    )
    assert _bound_metals(make_dative_bond(mol), 0) == {"Fe", "Ru"}


@pytest.mark.parametrize("ligand", ["COC", "CSC", "C[Se]C"])
def test_ether_competitors_on_the_same_side_share_one_direction(ligand):
    # The two methyl groups and all six hydrogens lie away from the metals.
    mol = _donor_with_metals(
        ligand,
        [
            (-0.9, 1.2, 0),
            (0, 0, 0),
            (-0.9, -1.2, 0),
            (-0.7, 1.8, 0.85),
            (-0.7, 1.8, -0.85),
            (-1.94, 0.9, 0),
            (-0.7, -1.8, 0.85),
            (-0.7, -1.8, -0.85),
            (-1.94, -0.9, 0),
        ],
        {"[Fe+2]": (1.2, 0, 1.7), "[Zn+2]": (2.0, 0.8, 0.15)},
    )
    assert _bound_metals(make_dative_bond(mol), 1) == {"Fe"}


@pytest.mark.parametrize("ligand", ["[SH-]", "[SeH-]"])
def test_terminal_anion_can_bridge_across_the_bond_axis(ligand):
    mol = _donor_with_metals(
        ligand,
        [(0, 0, 0), (-1.35, 0, 0)],
        {"[Fe+2]": (0, 0, 2.2), "[Zn+2]": (0, 0, -2.2)},
    )
    assert _bound_metals(make_dative_bond(mol), 0) == {"Fe", "Zn"}


@pytest.mark.parametrize("ligand", ["[CH2]", "[SiH2]"])
def test_resolved_bent_singlet_pair_selects_the_aligned_metal(ligand):
    mol = _donor_with_metals(
        ligand,
        [(0, 0, 0), (-0.6, 0.9, 0), (-0.6, -0.9, 0)],
        {"[Fe+2]": (1.2, 0, 1.65), "[Zn+2]": (2.3, 0, 0)},
    )
    donor = mol.GetAtomWithIdx(0)
    donor.SetNumRadicalElectrons(0)
    donor.SetIntProp(LONE_PAIR_COUNT_PROP, 1)
    result = make_dative_bond(mol)
    assert _bound_metals(result, 0) == {"Zn"}
    assert result.GetBondBetweenAtoms(0, 4).GetBondType() == Chem.BondType.DOUBLE
