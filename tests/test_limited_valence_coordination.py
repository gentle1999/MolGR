from __future__ import annotations

from pathlib import Path

import pytest
from rdkit import Chem, rdBase

from molgr import make_dative_bond, validate_smiles_roundtrip
from molgr.fallback.utils.consts import NON_METAL_DICT
from molgr.fallback.utils.electrons import LONE_PAIR_COUNT_PROP
from molgr.interface import xyz_to_rdmol
from molgr.utils.post_process import _has_limited_coordination_valence
from molgr.utils.smiles import normalize_smiles_hydrogens


def _contacts(atoms, positions, bonds=()):
    mol = Chem.RWMol()
    for element, charge in atoms:
        atom = Chem.Atom(element)
        atom.SetFormalCharge(charge)
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)
    for first, second, kind in bonds:
        mol.AddBond(first, second, kind)
    conf = Chem.Conformer(len(atoms))
    for index, point in enumerate(positions):
        conf.SetAtomPosition(index, point)
    mol.AddConformer(conf)
    mol.UpdatePropertyCache(strict=False)
    return mol.GetMol()


def _assert_native_roundtrip(mol):
    normalize_smiles_hydrogens(mol)
    smiles = validate_smiles_roundtrip(mol)
    params = Chem.SmilesParserParams()
    params.removeHs = False
    parsed = Chem.MolFromSmiles(smiles, params)
    assert parsed is not None
    assert parsed.GetNumAtoms() == mol.GetNumAtoms()
    assert Chem.GetFormalCharge(parsed) == Chem.GetFormalCharge(mol)


@pytest.mark.parametrize(
    "donor,charge,kind",
    [
        ("F", -1, Chem.BondType.SINGLE),
        ("O", -2, Chem.BondType.DOUBLE),
        ("N", -3, Chem.BondType.TRIPLE),
    ],
)
def test_anionic_donors_form_single_double_and_triple_bonds(donor, charge, kind):
    source = _contacts([("Al", 3), (donor, charge)], [(0, 0, 0), (1.7, 0, 0)])
    before = source.ToBinary(Chem.PropertyPickleOptions.AllProps)
    result = make_dative_bond(source)
    assert result.GetBondBetweenAtoms(0, 1).GetBondType() == kind
    assert result.GetAtomWithIdx(0).GetFormalCharge() == 3 + charge
    assert result.GetAtomWithIdx(1).GetFormalCharge() == 0
    assert Chem.GetFormalCharge(result) == Chem.GetFormalCharge(source)
    assert source.ToBinary(Chem.PropertyPickleOptions.AllProps) == before
    _assert_native_roundtrip(result)
    repeated = make_dative_bond(result)
    assert Chem.MolToSmiles(repeated) == Chem.MolToSmiles(result)


@pytest.mark.parametrize(
    "metal,charge",
    [
        ("Be", 2),
        ("Al", 3),
        ("Ga", 3),
        ("In", 3),
        ("Sn", 4),
        ("Pb", 2),
        ("Bi", 3),
        ("Cs", 1),
        ("Fr", 1),
    ],
)
def test_finite_valence_metals_get_covalent_representation(metal, charge):
    source = _contacts([(metal, charge), ("F", -1)], [(0, 0, 0), (1.7, 0, 0)])
    result = make_dative_bond(source)
    assert result.GetBondBetweenAtoms(0, 1).GetBondType() == Chem.BondType.SINGLE
    assert [a.GetFormalCharge() for a in result.GetAtoms()] == [charge - 1, 0]
    _assert_native_roundtrip(result)


def test_neutral_ammonia_donor_becomes_charge_separated_single_bond():
    source = _contacts(
        [("Al", 3), ("N", 0), ("H", 0), ("H", 0), ("H", 0)],
        [(0, 0, 0), (1.8, 0, 0), (2.2, 1, 0), (2.2, -0.5, 0.8), (2.2, -0.5, -0.8)],
        [(1, i, Chem.BondType.SINGLE) for i in (2, 3, 4)],
    )
    result = make_dative_bond(source)
    assert result.GetBondBetweenAtoms(0, 1).GetBondType() == Chem.BondType.SINGLE
    assert [a.GetFormalCharge() for a in result.GetAtoms()] == [2, 1, 0, 0, 0]
    assert result.GetNumAtoms() == source.GetNumAtoms()
    _assert_native_roundtrip(result)


def test_bridging_oxide_distributes_charge_across_both_main_group_metals():
    source = _contacts([("Al", 3), ("O", -2), ("Al", 3)], [(-1.7, 0, 0), (0, 0, 0), (1.7, 0, 0)])
    result = make_dative_bond(source)
    assert result.GetNumBonds() == 2
    assert all(b.GetBondType() == Chem.BondType.SINGLE for b in result.GetBonds())
    assert [a.GetFormalCharge() for a in result.GetAtoms()] == [2, 0, 2]
    _assert_native_roundtrip(result)


@pytest.mark.parametrize(
    "metal", ["Li", "Na", "K", "Mg", "Ca", "Tl", "Fe", "Cu", "Zn", "Cd", "Hg", "Pt", "La"]
)
def test_unrestricted_valence_metals_keep_dative_bonds_and_charges(metal):
    source = _contacts([(metal, 2), ("O", -2)], [(0, 0, 0), (1.7, 0, 0)])
    result = make_dative_bond(source)
    bond = result.GetBondBetweenAtoms(0, 1)
    assert bond.GetBondType() == Chem.BondType.DATIVE
    assert bond.GetBeginAtomIdx() == 1
    assert [a.GetFormalCharge() for a in result.GetAtoms()] == [2, -2]


def test_existing_dative_contacts_are_normalized_without_duplicate_bonds():
    source = _contacts(
        [("Al", 3), ("O", -2)], [(0, 0, 0), (1.7, 0, 0)], [(1, 0, Chem.BondType.DATIVE)]
    )
    result = make_dative_bond(source)
    assert result.GetNumBonds() == 1
    assert result.GetBondBetweenAtoms(0, 1).GetBondType() == Chem.BondType.DOUBLE
    _assert_native_roundtrip(result)


def test_singlet_lone_pair_donation_to_main_group_metal_uses_single_bond():
    source = _contacts(
        [("Al", 3), ("C", 0), ("H", 0), ("H", 0)],
        [(0, 0, 0), (1.8, 0, 0), (2.3, 0.9, 0), (2.3, -0.9, 0)],
        [(1, 2, Chem.BondType.SINGLE), (1, 3, Chem.BondType.SINGLE)],
    )
    source.GetAtomWithIdx(1).SetIntProp(LONE_PAIR_COUNT_PROP, 1)
    result = make_dative_bond(source)
    assert result.GetBondBetweenAtoms(0, 1).GetBondType() == Chem.BondType.SINGLE
    assert [a.GetFormalCharge() for a in result.GetAtoms()] == [2, 1, 0, 0]
    assert result.GetAtomWithIdx(1).GetIntProp(LONE_PAIR_COUNT_PROP) == 0
    assert source.GetAtomWithIdx(1).GetIntProp(LONE_PAIR_COUNT_PROP) == 1
    _assert_native_roundtrip(result)


@pytest.mark.parametrize("backend", ["cpp", "python"])
def test_potassium_contacts_and_charges_remain_unchanged(backend):
    xyz = (Path(__file__).parent / "data" / "xyz" / "test.xyz").read_text()
    mol = xyz_to_rdmol(xyz, 0, 1, backend=backend)
    # K is excluded from the conversion even when a donor also binds Ni.
    _assert_native_roundtrip(mol)
    carbon = mol.GetAtomWithIdx(91)
    potassium = mol.GetAtomWithIdx(120)
    bond = mol.GetBondBetweenAtoms(91, 120)
    assert carbon.GetSymbol() == "C" and potassium.GetSymbol() == "K"
    assert bond.GetBondType() == Chem.BondType.DATIVE
    assert carbon.GetFormalCharge() == -1
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "K":
            assert atom.GetFormalCharge() == 1
            assert all(b.GetBondType() == Chem.BondType.DATIVE for b in atom.GetBonds())
    assert Chem.GetFormalCharge(mol) == 0
    repeated = make_dative_bond(mol)
    assert Chem.MolToSmiles(repeated) == Chem.MolToSmiles(mol)


def test_element_scope_matches_observed_rdkit_acceptor_valence_failures():
    # Independent behavioral scan: require the isolated ion to pass, then add
    # neutral water donors. Only the acceptor can introduce a valence failure.
    observed = set()
    with rdBase.BlockLogs():
        for z in range(1, 119):
            if z in NON_METAL_DICT:
                continue
            for charge in range(-3, 7):
                mol = Chem.RWMol()
                metal = Chem.Atom(z)
                metal.SetFormalCharge(charge)
                metal.SetNoImplicit(True)
                mol.AddAtom(metal)
                try:
                    mol.UpdatePropertyCache(strict=True)
                except Chem.AtomValenceException:
                    continue
                for _ in range(8):
                    oxygen = Chem.Atom(8)
                    oxygen.SetNoImplicit(True)
                    oxygen.SetNumExplicitHs(2)
                    donor = mol.AddAtom(oxygen)
                    mol.AddBond(donor, 0, Chem.BondType.DATIVE)
                    try:
                        mol.UpdatePropertyCache(strict=True)
                    except Chem.AtomValenceException:
                        observed.add(z)
                        break
    selected = {z for z in range(1, 119) if _has_limited_coordination_valence(z)}
    assert selected == observed
    assert 19 not in selected  # K must never be included just because it is s-block.


def test_rubidium_scope_tracks_installed_rdkit_rules():
    source = _contacts([("Rb", 1), ("F", -1)], [(0, 0, 0), (1.7, 0, 0)])
    result = make_dative_bond(source)
    if _has_limited_coordination_valence(37):
        assert result.GetBondBetweenAtoms(0, 1).GetBondType() == Chem.BondType.SINGLE
        assert [a.GetFormalCharge() for a in result.GetAtoms()] == [0, 0]
    else:
        assert result.GetBondBetweenAtoms(0, 1).GetBondType() == Chem.BondType.DATIVE
        assert [a.GetFormalCharge() for a in result.GetAtoms()] == [1, -1]
    _assert_native_roundtrip(result)
