"""The compact data boundary must not change H, isotope or coordination semantics."""

import pytest
from rdkit import Chem

from molgr import _core
from molgr.fallback.utils.electrons import (
    LONE_PAIR_COUNT_PROP,
    UNPAIRED_ELECTRON_COUNT_PROP,
    UNRESOLVED_TWO_ELECTRON_CENTER_PROP,
)
from molgr.utils.converter import (
    METAL_UNPAIRED_ELECTRONS_PROP,
    mol_data_to_pybel,
    mol_data_to_rdkit,
    pybel_to_rdmol,
    rdkit_to_mol_data,
)


def _native_projection(data):
    pointer = _core.utils.molecule_data_to_obmol_ptr(data)
    try:
        return _core.utils.extract_molecule_data(pointer)
    finally:
        _core.free_obmol_ptr(pointer)


@pytest.mark.parametrize(
    "smiles", ["", "CCO", "[13CH3][2H]", "[NH4+]", "c1cc[nH]c1", "[CH3]", "[CH2-]C=C"]
)
def test_rdkit_input_preserves_explicit_h_isotopes_and_electronic_graph(smiles):
    source = Chem.MolFromSmiles(smiles)
    before = source.ToBinary(Chem.PropertyPickleOptions.AllProps)
    expected = Chem.AddHs(source)
    data = rdkit_to_mol_data(source)
    restored = mol_data_to_rdkit(data)

    assert source.ToBinary(Chem.PropertyPickleOptions.AllProps) == before
    assert len(data.atoms) == expected.GetNumAtoms() == restored.GetNumAtoms()
    assert len(data.bonds) == expected.GetNumBonds() == restored.GetNumBonds()
    # Compare the indexed graph: a Kekule form with aromatic flags can produce
    # a different canonical SMILES ordering from the original aromatic form.
    for original in expected.GetBonds():
        bond = restored.GetBondBetweenAtoms(original.GetBeginAtomIdx(), original.GetEndAtomIdx())
        assert bond is not None
        assert bond.GetIsAromatic() == original.GetIsAromatic()
        if not original.GetIsAromatic():
            assert bond.GetBondType() == original.GetBondType()
    for original, atom in zip(expected.GetAtoms(), restored.GetAtoms()):
        assert atom.GetAtomicNum() == original.GetAtomicNum()
        assert atom.GetIsotope() == original.GetIsotope()
        assert atom.GetFormalCharge() == original.GetFormalCharge()
        assert atom.GetNumRadicalElectrons() == original.GetNumRadicalElectrons()
        assert atom.GetNumImplicitHs() == atom.GetNumExplicitHs() == 0
        assert atom.GetNoImplicit()
    # Native radical counts do not fabricate explicit MolGR electron annotations.
    assert all(atom.electron_property_presence == 0 for atom in data.atoms)
    assert data.total_charge == sum(atom.GetFormalCharge() for atom in expected.GetAtoms())
    assert data.total_radical_num == sum(
        atom.GetNumRadicalElectrons() for atom in expected.GetAtoms()
    )


@pytest.mark.parametrize("smiles", ["N->[Cu]", "[Cu]<-N"])
def test_dative_direction_survives_rdkit_roundtrip(smiles):
    source = Chem.MolFromSmiles(smiles)
    data = rdkit_to_mol_data(source)
    bonds = [bond for bond in data.bonds if bond.kind == _core.utils.BondKind.Dative]
    assert len(bonds) == 1
    bond = bonds[0]
    assert bond.order == 0
    assert data.atoms[bond.begin_atom_idx - 1].atomic_num == 7
    assert data.atoms[bond.end_atom_idx - 1].atomic_num == 29
    restored = mol_data_to_rdkit(data)
    rd_bond = restored.GetBondBetweenAtoms(bond.begin_atom_idx - 1, bond.end_atom_idx - 1)
    assert rd_bond.GetBondType() == Chem.BondType.DATIVE
    assert rd_bond.GetBeginAtom().GetAtomicNum() == 7
    assert rd_bond.GetEndAtom().GetAtomicNum() == 29
    assert restored.GetNumAtoms() == 5  # NH3 and Cu; no H inferred after projection.


@pytest.mark.parametrize("route", ["native", "pybel"])
@pytest.mark.parametrize("kind", ["Dative", "CoordinationUnspecified"])
def test_ob_projection_omits_coordination_and_preserves_source(route, kind):
    data = rdkit_to_mol_data(Chem.MolFromSmiles("[15NH3]->[Cu]"))
    bonds = data.bonds
    bonds[0].kind = getattr(_core.utils.BondKind, kind)
    data.bonds = bonds
    if route == "native":
        projected = _native_projection(data)
    else:
        obmol = mol_data_to_pybel(data).OBMol
        assert all(obmol.GetAtom(i + 1).GetImplicitHCount() == 0 for i in range(5))
        projected = _core.utils.extract_molecule_data(int(obmol.this))
    assert len(projected.atoms) == len(data.atoms) == 5
    assert projected.atoms[0].isotope == 15
    assert len(projected.bonds) == 3
    assert all(b.kind == _core.utils.BondKind.Covalent for b in projected.bonds)
    assert all(2 not in (b.begin_atom_idx, b.end_atom_idx) for b in projected.bonds)
    assert len(data.bonds) == 4
    assert data.bonds[0].kind == getattr(_core.utils.BondKind, kind)
    assert data.bonds[0].begin_atom_idx == 1
    assert data.bonds[0].end_atom_idx == 2


@pytest.mark.parametrize("route", ["native", "pybel"])
def test_ob_projection_preserves_isotopic_hydrogen_and_aromatic_topology(route):
    data = rdkit_to_mol_data(Chem.MolFromSmiles("[2H]c1cc[nH]c1"))
    if route == "native":
        restored = mol_data_to_rdkit(_native_projection(data))
    else:
        restored = pybel_to_rdmol(mol_data_to_pybel(data))
    assert Chem.MolToSmiles(restored) == Chem.MolToSmiles(mol_data_to_rdkit(data))
    assert restored.GetNumAtoms() == len(data.atoms)


def test_rdkit_input_retains_independent_annotations_and_presence():
    source = Chem.MolFromSmiles("[Fe+2].[CH3].[NH3]")
    metal, carbon, nitrogen = source.GetAtoms()
    metal.SetIntProp(METAL_UNPAIRED_ELECTRONS_PROP, 4)
    metal.SetIntProp(LONE_PAIR_COUNT_PROP, 0)
    carbon.SetIntProp(UNPAIRED_ELECTRON_COUNT_PROP, 1)
    carbon.SetBoolProp(UNRESOLVED_TWO_ELECTRON_CENTER_PROP, False)
    nitrogen.SetIntProp(LONE_PAIR_COUNT_PROP, 1)
    data = rdkit_to_mol_data(source)
    assert [a.electron_property_presence for a in data.atoms[:3]] == [3, 5, 2]
    assert [a.radical_num for a in data.atoms[:3]] == [4, 1, 0]
    assert data.total_radical_num == 5
    assert all(a.electron_property_presence == 0 for a in data.atoms[3:])
    restored = mol_data_to_rdkit(data)
    assert restored.GetAtomWithIdx(0).GetIntProp(METAL_UNPAIRED_ELECTRONS_PROP) == 4
    assert restored.GetAtomWithIdx(0).GetNumRadicalElectrons() == 0
    assert restored.GetAtomWithIdx(0).GetIntProp(LONE_PAIR_COUNT_PROP) == 0
    assert not restored.GetAtomWithIdx(1).GetBoolProp(UNRESOLVED_TWO_ELECTRON_CENTER_PROP)
    assert not restored.GetAtomWithIdx(2).HasProp(UNPAIRED_ELECTRON_COUNT_PROP)


def test_rdkit_input_rejects_query_and_unsupported_bond_types():
    with pytest.raises(ValueError, match="query"):
        rdkit_to_mol_data(Chem.MolFromSmarts("[#6]"))
    source = Chem.MolFromSmiles("CC")
    source.GetBondWithIdx(0).SetBondType(Chem.BondType.ZERO)
    with pytest.raises(ValueError, match="Unsupported.*bond type"):
        rdkit_to_mol_data(source)


def test_undirected_coordination_is_not_invented_on_rdkit_output():
    data = rdkit_to_mol_data(Chem.MolFromSmiles("N->[Cu]"))
    bonds = data.bonds
    bonds[0].kind = _core.utils.BondKind.CoordinationUnspecified
    data.bonds = bonds
    with pytest.raises(ValueError, match="direction"):
        mol_data_to_rdkit(data)


def test_batch_transfer_preserves_coordinates_and_can_omit_them():
    source = Chem.AddHs(Chem.MolFromSmiles("[13CH3][C@H](F)Cl"))
    conf = Chem.Conformer(source.GetNumAtoms())
    for i in range(source.GetNumAtoms()):
        conf.SetAtomPosition(i, (i + 0.25, -i - 0.5, 0.75 * i))
    source.AddConformer(conf)
    before = source.ToBinary(Chem.PropertyPickleOptions.AllProps)
    kept = rdkit_to_mol_data(source)
    omitted = rdkit_to_mol_data(source, include_coordinates=False)
    for i, atom in enumerate(kept.atoms):
        assert (atom.x, atom.y, atom.z) == (i + 0.25, -i - 0.5, 0.75 * i)
    assert all((a.x, a.y, a.z) == (0, 0, 0) for a in omitted.atoms)
    assert len(kept.stereo) == len(omitted.stereo) == 1
    assert source.ToBinary(Chem.PropertyPickleOptions.AllProps) == before


def test_disabled_stereo_skips_perception_and_extraction(monkeypatch):
    from molgr.utils import converter, equivalence

    source = Chem.MolFromSmiles("F/C=C/[C@H](Cl)Br")
    before = source.ToBinary(Chem.PropertyPickleOptions.AllProps)

    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled stereo should not be processed")

    monkeypatch.setattr(Chem, "AssignChiralTypesFromBondDirs", forbidden)
    monkeypatch.setattr(Chem, "SetBondStereoFromDirections", forbidden)
    monkeypatch.setattr(converter, "extract_rdkit_stereo", forbidden)
    data, _ = equivalence._prepare_input(source, False)
    assert not data.stereo and not data.stereo_groups
    assert source.ToBinary(Chem.PropertyPickleOptions.AllProps) == before


def test_batch_transfer_rejects_invalid_presence_and_coordinate_shape():
    # Invalid presence must not wrap into a valid uint8_t mask.
    with pytest.raises(ValueError, match="presence"):
        _core.utils._molecule_data_from_rows([(6, 0, 0, 0, 0, False, 256)], [])
    with pytest.raises(ValueError, match="shape"):
        _core.utils._molecule_data_from_rows([(6, 0, 0, 0, 0, False, 0)], [], [[1, 2]])


def test_disabled_stereo_matches_explicitly_stripped_kekule_graph():
    # ACA FUI: stereo tags can influence RDKit's choice of aromatic bond orders,
    # even when no stereo records are exported. Bounded search sees those orders.
    source = Chem.MolFromSmiles(
        "CC1c2cccc(/C(C)=N/NC(=O)c3ccccc3)[n]2->[Zn+2]23(<-[OH2])"
        "(<-[O]=C(N[N]->2=1)c1ccccc1)<-[O-][N+](=O)[O-]->3"
    )
    before = source.ToBinary(Chem.PropertyPickleOptions.AllProps)
    stripped = Chem.Mol(source)
    Chem.RemoveStereochemistry(stripped)
    expected = rdkit_to_mol_data(stripped, perceive_radicals=True)
    actual = rdkit_to_mol_data(source, perceive_radicals=True, include_stereo=False)
    assert [(b.begin_atom_idx, b.end_atom_idx, b.order) for b in actual.bonds] == [
        (b.begin_atom_idx, b.end_atom_idx, b.order) for b in expected.bonds
    ]
    assert [(a.formal_charge, a.radical_num) for a in actual.atoms] == [
        (a.formal_charge, a.radical_num) for a in expected.atoms
    ]
    assert not actual.stereo and not actual.stereo_groups
    assert source.ToBinary(Chem.PropertyPickleOptions.AllProps) == before


@pytest.mark.parametrize("use_chirality", [False, True])
def test_preparation_reuses_decoded_annotations_without_changing_source(use_chirality):
    from molgr.utils.equivalence import UNKNOWN_ELECTRON_METADATA, _prepare_input

    source = Chem.MolFromSmiles("[13CH3][C@H](F)Cl.[NH2].[Cu+2]")
    carbon, _, fluorine, _, nitrogen, copper = source.GetAtoms()
    carbon.SetProp(LONE_PAIR_COUNT_PROP, "0")
    carbon.SetProp(UNRESOLVED_TWO_ELECTRON_CENTER_PROP, " false ")
    fluorine.SetProp(LONE_PAIR_COUNT_PROP, "invalid")
    fluorine.SetProp(UNRESOLVED_TWO_ELECTRON_CENTER_PROP, "invalid")
    nitrogen.SetProp(UNPAIRED_ELECTRON_COUNT_PROP, "3")
    copper.SetProp(METAL_UNPAIRED_ELECTRONS_PROP, "2")
    copper.SetProp(LONE_PAIR_COUNT_PROP, "-1")
    # The generic property remains in raw diagnostics, but metal assignments
    # come from the metal-specific field.
    copper.SetProp(UNPAIRED_ELECTRON_COUNT_PROP, "99")
    before = source.ToBinary(Chem.PropertyPickleOptions.AllProps)

    repaired = Chem.Mol(source)
    repaired.GetAtomWithIdx(0).SetBoolProp(UNRESOLVED_TWO_ELECTRON_CENTER_PROP, False)
    repaired.GetAtomWithIdx(2).ClearProp(LONE_PAIR_COUNT_PROP)
    repaired.GetAtomWithIdx(2).ClearProp(UNRESOLVED_TWO_ELECTRON_CENTER_PROP)
    repaired.GetAtomWithIdx(4).SetNumRadicalElectrons(3)
    repaired.GetAtomWithIdx(5).ClearProp(LONE_PAIR_COUNT_PROP)
    expected = rdkit_to_mol_data(
        repaired,
        perceive_radicals=True,
        include_stereo=use_chirality,
        include_coordinates=False,
    )
    actual, raw = _prepare_input(source, use_chirality)

    def snapshot(data):
        return (
            [
                (
                    a.atomic_num,
                    a.isotope,
                    a.formal_charge,
                    a.radical_num,
                    a.lone_pair_count,
                    a.unresolved_two_electron_center,
                    a.electron_property_presence,
                )
                for a in data.atoms
            ],
            [(b.begin_atom_idx, b.end_atom_idx, b.order, b.aromatic, b.kind) for b in data.bonds],
            [(s.kind, s.center, s.end, s.configuration, s.refs) for s in data.stereo],
            data.total_charge,
            data.total_radical_num,
        )

    assert snapshot(actual) == snapshot(expected)
    assert actual.atoms[0].electron_property_presence == 6  # known zero/false
    assert actual.atoms[2].electron_property_presence == 0  # invalid stays unknown
    assert actual.atoms[4].radical_num == 3
    assert actual.atoms[5].radical_num == 2
    assert raw.rdkit_radical_electrons == sum(a.GetNumRadicalElectrons() for a in source.GetAtoms())
    assert raw.active_lone_pair_properties == (
        (0, 0),
        (2, UNKNOWN_ELECTRON_METADATA),
        (5, UNKNOWN_ELECTRON_METADATA),
    )
    assert source.ToBinary(Chem.PropertyPickleOptions.AllProps) == before


def test_preparation_preserves_isotopes_restored_as_new_hydrogen_nodes():
    from molgr.utils.equivalence import _prepare_input

    params = Chem.RemoveHsParameters()
    params.removeAndTrackIsotopes = True
    source = Chem.RemoveHs(Chem.MolFromSmiles("[2H]C([3H])O"), params)
    source.GetAtomWithIdx(0).SetIntProp(LONE_PAIR_COUNT_PROP, 0)
    before = source.ToBinary(Chem.PropertyPickleOptions.AllProps)
    expected = Chem.AddHs(source)
    data, _ = _prepare_input(source, False)
    assert [(a.atomic_num, a.isotope) for a in data.atoms] == [
        (a.GetAtomicNum(), a.GetIsotope()) for a in expected.GetAtoms()
    ]
    assert sorted(a.isotope for a in data.atoms if a.isotope) == [2, 3]
    assert data.atoms[0].electron_property_presence == 2
    assert all(a.electron_property_presence == 0 for a in data.atoms[source.GetNumAtoms() :])
    assert source.ToBinary(Chem.PropertyPickleOptions.AllProps) == before


def test_preparation_rejects_query_and_unsupported_bonds():
    from molgr.utils.equivalence import _prepare_input

    with pytest.raises(ValueError, match="query"):
        _prepare_input(Chem.MolFromSmarts("[#6]"), False)
    source = Chem.MolFromSmiles("CC")
    source.GetBondWithIdx(0).SetBondType(Chem.BondType.ZERO)
    with pytest.raises(ValueError, match="Unsupported.*bond type"):
        _prepare_input(source, False)
