import pytest


# RDKit must initialize before pybel (also required by molgr.__init__).
# isort: off
from rdkit import Chem
from openbabel import openbabel as ob
from openbabel import pybel
# isort: on

from molgr import _core
from molgr.utils.converter import (
    mol_data_to_pybel,
    mol_data_to_rdkit,
    pybel_to_rdmol,
    rdkit_to_mol_data,
)


def _smiles(mol):
    return Chem.MolToSmiles(Chem.RemoveHs(mol))


def _native_roundtrip(data):
    ptr = _core.utils.molecule_data_to_obmol_ptr(data)
    try:
        return mol_data_to_rdkit(_core.utils.extract_molecule_data(ptr))
    finally:
        _core.free_obmol_ptr(ptr)


def _reorder(data):
    """Change both atom numbering and adjacency insertion order."""
    count = len(data.atoms)
    mapping = {i: count + 1 - i for i in range(1, count + 1)}
    mapping[0] = 0
    data.atoms = list(reversed(data.atoms))
    bonds = data.bonds
    for b in bonds:
        b.begin_atom_idx, b.end_atom_idx = mapping[b.begin_atom_idx], mapping[b.end_atom_idx]
        if b.kind == _core.utils.BondKind.Covalent:
            b.begin_atom_idx, b.end_atom_idx = b.end_atom_idx, b.begin_atom_idx
    data.bonds = list(reversed(bonds))
    records = data.stereo
    for s in records:
        s.center, s.end = mapping[s.center], mapping[s.end]
        s.refs = [mapping[i] for i in s.refs]
    data.stereo = records
    groups = data.stereo_groups
    for g in groups:
        g.atoms = [mapping[i] for i in g.atoms]
    data.stereo_groups = groups


COMMON = [
    "N[C@@H](C)O",
    "N[C@H](C)O",
    "F[C@](Cl)(Br)I",
    "F[C@@](Cl)(Br)I",
    "[2H][C@H](F)Cl",
    "C[S@](=O)CC",
    "C[S@@](=O)CC",
    "C1C[C@H](F)[C@@H](Cl)C1",
    "F/C=C/F",
    "F/C=C\\F",
    "Cl/C(F)=C(Br)/I",
    "C/C=N/O",
    "F/C=C/C=C/F",
    "Cl[Pt@SP1](Cl)(N)N",
    "Cl[Pt@SP2](Cl)(N)N",
    "Cl[Pt@SP3](Cl)(N)N",
]


@pytest.mark.parametrize("smiles", COMMON)
@pytest.mark.parametrize("route", ["rdkit", "native", "pybel"])
def test_stereo_survives_explicit_h_reordering_and_conversion(smiles, route):
    source = Chem.MolFromSmiles(smiles)
    before = source.ToBinary(Chem.PropertyPickleOptions.AllProps)
    data = rdkit_to_mol_data(source)
    assert data.stereo
    _reorder(data)
    if route == "native":
        restored = _native_roundtrip(data)
    elif route == "pybel":
        restored = pybel_to_rdmol(mol_data_to_pybel(data))
    else:
        restored = mol_data_to_rdkit(data)
    assert _smiles(restored) == _smiles(source)
    assert source.ToBinary(Chem.PropertyPickleOptions.AllProps) == before


@pytest.mark.parametrize("smiles", COMMON)
def test_ob_extraction_uses_atom_ids_not_indices(smiles):
    # Deleting a disconnected prefix makes IDs differ from index - 1.
    source = pybel.readstring("smi", "[Xe]." + smiles).OBMol
    source.AddHydrogens()
    source.DeleteAtom(source.GetAtom(1))
    assert source.GetAtom(1).GetId() != source.GetAtom(1).GetIdx() - 1
    data = _core.utils.extract_molecule_data(int(source.this))
    _reorder(data)
    restored = mol_data_to_rdkit(data)
    assert _smiles(restored) == _smiles(Chem.MolFromSmiles(smiles))


@pytest.mark.parametrize(
    "shape,count,template",
    [
        ("SP", 3, "F[Pt@SP{n}](Cl)(Br)I"),
        ("TB", 20, "F[As@TB{n}](Cl)(Br)(I)N"),
        ("OH", 30, "F[Co@OH{n}](Cl)(Br)(I)(N)P"),
    ],
)
def test_all_non_tetrahedral_permutations_survive_reordering(shape, count, template):
    signatures = set()
    for permutation in range(1, count + 1):
        source = Chem.MolFromSmiles(template.format(n=permutation))
        data = rdkit_to_mol_data(source)
        _reorder(data)
        restored = mol_data_to_rdkit(data)
        expected = _smiles(source)
        assert _smiles(restored) == expected, (shape, permutation)
        signatures.add(expected)
    assert len(signatures) == count


@pytest.mark.parametrize("group", ["a:1,3", "&1:1,3", "o1:1,3"])
def test_enhanced_groups_survive_and_relative_groups_are_not_projected_as_absolute(group):
    source = Chem.MolFromSmiles(f"C[C@H](O)[C@@H](F)Cl |{group}|")
    data = rdkit_to_mol_data(source)
    _reorder(data)
    restored = mol_data_to_rdkit(data)
    assert Chem.MolToCXSmiles(
        Chem.RemoveHs(restored), Chem.SmilesWriteParams(), Chem.CXSmilesFields.CX_ENHANCEDSTEREO
    ) == Chem.MolToCXSmiles(
        Chem.RemoveHs(Chem.AddHs(source)),
        Chem.SmilesWriteParams(),
        Chem.CXSmilesFields.CX_ENHANCEDSTEREO,
    )
    assert len(restored.GetStereoGroups()) == 1
    if group[0] != "a":
        projected = mol_data_to_pybel(data)
        assert not projected.OBMol.HasData(ob.StereoData)
        assert len(data.stereo) == 2 and len(data.stereo_groups) == 1


@pytest.mark.parametrize("route", ["rdkit", "native", "pybel"])
def test_explicitly_unknown_double_bond_stereo_is_not_dropped(route):
    source = Chem.MolFromSmiles("FC=CF")
    source.GetBondWithIdx(1).SetStereo(Chem.BondStereo.STEREOANY)
    data = rdkit_to_mol_data(source)
    assert data.stereo[0].configuration == 0
    if route == "native":
        restored = _native_roundtrip(data)
    elif route == "pybel":
        restored = pybel_to_rdmol(mol_data_to_pybel(data))
    else:
        restored = mol_data_to_rdkit(data)
    bond = next(b for b in restored.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
    assert bond.GetStereo() == Chem.BondStereo.STEREOANY


def test_unknown_tetrahedral_marker_survives():
    source = Chem.MolFromSmiles("NC(C)O")
    source.GetAtomWithIdx(1).SetBoolProp("_UnknownStereo", True)
    data = rdkit_to_mol_data(source)
    assert data.stereo[0].configuration == 0
    for restored in (mol_data_to_rdkit(data), _native_roundtrip(data)):
        assert restored.GetAtomWithIdx(1).GetBoolProp("_UnknownStereo")
        assert restored.GetAtomWithIdx(1).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED


def test_coordination_projection_does_not_leave_a_stale_stereo_frame():
    source = Chem.MolFromSmiles("N->[Pt@SP1](<-P)(<-F)<-Cl")
    assert source is not None
    data = rdkit_to_mol_data(source)
    assert data.stereo
    assert _smiles(mol_data_to_rdkit(data)) == _smiles(source)
    projected = mol_data_to_pybel(data)
    assert not projected.OBMol.HasData(ob.StereoData)
    assert data.stereo[0].kind == _core.utils.StereoKind.SquarePlanar


def test_stereo_distinguishes_enantiomers_and_geometric_isomers():
    for left, right in (("N[C@H](C)O", "N[C@@H](C)O"), ("F/C=C/F", "F/C=C\\F")):
        assert _smiles(_native_roundtrip(rdkit_to_mol_data(Chem.MolFromSmiles(left)))) != _smiles(
            _native_roundtrip(rdkit_to_mol_data(Chem.MolFromSmiles(right)))
        )


def test_unsupported_stereo_is_explicit_not_silently_erased():
    source = Chem.MolFromSmiles("CCC")
    source.GetAtomWithIdx(1).SetChiralTag(Chem.ChiralType.CHI_ALLENE)
    with pytest.raises(ValueError, match="Unsupported atom stereo"):
        rdkit_to_mol_data(source)
