"""
Author: TMJ
Date: 2026-02-27 14:45:40
LastEditors: TMJ
LastEditTime: 2026-02-27 22:34:34
Description: 请填写简介
"""
# pyright: reportMissingImports=false

import io
import sys
from pathlib import Path

import pytest
from rdkit import Chem

from molgr.fallback.utils.electrons import (
    LONE_PAIR_COUNT_PROP,
    UNPAIRED_ELECTRON_COUNT_PROP,
    UNRESOLVED_TWO_ELECTRON_CENTER_PROP,
    set_lone_pair_count,
    set_unpaired_electron_count,
    set_unresolved_two_electron_center,
)
from molgr.utils.converter import (
    METAL_UNPAIRED_ELECTRONS_PROP,
    get_atom_lone_pair_count,
    get_atom_unpaired_electrons,
    has_atom_unresolved_two_electron_center,
    mol_data_to_pybel,
    mol_data_to_rdkit,
    pybel_to_rdmol,
    refresh_atom_electron_property_lists,
)


pytest.importorskip("rdkit")
pytest.importorskip("openbabel")


def _openbabel_metal_radical_molecule():
    from openbabel import openbabel as ob
    from openbabel import pybel

    obmol = ob.OBMol()
    iron = obmol.NewAtom()
    iron.SetAtomicNum(26)
    iron.SetFormalCharge(2)
    set_unpaired_electron_count(iron, 2)
    set_lone_pair_count(iron, 1)
    iron.SetVector(0.0, 0.0, 0.0)
    carbon = obmol.NewAtom()
    carbon.SetAtomicNum(6)
    set_unpaired_electron_count(carbon, 1)
    set_unresolved_two_electron_center(carbon, True)
    carbon.SetVector(3.0, 0.0, 0.0)
    return pybel.Molecule(obmol)


def _openbabel_two_coordinate_boron_anion():
    from openbabel import openbabel as ob
    from openbabel import pybel

    obmol = ob.OBMol()
    boron = obmol.NewAtom()
    boron.SetAtomicNum(5)
    boron.SetFormalCharge(-1)
    set_unpaired_electron_count(boron, 0)
    boron.SetVector(0.0, 0.0, 0.0)
    for x in (-1.3, 1.3):
        oxygen = obmol.NewAtom()
        oxygen.SetAtomicNum(8)
        oxygen.SetVector(x, 0.0, 0.0)
        obmol.AddBond(boron.GetIdx(), oxygen.GetIdx(), 1)
    return pybel.Molecule(obmol)


def _assert_metal_unpaired_electron_representation(mol) -> None:
    iron = mol.GetAtomWithIdx(0)
    carbon = mol.GetAtomWithIdx(1)

    assert iron.GetNumRadicalElectrons() == 0
    assert iron.GetIntProp(METAL_UNPAIRED_ELECTRONS_PROP) == 2
    assert get_atom_unpaired_electrons(iron) == 2
    assert get_atom_lone_pair_count(iron) == 1
    assert carbon.GetNumRadicalElectrons() == 1
    assert carbon.GetIntProp(UNPAIRED_ELECTRON_COUNT_PROP) == 1
    assert not carbon.HasProp(METAL_UNPAIRED_ELECTRONS_PROP)
    assert get_atom_unpaired_electrons(carbon) == 1
    assert has_atom_unresolved_two_electron_center(carbon)
    assert mol.HasProp(f"atom.iprop.{LONE_PAIR_COUNT_PROP}")
    assert mol.HasProp(f"atom.bprop.{UNRESOLVED_TWO_ELECTRON_CENTER_PROP}")


def test_rdkit_converters_store_metal_spin_outside_formal_radicals() -> None:
    from molgr import _core

    omol = _openbabel_metal_radical_molecule()
    mol_ptr = int(getattr(omol.OBMol, "this", omol.OBMol))
    mol_data = _core.utils.extract_molecule_data(mol_ptr)

    _assert_metal_unpaired_electron_representation(mol_data_to_rdkit(mol_data, kekulize=False))
    _assert_metal_unpaired_electron_representation(pybel_to_rdmol(omol, kekulize=False))


def test_metal_unpaired_electron_property_survives_sdf_round_trip() -> None:
    mol = pybel_to_rdmol(_openbabel_metal_radical_molecule(), kekulize=False)
    buffer = io.StringIO()
    writer = Chem.SDWriter(buffer)
    writer.write(mol)
    writer.flush()

    restored = next(
        Chem.ForwardSDMolSupplier(
            io.BytesIO(buffer.getvalue().encode()),
            sanitize=False,
            removeHs=False,
        )
    )
    assert restored is not None
    iron = restored.GetAtomWithIdx(0)
    assert iron.GetNumRadicalElectrons() == 0
    assert get_atom_unpaired_electrons(iron) == 2
    assert get_atom_lone_pair_count(iron) == 1
    assert int(restored.GetAtomWithIdx(1).GetProp(UNPAIRED_ELECTRON_COUNT_PROP)) == 1
    assert has_atom_unresolved_two_electron_center(restored.GetAtomWithIdx(1))


def test_rdkit_converters_preserve_explicit_closed_shell_boron_anion() -> None:
    from molgr import _core

    omol = _openbabel_two_coordinate_boron_anion()
    mol_ptr = int(getattr(omol.OBMol, "this", omol.OBMol))
    mol_data = _core.utils.extract_molecule_data(mol_ptr)

    for converted in (mol_data_to_rdkit(mol_data), pybel_to_rdmol(omol)):
        boron = converted.GetAtomWithIdx(0)
        assert boron.GetFormalCharge() == -1
        assert boron.GetDegree() == 2
        assert boron.GetNumRadicalElectrons() == 0
        assert boron.GetIntProp(UNPAIRED_ELECTRON_COUNT_PROP) == 0
        assert get_atom_unpaired_electrons(boron) == 0


def _openbabel_known_zero_electron_molecule():
    from openbabel import openbabel as ob
    from openbabel import pybel

    obmol = ob.OBMol()
    for atomic_num, x in ((26, 0.0), (6, 3.0)):
        atom = obmol.NewAtom()
        atom.SetAtomicNum(atomic_num)
        atom.SetVector(x, 0.0, 0.0)
    iron = obmol.GetAtom(1)
    set_unpaired_electron_count(iron, 0)
    set_lone_pair_count(iron, 0)
    set_unresolved_two_electron_center(iron, False)
    return pybel.Molecule(obmol)


def _assert_known_zero_electron_properties(mol) -> None:
    iron = mol.GetAtomWithIdx(0)
    assert iron.HasProp(METAL_UNPAIRED_ELECTRONS_PROP)
    assert iron.GetIntProp(METAL_UNPAIRED_ELECTRONS_PROP) == 0
    assert iron.HasProp(LONE_PAIR_COUNT_PROP)
    assert iron.GetIntProp(LONE_PAIR_COUNT_PROP) == 0
    assert iron.HasProp(UNRESOLVED_TWO_ELECTRON_CENTER_PROP)
    assert not iron.GetBoolProp(UNRESOLVED_TWO_ELECTRON_CENTER_PROP)
    assert iron.GetNumRadicalElectrons() == 0
    assert mol.HasProp(f"atom.iprop.{LONE_PAIR_COUNT_PROP}")
    assert mol.HasProp(f"atom.bprop.{UNRESOLVED_TWO_ELECTRON_CENTER_PROP}")


def test_pybel_converter_preserves_known_zero_and_absent_electron_properties_in_binary() -> None:
    mol = pybel_to_rdmol(_openbabel_known_zero_electron_molecule(), kekulize=False)
    restored = Chem.Mol(mol.ToBinary(Chem.PropertyPickleOptions.AllProps))
    for candidate in (mol, restored):
        _assert_known_zero_electron_properties(candidate)
        carbon = candidate.GetAtomWithIdx(1)
        assert not carbon.HasProp(METAL_UNPAIRED_ELECTRONS_PROP)
        assert not carbon.HasProp(UNPAIRED_ELECTRON_COUNT_PROP)
        assert not carbon.HasProp(LONE_PAIR_COUNT_PROP)
        assert not carbon.HasProp(UNRESOLVED_TWO_ELECTRON_CENTER_PROP)


@pytest.mark.parametrize("via_pybel", [False, True])
def test_molecule_data_converters_preserve_known_zero_electron_properties_in_binary(
    via_pybel: bool,
) -> None:
    from molgr import _core

    omol = _openbabel_known_zero_electron_molecule()
    mol_ptr = int(getattr(omol.OBMol, "this", omol.OBMol))
    mol_data = _core.utils.extract_molecule_data(mol_ptr)
    if via_pybel:
        mol = pybel_to_rdmol(mol_data_to_pybel(mol_data), kekulize=False)
    else:
        mol = mol_data_to_rdkit(mol_data, kekulize=False)
    restored = Chem.Mol(mol.ToBinary(Chem.PropertyPickleOptions.AllProps))
    for candidate in (mol, restored):
        _assert_known_zero_electron_properties(candidate)
        carbon = candidate.GetAtomWithIdx(1)
        assert not carbon.HasProp(UNPAIRED_ELECTRON_COUNT_PROP)
        assert not carbon.HasProp(LONE_PAIR_COUNT_PROP)
        assert not carbon.HasProp(UNRESOLVED_TWO_ELECTRON_CENTER_PROP)


@pytest.mark.parametrize("route", ["direct", "data", "data_pybel", "native_roundtrip"])
def test_converters_leave_unannotated_reference_electron_fields_absent(route: str) -> None:
    from openbabel import openbabel as ob

    omol = _openbabel_known_zero_electron_molecule()
    iron = omol.OBMol.GetAtom(1)
    for name in (
        UNPAIRED_ELECTRON_COUNT_PROP,
        LONE_PAIR_COUNT_PROP,
        UNRESOLVED_TWO_ELECTRON_CENTER_PROP,
    ):
        iron.DeleteData(name)
    assert all(not atom.HasData(LONE_PAIR_COUNT_PROP) for atom in ob.OBMolAtomIter(omol.OBMol))
    from molgr import _core

    if route == "direct":
        mol = pybel_to_rdmol(omol, kekulize=False)
    else:
        data = _core.utils.extract_molecule_data(int(omol.OBMol.this))
        if route == "native_roundtrip":
            ptr = _core.utils.molecule_data_to_obmol_ptr(data)
            try:
                data = _core.utils.extract_molecule_data(ptr)
            finally:
                _core.free_obmol_ptr(ptr)
        if route == "data_pybel":
            mol = pybel_to_rdmol(mol_data_to_pybel(data), kekulize=False)
        else:
            mol = mol_data_to_rdkit(data, kekulize=False)
    restored = Chem.Mol(mol.ToBinary(Chem.PropertyPickleOptions.AllProps))
    for candidate in (mol, restored):
        for atom in candidate.GetAtoms():
            assert not atom.HasProp(METAL_UNPAIRED_ELECTRONS_PROP)
            assert not atom.HasProp(UNPAIRED_ELECTRON_COUNT_PROP)
            assert not atom.HasProp(LONE_PAIR_COUNT_PROP)
            assert not atom.HasProp(UNRESOLVED_TWO_ELECTRON_CENTER_PROP)
        assert not candidate.HasProp(f"atom.iprop.{METAL_UNPAIRED_ELECTRONS_PROP}")
        assert not candidate.HasProp(f"atom.iprop.{UNPAIRED_ELECTRON_COUNT_PROP}")
        assert not candidate.HasProp(f"atom.iprop.{LONE_PAIR_COUNT_PROP}")
        assert not candidate.HasProp(f"atom.bprop.{UNRESOLVED_TWO_ELECTRON_CENTER_PROP}")


@pytest.mark.parametrize("route", ["direct", "data", "data_pybel", "native_roundtrip"])
def test_mixed_atom_electron_presence_survives_conversion(route: str) -> None:
    from openbabel import openbabel as ob
    from openbabel import pybel

    from molgr import _core

    source = ob.OBMol()
    for idx in range(4):
        atom = source.NewAtom()
        atom.SetAtomicNum(6)
        atom.SetVector(float(idx * 3), 0.0, 0.0)
    set_unpaired_electron_count(source.GetAtom(1), 0)
    set_lone_pair_count(source.GetAtom(2), 0)
    set_unresolved_two_electron_center(source.GetAtom(3), False)
    set_unpaired_electron_count(source.GetAtom(4), 2)
    set_lone_pair_count(source.GetAtom(4), 1)
    set_unresolved_two_electron_center(source.GetAtom(4), True)
    if route == "direct":
        converted = pybel_to_rdmol(pybel.Molecule(source), kekulize=False)
    else:
        data = _core.utils.extract_molecule_data(int(source.this))
        if route == "native_roundtrip":
            ptr = _core.utils.molecule_data_to_obmol_ptr(data)
            try:
                data = _core.utils.extract_molecule_data(ptr)
            finally:
                _core.free_obmol_ptr(ptr)
        if route == "data_pybel":
            converted = pybel_to_rdmol(mol_data_to_pybel(data), kekulize=False)
        else:
            converted = mol_data_to_rdkit(data, kekulize=False)
    assert converted.GetAtomWithIdx(0).GetNumRadicalElectrons() == 0
    assert converted.GetAtomWithIdx(3).GetNumRadicalElectrons() == 2
    buffer = io.StringIO()
    with Chem.SDWriter(buffer) as writer:
        writer.write(converted)
    restored = next(Chem.ForwardSDMolSupplier(
        io.BytesIO(buffer.getvalue().encode()), sanitize=False, removeHs=False
    ))
    assert restored is not None
    fields = (
        UNPAIRED_ELECTRON_COUNT_PROP,
        LONE_PAIR_COUNT_PROP,
        UNRESOLVED_TWO_ELECTRON_CENTER_PROP,
    )
    expected = (
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, True),
    )
    for mol in (converted, restored):
        for idx, expected_presence in enumerate(expected):
            atom = mol.GetAtomWithIdx(idx)
            assert tuple(bool(atom.HasProp(name)) for name in fields) == expected_presence
        # SDF integer lists may use a wider C++ integer type on Windows.
        # Assert the serialized value, not RDKit's incidental storage width.
        assert int(mol.GetAtomWithIdx(0).GetProp(UNPAIRED_ELECTRON_COUNT_PROP)) == 0
        assert get_atom_lone_pair_count(mol.GetAtomWithIdx(1)) == 0
        assert not has_atom_unresolved_two_electron_center(mol.GetAtomWithIdx(2))
        assert int(mol.GetAtomWithIdx(3).GetProp(UNPAIRED_ELECTRON_COUNT_PROP)) == 2
        assert get_atom_lone_pair_count(mol.GetAtomWithIdx(3)) == 1
        assert has_atom_unresolved_two_electron_center(mol.GetAtomWithIdx(3))


def test_postprocess_consumed_nonmetal_radicals_update_explicit_sdf_count() -> None:
    from openbabel import openbabel as ob
    from openbabel import pybel

    from molgr.utils.post_process import make_dative_bond

    source = ob.OBMol()
    for atomic_num, x in ((26, 0.0), (6, 1.8), (6, 2.8)):
        atom = source.NewAtom()
        atom.SetAtomicNum(atomic_num)
        atom.SetVector(x, 0.0, 0.0)
    source.AddBond(2, 3, 1)
    set_unpaired_electron_count(source.GetAtom(2), 3)
    converted = pybel_to_rdmol(pybel.Molecule(source), sanitize=False, kekulize=False)
    result = make_dative_bond(converted)
    bond = result.GetBondBetweenAtoms(0, 1)
    assert bond is not None
    assert bond.GetBondType() == Chem.BondType.TRIPLE
    assert result.GetAtomWithIdx(1).GetNumRadicalElectrons() == 0
    assert result.GetAtomWithIdx(1).GetIntProp(UNPAIRED_ELECTRON_COUNT_PROP) == 0
    buffer = io.StringIO()
    with Chem.SDWriter(buffer) as writer:
        writer.write(result)
    restored = next(Chem.ForwardSDMolSupplier(
        io.BytesIO(buffer.getvalue().encode()), sanitize=False, removeHs=False
    ))
    assert restored is not None
    assert int(restored.GetAtomWithIdx(1).GetProp(UNPAIRED_ELECTRON_COUNT_PROP)) == 0


def test_refreshed_electron_lists_follow_edits_and_atom_reordering_in_sdf() -> None:
    mol = pybel_to_rdmol(_openbabel_metal_radical_molecule(), kekulize=False)
    iron = mol.GetAtomWithIdx(0)
    carbon = mol.GetAtomWithIdx(1)
    iron.SetIntProp(METAL_UNPAIRED_ELECTRONS_PROP, 4)
    iron.ClearProp(LONE_PAIR_COUNT_PROP)
    carbon.SetIntProp(LONE_PAIR_COUNT_PROP, 2)
    carbon.SetBoolProp(UNRESOLVED_TWO_ELECTRON_CENTER_PROP, False)
    mol.SetProp("unrelated_metadata", "preserved")
    reordered = Chem.RenumberAtoms(mol, [1, 0])
    # Simulate an editor retaining molecular metadata, including stale lists.
    for name in mol.GetPropNames():
        reordered.SetProp(name, mol.GetProp(name))
    mol = reordered
    refresh_atom_electron_property_lists(mol)
    refresh_atom_electron_property_lists(mol)
    assert get_atom_unpaired_electrons(mol.GetAtomWithIdx(0)) == 1

    buffer = io.StringIO()
    with Chem.SDWriter(buffer) as writer:
        writer.write(mol)
    restored = next(
        Chem.ForwardSDMolSupplier(
            io.BytesIO(buffer.getvalue().encode()), sanitize=False, removeHs=False
        )
    )
    assert restored is not None
    carbon, iron = restored.GetAtomWithIdx(0), restored.GetAtomWithIdx(1)
    assert carbon.GetAtomicNum() == 6
    assert int(carbon.GetProp(UNPAIRED_ELECTRON_COUNT_PROP)) == 1
    assert get_atom_lone_pair_count(carbon) == 2
    assert carbon.HasProp(UNRESOLVED_TWO_ELECTRON_CENTER_PROP)
    assert not has_atom_unresolved_two_electron_center(carbon)
    assert iron.GetAtomicNum() == 26
    assert iron.GetFormalCharge() == 2
    assert iron.GetNumRadicalElectrons() == 0
    assert get_atom_unpaired_electrons(iron) == 4
    assert get_atom_lone_pair_count(iron) == 0
    assert restored.GetProp("unrelated_metadata") == "preserved"


def test_refresh_removes_orphaned_electron_lists_without_reassigning_atoms() -> None:
    mol = pybel_to_rdmol(_openbabel_metal_radical_molecule(), kekulize=False)
    fields = (
        (METAL_UNPAIRED_ELECTRONS_PROP, "iprop"),
        (UNPAIRED_ELECTRON_COUNT_PROP, "iprop"),
        (LONE_PAIR_COUNT_PROP, "iprop"),
        (UNRESOLVED_TWO_ELECTRON_CENTER_PROP, "bprop"),
    )
    for name, kind in fields:
        assert mol.HasProp(f"atom.{kind}.{name}")
        for atom in mol.GetAtoms():
            atom.ClearProp(name)
    # A pure metadata refresh must not impose MolGR's metal radical convention.
    mol.GetAtomWithIdx(0).SetNumRadicalElectrons(3)
    refresh_atom_electron_property_lists(mol)

    for name, kind in fields:
        assert not mol.HasProp(f"atom.{kind}.{name}")
        assert not any(atom.HasProp(name) for atom in mol.GetAtoms())
    assert mol.GetAtomWithIdx(0).GetNumRadicalElectrons() == 3
    assert mol.GetAtomWithIdx(0).GetFormalCharge() == 2
    assert mol.GetAtomWithIdx(1).GetNumRadicalElectrons() == 1


def test_mol_data_to_rdkit_sets_positions_by_atom_index() -> None:
    from openbabel import pybel

    from molgr import _core  # type: ignore
    from molgr.utils.converter import mol_data_to_rdkit

    xyz_block = """3
pos
C 0.0 0.0 0.0
O 1.5 -2.0 3.0
N -4.0 5.0 -6.0
"""

    omol = pybel.readstring("xyz", xyz_block)

    def _get_ptr(obmol) -> int:
        this = getattr(obmol, "this", None)
        if this is not None:
            return int(this)  # type: ignore[arg-type]
        return int(obmol)  # type: ignore[arg-type]

    mol_ptr = _get_ptr(omol.OBMol)
    md = _core.utils.extract_molecule_data(mol_ptr)

    mol = mol_data_to_rdkit(md, sanitize=True)
    assert mol.GetNumAtoms() == 3
    conf = mol.GetConformer()

    p0 = conf.GetAtomPosition(0)
    p1 = conf.GetAtomPosition(1)
    p2 = conf.GetAtomPosition(2)

    assert p0.x == pytest.approx(0.0)
    assert p0.y == pytest.approx(0.0)
    assert p0.z == pytest.approx(0.0)

    assert p1.x == pytest.approx(1.5)
    assert p1.y == pytest.approx(-2.0)
    assert p1.z == pytest.approx(3.0)

    assert p2.x == pytest.approx(-4.0)
    assert p2.y == pytest.approx(5.0)
    assert p2.z == pytest.approx(-6.0)


@pytest.mark.parametrize("case_idx", [1])
def test_mol_data_to_rdkit_matches_fallback_pybel_conversion_for_hard_cases(case_idx: int) -> None:
    from molgr import _core  # type: ignore
    from molgr.fallback import xyz2omol
    from molgr.utils.converter import pybel_to_rdmol
    from molgr.utils.equivalence import check_equivalence

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.molgr_cases_smiles_csv import load_smiles_csv_cases

    cases = load_smiles_csv_cases(Path("tests/test_cases.csv"))
    case = next(row for row in cases if int(row["case_idx"]) == case_idx)

    xyz_block = str(case["xyz_block"])
    total_charge = int(case["total_charge"])
    total_radical_electrons = int(case["total_radical_electrons"])

    fallback_omol = xyz2omol(
        xyz_block,
        total_charge=total_charge,
        total_radical_electrons=total_radical_electrons,
    )
    assert fallback_omol is not None

    cpp_mol_data = _core.pipeline.reconstruct_with_metals.xyz2omol(
        xyz_block,
        total_charge,
        total_radical_electrons,
    )
    assert cpp_mol_data is not None

    fallback_rdmol = pybel_to_rdmol(fallback_omol)
    cpp_rdmol = mol_data_to_rdkit(cpp_mol_data)

    equivalent, info = check_equivalence(
        cpp_rdmol,
        fallback_rdmol,
        use_chirality=True,
        max_resonance=100,
    )
    assert equivalent, info.reason
