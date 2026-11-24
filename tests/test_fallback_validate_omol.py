"""
Author: TMJ
Date: 2026-02-25 00:41:17
LastEditors: TMJ
LastEditTime: 2026-02-25 13:45:25
Description: 请填写简介
"""
# pyright: reportMissingImports=false

from typing import Tuple

import pytest


pytest.importorskip("openbabel")

from openbabel import openbabel as ob
from openbabel import pybel

from molgr.fallback.stages.preprocess import validate_omol
from molgr.fallback.utils.electrons import (
    get_lone_pair_count,
    get_unpaired_electron_count,
    has_unresolved_two_electron_center,
    set_lone_pair_count,
    set_unpaired_electron_count,
    set_unresolved_two_electron_center,
)


def test_validate_omol_distinguishes_lone_pair_from_two_unpaired_electrons() -> None:
    obmol = ob.OBMol()
    a = obmol.NewAtom()
    a.SetAtomicNum(6)
    a.SetFormalCharge(0)

    set_lone_pair_count(a, 1)
    mol = pybel.Molecule(obmol)
    assert validate_omol(mol, 0, 0) is True

    set_lone_pair_count(a, 0)
    set_unpaired_electron_count(a, 2)
    mol = pybel.Molecule(obmol)
    assert validate_omol(mol, 0, 0) is False
    assert validate_omol(mol, 0, 2) is True

    set_unpaired_electron_count(a, 1)
    mol = pybel.Molecule(obmol)
    assert validate_omol(mol, 0, 1) is True


@pytest.mark.parametrize(
    ("total_radical_electrons", "expected_occupancy"),
    [(0, (0, 1)), (2, (2, 0))],
)
def test_validate_omol_resolves_two_electron_center_for_python_and_cpp(
    total_radical_electrons: int,
    expected_occupancy: Tuple[int, int],
) -> None:
    def make_molecule() -> pybel.Molecule:
        obmol = ob.OBMol()
        atom = obmol.NewAtom()
        atom.SetAtomicNum(6)
        set_unresolved_two_electron_center(atom, True)
        return pybel.Molecule(obmol)

    mol = make_molecule()
    assert validate_omol(mol, 0, total_radical_electrons) is True
    atom = mol.OBMol.GetAtom(1)
    assert (get_unpaired_electron_count(atom), get_lone_pair_count(atom)) == expected_occupancy
    assert not has_unresolved_two_electron_center(atom)

    from molgr import _core  # type: ignore

    cpp_mol = make_molecule()
    mol_ptr = int(getattr(cpp_mol.OBMol, "this", cpp_mol.OBMol))
    assert (
        _core.dev.stages.preprocess.validate_omol_ptr(
            mol_ptr,
            0,
            total_radical_electrons,
        )
        is True
    )
    atom = cpp_mol.OBMol.GetAtom(1)
    assert (get_unpaired_electron_count(atom), get_lone_pair_count(atom)) == expected_occupancy
    assert not has_unresolved_two_electron_center(atom)


def test_validate_omol_leaves_unresolved_center_unchanged_when_budget_is_impossible() -> None:
    obmol = ob.OBMol()
    atom = obmol.NewAtom()
    atom.SetAtomicNum(6)
    set_unresolved_two_electron_center(atom, True)
    mol = pybel.Molecule(obmol)

    assert validate_omol(mol, 0, 1) is False
    assert has_unresolved_two_electron_center(atom)
    assert (get_unpaired_electron_count(atom), get_lone_pair_count(atom)) == (0, 0)

    from molgr import _core  # type: ignore

    mol_ptr = int(getattr(mol.OBMol, "this", mol.OBMol))
    assert _core.dev.stages.preprocess.validate_omol_ptr(mol_ptr, 0, 1) is False
    assert has_unresolved_two_electron_center(atom)
    assert (get_unpaired_electron_count(atom), get_lone_pair_count(atom)) == (0, 0)


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize(
    ("order", "length", "angle", "substituted_oxygen", "expected"),
    [
        (3, 1.23, 120.0, False, False),  # Bent acyl cannot become acylium.
        (3, 1.23, 180.0, False, True),  # Preserve linear acylium.
        (3, 1.15, 120.0, False, False),  # Short C--O does not permit a bent triple.
        (3, 1.194, 131.3, False, False),  # Carbamoyl geometry in SUPTON.
        (3, 1.10, 120.0, False, False),
        (3, 1.15, 180.0, False, True),  # Short linear acylium remains available.
        (3, 1.15, None, False, True),  # Short isolated CO remains available.
        (2, 1.194, 131.3, False, True),  # Preserve the metal-bound acyl alternative.
        (3, 1.0, 120.0, False, True),  # Implausibly short geometry abstains.
        (3, 1.23, None, False, True),  # Isolated CO has no carbon angle.
        (2, 1.23, 120.0, False, True),  # The acyl double bond remains available.
        (2, 1.45, 120.0, True, False),  # Long ether-like C--O cannot be C=O.
        (2, 1.30, 120.0, True, True),
        (2, 1.45, 120.0, False, True),  # Terminal carbonyl uses its own scoring.
        (1, 1.45, 120.0, True, True),
        (3, 0.0, 120.0, False, True),  # Missing geometry must abstain.
    ],
)
def test_local_co_geometry_candidate_validation(
    backend, order, length, angle, substituted_oxygen, expected
):
    import math

    from molgr import _core

    mol = ob.OBMol()
    carbon = mol.NewAtom()
    carbon.SetAtomicNum(6)
    carbon.SetVector(0, 0, 0)
    oxygen = mol.NewAtom()
    oxygen.SetAtomicNum(8)
    oxygen.SetVector(length, 0, 0)
    mol.AddBond(1, 2, order)
    if angle is not None:
        neighbor = mol.NewAtom()
        neighbor.SetAtomicNum(6)
        neighbor.SetVector(
            1.5 * math.cos(math.radians(angle)), 1.5 * math.sin(math.radians(angle)), 0
        )
        mol.AddBond(1, 3, 1)
    if substituted_oxygen:
        neighbor = mol.NewAtom()
        neighbor.SetAtomicNum(6)
        neighbor.SetVector(length + 1, 1, 0)
        mol.AddBond(2, neighbor.GetIdx(), 1)
    # Rejected geometry must not resolve deferred labels as a side effect.
    set_unresolved_two_electron_center(carbon, True)
    if backend == "python":
        result = validate_omol(pybel.Molecule(mol), 0, 0)
    else:
        result = _core.dev.stages.preprocess.validate_omol_ptr(int(mol.this), 0, 0)
    assert result is expected
    assert has_unresolved_two_electron_center(carbon) is not expected


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize(
    ("no_length", "cn_length", "order", "extra_hydrogen", "expected"),
    [
        (1.333, 1.313, 2, None, False),
        (1.335, 1.299, 2, None, False),
        (1.333, 1.313, 1, None, True),
        (1.22, 1.42, 2, None, True),
        (1.333, 1.42, 2, None, True),
        (1.333, None, 2, None, True),
        (1.333, 1.313, 2, 1, True),
        (1.333, 1.313, 2, 2, True),
        (0.0, 1.313, 2, None, True),
        (1.333, 0.0, 2, None, True),
    ],
)
def test_nitroso_geometry_does_not_exclude_oximate_or_other_no_sites(
    backend, no_length, cn_length, order, extra_hydrogen, expected
):
    from molgr import _core

    mol = ob.OBMol()
    nitrogen = mol.NewAtom()
    nitrogen.SetAtomicNum(7)
    nitrogen.SetVector(0, 0, 0)
    oxygen = mol.NewAtom()
    oxygen.SetAtomicNum(8)
    oxygen.SetVector(no_length, 0, 0)
    mol.AddBond(1, 2, order)
    if cn_length is not None:
        carbon = mol.NewAtom()
        carbon.SetAtomicNum(6)
        carbon.SetVector(0, cn_length, 0)
        mol.AddBond(1, 3, 2 if order == 1 else 1)
    if extra_hydrogen is not None:
        hydrogen = mol.NewAtom()
        hydrogen.SetAtomicNum(1)
        hydrogen.SetVector(0, 0, 1)
        mol.AddBond(extra_hydrogen, hydrogen.GetIdx(), 1)
    set_unresolved_two_electron_center(nitrogen, True)
    if backend == "python":
        result = validate_omol(pybel.Molecule(mol), 0, 0)
    else:
        result = _core.dev.stages.preprocess.validate_omol_ptr(int(mol.this), 0, 0)
    assert result is expected
    assert has_unresolved_two_electron_center(nitrogen) is not expected


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize(
    "no_length,cn_length,cap_z,cap_degree,cap_order,no_order,expected",
    [
        (1.345, 1.332, 5, 4, 1, 2, False),
        (1.345, 1.332, 5, 4, 1, 1, True),
        (1.22, 1.332, 5, 4, 1, 2, True),
        (1.345, 1.42, 5, 4, 1, 2, True),
        (1.345, 1.332, 5, 3, 1, 2, True),
        (1.345, 1.332, 5, 4, 2, 2, True),
        (1.345, 1.332, 14, 4, 1, 2, True),
        (1.345, 1.332, 6, 4, 1, 2, True),
        (1.345, 1.332, 1, 1, 1, 2, True),
        (0.0, 1.332, 5, 4, 1, 2, True),
        (float("nan"), 1.332, 5, 4, 1, 2, True),
    ],
)
def test_borate_capped_oxime_geometry_scope_matches_backends(
    backend, no_length, cn_length, cap_z, cap_degree, cap_order, no_order, expected
):
    from molgr import _core

    mol = ob.OBMol()
    for atomic_num, xyz in (
        (7, (0, 0, 0)),
        (8, (no_length, 0, 0)),
        (6, (0, cn_length, 0)),
        (cap_z, (2.7, 0, 0)),
    ):
        atom = mol.NewAtom()
        atom.SetAtomicNum(atomic_num)
        atom.SetImplicitHCount(0)
        atom.SetVector(*xyz)
    mol.AddBond(1, 2, no_order)
    mol.AddBond(1, 3, 2 if no_order == 1 else 1)
    mol.AddBond(2, 4, 1)
    for i in range(cap_degree - 1):
        atom = mol.NewAtom()
        atom.SetAtomicNum(9)
        atom.SetImplicitHCount(0)
        atom.SetVector(2.7, i + 1, 0)
        mol.AddBond(4, atom.GetIdx(), cap_order if i == 0 else 1)
    nitrogen = mol.GetAtom(1)
    set_unresolved_two_electron_center(nitrogen, True)
    if backend == "python":
        result = validate_omol(pybel.Molecule(mol), 0, 0)
    else:
        result = _core.dev.stages.preprocess.validate_omol_ptr(int(mol.this), 0, 0)
    assert result is expected
    assert has_unresolved_two_electron_center(nitrogen) is not expected


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize(
    "nn_length,left_cn,right_cn,nn_order,left_z,extra_h,implicit_h,expected",
    [
        (1.36, 1.31, 1.32, 2, 6, False, False, False),
        (1.36, 1.31, 1.32, 1, 6, False, False, True),
        (1.27, 1.31, 1.32, 2, 6, False, False, True),
        (1.36, 1.41, 1.32, 2, 6, False, False, True),
        (1.36, 1.31, 1.42, 2, 6, False, False, True),
        (1.36, 1.31, 1.32, 2, 1, False, False, True),
        (1.36, 1.31, 1.32, 2, 6, True, False, True),
        (1.36, 1.31, 1.32, 2, 6, False, True, True),
        (0.0, 1.31, 1.32, 2, 6, False, False, True),
        (float("nan"), 1.31, 1.32, 2, 6, False, False, True),
        (1.30, 1.31, 1.32, 2, 6, False, False, True),
        (1.50, 1.31, 1.32, 2, 6, False, False, True),
        (1.36, 0.0, 1.32, 2, 6, False, False, True),
        (1.36, 1.343, 1.32, 2, 6, False, False, False),
        (1.33, 1.3498, 1.326, 2, 6, False, False, False),
        (1.36, 1.35, 1.32, 2, 6, False, False, True),
    ],
)
def test_hydrazone_nn_geometry_requires_both_short_cn_bonds(
    backend, nn_length, left_cn, right_cn, nn_order, left_z, extra_h, implicit_h, expected
):
    from molgr import _core

    mol = ob.OBMol()
    for atomic_num, xyz in (
        (7, (0, 0, 0)),
        (7, (nn_length, 0, 0)),
        (left_z, (0, left_cn, 0)),
        (6, (nn_length, -right_cn, 0)),
    ):
        atom = mol.NewAtom()
        atom.SetAtomicNum(atomic_num)
        atom.SetImplicitHCount(0)
        atom.SetVector(*xyz)
    mol.AddBond(1, 2, nn_order)
    mol.AddBond(1, 3, 2 if nn_order == 1 else 1)
    mol.AddBond(2, 4, 2 if nn_order == 1 else 1)
    nitrogen = mol.GetAtom(1)
    if extra_h:
        atom = mol.NewAtom()
        atom.SetAtomicNum(1)
        atom.SetImplicitHCount(0)
        atom.SetVector(0, 0, 1)
        mol.AddBond(1, atom.GetIdx(), 1)
    if implicit_h:
        nitrogen.SetImplicitHCount(1)
    set_unresolved_two_electron_center(nitrogen, True)
    if backend == "python":
        result = validate_omol(pybel.Molecule(mol), 0, 0)
    else:
        result = _core.dev.stages.preprocess.validate_omol_ptr(int(mol.this), 0, 0)
    assert result is expected
    # Rejected intermediates remain untouched and can continue exploration.
    assert has_unresolved_two_electron_center(nitrogen) is not expected


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize("ring_size", [4, 5, 12])
def test_hydrazone_nn_screen_abstains_on_covalent_rings(backend, ring_size):
    from molgr import _core

    mol = ob.OBMol()
    for atomic_num, xyz in (
        (7, (0, 0, 0)),
        (7, (1.36, 0, 0)),
        (6, (0, 1.31, 0)),
        (6, (1.36, -1.32, 0)),
    ):
        atom = mol.NewAtom()
        atom.SetAtomicNum(atomic_num)
        atom.SetImplicitHCount(0)
        atom.SetVector(*xyz)
    mol.AddBond(1, 2, 2)
    mol.AddBond(1, 3, 1)
    mol.AddBond(2, 4, 1)
    previous = 3
    for i in range(ring_size - 4):
        atom = mol.NewAtom()
        atom.SetAtomicNum(6)
        atom.SetImplicitHCount(0)
        atom.SetVector(i + 2, 2, 0)
        mol.AddBond(previous, atom.GetIdx(), 1)
        previous = atom.GetIdx()
    mol.AddBond(previous, 4, 1)
    # Do not precompute rings here: validation must request ring perception.
    nitrogen = mol.GetAtom(1)
    set_unresolved_two_electron_center(nitrogen, True)
    if backend == "python":
        result = validate_omol(pybel.Molecule(mol), 0, 0)
    else:
        result = _core.dev.stages.preprocess.validate_omol_ptr(int(mol.this), 0, 0)
    assert result is True
    assert not has_unresolved_two_electron_center(nitrogen)


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize(
    "kind,expected",
    [
        ("bent", False),
        ("linear", True),
        ("near_linear", True),
        ("below_angle_bound", False),
        ("degenerate", True),
        ("long", True),
        ("missing", True),
        ("nonfinite", True),
        ("nitrite", True),
        ("substituted_n", True),
        ("substituted_o", True),
        ("radical_n", True),
        ("radical_o", True),
        ("deferred_n", True),
        ("deferred_o", True),
    ],
)
def test_nitronium_geometry_requires_resolved_closed_shell_terminal_no2(backend, kind, expected):
    import math

    from molgr import _core

    angle = {"linear": 180, "near_linear": 151, "below_angle_bound": 149, "degenerate": 0}.get(
        kind, 125
    )
    length = {"long": 1.5, "missing": 0, "nonfinite": float("nan")}.get(kind, 1.23)
    mol = ob.OBMol()
    for z, position in (
        (7, (0, 0, 0)),
        (8, (1.23, 0, 0)),
        (8, (length * math.cos(math.radians(angle)), length * math.sin(math.radians(angle)), 0)),
        (6, (5, 5, 5)),
    ):
        atom = mol.NewAtom()
        atom.SetAtomicNum(z)
        atom.SetImplicitHCount(0)
        atom.SetVector(*position)
    mol.GetAtom(1).SetFormalCharge(1)
    mol.AddBond(1, 2, 2)
    mol.AddBond(1, 3, 2)
    if kind in ("nitrite", "substituted_n"):
        mol.GetAtom(3).SetFormalCharge(-1)
        mol.GetBond(1, 3).SetBondOrder(1)
        mol.GetAtom(1).SetFormalCharge(1 if kind == "substituted_n" else 0)
    if kind in ("substituted_n", "substituted_o"):
        atom = mol.NewAtom()
        atom.SetAtomicNum(6)
        atom.SetImplicitHCount(0)
        mol.AddBond(1 if kind == "substituted_n" else 2, atom.GetIdx(), 1)
        if kind == "substituted_o":
            mol.GetAtom(2).SetFormalCharge(1)
    if kind.startswith("radical"):
        set_unpaired_electron_count(mol.GetAtom(1 if kind.endswith("n") else 2), 2)
    if kind.startswith("deferred"):
        set_unresolved_two_electron_center(mol.GetAtom(1 if kind.endswith("n") else 2), True)
    sentinel = mol.GetAtom(4)
    set_unresolved_two_electron_center(sentinel, True)
    charge = sum(a.GetFormalCharge() for a in ob.OBMolAtomIter(mol))
    radicals = sum(get_unpaired_electron_count(a) for a in ob.OBMolAtomIter(mol))
    if backend == "python":
        result = validate_omol(pybel.Molecule(mol), charge, radicals)
    else:
        result = _core.dev.stages.preprocess.validate_omol_ptr(int(mol.this), charge, radicals)
    assert result is expected
    assert has_unresolved_two_electron_center(sentinel) is not expected
