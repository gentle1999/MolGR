"""Reject a long pyramidal Si=C without pruning short or unsupported silenes."""

import math

import pytest
from openbabel import openbabel as ob
from openbabel import pybel

from molgr import _core
from molgr.fallback.stages.preprocess import validate_omol
from molgr.fallback.utils.bond_geometry import silicon_carbon_order_allowed
from molgr.fallback.utils.electrons import (
    get_unpaired_electron_count,
    set_lone_pair_count,
    set_unpaired_electron_count,
    set_unresolved_two_electron_center,
)
from molgr.fallback.utils.resonance import build_resonance_state_key


def _silene(change):
    m = ob.OBMol()
    d = {"short": 1.76, "lower_boundary": 1.88, "upper_boundary": 2.10, "too_long": 2.20}.get(
        change, 1.90
    )
    positions = [
        (0, 0, 0),
        (d, 0, 0),
        (-0.5, math.sqrt(1.5), math.sqrt(0.5)),
        (-0.5, -math.sqrt(1.5), math.sqrt(0.5)),
        (d + 0.5, 0.9, 0),
        (d + 0.5, -0.9, 0),
    ]
    if change == "planar":
        positions[2:4] = [(-0.75, 1.299, 0), (-0.75, -1.299, 0)]
    if change in ("angle_sum_below", "angle_sum_above"):
        angle = math.radians((334 if change == "angle_sum_below" else 336) / 3)
        c, s = math.cos(angle), math.sin(angle)
        y = (c - c * c) / s
        positions[2:4] = [
            (1.5 * c, 1.5 * s, 0),
            (1.5 * c, 1.5 * y, 1.5 * math.sqrt(1 - c * c - y * y)),
        ]
    if change == "degenerate":
        positions = [(0, 0, 0)] * 6
    if change == "bad_angles":
        positions[2] = (1.5, 0, 0)
    for z, pos in zip([14, 6, 1, 1, 1, 1], positions):
        a = m.NewAtom()
        a.SetAtomicNum(z)
        a.SetImplicitHCount(0)
        a.SetVector(*pos)
    m.AddBond(1, 2, 1 if change == "single" else 2)
    for i, j in [(1, 3), (1, 4), (2, 5), (2, 6)]:
        m.AddBond(i, j, 1)
    if change == "si_charge":
        m.GetAtom(1).SetFormalCharge(-1)
    if change == "c_charge":
        m.GetAtom(2).SetFormalCharge(-1)
    if change == "si_radical":
        set_unpaired_electron_count(m.GetAtom(1), 1)
    if change == "c_radical":
        set_unpaired_electron_count(m.GetAtom(2), 1)
    if change == "si_deferred":
        set_unresolved_two_electron_center(m.GetAtom(1), True)
    if change == "c_deferred":
        set_unresolved_two_electron_center(m.GetAtom(2), True)
    if change == "si_implicit_h":
        m.GetAtom(1).SetImplicitHCount(1)
    if change == "c_implicit_h":
        m.GetAtom(2).SetImplicitHCount(1)
    if change == "si_hetero_substitution":
        m.GetAtom(3).SetAtomicNum(9)
    if change == "c_hetero_substitution":
        m.GetAtom(5).SetAtomicNum(9)
    if change == "not_silicon":
        m.GetAtom(1).SetAtomicNum(6)
    if change == "hypercoordinate":
        a = m.NewAtom()
        a.SetAtomicNum(1)
        a.SetImplicitHCount(0)
        a.SetVector(0, 0, -1.5)
        m.AddBond(1, 7, 1)
    if change == "ring":
        m.AddBond(3, 5, 1)
    return m


CHANGES = [
    "long_pyramidal",
    "short",
    "lower_boundary",
    "upper_boundary",
    "too_long",
    "planar",
    "degenerate",
    "bad_angles",
    "single",
    "si_charge",
    "c_charge",
    "si_radical",
    "c_radical",
    "si_deferred",
    "c_deferred",
    "si_implicit_h",
    "c_implicit_h",
    "si_hetero_substitution",
    "c_hetero_substitution",
    "not_silicon",
    "hypercoordinate",
    "ring",
    "angle_sum_below",
    "angle_sum_above",
]


@pytest.mark.parametrize("change", CHANGES)
def test_geometry_scope_and_native_validation(change):
    m = _silene(change)
    expected = change not in ("long_pyramidal", "angle_sum_below")
    a, b = m.GetAtom(1), m.GetAtom(2)
    order = m.GetBond(a, b).GetBondOrder()
    assert silicon_carbon_order_allowed(a, b, order) == expected
    assert silicon_carbon_order_allowed(b, a, order) == expected
    q = sum(a.GetFormalCharge() for a in ob.OBMolAtomIter(m))
    spin = sum(get_unpaired_electron_count(a) for a in ob.OBMolAtomIter(m))
    for backend in ["python", "cpp"]:
        copy = ob.OBMol(m)
        before = build_resonance_state_key(pybel.Molecule(copy))
        actual = (
            validate_omol(pybel.Molecule(copy), q, spin)
            if backend == "python"
            else _core.dev.stages.preprocess.validate_omol_ptr(int(copy.this), q, spin)
        )
        assert actual == expected, (change, backend)
        if not expected:
            assert build_resonance_state_key(pybel.Molecule(copy)) == before


def _silyloxonium(variant):
    mol = ob.OBMol()
    co = {"long_co": 1.45, "short_co": 1.10}.get(variant, 1.28)
    so = {"short_sio": 1.58, "far_sio": 2.30}.get(variant, 1.88)
    positions = [
        (0, 0, 0),
        (co, 0, 0),
        (co, so, 0),
        (-0.5, 0.8, 0),
        (-0.5, -0.8, 0),
        (co + 0.8, so + 0.5, 0.7),
        (co + 0.8, so + 0.5, -0.7),
    ]
    for z, pos in zip([6, 8, 14, 1, 1, 1, 1], positions):
        atom = mol.NewAtom()
        atom.SetAtomicNum(z)
        atom.SetImplicitHCount(0)
        atom.SetVector(*pos)
    for i, j, order in [(1, 2, 1), (2, 3, 2), (1, 4, 1), (1, 5, 1), (3, 6, 1), (3, 7, 1)]:
        mol.AddBond(i, j, order)
    mol.GetAtom(1).SetFormalCharge(-1)
    mol.GetAtom(2).SetFormalCharge(1)
    if variant == "already_normalized":
        mol.GetAtom(1).SetFormalCharge(0)
        mol.GetAtom(3).SetFormalCharge(-1)
        mol.GetBond(1, 2).SetBondOrder(2)
        mol.GetBond(2, 3).SetBondOrder(1)
    if variant == "neutral_oxygen":
        mol.GetAtom(2).SetFormalCharge(0)
    if variant == "positive_carbon":
        mol.GetAtom(1).SetFormalCharge(1)
    if variant == "negative_silicon":
        mol.GetAtom(3).SetFormalCharge(-1)
    if variant == "not_silicon":
        mol.GetAtom(3).SetAtomicNum(6)
    if variant == "hypercoordinate":
        atom = mol.NewAtom()
        atom.SetAtomicNum(1)
        atom.SetImplicitHCount(0)
        atom.SetVector(co, so + 1, 0)
        mol.AddBond(3, atom.GetIdx(), 1)
    for idx, element in [(1, "c"), (2, "o"), (3, "si")]:
        atom = mol.GetAtom(idx)
        if variant == element + "_radical":
            set_unpaired_electron_count(atom, 1)
        if variant == element + "_deferred":
            set_unresolved_two_electron_center(atom, True)
        if variant == element + "_lone_pair":
            set_lone_pair_count(atom, 1)
        if variant == element + "_implicit":
            atom.SetImplicitHCount(1)
    if variant == "missing":
        for atom in ob.OBMolAtomIter(mol):
            atom.SetVector(0, 0, 0)
    if variant == "rigid":
        for atom in ob.OBMolAtomIter(mol):
            atom.SetVector(-atom.GetY() + 3, atom.GetX() - 4, atom.GetZ() + 5)
    return mol


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize(
    "variant",
    [
        "stretched",
        "rigid",
        "short_sio",
        "far_sio",
        "long_co",
        "short_co",
        "missing",
        "already_normalized",
        "neutral_oxygen",
        "positive_carbon",
        "negative_silicon",
        "not_silicon",
        "hypercoordinate",
        *[
            element + "_" + state
            for element in ["c", "o", "si"]
            for state in ["radical", "deferred", "lone_pair", "implicit"]
        ],
    ],
)
def test_silyloxonium_cleanup_preserves_charge_and_supported_geometry(backend, variant):
    from molgr.fallback.stages.clean import clean_resonances

    mol = _silyloxonium(variant)
    before = build_resonance_state_key(pybel.Molecule(mol))
    coordinates = [(a.GetX(), a.GetY(), a.GetZ()) for a in ob.OBMolAtomIter(mol)]
    charge = sum(a.GetFormalCharge() for a in ob.OBMolAtomIter(mol))
    if backend == "python":
        _, hit = clean_resonances(pybel.Molecule(mol))
    else:
        hit = _core.dev.stages.clean.clean_resonances_ptr(int(mol.this))
    assert coordinates == [(a.GetX(), a.GetY(), a.GetZ()) for a in ob.OBMolAtomIter(mol)]
    assert charge == sum(a.GetFormalCharge() for a in ob.OBMolAtomIter(mol))
    if variant in ("stretched", "rigid"):
        assert hit
        assert [mol.GetAtom(i).GetFormalCharge() for i in (1, 2, 3)] == [0, 1, -1]
        assert [mol.GetBond(i, j).GetBondOrder() for i, j in [(1, 2), (2, 3)]] == [2, 1]
        assert validate_omol(pybel.Molecule(mol), charge, 0)
    else:
        assert build_resonance_state_key(pybel.Molecule(mol)) == before
