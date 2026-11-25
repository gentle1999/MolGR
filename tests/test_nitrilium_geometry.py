import math

import pytest
from openbabel import openbabel as ob
from openbabel import pybel

from molgr import _core
from molgr.fallback.stages.preprocess import validate_omol
from molgr.fallback.utils.bond_geometry import nitrilium_geometry_allowed
from molgr.fallback.utils.electrons import (
    get_unpaired_electron_count,
    set_unpaired_electron_count,
    set_unresolved_two_electron_center,
)
from molgr.fallback.utils.resonance import build_resonance_state_key


def molecule(change):
    m = pybel.readstring("smi", "CC#[N+]C")
    m.addh()
    m = m.OBMol
    d = {"short": 1.18, "lower_boundary": 1.26, "upper_boundary": 1.50, "too_long": 1.6}.get(
        change, 1.304
    )
    ca = 170 if change in ["linear", "carbon_linear"] else 115
    na = 170 if change in ["linear", "nitrogen_linear"] else 122
    if change == "angle_outside":
        ca = 151
    if change == "angle_inside":
        ca = 149
    ca, na = math.radians(ca), math.radians(na)
    xyz = [
        (1.45 * math.cos(ca), 1.45 * math.sin(ca), 0),
        (0, 0, 0),
        (d, 0, 0),
        (d - 1.45 * math.cos(na), 1.45 * math.sin(na), 0),
    ]
    for i, pos in enumerate(xyz, 1):
        m.GetAtom(i).SetVector(*pos)
    for a in ob.OBMolAtomIter(m):
        a.SetImplicitHCount(0)
        if a.GetAtomicNum() == 1:
            parent = next(ob.OBAtomAtomIter(a))
            a.SetVector(parent.GetX(), parent.GetY(), 1 if a.GetIdx() % 2 else -1)
    c, n = m.GetAtom(2), m.GetAtom(3)
    if change == "missing_geometry":
        for a in ob.OBMolAtomIter(m):
            a.SetVector(0, 0, 0)
    if change == "carbon_charge":
        c.SetFormalCharge(-1)
    if change == "nitrogen_charge":
        n.SetFormalCharge(0)
    if change == "carbon_radical":
        set_unpaired_electron_count(c, 1)
    if change == "nitrogen_radical":
        set_unpaired_electron_count(n, 1)
    if change == "carbon_deferred":
        set_unresolved_two_electron_center(c, True)
    if change == "nitrogen_deferred":
        set_unresolved_two_electron_center(n, True)
    if change == "implicit_h":
        n.SetImplicitHCount(1)
    if change == "hetero_neighbor":
        m.GetAtom(4).SetAtomicNum(8)
    if change == "ring":
        m.AddBond(1, 4, 1)
        m.SetRingAtomsAndBondsPerceived(False)
    if change == "double":
        m.GetBond(2, 3).SetBondOrder(2)
    if change == "terminal_nitrile":
        m.DeleteBond(m.GetBond(3, 4))
        n.SetFormalCharge(0)
    return m


@pytest.mark.parametrize(
    "change",
    [
        "bent_long",
        "short",
        "lower_boundary",
        "upper_boundary",
        "too_long",
        "linear",
        "carbon_linear",
        "nitrogen_linear",
        "angle_outside",
        "angle_inside",
        "missing_geometry",
        "carbon_charge",
        "nitrogen_charge",
        "carbon_radical",
        "nitrogen_radical",
        "carbon_deferred",
        "nitrogen_deferred",
        "implicit_h",
        "hetero_neighbor",
        "ring",
        "double",
        "terminal_nitrile",
    ],
)
def test_scope_and_native_validator_parity(change):
    m = molecule(change)
    c, n = m.GetAtom(2), m.GetAtom(3)
    order = m.GetBond(c, n).GetBondOrder()
    expected = change not in ["bent_long", "angle_inside"]
    assert nitrilium_geometry_allowed(c, n, order) == expected
    assert nitrilium_geometry_allowed(n, c, order) == expected
    q = sum(a.GetFormalCharge() for a in ob.OBMolAtomIter(m))
    s = sum(get_unpaired_electron_count(a) for a in ob.OBMolAtomIter(m))
    results = []
    for backend in ["python", "cpp"]:
        copy = ob.OBMol(m)
        before = build_resonance_state_key(pybel.Molecule(copy))
        result = (
            validate_omol(pybel.Molecule(copy), q, s)
            if backend == "python"
            else _core.dev.stages.preprocess.validate_omol_ptr(int(copy.this), q, s)
        )
        results.append(result)
        if not expected:
            assert not result and build_resonance_state_key(pybel.Molecule(copy)) == before
    assert results[0] == results[1], (change, results)
    if change in ["linear", "short", "carbon_linear", "nitrogen_linear"]:
        assert all(results)
