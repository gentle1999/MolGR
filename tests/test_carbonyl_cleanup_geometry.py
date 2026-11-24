"""Native cleanup preserves a short carbonyl beside a long C-C bond."""

import math

import pytest
from openbabel import openbabel as ob

from molgr import _core


def molecule(co=1.22, cc=1.50, angle=120):
    m = ob.OBMol()
    theta = math.radians(angle)
    positions = [
        (0, 0, 0),
        (co, 0, 0),
        (cc * math.cos(theta), cc * math.sin(theta), 0),
        (-0.5, -0.8, 0),
        (-1.5, 1.2, 0),
        (-0.3, 2.1, 0),
    ]
    for z, pos in zip([6, 8, 6, 1, 1, 1], positions):
        a = m.NewAtom()
        a.SetAtomicNum(z)
        a.SetImplicitHCount(0)
        a.SetVector(*pos)
    for i, j, k in [(1, 2, 2), (1, 3, 1), (1, 4, 1), (3, 5, 1), (3, 6, 1)]:
        m.AddBond(i, j, k)
    m.GetAtom(3).SetFormalCharge(-1)
    return m


@pytest.mark.parametrize(
    "options,protect",
    [
        ({}, True),
        ({"cc": 1.53}, True),
        ({"cc": 1.43}, False),
        ({"co": 1.28}, False),
        ({"cc": 1.7}, False),
        ({"angle": 90}, False),
        ({"angle": 160}, False),
        ({"co": 0}, False),
        ({"cc": 0}, False),
    ],
)
def test_native_cleanup_retains_short_carbonyl_when_cc_is_long(options, protect):
    m = molecule(**options)
    before = [(a.GetX(), a.GetY(), a.GetZ()) for a in ob.OBMolAtomIter(m)]
    _core.dev.stages.clean.clean_resonances_ptr(int(m.this))
    assert before == [(a.GetX(), a.GetY(), a.GetZ()) for a in ob.OBMolAtomIter(m)]
    assert sum(a.GetFormalCharge() for a in ob.OBMolAtomIter(m)) == -1
    assert m.GetBond(1, 2).GetBondOrder() == (2 if protect else 1)
    assert m.GetBond(1, 3).GetBondOrder() == (1 if protect else 2)
    assert [m.GetAtom(i).GetFormalCharge() for i in [1, 2, 3]] == (
        [0, 0, -1] if protect else [0, -1, 0]
    )


def test_native_cleanup_keeps_cyclic_enolate_resonance_available():
    """A local carbonyl guard must not constrain whole-ring electron allocation."""
    m = ob.OBMol()
    radius = 1.5 / (2 * math.sin(math.pi / 5))
    for i in range(5):
        atom = m.NewAtom()
        atom.SetAtomicNum(6)
        atom.SetImplicitHCount(0)
        theta = 2 * math.pi * i / 5
        atom.SetVector(radius * math.cos(theta), radius * math.sin(theta), 0)
    oxygen = m.NewAtom()
    oxygen.SetAtomicNum(8)
    oxygen.SetImplicitHCount(0)
    oxygen.SetVector(radius + 1.22, 0, 0)
    for i in range(5):
        m.AddBond(i + 1, (i + 1) % 5 + 1, 1)
    m.AddBond(1, 6, 2)
    m.GetAtom(2).SetFormalCharge(-1)
    for idx, count in [(2, 1), (3, 2), (4, 2), (5, 2)]:
        carbon = m.GetAtom(idx)
        for j in range(count):
            hydrogen = m.NewAtom()
            hydrogen.SetAtomicNum(1)
            hydrogen.SetImplicitHCount(0)
            hydrogen.SetVector(carbon.GetX(), carbon.GetY(), 1.09 if j == 0 else -1.09)
            m.AddBond(idx, hydrogen.GetIdx(), 1)
    before = [(a.GetX(), a.GetY(), a.GetZ()) for a in ob.OBMolAtomIter(m)]
    _core.dev.stages.clean.clean_resonances_ptr(int(m.this))
    assert before == [(a.GetX(), a.GetY(), a.GetZ()) for a in ob.OBMolAtomIter(m)]
    assert m.GetBond(1, 6).GetBondOrder() == 1
    assert m.GetBond(1, 2).GetBondOrder() == 2
    assert m.GetAtom(6).GetFormalCharge() == -1
    assert sum(a.GetFormalCharge() for a in ob.OBMolAtomIter(m)) == -1
