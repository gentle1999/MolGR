"""Geometry screens must retain alternatives outside their complete local motif."""

import math

import pytest
from openbabel import openbabel as ob
from openbabel import pybel

from molgr import _core
from molgr.fallback.stages import clean
from molgr.fallback.stages.preprocess import validate_omol
from molgr.fallback.utils.electrons import (
    get_unpaired_electron_count,
    set_unpaired_electron_count,
    set_unresolved_two_electron_center,
)
from molgr.fallback.utils.resonance import build_resonance_state_key


def _atom(mol, z, position):
    atom = mol.NewAtom()
    atom.SetAtomicNum(z)
    atom.SetImplicitHCount(0)
    atom.SetVector(*position)
    return atom.GetIdx()


def _validate(mol, backend):
    charge = sum(a.GetFormalCharge() for a in ob.OBMolAtomIter(mol))
    radicals = sum(get_unpaired_electron_count(a) for a in ob.OBMolAtomIter(mol))
    if backend == "python":
        return validate_omol(pybel.Molecule(mol), charge, radicals)
    return _core.dev.stages.preprocess.validate_omol_ptr(int(mol.this), charge, radicals)


def _pyramidal_nn(*, length=1.42, angle=110, order=2):
    mol = ob.OBMol()
    donor = _atom(mol, 7, (0, 0, 0))
    mol.GetAtom(donor).SetFormalCharge(1)
    z = math.sqrt(max(0, (math.cos(math.radians(angle)) + 0.5) / 1.5))
    radius = math.sqrt(1 - z * z)
    for i in range(3):
        phi = i * 2 * math.pi / 3
        distance = length if i == 0 else 1.46
        atom = _atom(
            mol,
            7 if i == 0 else 6,
            (distance * radius * math.cos(phi), distance * radius * math.sin(phi), distance * z),
        )
        mol.AddBond(donor, atom, order if i == 0 else 1)
    carbon = _atom(mol, 6, (2.5, 0, 0.5))
    mol.AddBond(2, carbon, 1)
    return mol


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize(
    "options,expected",
    [
        ({}, False),
        ({"length": 1.379}, True),
        ({"length": 1.381}, False),
        ({"length": 1.599}, False),
        ({"length": 1.601}, True),
        ({"angle": 113}, False),
        ({"angle": 114}, True),
        ({"angle": 120}, True),
        ({"order": 1}, True),
        ({"order": 3}, True),
    ],
)
def test_pyramidal_hydrazonium_needs_joint_length_and_angle_evidence(backend, options, expected):
    mol = _pyramidal_nn(**options)
    before = build_resonance_state_key(pybel.Molecule(mol))
    assert _validate(mol, backend) is expected
    assert build_resonance_state_key(pybel.Molecule(mol)) == before


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize(
    "variant",
    [
        "neutral",
        "negative_other",
        "radical",
        "unresolved",
        "implicit_h",
        "hetero_substitution",
        "ring",
    ],
)
def test_pyramidal_hydrazonium_abstains_on_other_local_states(backend, variant):
    mol = _pyramidal_nn()
    if variant == "neutral":
        mol.GetAtom(1).SetFormalCharge(0)
    elif variant == "negative_other":
        mol.GetAtom(2).SetFormalCharge(-1)
    elif variant == "radical":
        set_unpaired_electron_count(mol.GetAtom(1), 1)
    elif variant == "unresolved":
        set_unresolved_two_electron_center(mol.GetAtom(1), True)
    elif variant == "implicit_h":
        mol.GetAtom(1).SetImplicitHCount(1)
    elif variant == "hetero_substitution":
        mol.GetAtom(3).SetAtomicNum(8)
    else:
        mol.AddBond(5, 3, 1)
    assert _validate(mol, backend)


def _pyramidal_pc(*, length=1.84, angle=105, order=2):
    mol = ob.OBMol()
    p = _atom(mol, 15, (0, 0, 0))
    mol.GetAtom(p).SetFormalCharge(1)
    z = math.sqrt(max(0, (math.cos(math.radians(angle)) + 0.5) / 1.5))
    radius = math.sqrt(1 - z * z)
    for i in range(3):
        phi = i * 2 * math.pi / 3
        distance = length if i == 0 else 1.84
        c = _atom(
            mol,
            6,
            (distance * radius * math.cos(phi), distance * radius * math.sin(phi), distance * z),
        )
        mol.AddBond(p, c, order if i == 0 else 1)
    return mol


@pytest.mark.parametrize(
    "length,angle,blocked", [(1.84, 105, True), (1.70, 105, False), (1.84, 120, False)]
)
@pytest.mark.parametrize("alternative", [False, True])
def test_cpp_positive_assignment_checks_pc_geometry_before_spending_charge(
    length, angle, blocked, alternative
):
    mol = _pyramidal_pc(length=length, angle=angle, order=1)
    mol.GetAtom(1).SetFormalCharge(0)
    set_unpaired_electron_count(mol.GetAtom(2), 1)
    for sign in (-1, 1):
        mol.AddBond(2, _atom(mol, 1, (length + 0.5, sign, 0)), 1)
    carbon = _atom(mol, 6, (8, 0, 0))
    set_unpaired_electron_count(mol.GetAtom(carbon), 1)
    for sign in (-1, 1):
        mol.AddBond(carbon, _atom(mol, 1, (8, sign, 0)), 1)
    if alternative:
        amine = _atom(mol, 7, (9.3, 0, 0))
        mol.AddBond(carbon, amine, 1)
        for sign in (-1, 1):
            mol.AddBond(amine, _atom(mol, 1, (9.3, sign, 0)), 1)
    else:
        mol.AddBond(carbon, _atom(mol, 1, (9, 0, 0)), 1)
    before = build_resonance_state_key(pybel.Molecule(mol))
    remaining, hit = _core.dev.stages.eliminate.eliminate_positive_charges_ptr(int(mol.this), 0)
    if blocked and not alternative:
        assert not hit and remaining == 0
        assert build_resonance_state_key(pybel.Molecule(mol)) == before
        return
    assert hit and remaining == -1
    assert mol.GetBond(1, 2).GetBondOrder() == (1 if blocked else 2)
    assert mol.GetAtom(1).GetFormalCharge() == (0 if blocked else 1)
    assert get_unpaired_electron_count(mol.GetAtom(2)) == (1 if blocked else 0)
    if alternative:
        assert mol.GetBond(carbon, amine).GetBondOrder() == (2 if blocked else 1)
        assert mol.GetAtom(amine).GetFormalCharge() == (1 if blocked else 0)


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize(
    "options,expected",
    [
        ({}, False),
        ({"length": 1.70}, True),
        ({"length": 1.779}, True),
        ({"length": 1.781}, False),
        ({"length": 2.099}, False),
        ({"length": 2.101}, True),
        ({"length": 0}, True),
        ({"angle": 99}, True),
        ({"angle": 101}, False),
        ({"angle": 109}, False),
        ({"angle": 111}, True),
        ({"angle": 120}, True),
        ({"order": 1}, True),
        ({"order": 3}, True),
    ],
)
def test_pyramidal_pc_requires_joint_geometry_and_electronic_state(backend, options, expected):
    mol = _pyramidal_pc(**options)
    before = build_resonance_state_key(pybel.Molecule(mol))
    assert _validate(mol, backend) is expected
    assert build_resonance_state_key(pybel.Molecule(mol)) == before


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize(
    "variant",
    [
        "neutral_p",
        "negative_carbon",
        "positive_carbon",
        "hetero_neighbor",
        "ring",
        "p_radical",
        "c_radical",
        "p_implicit_h",
        "c_implicit_h",
        "missing",
        "extra_neighbor",
        "single_bonds_lone_pair",
    ],
)
def test_pyramidal_pc_preserves_other_phosphorus_states(backend, variant):
    mol = _pyramidal_pc()
    p, c = mol.GetAtom(1), mol.GetAtom(2)
    if variant == "neutral_p":
        p.SetFormalCharge(0)
    elif variant == "negative_carbon":
        c.SetFormalCharge(-1)
    elif variant == "positive_carbon":
        c.SetFormalCharge(1)
    elif variant == "hetero_neighbor":
        mol.GetAtom(3).SetAtomicNum(7)
    elif variant == "ring":
        mol.AddBond(2, 3, 1)
    elif variant.endswith("radical"):
        set_unpaired_electron_count(p if variant.startswith("p_") else c, 1)
    elif variant.endswith("implicit_h"):
        (p if variant.startswith("p_") else c).SetImplicitHCount(1)
    elif variant == "missing":
        for atom in ob.OBMolAtomIter(mol):
            atom.SetVector(0, 0, 0)
    elif variant == "extra_neighbor":
        mol.AddBond(1, _atom(mol, 6, (0, 0, -1.84)), 1)
    elif variant == "single_bonds_lone_pair":
        mol.GetBond(1, 2).SetBondOrder(1)
    assert _validate(mol, backend)


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize("index", [1, 2])
def test_pyramidal_pc_defers_unresolved_electrons(backend, index):
    mol = _pyramidal_pc()
    set_unresolved_two_electron_center(mol.GetAtom(index), True)
    assert _validate(mol, backend)


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_pyramidal_pc_is_invariant_to_rotation_and_translation(backend):
    mol = _pyramidal_pc()
    for atom in ob.OBMolAtomIter(mol):
        x, y, z = atom.GetX(), atom.GetY(), atom.GetZ()
        atom.SetVector(z + 12, x - 4, y + 7)
    assert not _validate(mol, backend)


def _diazonium(*, nn=1.28, cn=1.43, angle=125, order=3):
    mol = ob.OBMol()
    theta = math.radians(angle)
    _atom(mol, 7, (nn, 0, 0))
    _atom(mol, 7, (0, 0, 0))
    _atom(mol, 6, (cn * math.cos(theta), cn * math.sin(theta), 0))
    mol.GetAtom(2).SetFormalCharge(1)
    mol.AddBond(1, 2, order)
    mol.AddBond(2, 3, 1)
    return mol


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize(
    "options,expected",
    [
        ({}, False),
        ({"angle": 135.4}, False),
        ({"angle": 149}, False),
        ({"angle": 151}, True),
        ({"angle": 180}, True),
        ({"nn": 1.10}, True),
        ({"nn": 1.24}, True),
        ({"nn": 1.50}, True),
        ({"cn": 1.14}, True),
        ({"cn": 1.61}, True),
        ({"nn": 0}, True),
        ({"cn": 0}, True),
        ({"order": 2}, True),
    ],
)
def test_bent_diazonium_requires_joint_length_angle_and_order_evidence(backend, options, expected):
    mol = _diazonium(**options)
    before = build_resonance_state_key(pybel.Molecule(mol))
    assert _validate(mol, backend) is expected
    assert build_resonance_state_key(pybel.Molecule(mol)) == before


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize(
    "variant",
    [
        "radical",
        "deferred",
        "neutral_center",
        "charged_terminal",
        "implicit_h",
        "capped_terminal",
        "noncarbon",
        "ring",
        "missing",
    ],
)
def test_bent_diazonium_abstains_outside_resolved_acyclic_motif(backend, variant):
    mol = _diazonium()
    if variant == "radical":
        set_unpaired_electron_count(mol.GetAtom(1), 1)
    elif variant == "deferred":
        set_unresolved_two_electron_center(mol.GetAtom(1), True)
    elif variant == "neutral_center":
        mol.GetAtom(2).SetFormalCharge(0)
    elif variant == "charged_terminal":
        mol.GetAtom(1).SetFormalCharge(-1)
    elif variant == "implicit_h":
        mol.GetAtom(2).SetImplicitHCount(1)
    elif variant == "capped_terminal":
        cap = _atom(mol, 6, (2.7, 0, 0))
        mol.AddBond(1, cap, 1)
    elif variant == "noncarbon":
        mol.GetAtom(3).SetAtomicNum(1)
    elif variant == "ring":
        mol.AddBond(1, 3, 1)
    else:
        for atom in ob.OBMolAtomIter(mol):
            atom.SetVector(0, 0, 0)
    assert _validate(mol, backend)


def _cnn_cumulene(*, nn=1.382, cn=1.310, angle=114.4):
    mol = _diazonium(nn=nn, cn=cn, angle=angle, order=2)
    mol.GetBond(2, 3).SetBondOrder(2)
    mol.AddBond(1, _atom(mol, 6, (nn + 1.5, 0, 0)), 1)
    carbon = mol.GetAtom(3)
    for sign in (-1, 1):
        cap = _atom(mol, 6, (carbon.GetX(), carbon.GetY(), sign * 1.5))
        mol.AddBond(3, cap, 1)
    return mol


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize("angle,expected", [(124, False), (149, False), (151, True), (180, True)])
def test_terminal_diazo_cumulene_requires_linear_central_n(backend, angle, expected):
    mol = _cnn_cumulene(angle=angle)
    outer = mol.GetAtom(1)
    cap = next(a for a in ob.OBAtomAtomIter(outer) if a.GetAtomicNum() == 6)
    mol.DeleteAtom(cap)
    mol.GetAtom(1).SetFormalCharge(-1)
    before = build_resonance_state_key(pybel.Molecule(mol))
    assert _validate(mol, backend) is expected
    assert build_resonance_state_key(pybel.Molecule(mol)) == before


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize("motif", ["nitronium", "cnn"])
@pytest.mark.parametrize("angle", [120, 180])
@pytest.mark.parametrize("alternative", [False, True])
def test_positive_charge_assignment_skips_invalid_geometry(backend, motif, angle, alternative):
    from molgr.fallback.stages.eliminate import eliminate_positive_charges

    if motif == "cnn":
        mol = _cnn_cumulene(angle=angle)
        radical, donor = 1, 2
        mol.GetBond(radical, donor).SetBondOrder(1)
        mol.GetAtom(donor).SetFormalCharge(0)
    else:
        mol = ob.OBMol()
        donor = _atom(mol, 7, (0, 0, 0))
        radical = _atom(mol, 8, (1.25, 0, 0))
        other = _atom(
            mol, 8, (1.25 * math.cos(math.radians(angle)), 1.25 * math.sin(math.radians(angle)), 0)
        )
        mol.AddBond(donor, radical, 1)
        mol.AddBond(donor, other, 2)
    set_unpaired_electron_count(mol.GetAtom(radical), 1)
    carbon = _atom(mol, 6, (8, 0, 0))
    set_unpaired_electron_count(mol.GetAtom(carbon), 1)
    if alternative:
        amine = _atom(mol, 7, (9.3, 0, 0))
        mol.AddBond(carbon, amine, 1)
        for i in range(2):
            mol.AddBond(amine, _atom(mol, 1, (9.3, i * 2 - 1, 0)), 1)
            mol.AddBond(carbon, _atom(mol, 1, (8, i * 2 - 1, 0)), 1)
    else:
        for position in ((8, 1, 0), (8, -1, 0), (9, 0, 0)):
            mol.AddBond(carbon, _atom(mol, 1, position), 1)
    # A zero budget excludes direct radical-to-cation fallback. Two
    # radicals permit exactly one donor action; a rejected action consumes none.
    initial_charge = 0
    before = build_resonance_state_key(pybel.Molecule(mol))
    if backend == "python":
        _, remaining, hit = eliminate_positive_charges(pybel.Molecule(mol), initial_charge)
    else:
        remaining, hit = _core.dev.stages.eliminate.eliminate_positive_charges_ptr(
            int(mol.this), initial_charge
        )
    if not alternative and angle == 120:
        assert not hit
        assert remaining == initial_charge
        assert build_resonance_state_key(pybel.Molecule(mol)) == before
        return
    assert hit
    assert remaining == initial_charge - 1
    assert mol.GetBond(radical, donor).GetBondOrder() == (1 if angle == 120 else 2)
    assert mol.GetAtom(donor).GetFormalCharge() == (0 if angle == 120 else 1)
    assert get_unpaired_electron_count(mol.GetAtom(radical)) == (1 if angle == 120 else 0)
    if alternative:
        assert mol.GetBond(carbon, amine).GetBondOrder() == (2 if angle == 120 else 1)
        assert mol.GetAtom(amine).GetFormalCharge() == (1 if angle == 120 else 0)


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize(
    "options,expected",
    [
        ({}, False),
        ({"angle": 149}, False),
        ({"angle": 151}, True),
        ({"angle": 180}, True),
        ({"nn": 1.19}, True),
        ({"nn": 1.51}, True),
        ({"cn": 1.14}, True),
        ({"cn": 1.51}, True),
        ({"nn": 0}, True),
        ({"cn": 0}, True),
    ],
)
def test_bent_cnn_cumulene_requires_joint_geometry(backend, options, expected):
    mol = _cnn_cumulene(**options)
    before = build_resonance_state_key(pybel.Molecule(mol))
    assert _validate(mol, backend) is expected
    assert build_resonance_state_key(pybel.Molecule(mol)) == before


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize(
    "variant",
    [
        "neutral_center",
        "diazo",
        "charged_carbon",
        "single_cn",
        "single_nn",
        "ring",
        "noncarbon",
        "carbon_degree",
        "missing",
        "rigid_transform",
    ],
)
def test_bent_cnn_cumulene_preserves_other_electronic_structures(backend, variant):
    mol = _cnn_cumulene(angle=180) if variant == "diazo" else _cnn_cumulene()
    if variant == "neutral_center":
        mol.GetAtom(2).SetFormalCharge(0)
    elif variant == "diazo":
        mol.GetAtom(1).SetFormalCharge(-1)
        mol.DeleteBond(mol.GetBond(1, 4))
    elif variant == "charged_carbon":
        mol.GetAtom(3).SetFormalCharge(-1)
    elif variant == "single_cn":
        mol.GetBond(2, 3).SetBondOrder(1)
    elif variant == "single_nn":
        mol.GetBond(1, 2).SetBondOrder(1)
    elif variant == "ring":
        mol.AddBond(4, 5, 1)
    elif variant == "noncarbon":
        mol.GetAtom(3).SetAtomicNum(14)
    elif variant == "carbon_degree":
        mol.DeleteBond(mol.GetBond(3, 5))
    elif variant == "missing":
        for atom in ob.OBMolAtomIter(mol):
            atom.SetVector(0, 0, 0)
    elif variant == "rigid_transform":
        for atom in ob.OBMolAtomIter(mol):
            atom.SetVector(atom.GetZ() + 7, atom.GetX() - 4, atom.GetY() + 12)
    assert _validate(mol, backend) is (variant != "rigid_transform")


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize("index", [1, 2, 3])
@pytest.mark.parametrize("label", ["radical", "deferred", "implicit_h"])
def test_bent_cnn_cumulene_abstains_for_unresolved_or_implicit_labels(backend, index, label):
    mol = _cnn_cumulene()
    atom = mol.GetAtom(index)
    if label == "radical":
        set_unpaired_electron_count(atom, 1)
    elif label == "deferred":
        set_unresolved_two_electron_center(atom, True)
    else:
        atom.SetImplicitHCount(1)
    assert _validate(mol, backend)


def _short_azo_hydrazone(*, nn=1.289, left_cn=1.398, right_cn=1.362):
    mol = _cnn_cumulene(nn=nn, cn=left_cn)
    mol.GetAtom(2).SetFormalCharge(0)
    mol.GetBond(1, 2).SetBondOrder(1)
    mol.GetBond(1, 4).SetBondOrder(2)
    mol.GetAtom(4).SetVector(nn + right_cn, 0, 0)
    for sign in (-1, 1):
        mol.AddBond(4, _atom(mol, 6, (nn + right_cn, sign * 1.5, 0)), 1)
    return mol


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize(
    "options,expected",
    [
        ({}, False),
        ({"nn": 1.32}, True),
        ({"nn": 1.19}, True),
        ({"left_cn": 1.34}, True),
        ({"right_cn": 1.34}, True),
        ({"left_cn": 1.51}, True),
        ({"right_cn": 1.51}, True),
        ({"nn": 0}, True),
        ({"left_cn": 0}, True),
    ],
)
def test_short_nn_requires_both_long_cn_bonds(backend, options, expected):
    mol = _short_azo_hydrazone(**options)
    before = build_resonance_state_key(pybel.Molecule(mol))
    assert _validate(mol, backend) is expected
    assert build_resonance_state_key(pybel.Molecule(mol)) == before


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize("index", [1, 2, 3, 4])
@pytest.mark.parametrize("label", ["charged", "radical", "deferred", "implicit_h"])
def test_short_nn_preserves_other_electronic_states(backend, index, label):
    mol = _short_azo_hydrazone()
    atom = mol.GetAtom(index)
    if label == "charged":
        atom.SetFormalCharge(-1)
    elif label == "radical":
        set_unpaired_electron_count(atom, 1)
    elif label == "deferred":
        set_unresolved_two_electron_center(atom, True)
    else:
        atom.SetImplicitHCount(1)
    assert _validate(mol, backend)


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize("variant", ["ring", "azo", "substituted_n", "rigid_transform"])
def test_short_nn_retains_azo_and_cyclic_alternatives(backend, variant):
    mol = _short_azo_hydrazone()
    if variant == "ring":
        mol.AddBond(5, 7, 1)
    elif variant == "azo":
        mol.GetBond(1, 2).SetBondOrder(2)
        mol.GetBond(2, 3).SetBondOrder(1)
        mol.GetBond(1, 4).SetBondOrder(1)
    elif variant == "substituted_n":
        mol.AddBond(1, _atom(mol, 1, (1.0, 1.0, 0)), 1)
    else:
        for atom in ob.OBMolAtomIter(mol):
            atom.SetVector(atom.GetZ() + 7, atom.GetX() - 4, atom.GetY() + 12)
    assert _validate(mol, backend) is (variant != "rigid_transform")


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize("index", [5, 7])
@pytest.mark.parametrize("element", [7, 8, 16])
def test_short_nn_abstains_for_additional_heteroatom_conjugation(backend, index, element):
    mol = _short_azo_hydrazone()
    mol.GetAtom(index).SetAtomicNum(element)
    assert _validate(mol, backend)


def _thiohydrazone(*, nn=1.33, cn=1.32, cs=1.74, left_element=6):
    mol = ob.OBMol()
    for z, pos in [
        (left_element, (-1.41, 0, 0)),
        (7, (0, 0, 0)),
        (7, (nn, 0, 0)),
        (6, (nn + cn, 0, 0)),
        (16, (nn + cn, cs, 0)),
        (6, (nn + cn, -1.50, 0)),
    ]:
        _atom(mol, z, pos)
    for i, j, order in [(1, 2, 1), (2, 3, 2), (3, 4, 1), (4, 5, 2), (4, 6, 1)]:
        mol.AddBond(i, j, order)
    return mol


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize(
    "options,expected",
    [
        ({}, False),
        ({"nn": 1.28}, True),
        ({"nn": 1.52}, True),
        ({"cn": 1.39}, True),
        ({"cn": 1.17}, True),
        ({"cs": 1.65}, True),
        ({"cs": 1.95}, True),
        ({"left_element": 1}, True),
    ],
)
def test_thiohydrazone_requires_all_three_lengths_and_carbon_substitution(
    backend, options, expected
):
    assert _validate(_thiohydrazone(**options), backend) is expected


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize(
    "variant", ["radical", "deferred", "charged", "implicit_h", "capped_s", "ring", "missing"]
)
def test_thiohydrazone_screen_abstains_on_other_electronic_or_topological_states(backend, variant):
    mol = _thiohydrazone()
    if variant == "radical":
        set_unpaired_electron_count(mol.GetAtom(2), 1)
    elif variant == "deferred":
        set_unresolved_two_electron_center(mol.GetAtom(2), True)
    elif variant == "charged":
        mol.GetAtom(2).SetFormalCharge(1)
    elif variant == "implicit_h":
        mol.GetAtom(5).SetImplicitHCount(1)
    elif variant == "capped_s":
        cap = _atom(mol, 6, (3, 3, 0))
        mol.AddBond(5, cap, 1)
    elif variant == "ring":
        mol.AddBond(1, 6, 1)
    else:
        for atom in ob.OBMolAtomIter(mol):
            atom.SetVector(0, 0, 0)
    assert _validate(mol, backend)


def _paired_carbonyl(*, cc=1.49, co=1.22, angle=123, ring_cc=1.415, ring_co=1.32, cyclic=True):
    mol = ob.OBMol()
    height = ring_cc * math.sqrt(3) / 2
    for pos in [
        (0, 0, 0),
        (ring_cc, 0, 0),
        (1.5 * ring_cc, height, 0),
        (ring_cc, 2 * height, 0),
        (0, 2 * height, 0),
        (-0.5 * ring_cc, height, 0),
    ]:
        _atom(mol, 6, pos)
    for i, j, order in [(1, 2, 1), (2, 3, 1), (3, 4, 2), (4, 5, 1), (5, 6, 2)]:
        mol.AddBond(i, j, order)
    if cyclic:
        mol.AddBond(6, 1, 1)
    center = _atom(mol, 6, (-cc, 0, 0))
    radians = math.radians(angle)
    oxygen = _atom(mol, 8, (-cc + co * math.cos(radians), co * math.sin(radians), 0))
    other_oxygen = _atom(mol, 8, (ring_cc, -ring_co, 0))
    mol.AddBond(1, center, 2)
    mol.AddBond(center, oxygen, 2)
    mol.AddBond(2, other_oxygen, 2)
    return mol


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize(
    "options,expected",
    [
        ({}, False),
        ({"angle": 170}, True),
        ({"angle": 80}, True),
        ({"cc": 1.33}, True),
        ({"cc": 1.70}, True),
        ({"co": 1.42}, True),
        ({"ring_co": 1.22}, True),
        ({"ring_co": 1.46}, True),
        ({"ring_cc": 1.50}, True),
        ({"ring_cc": 1.33}, True),
        ({"cyclic": False}, True),
    ],
)
def test_bent_ketene_requires_neighboring_ring_carbonyl_evidence(backend, options, expected):
    assert _validate(_paired_carbonyl(**options), backend) is expected


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize(
    "variant", ["radical", "deferred", "charged", "implicit_h", "capped_o", "missing", "nan"]
)
def test_paired_carbonyl_screen_abstains_outside_closed_shell_explicit_motif(backend, variant):
    mol = _paired_carbonyl()
    if variant == "radical":
        set_unpaired_electron_count(mol.GetAtom(7), 1)
    elif variant == "deferred":
        set_unresolved_two_electron_center(mol.GetAtom(7), True)
    elif variant == "charged":
        mol.GetAtom(9).SetFormalCharge(-1)
    elif variant == "implicit_h":
        mol.GetAtom(9).SetImplicitHCount(1)
    elif variant == "capped_o":
        cap = _atom(mol, 6, (3, -2, 0))
        mol.AddBond(9, cap, 1)
    else:
        for atom in ob.OBMolAtomIter(mol):
            atom.SetVector(0, 0, 0 if variant == "missing" else float("nan"))
    assert _validate(mol, backend)


def _radical_hydrazone_path(size, variant):
    """Closed-shell inner atoms, two terminal radicals, and explicit H only."""
    mol = ob.OBMol()
    if size == 6:
        numbers, lengths, pair = [8, 6, 7, 7, 6, 6], [1.30, 1.32, 1.40, 1.32, 1.40], (3, 4)
    else:
        numbers, lengths, pair = [7, 7, 6, 6], [1.40, 1.32, 1.40], (1, 2)
    position = 0.0
    for index, z in enumerate(numbers):
        if index:
            position += lengths[index - 1]
        _atom(mol, z, (position, 0, 0))
    for i in range(1, size):
        mol.AddBond(i, i + 1, 1 if i % 2 else 2)
    if size == 4:
        cap = _atom(mol, 6, (-1.32, 0, 0))
        mol.AddBond(1, cap, 1)
    # Complete carbon valences with explicit H while retaining a radical at
    # the terminal carbon. The nitrogen endpoint in the four-atom path also
    # has valence two and one real unpaired electron.
    for atom in list(ob.OBMolAtomIter(mol)):
        if atom.GetAtomicNum() != 6:
            continue
        target = 3 if atom.GetIdx() == size else 4
        for j in range(target - atom.GetExplicitValence()):
            hydrogen = _atom(mol, 1, (atom.GetX(), j + 1, 0))
            mol.AddBond(atom.GetIdx(), hydrogen, 1)
    for idx in (1, size):
        set_unpaired_electron_count(mol.GetAtom(idx), 1)
    first, second = (mol.GetAtom(i) for i in pair)
    if variant in ("short_nn", "long_nn", "long_cn"):
        # Move a whole suffix to vary only one local bond length.
        split = pair[1] if variant != "long_cn" else pair[1] + 1
        delta = {"short_nn": -0.12, "long_nn": 0.12, "long_cn": 0.10}[variant]
        for atom in ob.OBMolAtomIter(mol):
            if split <= atom.GetIdx() <= size:
                atom.SetVector(atom.GetX() + delta, atom.GetY(), atom.GetZ())
    elif variant == "implicit_h":
        second.SetImplicitHCount(1)
    elif variant == "ring":
        mol.AddBond(1 if size == 6 else cap, size, 1)
    elif variant == "missing":
        for atom in ob.OBMolAtomIter(mol):
            atom.SetVector(0, 0, 0)
    elif variant == "rigid":
        for atom in ob.OBMolAtomIter(mol):
            atom.SetVector(-atom.GetY() + 3, atom.GetX() - 4, atom.GetZ() + 5)
    return mol


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize("size", [4, 6])
@pytest.mark.parametrize(
    "variant,allowed",
    [
        ("conflict", False),
        ("rigid", False),
        ("short_nn", True),
        ("long_nn", True),
        ("long_cn", True),
        ("implicit_h", True),
        ("ring", True),
        ("missing", True),
    ],
)
def test_radical_cleanup_preserves_geometry_compatible_hydrazone(backend, size, variant, allowed):
    from molgr.fallback.stages import clean
    from molgr.fallback.utils.resonance import build_resonance_state_key

    mol = _radical_hydrazone_path(size, variant)
    before = build_resonance_state_key(pybel.Molecule(mol))
    coordinates = [(a.GetX(), a.GetY(), a.GetZ()) for a in ob.OBMolAtomIter(mol)]
    if backend == "python":
        _, hit = getattr(clean, f"clean_1_{size}_radicals")(pybel.Molecule(mol), 0, 0)
    else:
        hit = getattr(_core.dev.stages.clean, f"clean_1_{size}_radicals_ptr")(int(mol.this), 0, 0)
    assert hit is allowed
    assert coordinates == [(a.GetX(), a.GetY(), a.GetZ()) for a in ob.OBMolAtomIter(mol)]
    if not allowed:
        # Rejection must not consume electrons, edit the source graph or its
        # aromatic flags, or prevent the normal charge-assignment search.
        assert build_resonance_state_key(pybel.Molecule(mol)) == before
        assert [get_unpaired_electron_count(mol.GetAtom(i)) for i in (1, size)] == [1, 1]
    else:
        assert [mol.GetBond(i, i + 1).GetBondOrder() for i in range(1, size)] == [
            2 if i % 2 else 1 for i in range(1, size)
        ]


def _rule_two_hydrazone(variant):
    mol = ob.OBMol()
    for z, p in [
        (8, (0, 0, 0)),
        (6, (1.22, 0, 0)),
        (6, (2.70, 0, 0)),
        (7, (4.02, 0, 0)),
        (7, (5.42, 0, 0)),
        (6, (1.22, 1.4, 0)),
        (6, (6.74, 0, 0)),
        (6, (2.7, 1.4, 0)),
    ]:
        a = mol.NewAtom()
        a.SetAtomicNum(z)
        a.SetVector(*p)
        a.SetImplicitHCount(0)
    for i, j, order in [
        (1, 2, 2),
        (2, 3, 1),
        (3, 4, 2),
        (4, 5, 1),
        (2, 6, 1),
        (5, 7, 1),
        (3, 8, 1),
    ]:
        mol.AddBond(i, j, order)
    mol.GetAtom(5).SetFormalCharge(-1)
    if variant == "ring":
        mol.AddBond(7, 8, 1)
    for a in list(ob.OBMolAtomIter(mol)):
        if a.GetAtomicNum() != 6:
            continue
        for n in range(4 - a.GetExplicitValence()):
            h = mol.NewAtom()
            h.SetAtomicNum(1)
            h.SetImplicitHCount(0)
            h.SetVector(a.GetX(), a.GetY() + n + 0.8, 1)
            mol.AddBond(a.GetIdx(), h.GetIdx(), 1)
    if variant == "short_nn":
        mol.GetAtom(5).SetVector(5.29, 0, 0)
    elif variant == "long_nn":
        mol.GetAtom(5).SetVector(5.56, 0, 0)
    elif variant == "long_cn":
        mol.GetAtom(7).SetVector(6.9, 0, 0)
    elif variant == "implicit_h":
        mol.GetAtom(5).SetImplicitHCount(1)
    elif variant == "missing":
        for a in ob.OBMolAtomIter(mol):
            a.SetVector(0, 0, 0)
    elif variant == "rigid":
        for a in ob.OBMolAtomIter(mol):
            a.SetVector(-a.GetY() + 3, a.GetX() - 4, a.GetZ() + 5)
    elif variant == "endpoint_charge":
        # Only the fully proposed neutral N=N product triggers the existing
        # short C--N / long C=S branch; checking the old N- label misses it.
        hs = [a.GetIdx() for a in ob.OBAtomAtomIter(mol.GetAtom(7)) if a.GetAtomicNum() == 1]
        for idx in sorted(hs[:2], reverse=True):
            mol.DeleteAtom(mol.GetAtom(idx))
        for idx, x in [(4, 4.15), (5, 5.55), (7, 6.87)]:
            mol.GetAtom(idx).SetVector(x, 0, 0)
        s = mol.NewAtom()
        s.SetAtomicNum(16)
        s.SetImplicitHCount(0)
        s.SetVector(8.65, 0, 0)
        mol.AddBond(7, s.GetIdx(), 2)
    return mol


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize(
    "variant,allowed",
    [
        ("conflict", False),
        ("rigid", False),
        ("short_nn", True),
        ("long_nn", True),
        # Implicit H on the negative endpoint already fails the original valence
        # guard: it has no capacity for the extra bond-order unit.
        ("long_cn", True),
        ("implicit_h", False),
        ("ring", True),
        ("missing", True),
    ],
)
def test_rule_two_keeps_feasible_electronic_assignment(backend, variant, allowed):
    mol = _rule_two_hydrazone(variant)
    before = build_resonance_state_key(pybel.Molecule(mol))
    coords = [(a.GetX(), a.GetY(), a.GetZ()) for a in ob.OBMolAtomIter(mol)]
    if backend == "python":
        _, hit = clean.clean_resonances(pybel.Molecule(mol))
    else:
        hit = _core.dev.stages.clean.clean_resonances_ptr(int(mol.this))
    assert hit is allowed
    assert coords == [(a.GetX(), a.GetY(), a.GetZ()) for a in ob.OBMolAtomIter(mol)]
    assert sum(a.GetFormalCharge() for a in ob.OBMolAtomIter(mol)) == -1
    if not allowed:
        assert build_resonance_state_key(pybel.Molecule(mol)) == before
    else:
        assert [mol.GetBond(i, j).GetBondOrder() for i, j in [(1, 2), (2, 3), (3, 4), (4, 5)]] == [
            1,
            2,
            1,
            2,
        ]
        assert mol.GetAtom(1).GetFormalCharge() == -1 and mol.GetAtom(5).GetFormalCharge() == 0


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_normalization_checks_proposed_endpoint_charges(backend):
    mol = _rule_two_hydrazone("endpoint_charge")
    coords = [(a.GetX(), a.GetY(), a.GetZ()) for a in ob.OBMolAtomIter(mol)]
    if backend == "python":
        clean.clean_resonances(pybel.Molecule(mol))
    else:
        _core.dev.stages.clean.clean_resonances_ptr(int(mol.this))
    # Rule 2 must abstain. A later valid N--C=S -> N=C--S- normalization
    # remains available, so the complete rule family may still change the graph.
    assert mol.GetBond(4, 5).GetBondOrder() == 1
    assert mol.GetBond(1, 2).GetBondOrder() == 2
    assert sum(a.GetFormalCharge() for a in ob.OBMolAtomIter(mol)) == -1
    assert coords == [(a.GetX(), a.GetY(), a.GetZ()) for a in ob.OBMolAtomIter(mol)]


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize("size", [4, 6])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "variant,allowed",
    [
        ("oximate", False),
        ("short_no", True),
        ("long_cn", True),
        ("missing_coordinates", True),
        ("substituted_oxygen", True),
    ],
)
def test_radical_cleanup_preserves_oximate_path(backend, size, reverse, variant, allowed):
    """Do not consume oximate radicals into a product rejected by final geometry."""
    mol = ob.OBMol()
    numbers = [8, 7, 6, 6] if size == 4 else [8, 7, 6, 6, 7, 8]
    lengths = [1.33, 1.32, 1.46] if size == 4 else [1.33, 1.32, 1.46, 1.32, 1.29]
    if variant == "short_no":
        lengths[0] = 1.22
    elif variant == "long_cn":
        lengths[1] = 1.45
    position = 0.0
    positions = []
    for i in range(size):
        if i:
            position += lengths[i - 1]
        positions.append((position, 0, 0))
    order = list(range(size))
    if reverse:
        order.reverse()
    indices = {i: _atom(mol, numbers[i], positions[i]) for i in order}
    for i in range(size - 1):
        mol.AddBond(indices[i], indices[i + 1], 1 if i % 2 == 0 else 2)
    for i in (0, size - 1):
        set_unpaired_electron_count(mol.GetAtom(indices[i]), 1)
    for i, z in enumerate(numbers):
        if z != 6:
            continue
        atom = mol.GetAtom(indices[i])
        target = 3 if i == size - 1 else 4
        for j in range(target - atom.GetExplicitValence()):
            h = _atom(mol, 1, (atom.GetX(), j + 1, 0))
            mol.AddBond(indices[i], h, 1)
    if variant == "substituted_oxygen":
        h = _atom(mol, 1, (-1, 0, 0))
        mol.AddBond(indices[0], h, 1)
    elif variant == "missing_coordinates":
        for atom in ob.OBMolAtomIter(mol):
            atom.SetVector(0, 0, 0)
    before = build_resonance_state_key(pybel.Molecule(mol))
    coordinates = [(a.GetX(), a.GetY(), a.GetZ()) for a in ob.OBMolAtomIter(mol)]
    if backend == "python":
        _, hit = getattr(clean, f"clean_1_{size}_radicals")(pybel.Molecule(mol), 0, 0)
    else:
        hit = getattr(_core.dev.stages.clean, f"clean_1_{size}_radicals_ptr")(int(mol.this), 0, 0)
    assert hit is allowed
    assert coordinates == [(a.GetX(), a.GetY(), a.GetZ()) for a in ob.OBMolAtomIter(mol)]
    if not allowed:
        assert build_resonance_state_key(pybel.Molecule(mol)) == before
        assert [get_unpaired_electron_count(mol.GetAtom(indices[i])) for i in (0, size - 1)] == [
            1,
            1,
        ]
    else:
        assert [
            mol.GetBond(indices[i], indices[i + 1]).GetBondOrder() for i in range(size - 1)
        ] == [2 if i % 2 == 0 else 1 for i in range(size - 1)]
