"""Local pi-plane geometry must not depend on atom numbering or coordinate frame."""

from __future__ import annotations

import math
import random
from dataclasses import replace

import pytest
from rdkit import Chem

from molgr.utils.equivalence import evaluate_equivalence
from scripts.tmqmg_fixture_data import fixture, reference_graph


# Load RDKit before OpenBabel to avoid shared-library initialization conflicts.
# isort: split
from openbabel import openbabel as ob
from openbabel import pybel

from molgr import _core
from molgr.config import MolGRConfig
from molgr.fallback.utils.organic_topology import compute_organic_topology_metrics


def _diene(angle, *, scale=1.0, permutation=0, reverse_bonds=False):
    """Explicit-H butadiene with independently rotated terminal pi planes."""
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    points = [
        (-0.7, 1.2, 0),
        (0, 0, 0),
        (1.4, 0, 0),
        (2.1, 1.2 * c, 1.2 * s),
        (-1.7, 1.2, 0),
        (-0.4, 2.2, 0),
        (-0.6, -0.9, 0),
        (2, -0.9 * c, -0.9 * s),
        (3.1, 1.2 * c, 1.2 * s),
        (1.8, 2.1 * c, 2.1 * s),
    ]
    bonds = [
        (0, 1, 2),
        (1, 2, 1),
        (2, 3, 2),
        (0, 4, 1),
        (0, 5, 1),
        (1, 6, 1),
        (2, 7, 1),
        (3, 8, 1),
        (3, 9, 1),
    ]
    order = list(range(len(points)))
    if permutation:
        random.Random(permutation).shuffle(order)
    inverse = {old: new + 1 for new, old in enumerate(order)}
    mol = ob.OBMol()
    for old in order:
        a = mol.NewAtom()
        a.SetAtomicNum(6 if old < 4 else 1)
        a.SetImplicitHCount(0)
        x, y, z = points[old]
        # Rotate and translate as well as scaling; chemistry is unchanged.
        a.SetVector(scale * z + 7, scale * x - 3, scale * y + 2)
    for i, j, order in reversed(bonds) if reverse_bonds else bonds:
        mol.AddBond(inverse[i], inverse[j], order)
    return pybel.Molecule(mol)


def _counts(mol, config=None):
    resolved = config or MolGRConfig()
    python = compute_organic_topology_metrics(mol, config=resolved.organic_topology)
    cpp = _core.dev.utils.compute_organic_topology_metrics_ptr(int(mol.OBMol.this), config=resolved)
    values = python.conjugated_atom_count, python.conjugated_bond_count
    assert values == (cpp["conjugated_atom_count"], cpp["conjugated_bond_count"])
    return values


@pytest.mark.parametrize(
    "angle,expected", [(0, 4), (30, 4), (44.9, 4), (45.1, 0), (90, 0), (150, 4), (180, 4)]
)
@pytest.mark.parametrize("scale", [0.2, 1, 4])
def test_pi_overlap_tracks_local_planes_in_any_coordinate_frame(angle, expected, scale):
    for permutation in (0, 1, 2, 3):
        mol = _diene(
            angle, scale=scale, permutation=permutation, reverse_bonds=bool(permutation % 2)
        )
        before = [(a.OBAtom.GetAtomicNum(), a.coords) for a in mol.atoms]
        assert _counts(mol) == (expected, 3 if expected else 0)
        assert [(a.OBAtom.GetAtomicNum(), a.coords) for a in mol.atoms] == before
        assert all(a.OBAtom.GetImplicitHCount() == 0 for a in mol.atoms)


def test_pi_overlap_configuration_is_shared_by_backends():
    mol = _diene(60)
    assert _counts(mol) == (0, 0)
    config = MolGRConfig()
    config.organic_topology = replace(config.organic_topology, conjugation_min_plane_overlap=0.2)
    assert _counts(mol, config) == (4, 3)


def test_missing_geometry_does_not_invent_an_orthogonal_plane():
    mol = _diene(90)
    for a in mol.atoms:
        a.OBAtom.SetVector(0, 0, 0)
    assert _counts(mol) == (4, 3)


def _carbon_pi_center(angle, kind, *, permutation=0, triple=False):
    from molgr.fallback.utils.electrons import set_lone_pair_count, set_unpaired_electron_count

    cosine, sine = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    points = [(-1.4, 0, 0), (0, 0, 0), (1.2, 0, 0) if triple else (0.7, 1.2, 0)]
    bonds = [(0, 1, 1), (1, 2, 3 if triple else 2)]
    hydrogens = [(0, (-1.9, 0.866 * cosine, 0.866 * sine))]
    if kind != "carbene":
        hydrogens.append((0, (-1.9, -0.866 * cosine, -0.866 * sine)))
    hydrogens += (
        [(2, (2.2, 0, 0))]
        if triple
        else [(1, (-0.5, -0.866, 0)), (2, (1.7, 1.2, 0)), (2, (0.2, 2.066, 0))]
    )
    for parent, point in hydrogens:
        bonds.append((parent, len(points), 1))
        points.append(point)
    order = list(range(len(points)))
    if permutation:
        random.Random(permutation).shuffle(order)
    inverse = {old: new + 1 for new, old in enumerate(order)}
    mol = ob.OBMol()
    for old in order:
        atom = mol.NewAtom()
        atom.SetAtomicNum(6 if old < 3 else 1)
        atom.SetImplicitHCount(0)
        x, y, z = points[old]
        atom.SetVector(z + 5, x - 2, y + 3)
    for i, j, bond_order in reversed(bonds) if permutation else bonds:
        mol.AddBond(inverse[i], inverse[j], bond_order)
    center = mol.GetAtom(inverse[0])
    center.SetFormalCharge({"anion": -1, "cation": 1}.get(kind, 0))
    set_unpaired_electron_count(center, int(kind == "radical"))
    set_lone_pair_count(center, int(kind in ("anion", "carbene")))
    return pybel.Molecule(mol)


@pytest.mark.parametrize("kind", ["anion", "cation", "radical", "carbene"])
@pytest.mark.parametrize("angle,expected", [(0, 3), (44.9, 3), (45.1, 0), (90, 0), (180, 3)])
def test_carbon_pi_center_labels_do_not_bypass_local_plane_geometry(kind, angle, expected):
    for permutation in (0, 1, 2):
        mol = _carbon_pi_center(angle, kind, permutation=permutation)
        assert _counts(mol) == (expected, 2 if expected else 0)


def test_carbon_pi_center_uses_configured_overlap_and_abstains_without_geometry():
    mol = _carbon_pi_center(60, "anion")
    assert _counts(mol) == (0, 0)
    config = MolGRConfig()
    config.organic_topology = replace(config.organic_topology, conjugation_min_plane_overlap=0.2)
    assert _counts(mol, config) == (3, 2)
    for atom in mol.atoms:
        atom.OBAtom.SetVector(0, 0, 0)
    assert _counts(mol) == (3, 2)


@pytest.mark.parametrize("angle", [0, 45, 90, 135])
def test_carbon_pi_center_can_use_either_triple_bond_channel(angle):
    assert _counts(_carbon_pi_center(angle, "anion", triple=True)) == (3, 2)


def _nitrogen_pi_donor(angle, *, charge_separated=False, triple=False):
    from molgr.fallback.utils.electrons import set_lone_pair_count

    mol = _carbon_pi_center(angle, "anion", triple=triple)
    nitrogen = mol.OBMol.GetAtom(1)
    nitrogen.SetAtomicNum(7)
    nitrogen.SetFormalCharge(0)
    if charge_separated:
        assert not triple
        nitrogen.SetFormalCharge(1)
        set_lone_pair_count(nitrogen, 0)
        mol.OBMol.GetBond(1, 2).SetBondOrder(2)
        mol.OBMol.GetBond(2, 3).SetBondOrder(1)
        carbon = mol.OBMol.GetAtom(3)
        carbon.SetFormalCharge(-1)
        set_lone_pair_count(carbon, 1)
    return mol


@pytest.mark.parametrize("angle,expected", [(0, 3), (30, 3), (44.9, 3), (45.1, 0), (90, 0)])
def test_enamine_and_iminium_contributors_share_pi_geometry(angle, expected):
    """Moving the N pair into a bond cannot manufacture a conjugation reward."""
    for charge_separated in (False, True):
        mol = _nitrogen_pi_donor(angle, charge_separated=charge_separated)
        for seed in range(3):
            copy = ob.OBMol(mol.OBMol)
            order = list(range(1, copy.NumAtoms() + 1))
            random.Random(seed).shuffle(order)
            copy.RenumberAtoms(ob.vectorInt(order))
            assert _counts(pybel.Molecule(copy)) == (expected, 2 if expected else 0)
            assert all(a.GetImplicitHCount() == 0 for a in ob.OBMolAtomIter(copy))


def test_neutral_nitrogen_respects_overlap_configuration_and_missing_geometry():
    mol = _nitrogen_pi_donor(60)
    assert _counts(mol) == (0, 0)
    config = MolGRConfig()
    config.organic_topology = replace(config.organic_topology, conjugation_min_plane_overlap=0.2)
    assert _counts(mol, config) == (3, 2)
    for atom in mol.atoms:
        atom.OBAtom.SetVector(0, 0, 0)
    assert _counts(mol) == (3, 2)


def test_strongly_pyramidal_nitrogen_does_not_receive_pi_overlap_reward():
    mol = _nitrogen_pi_donor(0)
    nitrogen = mol.OBMol.GetAtom(1)
    nitrogen.SetVector(nitrogen.GetX() + 0.6, nitrogen.GetY(), nitrogen.GetZ())
    from molgr.fallback.utils.organic_topology import _strongly_pyramidal_pi_endpoint

    assert _strongly_pyramidal_pi_endpoint(nitrogen)
    assert _counts(mol) == (0, 0)


@pytest.mark.parametrize("angle", [0, 45, 90, 135])
def test_neutral_nitrogen_can_use_either_triple_bond_channel(angle):
    assert _counts(_nitrogen_pi_donor(angle, triple=True)) == (3, 2)


@pytest.mark.parametrize(
    "element,angle_sum,expected",
    [
        (6, 321.9, True),
        (6, 329.9, True),
        (6, 330.1, False),
        (6, 339.3, False),
        (7, 321.9, True),
        (7, 359.5, False),
        (15, 321.9, False),
    ],
)
def test_strong_pyramid_guard_preserves_moderately_bent_pi_endpoints(element, angle_sum, expected):
    from molgr.fallback.utils.organic_topology import _strongly_pyramidal_pi_endpoint

    cosine = math.cos(math.radians(angle_sum / 3))
    height = math.sqrt((cosine + 0.5) / (1 - cosine))
    mol = ob.OBMol()
    center = mol.NewAtom()
    center.SetAtomicNum(element)
    center.SetImplicitHCount(0)
    center.SetVector(0, 0, height)
    for angle in (0, 120, 240):
        atom = mol.NewAtom()
        atom.SetAtomicNum(6)
        atom.SetImplicitHCount(0)
        atom.SetVector(math.cos(math.radians(angle)), math.sin(math.radians(angle)), 0)
        mol.AddBond(1, atom.GetIdx(), 1)
    assert _strongly_pyramidal_pi_endpoint(center) is expected
    for atom in ob.OBMolAtomIter(mol):
        atom.SetVector(0, 0, 0)
    assert not _strongly_pyramidal_pi_endpoint(center)


@pytest.mark.parametrize("angle", [0, 30, 90, 150])
def test_triple_bond_keeps_both_pi_channels(angle):
    """Nitrile-oxide conjugation cannot depend on one bent N--O direction."""
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    mol = ob.OBMol()
    for z, charge, position in [
        (8, -1, (-1.2, 0.05 * c, 0.05 * s)),
        (7, 1, (0, 0, 0)),
        (6, 0, (1.2, 0, 0)),
        (6, 0, (2.65, 0.1, 0)),
        (6, 0, (3.3, 1.25, 0)),
        (1, 0, (3.2, -0.8, 0)),
        (1, 0, (4.3, 1.2, 0)),
        (1, 0, (2.9, 2.2, 0)),
    ]:
        a = mol.NewAtom()
        a.SetAtomicNum(z)
        a.SetFormalCharge(charge)
        a.SetImplicitHCount(0)
        a.SetVector(*position)
    for i, j, order in [
        (1, 2, 1),
        (2, 3, 3),
        (3, 4, 1),
        (4, 5, 2),
        (4, 6, 1),
        (5, 7, 1),
        (5, 8, 1),
    ]:
        mol.AddBond(i, j, order)
    assert _counts(pybel.Molecule(mol)) == (5, 4)


def test_approved_nisnio_graph_conjugation_is_invariant_to_atom_order():

    graph = reference_graph("NISNIO")
    # Organic metrics operate on the detached ligands. Preserve every explicit H.
    graph = Chem.RWMol(graph)
    for a in reversed(list(graph.GetAtoms())):
        if a.GetAtomicNum() == 74:
            graph.RemoveAtom(a.GetIdx())
    Chem.Kekulize(graph, clearAromaticFlags=True)
    mol = ob.OBMol()
    for a in graph.GetAtoms():
        atom = mol.NewAtom()
        atom.SetAtomicNum(a.GetAtomicNum())
        atom.SetFormalCharge(a.GetFormalCharge())
        atom.SetImplicitHCount(0)
        atom.SetVector(*graph.GetConformer().GetAtomPosition(a.GetIdx()))
    for bond in graph.GetBonds():
        mol.AddBond(
            bond.GetBeginAtomIdx() + 1, bond.GetEndAtomIdx() + 1, int(bond.GetBondTypeAsDouble())
        )
    for seed in range(6):
        copy = ob.OBMol(mol)
        order = list(range(1, copy.NumAtoms() + 1))
        random.Random(seed).shuffle(order)
        copy.RenumberAtoms(ob.vectorInt(order))
        assert _counts(pybel.Molecule(copy)) == (24, 23)


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_nisnio_reconstruction_preserves_approved_graph_after_renumbering(backend):
    from molgr.interface import xyz_to_rdmol

    case = fixture("NISNIO")
    lines = case["xyz"].splitlines()
    count = int(lines[0])
    for seed in range(4):
        positions = lines[2 : count + 2].copy()
        random.Random(seed).shuffle(positions)
        xyz = "\n".join([lines[0], lines[1], *positions]) + "\n"
        mol = xyz_to_rdmol(xyz, case["charge"], case["multiplicity"], backend=backend)
        assert [a.GetFormalCharge() for a in mol.GetAtoms() if a.GetAtomicNum() == 74] == [0]
        assert mol.GetNumAtoms() == count
        assert all(a.GetNumImplicitHs() == a.GetNumExplicitHs() == 0 for a in mol.GetAtoms())
        comparison = evaluate_equivalence(
            reference_graph("NISNIO"),
            mol,
            use_chirality=False,
            ignore_metal_coordination_bonds=True,
            max_resonance=100000,
        )
        assert comparison.decision.value == "equivalent", comparison
