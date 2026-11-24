"""Metal coordinates break ties during ligand anion assignment."""

from unittest.mock import patch

import pytest
from openbabel import openbabel as ob
from openbabel import pybel

from molgr.config import MolGRConfig
from molgr.fallback.pipeline import reconstruct_without_metals
from molgr.fallback.stages.eliminate import eliminate_high_positive_charge_atoms
from molgr.fallback.utils.coordination_geometry import metal_contact_distances
from molgr.fallback.utils.dataclasses import MetalSite
from molgr.fallback.utils.electrons import set_unpaired_electron_count
from molgr.interface import xyz_to_rdmol


def formate(side=1, reverse=False, far_length=1.25):
    oxygens = [
        "O -0.625 1.082532 0",
        f"O {-0.5 * far_length} {-0.8660254 * far_length} 0",
    ]
    if reverse:
        oxygens.reverse()
    return "\n".join(["5", "", f"Na -1.68 {side * 2.92} 0", "C 0 0 0", *oxygens, "H 1.1 0 0", ""])


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize("side,reverse", [(1, False), (-1, False), (1, True), (-1, True)])
def test_equalized_formate_assigns_the_coordinating_oxygen(backend, side, reverse):
    mol = xyz_to_rdmol(formate(side, reverse), 0, 1, backend=backend, make_dative_bonds=False)
    assert mol.GetAtomWithIdx(0).GetFormalCharge() == 1
    for i in (2, 3):
        near = mol.GetConformer().GetAtomPosition(i).y * side > 0
        assert mol.GetAtomWithIdx(i).GetFormalCharge() == (-1 if near else 0)
        assert mol.GetBondBetweenAtoms(1, i).GetBondTypeAsDouble() == (1 if near else 2)


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_distinct_co_lengths_preserve_perceived_carboxyl_assignment(backend):
    mol = xyz_to_rdmol(formate(far_length=1.32), 0, 1, backend=backend, make_dative_bonds=False)
    assert mol.GetAtomWithIdx(3).GetFormalCharge() == -1
    assert mol.GetBondBetweenAtoms(1, 2).GetBondTypeAsDouble() == 2


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize(
    "first,second,charged",
    [
        ("F -2.26 0 0", "F 2.00 0 0", 2),  # two accessible sites, different distances
        ("F -2.00 0 0", "F 2.26 0 0", 1),
        ("F -2.00 0 0", "F 2.00 0 0", 1),  # equal distances retain atom order
        ("F -8.00 0 0", "F 7.00 0 0", 1),  # both outside the coordination sphere
        ("Cl -2.00 0 0", "F 6.00 0 0", 2),  # the F SMARTS tier outranks proximity
    ],
)
def test_negative_charge_budget_ties_use_visible_distance(backend, first, second, charged):
    xyz = f"3\n\nNa 0 0 0\n{first}\n{second}\n"
    mol = xyz_to_rdmol(xyz, 0, 2, backend=backend, make_dative_bonds=False)
    assert mol.GetAtomWithIdx(charged).GetFormalCharge() == -1
    assert mol.GetAtomWithIdx(3 - charged).GetNumRadicalElectrons() == 1


def test_equivalent_formate_choice_does_not_add_a_resonance_walk():
    walk = reconstruct_without_metals.walk_radical_resonances
    with patch.object(reconstruct_without_metals, "walk_radical_resonances", wraps=walk) as counter:
        xyz_to_rdmol(formate(), 0, 1, backend="python")
        actual = counter.call_count
        counter.reset_mock()
        with patch.object(reconstruct_without_metals, "metal_contact_distances", return_value=()):
            xyz_to_rdmol(formate(), 0, 1, backend="python")
        assert actual == counter.call_count > 0


def test_multiple_metals_use_the_nearest_unobstructed_contact():
    mol = ob.OBMol()
    for x in (0.0, 1.0):
        atom = mol.NewAtom()
        atom.SetAtomicNum(8)
        atom.SetVector(x, 0, 0)
    left = MetalSite(11, -1.5, 0, 0)
    right = MetalSite(11, 2.9, 0, 0)
    config = MolGRConfig().metal_scoring
    assert metal_contact_distances(mol, [left], config)[2] > 2500
    for sites in ([left, right], [right, left]):
        assert metal_contact_distances(mol, sites, config)[1:] == (1500, 1900)


def test_local_compensation_uses_distance_after_electronegativity():
    mol = ob.OBMol()
    center = mol.NewAtom()
    center.SetAtomicNum(16)
    center.SetFormalCharge(1)
    for element in (8, 16, 8):
        neighbor = mol.NewAtom()
        neighbor.SetAtomicNum(element)
        set_unpaired_electron_count(neighbor, 1)
        mol.AddBond(1, neighbor.GetIdx(), 1)
    _, remaining, hit = eliminate_high_positive_charge_atoms(
        pybel.Molecule(mol), -1, (0, 0, 2400, 1800, 2100)
    )
    assert hit and remaining == 0
    assert [a.GetFormalCharge() for a in ob.OBMolAtomIter(mol)] == [1, 0, 0, -1]
