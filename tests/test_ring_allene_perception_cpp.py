"""Stale aromatic flags must not suppress ring-cumulene geometry checks."""

import math

import pytest
from openbabel import openbabel as ob
from openbabel import pybel
from rdkit import Chem

from molgr import MolGRConfig, _core
from molgr.fallback.utils.electrons import (
    set_unpaired_electron_count,
    set_unresolved_two_electron_center,
)
from molgr.interface import xyz_to_rdmol
from molgr.utils.equivalence import evaluate_equivalence
from scripts.tmqmg_fixture_data import fixture, reference_graph


@pytest.mark.parametrize(
    "case_id,rigid_motion",
    [
        ("BIBLAB", False),
        ("BIBLAB", True),
        ("NICRUP", False),
        ("TAYFUY", False),
        ("TAYFUY", True),
        ("OFOBEQ", False),
    ],
)
def test_native_ring_allene_geometry_uses_current_bond_orders(case_id, rigid_motion):
    case = dict(fixture(case_id))
    xyz = Chem.MolFromXYZBlock(case["xyz"])
    text = case["xyz"]
    if rigid_motion:
        for i, pos in enumerate(xyz.GetConformer().GetPositions()):
            xyz.GetConformer().SetAtomPosition(i, pos[[1, 2, 0]] + (15, -20, 10))
        text = Chem.MolToXYZBlock(xyz)
    cfg = MolGRConfig()
    cfg.cpp_backend.max_threads = 1
    cfg.cpp_backend.target_bucket_parallel_max_threads = 1
    pool = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        text, case["charge"], case["multiplicity"] - 1, config=cfg
    )["candidates"]
    selected = next(c for c in pool if c["selected"])
    key = "metal_discordance_bent_cumulated_ring_allene_count"
    if case_id == "TAYFUY":
        # Two carbanion shifts recover an aromatic Rh(III) contributor.
        assert selected["metal_states"][0]["valence"] == 3
        assert selected[key] == 0
    elif case_id != "OFOBEQ":
        assert selected["metal_states"][0]["valence"] == 3
        rejected = next(c for c in pool if c["metal_states"][0]["valence"] == 1)
        assert rejected[key] == 1
        # The scorer inspects a fresh copy: the original electronic candidate
        # and its cached aromatic flags are preserved for search provenance.
        stale_allenes = []
        for i, atom in enumerate(rejected["no_metal_atom_signature"], 1):
            doubles = [b for b in rejected["no_metal_bond_signature"] if i in b[:2] and b[2] == 2]
            if atom[0] == 6 and len(doubles) == 2 and all(b[3] for b in doubles):
                stale_allenes.append(i)
        assert stale_allenes
    else:
        # A supported allenylidene outside a ring remains eligible.
        assert selected["metal_states"][0]["valence"] == 1
        assert selected[key] == 0
    mol = xyz_to_rdmol(text, case["charge"], case["multiplicity"], backend="cpp", config=cfg)
    assert [a.GetAtomicNum() for a in mol.GetAtoms()] == [a.GetAtomicNum() for a in xyz.GetAtoms()]
    assert mol.GetConformer().GetPositions() == pytest.approx(xyz.GetConformer().GetPositions())
    assert Chem.GetFormalCharge(mol) == case["charge"]
    assert sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()) == case["multiplicity"] - 1
    assert not any(a.GetNumImplicitHs() or a.GetNumExplicitHs() for a in mol.GetAtoms())
    assert (
        evaluate_equivalence(
            reference_graph(case_id),
            mol,
            use_chirality=False,
            ignore_metal_coordination_bonds=True,
            max_resonance=100000,
        ).decision.value
        == "equivalent"
    )


@pytest.mark.parametrize(
    "defect,expected",
    [
        ("none", True),
        ("stale_aromaticity", True),
        ("rigid_motion", True),
        ("radical", False),
        ("unresolved", False),
        ("implicit_h", False),
        ("disconnected_charge", False),
        ("linear", False),
        ("open_ring", False),
    ],
)
def test_ring_allene_resonance_requires_a_local_closed_shell_witness(defect, expected):
    mol = pybel.readstring("smi", "C1=C=CC=C[CH-]1")
    mol.addh()
    source = mol.OBMol
    for i in range(1, 7):
        angle = (i - 1) * math.pi / 3
        source.GetAtom(i).SetVector(1.4 * math.cos(angle), 1.4 * math.sin(angle), 0)
    if defect == "radical":
        set_unpaired_electron_count(source.GetAtom(6), 1)
    elif defect == "unresolved":
        set_unresolved_two_electron_center(source.GetAtom(6), True)
    elif defect == "implicit_h":
        source.GetAtom(6).SetImplicitHCount(1)
    elif defect == "disconnected_charge":
        source.GetAtom(6).SetFormalCharge(0)
        isolated = source.NewAtom()
        isolated.SetAtomicNum(6)
        isolated.SetFormalCharge(-1)
        isolated.SetImplicitHCount(0)
    elif defect == "linear":
        a, b = source.GetAtom(1), source.GetAtom(3)
        source.GetAtom(2).SetVector((a.GetX() + b.GetX()) / 2, (a.GetY() + b.GetY()) / 2, 0)
    elif defect == "open_ring":
        source.DeleteBond(source.GetBond(4, 5))
    elif defect == "rigid_motion":
        for atom in ob.OBMolAtomIter(source):
            atom.SetVector(atom.GetY() + 10, atom.GetZ() - 20, atom.GetX() + 30)
    source.FindRingAtomsAndBonds()
    if defect == "stale_aromaticity":
        for atom in ob.OBMolAtomIter(source):
            atom.SetAromatic(atom.GetAtomicNum() == 6)
        for bond in ob.OBMolBondIter(source):
            bond.SetAromatic(
                bond.GetBeginAtom().GetAtomicNum() == bond.GetEndAtom().GetAtomicNum() == 6
            )
        source.SetAromaticPerceived(True)

    def signature():
        return (
            [
                (
                    a.GetFormalCharge(),
                    a.GetImplicitHCount(),
                    a.IsAromatic(),
                    a.GetX(),
                    a.GetY(),
                    a.GetZ(),
                )
                for a in ob.OBMolAtomIter(source)
            ],
            [
                (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondOrder(), b.IsAromatic())
                for b in ob.OBMolBondIter(source)
            ],
        )

    before = signature()
    for _ in range(2):
        assert (
            _core.dev.pipeline.reconstruct_with_metals.debug_ring_allene_resonance_ptr(
                int(source.this), 2
            )
            is expected
        )
        assert signature() == before
