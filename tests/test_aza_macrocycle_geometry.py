"""Geometry and electron-count boundaries for the approved BEBQIK topology."""

from dataclasses import replace
from unittest.mock import patch

import pytest
from openbabel import openbabel as ob
from openbabel import pybel

from molgr.fallback.pipeline import reconstruct_with_metals
from molgr.fallback.utils.bond_geometry import planar_aza_macrocycle_circuits
from molgr.fallback.utils.electrons import (
    set_unpaired_electron_count,
    set_unresolved_two_electron_center,
)
from molgr.fallback.utils.metals import scoring
from scripts.tmqmg_fixture_data import fixture


@pytest.fixture(scope="module")
def aza_candidates():
    case = fixture("BEBQIK")
    captured = []
    original = scoring.select_best_candidate

    def capture(pool, **kwargs):
        captured[:] = pool
        return original(pool, **kwargs)

    with patch.object(scoring, "select_best_candidate", capture):
        result = reconstruct_with_metals.xyz2omol_state(
            case["xyz"], case["charge"], case["multiplicity"] - 1
        )
    assert result is not None
    return {c.metal_states[0].valence: c for c in captured if c.metal_states[0].valence in (0, 2)}


def _copy(candidate):
    return replace(
        candidate,
        metadata=dict(candidate.metadata),
        no_metal_state=replace(
            candidate.no_metal_state,
            omol=pybel.Molecule(ob.OBMol(candidate.no_metal_state.omol.OBMol)),
        ),
    )


def test_aza_circuit_counts_pyrrolic_lone_pair_and_only_four_donors(aza_candidates):
    for valence, expected in [(0, 16), (2, 18)]:
        candidate = _copy(aza_candidates[valence])
        mol = candidate.no_metal_state.omol.OBMol
        [(ids, donors)] = list(planar_aza_macrocycle_circuits(mol))
        assert len(ids) == 15 and len(donors) == 4
        assert sum(mol.GetAtom(i).GetAtomicNum() == 7 for i in ids) == 6
        candidate.metadata["metal_visible_atom_groups"] = ",".join(map(str, sorted(donors)))
        assert scoring._macrocycle_electron_sites(candidate) == {tuple(sorted(ids)): expected}
        # A proper rigid motion must preserve recognition and electron counting.
        for atom in ob.OBMolAtomIter(mol):
            atom.SetVector(atom.GetY() + 100, atom.GetZ() - 50, atom.GetX() + 20)
        assert scoring._macrocycle_electron_sites(candidate) == {tuple(sorted(ids)): expected}
        candidate.metadata["metal_visible_atom_groups"] = ",".join(map(str, sorted(donors)[1:]))
        assert scoring._macrocycle_electron_sites(candidate) == {}


@pytest.mark.parametrize(
    "defect", ["implicit_h", "element", "length", "nonplanar", "missing", "nan", "fused_ring"]
)
def test_aza_geometry_abstains_outside_recognized_topology(aza_candidates, defect):
    mol = _copy(aza_candidates[2]).no_metal_state.omol.OBMol
    [(ids, _)] = list(planar_aza_macrocycle_circuits(mol))
    atom = mol.GetAtom(ids[0])
    if defect == "implicit_h":
        atom.SetImplicitHCount(1)
    elif defect == "element":
        atom.SetAtomicNum(15)
    elif defect == "length":
        for atom in ob.OBMolAtomIter(mol):
            atom.SetVector(atom.GetX() * 1.2, atom.GetY() * 1.2, atom.GetZ() * 1.2)
    elif defect == "nonplanar":
        atom.SetVector(atom.GetX(), atom.GetY(), atom.GetZ() + 0.8)
    elif defect in ("missing", "nan"):
        for atom in ob.OBMolAtomIter(mol):
            atom.SetVector(0, 0, 0 if defect == "missing" else float("nan"))
    else:
        fused = next(
            r for r in mol.GetSSSR() if len(r._path) == 6 and len(set(r._path) & set(ids)) == 3
        )
        outer = next(i for i in fused._path if i not in ids)
        mol.GetAtom(outer).SetAtomicNum(8)
    assert list(planar_aza_macrocycle_circuits(mol)) == []


@pytest.mark.parametrize("defect", ["radical", "deferred", "charge", "valence"])
def test_aza_electron_count_abstains_on_unresolved_or_unsupported_labels(aza_candidates, defect):
    candidate = _copy(aza_candidates[2])
    mol = candidate.no_metal_state.omol.OBMol
    [(ids, donors)] = list(planar_aza_macrocycle_circuits(mol))
    atom = mol.GetAtom(min(donors))
    if defect == "radical":
        set_unpaired_electron_count(atom, 1)
    elif defect == "deferred":
        set_unresolved_two_electron_center(atom, True)
    elif defect == "charge":
        atom.SetFormalCharge(1)
    else:
        next(iter(ob.OBAtomBondIter(atom))).SetBondOrder(3)
    assert list(planar_aza_macrocycle_circuits(mol)) == [(ids, donors)]
    assert scoring._macrocycle_electron_sites(candidate) == {}


def test_aza_single_candidate_does_not_force_a_charge_transfer(aza_candidates):
    candidate = _copy(aza_candidates[0])
    scoring._annotate_macrocycle_alternatives([candidate])
    assert candidate.metadata["metal_macrocycle_electron_conflict_count"] == 0
    assert candidate.metal_states[0].valence == 0
