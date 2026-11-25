"""Local C4 geometry and contributor selection in folded eight-carbon rings."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
from openbabel import openbabel as ob
from openbabel import pybel

from molgr.fallback.pipeline import reconstruct_with_metals
from molgr.fallback.utils.bond_geometry import folded_eight_carbon_segments
from molgr.fallback.utils.electrons import (
    set_unpaired_electron_count,
    set_unresolved_two_electron_center,
)
from molgr.fallback.utils.metals import scoring
from molgr.fallback.utils.no_metals.selection import _no_metal_candidate_selection_key
from scripts.tmqmg_fixture_data import fixture


KEY = "metal_folded_ring_geometry_conflict_count"


@pytest.fixture(scope="module")
def folded_candidates():
    case = fixture("BUSWOD")
    raw, retained = [], []
    original = scoring.select_ligand_alternatives

    def capture(pool, **kwargs):
        if not raw:
            raw.extend(pool)
        result = original(pool, **kwargs)
        retained[:] = result
        return result

    with patch.object(scoring, "select_ligand_alternatives", capture):
        result = reconstruct_with_metals.xyz2omol_state(
            case["xyz"], case["charge"], case["multiplicity"] - 1
        )
    assert result is not None
    return raw, {c.metal_states[0].valence: c for c in retained}


def _copy(candidate):
    clone = replace(
        candidate,
        metadata=dict(candidate.metadata),
        key_cache={},
        no_metal_state=replace(
            candidate.no_metal_state,
            metadata=dict(candidate.no_metal_state.metadata),
            omol=pybel.Molecule(ob.OBMol(candidate.no_metal_state.omol.OBMol)),
        ),
    )
    scoring._clear_folded_ring_geometry_conflicts([clone])
    return clone


def test_folded_ring_recognition_uses_local_planarity_and_is_rigid_motion_invariant(
    folded_candidates,
):
    _, retained = folded_candidates
    candidate = _copy(retained[0])
    mol = candidate.no_metal_state.omol.OBMol
    segments = list(folded_eight_carbon_segments(mol))
    assert len(segments) == 2
    assert all(len(ids) == 8 for ids in segments)
    reduced, neutral = scoring._folded_ring_electronic_sites(candidate)
    assert not reduced and len(neutral) == 2
    for atom in ob.OBMolAtomIter(mol):
        atom.SetVector(atom.GetY() + 100, atom.GetZ() - 50, atom.GetX() + 20)
    assert list(folded_eight_carbon_segments(mol)) == segments
    assert scoring._folded_ring_electronic_sites(candidate) == (reduced, neutral)
    assert len(scoring._folded_ring_electronic_sites(retained[4])[0]) == 2


@pytest.mark.parametrize(
    "defect",
    ["implicit_h", "element", "length", "flatten", "missing", "nan", "radical", "deferred"],
)
def test_folded_ring_geometry_abstains_outside_bounded_motif(folded_candidates, defect):
    mol = _copy(folded_candidates[1][0]).no_metal_state.omol.OBMol
    segments = list(folded_eight_carbon_segments(mol))
    for ids in segments:
        atom = mol.GetAtom(ids[0])
        if defect == "implicit_h":
            atom.SetImplicitHCount(1)
        elif defect == "element":
            atom.SetAtomicNum(7)
        elif defect == "radical":
            set_unpaired_electron_count(atom, 1)
        elif defect == "deferred":
            set_unresolved_two_electron_center(atom, True)
        elif defect == "flatten":
            points = np.array(
                [(mol.GetAtom(i).GetX(), mol.GetAtom(i).GetY(), mol.GetAtom(i).GetZ()) for i in ids]
            )
            center = points.mean(axis=0)
            _, _, axes = np.linalg.svd(points - center)
            flattened = points - np.outer((points - center) @ axes[-1], axes[-1])
            for index, point in zip(ids, flattened):
                mol.GetAtom(index).SetVector(*map(float, point))
    if defect in ("length", "missing", "nan"):
        for atom in ob.OBMolAtomIter(mol):
            if defect == "length":
                atom.SetVector(atom.GetX() * 1.2, atom.GetY() * 1.2, atom.GetZ() * 1.2)
            else:
                atom.SetVector(0, 0, 0 if defect == "missing" else float("nan"))
    assert list(folded_eight_carbon_segments(mol)) == []


@pytest.mark.parametrize("defect", ["visibility", "charge", "order"])
def test_folded_ring_electronic_features_require_complete_supported_assignment(
    folded_candidates, defect
):
    candidate = _copy(folded_candidates[1][0])
    mol = candidate.no_metal_state.omol.OBMol
    segments = list(folded_eight_carbon_segments(mol))
    if defect == "visibility":
        candidate.metadata["metal_visible_atom_groups"] = ""
    else:
        for ids in segments:
            if defect == "charge":
                mol.GetAtom(ids[0]).SetFormalCharge(1)
            else:
                mol.GetBond(ids[0], ids[1]).SetBondOrder(3)
    assert len(list(folded_eight_carbon_segments(mol))) == 2
    assert scoring._folded_ring_electronic_sites(candidate) == (set(), set())


@pytest.mark.parametrize(
    "defect",
    ["solo", "metal", "delta", "metal_spin", "ligand_spin", "nitrosyl", "independent_conflict"],
)
def test_folded_ring_partner_guards_do_not_force_charge_transfer(folded_candidates, defect):
    neutral, reduced = [_copy(folded_candidates[1][v]) for v in (0, 4)]
    candidates = [neutral, reduced]
    if defect == "solo":
        candidates = [neutral]
    elif defect in ("metal", "delta", "metal_spin"):
        attr, value = {
            "metal": ("idx", 1000),
            "delta": ("valence", 2),
            "metal_spin": ("radical_num", 2),
        }[defect]
        reduced.metal_states = (replace(reduced.metal_states[0], **{attr: value}),)
    elif defect == "ligand_spin":
        reduced.no_metal_radical_target += 2
    elif defect == "nitrosyl":
        reduced.metadata["nitrosyl_geometry_penalty"] += 1
    else:
        reduced.metadata["metal_discordance_structural_count"] += 1
    scoring._annotate_folded_ring_alternatives(candidates)
    assert neutral.metadata[KEY] == 0


def test_folded_ring_balances_two_local_reductions_and_repeats_without_accumulation(
    folded_candidates,
):
    neutral, reduced = [_copy(folded_candidates[1][v]) for v in (0, 4)]
    candidates = [neutral, reduced]
    for pool in (candidates, list(reversed(candidates)), candidates):
        scoring._annotate_folded_ring_alternatives(pool)
        assert neutral.metadata[KEY] == 2 and reduced.metadata[KEY] == 0
        assert neutral.metadata["metal_discordance_structural_count"] == 2
        assert reduced.metadata["metal_discordance_structural_count"] == 0


def _tied_contributors(folded_candidates):
    candidates = [_copy(c) for c in folded_candidates[0] if c.metal_states[0].valence == 4]
    keys = [_no_metal_candidate_selection_key(c.no_metal_state) for c in candidates]
    assert all(key == keys[0] for key in keys)
    return candidates


def test_exact_contributor_ties_prefer_fewer_metal_conflicts_before_atom_indices(folded_candidates):
    candidates = _tied_contributors(folded_candidates)
    counts = [c.metadata["metal_discordance_structural_count"] for c in candidates]
    assert min(counts) == 0 and max(counts) == 2
    [best] = scoring.select_ligand_alternatives(candidates)
    assert best.metadata["metal_discordance_structural_count"] == 0
    assert scoring.select_ligand_alternatives(list(reversed(candidates))) == [best]


@pytest.mark.parametrize("priority", ["organic_score", "nitrosyl", "cation"])
def test_contributor_tie_break_never_overrides_established_chemistry_priorities(
    folded_candidates, priority
):
    candidates = _tied_contributors(folded_candidates)
    good = min(candidates, key=lambda c: c.metadata["metal_discordance_structural_count"])
    bad = max(candidates, key=lambda c: c.metadata["metal_discordance_structural_count"])
    if priority == "organic_score":
        bad.no_metal_state.metadata["score"] -= 1
    elif priority == "nitrosyl":
        good.metadata["nitrosyl_geometry_penalty"] += 1
    else:
        good.metadata["metal_discordance_unsaturated_organic_cation_count"] += 1
    assert scoring.select_ligand_alternatives([good, bad]) == [bad]
