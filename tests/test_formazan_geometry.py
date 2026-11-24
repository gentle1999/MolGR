"""Bounded geometry-supported formazan alternatives and their search guards."""

from dataclasses import replace
from unittest.mock import patch

import pytest
from openbabel import openbabel as ob
from openbabel import pybel

from molgr.fallback.pipeline import reconstruct_with_metals
from molgr.fallback.utils.bond_geometry import planar_bis_phenoxy_formazan_sites
from molgr.fallback.utils.electrons import (
    set_unpaired_electron_count,
    set_unresolved_two_electron_center,
)
from molgr.fallback.utils.metals import scoring
from molgr.utils.converter import pybel_to_rdmol
from molgr.utils.equivalence import evaluate_equivalence
from scripts.tmqmg_fixture_data import fixture


KEY = "metal_formazan_geometry_conflict_count"


@pytest.fixture(scope="module")
def formazan_candidates():
    case = fixture("BIXBUG")
    captured = []
    original = scoring.select_best_candidate

    def capture(pool, **kwargs):
        result = original(pool, **kwargs)
        captured[:] = pool
        return result

    with patch.object(scoring, "select_best_candidate", capture):
        result = reconstruct_with_metals.xyz2omol_state(
            case["xyz"], case["charge"], case["multiplicity"] - 1
        )
    assert result is not None
    return {c.metal_states[0].valence: c for c in captured}


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
    scoring._clear_formazan_geometry_conflicts([clone])
    return clone


def test_geometry_recognition_is_independent_of_candidate_charge_and_rigid_motion(
    formazan_candidates,
):
    oxidized, reduced = [_copy(formazan_candidates[v]) for v in (1, 3)]
    mol = oxidized.no_metal_state.omol.OBMol
    sites = list(planar_bis_phenoxy_formazan_sites(mol))
    assert len(sites) == 1
    assert sites == list(planar_bis_phenoxy_formazan_sites(reduced.no_metal_state.omol.OBMol))
    site = sites[0][2]
    assert scoring._formazan_electronic_sites(oxidized) == {site: "oxidized"}
    assert scoring._formazan_electronic_sites(reduced) == {site: "reduced"}
    for atom in ob.OBMolAtomIter(mol):
        atom.SetVector(atom.GetY() + 100, atom.GetZ() - 50, atom.GetX() + 20)
    assert sites == list(planar_bis_phenoxy_formazan_sites(mol))


@pytest.mark.parametrize(
    "defect",
    ["implicit_h", "radical", "deferred", "element", "long_co", "twisted", "missing", "nan"],
)
def test_incomplete_or_conflicting_geometry_does_not_enable_preference(formazan_candidates, defect):
    candidate = _copy(formazan_candidates[1])
    mol = candidate.no_metal_state.omol.OBMol
    center, arms, _ = next(planar_bis_phenoxy_formazan_sites(mol))
    atom = mol.GetAtom(center)
    if defect == "implicit_h":
        atom.SetImplicitHCount(1)
    elif defect == "radical":
        set_unpaired_electron_count(atom, 1)
    elif defect == "deferred":
        set_unresolved_two_electron_center(atom, True)
    elif defect == "element":
        mol.GetAtom(arms[0][4]).SetAtomicNum(16)
    elif defect == "long_co":
        oxygen, carbon = mol.GetAtom(arms[0][4]), mol.GetAtom(arms[0][3])
        scale = 1.6 / oxygen.GetDistance(carbon)
        oxygen.SetVector(
            *(
                c + scale * (o - c)
                for o, c in zip(
                    (oxygen.GetX(), oxygen.GetY(), oxygen.GetZ()),
                    (carbon.GetX(), carbon.GetY(), carbon.GetZ()),
                )
            )
        )
    elif defect == "twisted":
        atom.SetVector(atom.GetX(), atom.GetY(), atom.GetZ() + 1)
    else:
        for atom in ob.OBMolAtomIter(mol):
            atom.SetVector(0, 0, 0 if defect == "missing" else float("nan"))
    assert list(planar_bis_phenoxy_formazan_sites(mol)) == []
    assert scoring._formazan_electronic_sites(candidate) == {}


@pytest.mark.parametrize("defect", ["visibility", "charge", "order"])
def test_geometry_does_not_accept_unsupported_electronic_labels(formazan_candidates, defect):
    candidate = _copy(formazan_candidates[1])
    mol = candidate.no_metal_state.omol.OBMol
    center, arms, _ = next(planar_bis_phenoxy_formazan_sites(mol))
    if defect == "visibility":
        candidate.metadata["metal_visible_atom_groups"] = ""
    elif defect == "charge":
        mol.GetAtom(center).SetFormalCharge(1)
    else:
        mol.GetBond(arms[0][0], arms[0][1]).SetBondOrder(3)
    assert len(list(planar_bis_phenoxy_formazan_sites(mol))) == 1
    assert scoring._formazan_electronic_sites(candidate) == {}


@pytest.mark.parametrize(
    "defect",
    [
        "solo",
        "metal",
        "delta",
        "metal_spin",
        "ligand_spin",
        "nitrosyl",
        "independent_conflict",
        "outside_charge",
        "outside_bond",
    ],
)
def test_partner_guards_preserve_other_constraints(formazan_candidates, defect):
    oxidized, reduced = [_copy(formazan_candidates[v]) for v in (1, 3)]
    candidates = [oxidized, reduced]
    if defect == "solo":
        candidates = [oxidized]
    elif defect in ("metal", "delta", "metal_spin"):
        attr, value = {
            "metal": ("idx", 1000),
            "delta": ("valence", 5),
            "metal_spin": ("radical_num", 2),
        }[defect]
        reduced.metal_states = (replace(reduced.metal_states[0], **{attr: value}),)
    elif defect == "ligand_spin":
        reduced.no_metal_radical_target += 2
    elif defect == "nitrosyl":
        reduced.metadata["nitrosyl_geometry_penalty"] += 1
    elif defect == "independent_conflict":
        reduced.metadata["metal_discordance_structural_count"] += 1
    else:
        mol = reduced.no_metal_state.omol.OBMol
        center, _, site = next(planar_bis_phenoxy_formazan_sites(mol))
        outside = next(a for a in ob.OBAtomAtomIter(mol.GetAtom(center)) if a.GetIdx() not in site)
        if defect == "outside_charge":
            outside.SetFormalCharge(1)
        else:
            mol.GetBond(center, outside.GetIdx()).SetBondOrder(2)
    scoring._annotate_formazan_alternatives(candidates)
    assert oxidized.metadata[KEY] == 0


def test_comparison_is_repeatable_and_reversible_when_partner_disappears(formazan_candidates):
    oxidized, reduced = [_copy(formazan_candidates[v]) for v in (1, 3)]
    for pool in ([oxidized, reduced], [reduced, oxidized], [oxidized, reduced]):
        scoring._annotate_formazan_alternatives(pool)
        assert oxidized.metadata[KEY] == 1
        assert reduced.metadata[KEY] == 0
        assert oxidized.metadata["metal_discordance_structural_count"] == 1
    scoring._annotate_formazan_alternatives([oxidized])
    assert oxidized.metadata[KEY] == 0
    assert oxidized.metadata["metal_discordance_structural_count"] == 0


@pytest.mark.parametrize("arm_index", [0, 1])
def test_reduced_site_accepts_negative_nitrogen_resonance_contributors(
    formazan_candidates, arm_index
):
    oxidized, reduced = [_copy(formazan_candidates[v]) for v in (1, 3)]
    mol = reduced.no_metal_state.omol.OBMol
    center, arms, site = next(planar_bis_phenoxy_formazan_sites(mol))
    inner, outer = arms[arm_index][:2]
    original = pybel_to_rdmol(reduced.no_metal_state.omol)
    # Move the central carbon lone pair through one N-N arm without changing
    # composition, connectivity, total charge or the metal assignment.
    mol.GetAtom(center).SetFormalCharge(0)
    mol.GetAtom(outer).SetFormalCharge(-1)
    mol.GetBond(center, inner).SetBondOrder(2)
    mol.GetBond(inner, outer).SetBondOrder(1)
    comparison = evaluate_equivalence(
        original,
        pybel_to_rdmol(reduced.no_metal_state.omol),
        use_chirality=False,
        max_resonance=100000,
    )
    assert comparison.decision.value == "equivalent"
    assert scoring._formazan_electronic_sites(reduced) == {site: "reduced"}
    scoring._annotate_formazan_alternatives([oxidized, reduced])
    assert oxidized.metadata[KEY] == 1
