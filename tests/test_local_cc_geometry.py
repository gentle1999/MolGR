"""Boundary and nonmutation controls for paired local C-C geometry preferences."""

from dataclasses import replace
from unittest.mock import patch

import pytest
from openbabel import openbabel as ob
from openbabel import pybel

from molgr.config import MolGRConfig
from molgr.fallback.utils.electrons import (
    get_lone_pair_count,
    get_unpaired_electron_count,
    has_unresolved_two_electron_center,
    set_unpaired_electron_count,
    set_unresolved_two_electron_center,
)
from molgr.fallback.utils.metals import conjugated_geometry, scoring
from molgr.fallback.utils.metals import local_cc_geometry as g
from molgr.interface import xyz_to_rdmol
from scripts.tmqmg_fixture_data import fixture


@pytest.fixture(scope="module", params=["YAFSAC", "YERCUU", "KAYWOA"])
def pool(request):
    row = fixture(request.param)
    captured = []
    old = scoring.select_best_candidate

    def capture(cs, **kwargs):
        winner = old(cs, **kwargs)
        captured.extend(cs)
        return winner

    cfg = MolGRConfig()
    cfg.resonance.max_depth = 3
    cfg.resonance.limited_discrepancy_max_discrepancy = 2
    with patch.object(scoring, "select_best_candidate", capture):
        xyz_to_rdmol(
            row["xyz"],
            row["charge"],
            row["multiplicity"],
            backend="python",
            config=cfg,
        )
    return request.param, {c.metal_states[0].valence: c for c in captured}


def copy(c):
    out = replace(
        c,
        metadata=dict(c.metadata),
        key_cache={},
        no_metal_state=replace(
            c.no_metal_state,
            metadata=dict(c.no_metal_state.metadata),
            omol=pybel.Molecule(ob.OBMol(c.no_metal_state.omol.OBMol)),
        ),
    )
    g.clear([out])
    return out


@pytest.fixture
def pair(pool):
    return pool[0], copy(pool[1][0]), copy(pool[1][2])


def signature(c):
    m = c.no_metal_state.omol.OBMol
    return (
        [
            (
                a.GetIdx(),
                a.GetAtomicNum(),
                a.GetFormalCharge(),
                get_unpaired_electron_count(a),
                get_lone_pair_count(a),
                has_unresolved_two_electron_center(a),
                a.GetX(),
                a.GetY(),
                a.GetZ(),
            )
            for a in ob.OBMolAtomIter(m)
        ],
        [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondOrder()) for b in ob.OBMolBondIter(m)],
    )


def apply(cs):
    g.clear(cs)
    g.annotate(cs)


def core(c):
    sites = {**g.pyrrole_link_sites(c), **g.allyl_carbene_sites(c)}
    return next(iter(sites))[1]


def test_supported_pair_is_order_independent_idempotent_and_nonmutating(pair):
    _, c, p = pair
    before = [signature(x) for x in [c, p]]
    base = [x.metadata["metal_discordance_count"] for x in [c, p]]
    for cs in [[c, p], [c, p], [p, c]]:
        apply(cs)
        assert c.metadata[g.KEY] == 1 and p.metadata[g.KEY] == 0
        assert [signature(x) for x in [c, p]] == before
        assert [x.metadata["metal_discordance_count"] for x in [c, p]] == [base[0] + 1, base[1]]
    apply([c])
    assert c.metadata[g.KEY] == 0


@pytest.mark.parametrize("pool", ["YERCUU"], indirect=True)
@pytest.mark.parametrize(
    "defect", ["radical", "deferred", "implicit_h", "path_budget", "variant_budget"]
)
def test_delocalized_partner_requires_complete_closed_shell_witness(pair, monkeypatch, defect):
    _, c, p = pair
    apply([c, p])
    assert c.metadata[g.KEY] == 1
    site = core(c)
    mol = p.no_metal_state.omol.OBMol
    # This accepted contributor stores the first local carbanion on its arm.
    assert mol.GetAtom(site[0]).GetFormalCharge() == 0
    paths = conjugated_geometry.charge_transfer_paths(mol, site[0], set(site))
    assert paths
    if defect == "path_budget":
        monkeypatch.setattr(conjugated_geometry, "MAX_PATH_STATES", 0)
    elif defect == "variant_budget":
        monkeypatch.setattr(conjugated_geometry, "MAX_VARIANTS", 0)
    else:
        for endpoint in {path[-1] for path in paths}:
            atom = mol.GetAtom(endpoint)
            if defect == "radical":
                set_unpaired_electron_count(atom, 1)
            elif defect == "deferred":
                set_unresolved_two_electron_center(atom, True)
            else:
                atom.SetImplicitHCount(1)
    before = [signature(x) for x in [c, p]]
    apply([c, p])
    assert c.metadata[g.KEY] == p.metadata[g.KEY] == 0
    assert [signature(x) for x in [c, p]] == before


@pytest.mark.parametrize(
    "defect",
    [
        "metal",
        "delta",
        "metal_spin",
        "ligand_spin",
        "nitrosyl",
        "independent",
        "outside_atom",
        "outside_bond",
    ],
)
def test_incompatible_partner_never_supplies_preference(pair, defect):
    _, c, p = pair
    if defect in ["metal", "delta", "metal_spin"]:
        key, value = {
            "metal": ("idx", 9999),
            "delta": ("valence", 4),
            "metal_spin": ("radical_num", 2),
        }[defect]
        p.metal_states = (replace(p.metal_states[0], **{key: value}),)
    elif defect == "ligand_spin":
        p.no_metal_radical_target += 2
    elif defect == "nitrosyl":
        p.metadata["nitrosyl_geometry_penalty"] = c.metadata.get("nitrosyl_geometry_penalty", 0) + 1
    elif defect == "independent":
        p.metadata["metal_discordance_structural_count"] = (
            c.metadata["metal_discordance_structural_count"] + 1
        )
    else:
        m = p.no_metal_state.omol.OBMol
        outside = next(a for a in ob.OBMolAtomIter(m) if a.GetAtomicNum() == 1)
        if defect == "outside_atom":
            outside.SetFormalCharge(1)
        else:
            next(ob.OBAtomBondIter(outside)).SetBondOrder(2)
    apply([c, p])
    assert c.metadata[g.KEY] == p.metadata[g.KEY] == 0


@pytest.mark.parametrize(
    "defect",
    ["missing_geometry", "radical", "deferred", "implicit_h", "heteroatom", "outside_window"],
)
def test_incomplete_motif_abstains(pair, defect):
    name, c, p = pair
    indices = core(c)
    for candidate in [c, p]:
        m = candidate.no_metal_state.omol.OBMol
        a = m.GetAtom(indices[0])
        if defect == "missing_geometry":
            for atom in ob.OBMolAtomIter(m):
                atom.SetVector(0, 0, 0)
        elif defect == "radical":
            set_unpaired_electron_count(a, 1)
        elif defect == "deferred":
            set_unresolved_two_electron_center(a, True)
        elif defect == "implicit_h":
            a.SetImplicitHCount(1)
        elif defect == "heteroatom":
            a.SetAtomicNum(8)
        elif name == "YAFSAC":
            # A long ring edge no longer supplies the complete compact pyrrole motif.
            a.SetVector(a.GetX() + 5, a.GetY(), a.GetZ())
        else:
            middle = m.GetAtom(indices[1])
            scale = 1.35 / a.GetDistance(middle)
            a.SetVector(
                *(
                    v + scale * (u - v)
                    for u, v in zip(
                        (a.GetX(), a.GetY(), a.GetZ()),
                        (middle.GetX(), middle.GetY(), middle.GetZ()),
                    )
                )
            )
    apply([c, p])
    assert c.metadata[g.KEY] == p.metadata[g.KEY] == 0


def test_cp_normalization_requires_actual_closed_shell_cyclopentadienyl(pair):
    name, c, p = pair
    if name != "YERCUU":
        return
    site = core(c)
    # Isolate Cp normalization from the independently tested attached-arm
    # resonance move: the selected partner can delocalize this carbon's charge.
    source = p.no_metal_state.omol.OBMol
    if source.GetAtom(site[0]).GetFormalCharge() == 0:
        witnesses = []
        for path in conjugated_geometry.charge_transfer_paths(source, site[0], set(site)):
            witness = copy(p)
            mol = witness.no_metal_state.omol.OBMol
            mol.GetAtom(path[0]).SetFormalCharge(-1)
            mol.GetAtom(path[-1]).SetFormalCharge(0)
            for i, j in zip(path, path[1:]):
                bond = mol.GetBond(i, j)
                bond.SetBondOrder(3 - bond.GetBondOrder())
            if g.normalized_outside(c, site) == g.normalized_outside(witness, site):
                witnesses.append(witness)
        assert witnesses, "A complete resonance witness must precede Cp normalization"
        p = witnesses[0]
    # Exercise a different valid Cp contributor explicitly; candidate selection
    # is allowed to return the same contributor on both sides.
    m = p.no_metal_state.omol.OBMol
    ring = next(
        list(r._path)
        for r in m.GetSSSR()
        if len(r._path) == 5
        and not set(r._path) & set(site)
        and all(m.GetAtom(i).GetAtomicNum() == 6 for i in r._path)
        and sum(m.GetAtom(i).GetFormalCharge() for i in r._path) == -1
    )
    target = next(i for i in ring if c.no_metal_state.omol.OBMol.GetAtom(i).GetFormalCharge() == 0)
    offset = ring.index(target)
    ring = ring[offset:] + ring[:offset]
    for i, idx in enumerate(ring):
        m.GetAtom(idx).SetFormalCharge(-1 if i == 0 else 0)
        m.GetBond(idx, ring[(i + 1) % 5]).SetBondOrder(2 if i in (1, 3) else 1)
    assert g.outside_signature(c, site) != g.outside_signature(p, site)
    assert g.normalized_outside(c, site) == g.normalized_outside(p, site)
    m.GetAtom(target).SetFormalCharge(0)
    assert g.normalized_outside(c, site) != g.normalized_outside(p, site)


@pytest.mark.parametrize(
    "defect", ["none", "charge", "radical", "implicit_h", "bond_order", "heteroatom"]
)
def test_cot_kekule_normalization_is_bounded_and_nonmutating(pair, defect):
    name, c, p = pair
    if name != "KAYWOA":
        return
    site = core(c)
    before = [signature(x) for x in [c, p]]
    assert g.outside_signature(c, site) != g.outside_signature(p, site)
    assert g.normalized_outside(c, site) == g.normalized_outside(p, site)
    assert [signature(x) for x in [c, p]] == before
    m = p.no_metal_state.omol.OBMol
    ring = next(list(x._path) for x in m.GetSSSR() if len(x._path) == 8)
    a = m.GetAtom(ring[0])
    if defect == "charge":
        a.SetFormalCharge(1)
    elif defect == "radical":
        set_unpaired_electron_count(a, 1)
    elif defect == "implicit_h":
        a.SetImplicitHCount(1)
    elif defect == "bond_order":
        m.GetBond(ring[0], ring[1]).SetBondOrder(3)
    elif defect == "heteroatom":
        a.SetAtomicNum(7)
    if defect != "none":
        assert g.normalized_outside(c, site) != g.normalized_outside(p, site)
    apply([c, p])
    assert c.metadata[g.KEY] == (1 if defect == "none" else 0)


def test_candidate_without_organic_graph_keeps_existing_score(pair):
    _, c, p = pair
    c.no_metal_state = None
    p.no_metal_state = None
    before = [x.metadata["metal_discordance_count"] for x in [c, p]]
    apply([c, p])
    assert [x.metadata[g.KEY] for x in [c, p]] == [0, 0]
    assert [x.metadata["metal_discordance_count"] for x in [c, p]] == before


@pytest.fixture(scope="module")
def aryl_pool():
    row = fixture("AYEVEF")
    captured = []
    original = scoring.select_best_candidate

    def capture(cs, **kwargs):
        winner = original(cs, **kwargs)
        captured.extend(cs)
        return winner

    with patch.object(scoring, "select_best_candidate", capture):
        xyz_to_rdmol(row["xyz"], row["charge"], row["multiplicity"], backend="python")
    return {c.metal_states[0].valence: c for c in captured}


@pytest.fixture
def aryl_pair(aryl_pool):
    pair = [copy(aryl_pool[v]) for v in (6, 4)]
    g.clear_aryl_carbenoid(pair)
    return pair


def apply_aryl(pair):
    g.clear_aryl_carbenoid(pair)
    g.annotate_aryl_carbenoid(pair)


def test_aryl_carbene_preference_retains_donor_penalty_and_graphs(aryl_pair):
    old, supported = aryl_pair
    before = [signature(c) for c in aryl_pair]
    baseline = [c.metadata["metal_discordance_count"] for c in aryl_pair]
    for pair in [aryl_pair, aryl_pair, list(reversed(aryl_pair))]:
        apply_aryl(pair)
        assert old.metadata[g.ARYL_CARBENOID_KEY] == 1
        assert supported.metadata[g.ARYL_CARBENOID_KEY] == 0
        assert [signature(c) for c in aryl_pair] == before
        assert [c.metadata["metal_discordance_count"] for c in aryl_pair] == [
            baseline[0] + 1,
            baseline[1],
        ]
        assert supported.metadata["metal_discordance_inner_visible_same_sign_charge_penalty"] == 0.5
    apply_aryl([old])
    assert old.metadata[g.ARYL_CARBENOID_KEY] == 0


@pytest.mark.parametrize(
    "defect",
    [
        "short_cc",
        "long_cc",
        "long_cs",
        "nonplanar",
        "unresolved",
        "radical",
        "hidden",
        "outside",
        "independent",
        "metal_spin",
        "ligand_spin",
        "redox_delta",
        "nitrosyl",
    ],
)
def test_aryl_carbene_preference_requires_geometry_and_compatible_partner(aryl_pair, defect):
    old, partner = aryl_pair
    core = next(iter(g.aryl_carbenoid_sites(partner)))
    m = partner.no_metal_state.omol.OBMol
    carbon, sulfur = (m.GetAtom(i) for i in core[-2:])
    aryl = next(a for a in ob.OBAtomAtomIter(carbon) if a.GetAtomicNum() == 6)
    if defect in ("short_cc", "long_cc"):
        distance = 1.45 if defect == "short_cc" else 1.65
        factor = distance / carbon.GetDistance(aryl)
        carbon.SetVector(
            *(
                origin + factor * (value - origin)
                for value, origin in zip(
                    (carbon.GetX(), carbon.GetY(), carbon.GetZ()),
                    (aryl.GetX(), aryl.GetY(), aryl.GetZ()),
                )
            )
        )
    elif defect == "long_cs":
        factor = 1.95 / carbon.GetDistance(sulfur)
        sulfur.SetVector(
            *(
                origin + factor * (value - origin)
                for value, origin in zip(
                    (sulfur.GetX(), sulfur.GetY(), sulfur.GetZ()),
                    (carbon.GetX(), carbon.GetY(), carbon.GetZ()),
                )
            )
        )
    elif defect == "nonplanar":
        atom = m.GetAtom(core[0])
        atom.SetVector(atom.GetX(), atom.GetY(), atom.GetZ() + 1)
    elif defect == "unresolved":
        set_unresolved_two_electron_center(carbon, True)
    elif defect == "radical":
        set_unpaired_electron_count(sulfur, 2)
    elif defect == "hidden":
        partner.metadata["metal_visible_atom_indices"] = ""
    elif defect == "outside":
        next(a for a in ob.OBMolAtomIter(m) if a.GetAtomicNum() == 1).SetFormalCharge(1)
    elif defect == "independent":
        partner.metadata["metal_discordance_structural_count"] += 1
    elif defect == "metal_spin":
        partner.metal_states = (replace(partner.metal_states[0], radical_num=2),)
    elif defect == "ligand_spin":
        partner.no_metal_radical_target += 2
    elif defect == "redox_delta":
        partner.metal_states = (replace(partner.metal_states[0], valence=2),)
    elif defect == "nitrosyl":
        partner.metadata["nitrosyl_geometry_penalty"] = 1
    apply_aryl(aryl_pair)
    assert old.metadata[g.ARYL_CARBENOID_KEY] == partner.metadata[g.ARYL_CARBENOID_KEY] == 0
