"""Compare validated C--C redox partners using scoped local geometry."""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import replace
from typing import Tuple, cast

from openbabel import openbabel as ob
from openbabel import pybel

from molgr.fallback.state import MetalCandidateState
from molgr.fallback.utils.electrons import get_lone_pair_count
from molgr.fallback.utils.metals.conjugated_geometry import (
    NormalizationBoundExceeded,
    OutsideSignature,
    charge_transfer_paths,
    equalized_aryl_rings,
    outside_signature,
    resolved,
)
from molgr.fallback.utils.ob_graph import (
    atom_bonds,
    atom_neighbors,
    indexed_bond_order,
    mol_atoms,
    mol_bonds,
    neighbor_bond_order,
    ring_paths,
)


LocalSite = Tuple[str, Tuple[int, ...]]


KEY = "metal_local_cc_redox_geometry_conflict_count"


def planar_ring(m: ob.OBMol, ids: Sequence[int]) -> bool:
    """Check five-membered pyrrole perimeter lengths and torsions for a conjugated ring."""
    ats = [cast(ob.OBAtom, m.GetAtom(i)) for i in ids]
    if any(not 1.30 < a.GetDistance(ats[(i + 1) % 5]) < 1.46 for i, a in enumerate(ats)):
        return False
    ts = [abs(m.GetTorsion(*(ids[(i + k) % 5] for k in range(4)))) for i in range(5)]
    return all(math.isfinite(t) and min(t, abs(180 - t)) < 15 for t in ts)


def pyrrole_link_sites(c: MetalCandidateState) -> dict[LocalSite, str]:
    """Compare a long inter-ring C=C with two linked pyrrolide anions carrying a single C–C bond."""
    if c.no_metal_state is None:
        return {}
    m = cast(ob.OBMol, c.no_metal_state.omol.OBMol)
    rings = []
    out = {}
    for perimeter in ring_paths(m):
        ids = tuple(perimeter)
        if len(ids) != 5:
            continue
        ats = [cast(ob.OBAtom, m.GetAtom(i)) for i in ids]
        if sorted(a.GetAtomicNum() for a in ats) != [6, 6, 6, 6, 7] or not planar_ring(m, ids):
            continue
        if any(
            not resolved(a) or a.GetExplicitDegree() != (2 if a.GetAtomicNum() == 7 else 3)
            for a in ats
        ):
            continue
        n = next(a for a in ats if a.GetAtomicNum() == 7)
        if any(
            a.GetFormalCharge() != 0 or a.GetTotalValence() != 4
            for a in ats
            if a.GetAtomicNum() == 6
        ):
            continue
        if (n.GetFormalCharge(), n.GetTotalValence()) not in [(0, 3), (-1, 2)]:
            continue
        rings.append((set(ids), n))
    for bond in mol_bonds(m):
        a, b = cast(ob.OBAtom, bond.GetBeginAtom()), cast(ob.OBAtom, bond.GetEndAtom())
        if a.GetAtomicNum() != 6 or b.GetAtomicNum() != 6 or not 1.47 < a.GetDistance(b) < 1.58:
            continue
        for left, n in rings:
            if a.GetIdx() not in left or b.GetIdx() in left or a.GetBond(n) is None:
                continue
            for right, o in rings:
                if b.GetIdx() not in right or left & right or b.GetBond(o) is None:
                    continue
                torsion = abs(m.GetTorsion(n.GetIdx(), a.GetIdx(), b.GetIdx(), o.GetIdx()))
                if not math.isfinite(torsion) or min(torsion, abs(180 - torsion)) >= 15:
                    continue
                key = ("pyrrole_link", tuple(sorted(left | right)))
                if bond.GetBondOrder() == 2 and n.GetFormalCharge() == o.GetFormalCharge() == 0:
                    out[key] = "inverted"
                if bond.GetBondOrder() == 1 and n.GetFormalCharge() == o.GetFormalCharge() == -1:
                    out[key] = "reduced"
    return out


def allyl_carbene_sites(c: MetalCandidateState) -> dict[LocalSite, str]:
    """Classify allylic assignments from bond-length asymmetry and endpoint charge and valence."""
    if c.no_metal_state is None:
        return {}
    m = cast(ob.OBMol, c.no_metal_state.omol.OBMol)
    out: dict[LocalSite, str] = {}
    for middle in mol_atoms(m):
        if (
            middle.GetAtomicNum() != 6
            or middle.GetExplicitDegree() != 3
            or middle.IsInRing()
            or not resolved(middle)
            or middle.GetFormalCharge() != 0
            or middle.GetTotalValence() != 4
        ):
            continue
        for first in atom_neighbors(middle):
            if (
                first.GetAtomicNum() != 6
                or first.GetExplicitDegree() != 3
                or first.IsInRing()
                or not resolved(first)
                or not 1.46 < first.GetDistance(middle) < 1.60
            ):
                continue
            for last in atom_neighbors(middle):
                if (
                    last.GetIdx() == first.GetIdx()
                    or last.GetAtomicNum() != 6
                    or last.GetExplicitDegree() != 2
                    or last.IsInRing()
                    or not resolved(last)
                    or not 1.28 < last.GetDistance(middle) < 1.40
                ):
                    continue
                if (
                    first.GetDistance(middle) - last.GetDistance(middle) <= 0.08
                    or not 95 < m.GetAngle(first, middle, last) < 145
                ):
                    continue
                if any(
                    n.GetAtomicNum() not in (1, 6)
                    for a in (first, middle, last)
                    for n in atom_neighbors(a)
                ):
                    continue
                orders = neighbor_bond_order(first, middle), neighbor_bond_order(middle, last)
                key = ("allyl_carbene", (first.GetIdx(), middle.GetIdx(), last.GetIdx()))
                if (
                    orders == (2, 1)
                    and first.GetFormalCharge() == last.GetFormalCharge() == 0
                    and (last.GetTotalValence() == 2 and get_lone_pair_count(last) == 1)
                ):
                    out[key] = "inverted"
                if (
                    orders == (1, 2)
                    and last.GetFormalCharge() == -1
                    and last.GetTotalValence() == 3
                    and (first.GetFormalCharge(), first.GetTotalValence()) in [(-1, 3), (0, 4)]
                ):
                    # A neutral first carbon requires an explicit alternating-
                    # path witness before it can support the local preference.
                    out[key] = "reduced"
    return out


def ortho_aryl_amido_carbanion_sites(candidate: MetalCandidateState) -> dict[LocalSite, str]:
    """Recognize the reduced aryl and the quinoid form of a visible N,C chelate."""
    if candidate.no_metal_state is None:
        return {}
    mol = cast(ob.OBMol, candidate.no_metal_state.omol.OBMol)
    visible = [
        {int(i) for i in group.split(",") if i}
        for group in str(candidate.metadata.get("metal_visible_atom_groups", "")).split(";")
    ]
    sites = {}
    rings = [list(path) for path in ring_paths(mol)]
    for ids, atoms, orders in equalized_aryl_rings(mol, rings):
        for i, left in enumerate(atoms):
            nitrogen = next((a for a in atom_neighbors(left) if a.GetIdx() not in ids), None)
            if (
                nitrogen is None
                or nitrogen.GetAtomicNum() != 7
                or nitrogen.GetExplicitDegree() != 2
                or not resolved(nitrogen)
            ):
                continue
            if not 1.38 < left.GetDistance(nitrogen) < 1.48:
                continue
            for right in (atoms[i - 1], atoms[(i + 1) % 6]):
                carbon = next((a for a in atom_neighbors(right) if a.GetIdx() not in ids), None)
                if (
                    carbon is None
                    or carbon.GetAtomicNum() != 6
                    or carbon.GetExplicitDegree() != 3
                    or not resolved(carbon)
                    or not 1.47 < right.GetDistance(carbon) < 1.60
                ):
                    continue
                donors = (nitrogen.GetIdx(), carbon.GetIdx())
                if not any(set(donors) <= group for group in visible):
                    continue
                if any(
                    n.GetAtomicNum() != 6 or neighbor_bond_order(a, n) != 1
                    for a in (nitrogen, carbon)
                    for n in atom_neighbors(a)
                    if n.GetIdx() not in (left.GetIdx(), right.GetIdx())
                ):
                    continue
                labels = [
                    (neighbor_bond_order(a, r), a.GetFormalCharge(), a.GetTotalValence())
                    for a, r in [(nitrogen, left), (carbon, right)]
                ]
                key = ("aryl_amido_carbanion", tuple(sorted(ids)) + donors)
                if labels == [(2, 0, 3), (2, 0, 4)] and sum(orders) == 8:
                    sites[key] = "inverted"
                if labels == [(1, -1, 2), (1, -1, 3)] and all(
                    orders[j] + orders[j - 1] == 3 for j in range(6)
                ):
                    sites[key] = "reduced"
    return sites


def normalized_outside(c: MetalCandidateState, site: Collection[int]) -> OutsideSignature:
    """Canonicalize isolated Cp(-) or neutral six/eight-cycle Kekule witnesses."""
    assert c.no_metal_state is not None
    source = cast(ob.OBMol, c.no_metal_state.omol.OBMol)
    m = ob.OBMol(source)
    core = set(site)
    rings = [list(path) for path in ring_paths(m)]
    membership = {idx: sum(idx in ring for ring in rings) for ring in rings for idx in ring}
    for ids in rings:
        if set(ids) & core or any(membership[idx] != 1 for idx in ids):
            continue
        if len(ids) in (6, 8):
            ats = [cast(ob.OBAtom, m.GetAtom(i)) for i in ids]
            size = len(ids)
            if any(
                a.GetAtomicNum() != 6
                or a.GetExplicitDegree() != 3
                or not resolved(a)
                or a.GetFormalCharge() != 0
                or a.GetTotalValence() != 4
                for a in ats
            ):
                continue
            if any(
                indexed_bond_order(m, ids[i], ids[(i + 1) % size]) not in (1, 2)
                or indexed_bond_order(m, ids[i], ids[(i + 1) % size])
                + indexed_bond_order(m, ids[i], ids[(i - 1) % size])
                != 3
                for i in range(size)
            ):
                continue
            if any(
                b.GetBondOrder() != 1
                for a in ats
                for b in atom_bonds(a)
                if b.GetNbrAtom(a).GetIdx() not in ids
            ):
                continue
            start = ids.index(min(ids))
            ids = ids[start:] + ids[:start]
            if ids[1] > ids[-1]:
                ids = [ids[0]] + list(reversed(ids[1:]))
            for i, idx in enumerate(ids):
                m.GetBond(idx, ids[(i + 1) % size]).SetBondOrder(2 if i % 2 == 0 else 1)
            continue
        if len(ids) != 5:
            continue
        ats = [cast(ob.OBAtom, m.GetAtom(i)) for i in ids]
        if any(
            a.GetAtomicNum() != 6
            or a.GetExplicitDegree() != 3
            or not resolved(a)
            or (a.GetFormalCharge(), a.GetTotalValence()) not in [(0, 4), (-1, 3)]
            for a in ats
        ):
            continue
        if sum(a.GetFormalCharge() for a in ats) != -1:
            continue
        if any(indexed_bond_order(m, ids[i], ids[(i + 1) % 5]) not in (1, 2) for i in range(5)):
            continue
        if any(
            indexed_bond_order(m, ids[i], ids[(i + 1) % 5])
            + indexed_bond_order(m, ids[i], ids[(i - 1) % 5])
            != (2 if ats[i].GetFormalCharge() == -1 else 3)
            for i in range(5)
        ):
            continue
        if any(
            b.GetBondOrder() != 1
            for a in ats
            for b in atom_bonds(a)
            if b.GetNbrAtom(a).GetIdx() not in ids
        ):
            continue
        # All contributors of this isolated five-cycle have the same outside labels.
        start = ids.index(min(ids))
        ids = ids[start:] + ids[:start]
        if ids[1] > ids[-1]:
            ids = [ids[0]] + list(reversed(ids[1:]))
        for i, idx in enumerate(ids):
            m.GetAtom(idx).SetFormalCharge(-1 if i == 0 else 0)
            m.GetBond(idx, ids[(i + 1) % 5]).SetBondOrder(2 if i in (1, 3) else 1)
    copy = replace(c, no_metal_state=replace(c.no_metal_state, omol=pybel.Molecule(m)))
    return outside_signature(copy, site)


def local_cc_outside_signatures(
    candidate: MetalCandidateState, key: LocalSite, state: str
) -> set[OutsideSignature]:
    """Compare full outside labels after bounded, copy-only resonance moves."""
    assert candidate.no_metal_state is not None
    kind, core = key
    if kind != "allyl_carbene":
        return {outside_signature(candidate, core)}
    source = cast(ob.OBMol, candidate.no_metal_state.omol.OBMol)
    first = cast(ob.OBAtom, source.GetAtom(core[0]))
    if state != "reduced" or first.GetFormalCharge() == -1:
        return {normalized_outside(candidate, core)}
    try:
        paths = charge_transfer_paths(source, core[0], set(core))
    except NormalizationBoundExceeded:
        return set()
    signatures = set()
    for path in paths:
        mol = ob.OBMol(source)
        mol.GetAtom(path[0]).SetFormalCharge(-1)
        mol.GetAtom(path[-1]).SetFormalCharge(0)
        for i, j in zip(path, path[1:]):
            bond = cast(ob.OBBond, mol.GetBond(i, j))
            bond.SetBondOrder(3 - bond.GetBondOrder())
        copy = replace(
            candidate,
            no_metal_state=replace(candidate.no_metal_state, omol=pybel.Molecule(mol)),
        )
        signatures.add(normalized_outside(copy, core))
    return signatures


def clear(candidates: Sequence[MetalCandidateState]) -> None:
    """Subtract earlier C–C geometry conflicts before comparing a different pool of metal states."""
    for c in candidates:
        old = c.metadata.get(KEY, 0)
        for key in ["metal_discordance_structural_count", "metal_discordance_count"]:
            c.metadata[key] = c.metadata.get(key, 0) - old
        c.metadata[KEY] = 0


def annotate(candidates: Sequence[MetalCandidateState]) -> None:
    """Penalize a localized C–C assignment only if its reduced partner preserves outside labels."""
    features = [
        {**pyrrole_link_sites(c), **allyl_carbene_sites(c), **ortho_aryl_amido_carbanion_sites(c)}
        for c in candidates
    ]
    counts = [c.metadata.get("metal_discordance_structural_count", 0) for c in candidates]
    cache = {}
    for i, c in enumerate(candidates):
        conflicts = set()
        for j, p in enumerate(candidates):
            if (
                len(c.metal_states) != 1
                or len(p.metal_states) != 1
                or counts[j] > counts[i]
                or c.no_metal_radical_target != p.no_metal_radical_target
            ):
                continue
            a, b = c.metal_states[0], p.metal_states[0]
            if (a.idx, a.radical_num, a.valence + 2) != (b.idx, b.radical_num, b.valence):
                continue
            if c.metadata.get("nitrosyl_geometry_penalty", 0) != p.metadata.get(
                "nitrosyl_geometry_penalty", 0
            ):
                continue
            for key, state in features[i].items():
                if state != "inverted" or features[j].get(key) != "reduced":
                    continue
                for index, candidate in [(i, c), (j, p)]:
                    if (index, key) not in cache:
                        cache[index, key] = local_cc_outside_signatures(
                            candidate, key, features[index][key]
                        )
                if cache[i, key] & cache[j, key]:
                    conflicts.add(key)
        c.metadata[KEY] = len(conflicts)
        for field in ("metal_discordance_structural_count", "metal_discordance_count"):
            c.metadata[field] += len(conflicts)


ARYL_CARBENOID_KEY = "metal_aryl_carbenoid_geometry_conflict_count"


def aryl_carbenoid_sites(c: MetalCandidateState) -> dict[tuple[int, ...], bool]:
    """Identify the two C–S bond-order assignments at a metal-visible aryl carbenoid."""
    if c.no_metal_state is None:
        return {}
    m = cast(ob.OBMol, c.no_metal_state.omol.OBMol)
    visible = {
        int(i) for i in str(c.metadata.get("metal_visible_atom_indices", "")).split(",") if i
    }
    out = {}
    for perimeter in ring_paths(m):
        ids = list(perimeter)
        if len(ids) != 6:
            continue
        ats = [cast(ob.OBAtom, m.GetAtom(i)) for i in ids]
        if any(
            a.GetAtomicNum() != 6
            or a.GetExplicitDegree() != 3
            or not resolved(a)
            or (a.GetFormalCharge(), a.GetTotalValence()) not in [(0, 4), (-1, 3)]
            for a in ats
        ):
            continue
        lengths = [ats[i].GetDistance(ats[(i + 1) % 6]) for i in range(6)]
        if not all(1.35 < x < 1.45 for x in lengths) or max(lengths) - min(lengths) > 0.06:
            continue
        torsions = [abs(m.GetTorsion(*(ids[(i + k) % 6] for k in range(4)))) for i in range(6)]
        if not all(math.isfinite(t) and min(t, abs(180 - t)) < 15 for t in torsions):
            continue
        orders = [indexed_bond_order(m, ids[i], ids[(i + 1) % 6]) for i in range(6)]
        if any(o not in (1, 2) for o in orders):
            continue
        for aryl in ats:
            for carbon in atom_neighbors(aryl):
                if (
                    carbon.GetIdx() in ids
                    or carbon.GetIdx() not in visible
                    or carbon.GetAtomicNum() != 6
                    or carbon.GetExplicitDegree() != 2
                    or carbon.GetFormalCharge() != -1
                    or carbon.GetTotalValence() != 3
                    or not resolved(carbon)
                    or not 1.47 < aryl.GetDistance(carbon) < 1.60
                ):
                    continue
                others = [n for n in atom_neighbors(carbon) if n.GetIdx() != aryl.GetIdx()]
                if len(others) != 1:
                    continue
                sulfur = others[0]
                if (
                    sulfur.GetAtomicNum() != 16
                    or sulfur.GetExplicitDegree() != 2
                    or not resolved(sulfur)
                    or not 1.60 < carbon.GetDistance(sulfur) < 1.85
                ):
                    continue
                rest = [n for n in atom_neighbors(sulfur) if n.GetIdx() != carbon.GetIdx()]
                if (
                    len(rest) != 1
                    or rest[0].GetAtomicNum() != 6
                    or neighbor_bond_order(sulfur, rest[0]) != 1
                ):
                    continue
                pair = (neighbor_bond_order(aryl, carbon), neighbor_bond_order(carbon, sulfur))
                key = tuple(sorted(ids)) + (carbon.GetIdx(), sulfur.GetIdx())
                charge = sum(a.GetFormalCharge() for a in ats)
                if (
                    pair == (2, 1)
                    and sulfur.GetFormalCharge() == 0
                    and sulfur.GetTotalValence() == 2
                    and charge == -1
                    and sum(orders) == 8
                ):
                    out[key] = False
                if (
                    pair == (1, 2)
                    and sulfur.GetFormalCharge() == 1
                    and sulfur.GetTotalValence() == 3
                    and charge == 0
                    and sum(orders) == 9
                    and all(orders[i] + orders[(i + 1) % 6] == 3 for i in range(6))
                ):
                    out[key] = True
    return out


def clear_aryl_carbenoid(candidates: Sequence[MetalCandidateState]) -> None:
    """Remove the aryl carbenoid contribution from both structural and total discordance."""
    for c in candidates:
        previous = c.metadata.get(ARYL_CARBENOID_KEY, 0)
        for key in ("metal_discordance_structural_count", "metal_discordance_count"):
            c.metadata[key] = c.metadata.get(key, 0) - previous
        c.metadata[ARYL_CARBENOID_KEY] = 0


def annotate_aryl_carbenoid(cs: Sequence[MetalCandidateState]) -> None:
    """Prefer a supported aryl carbene over a long exocyclic carbanion C=C.

    Require a complete alternative two-electron assignment and identical outside
    labels (allowing isolated-ring resonance). The donor cation keeps its normal
    penalty; only the incompatible long bond receives an additional conflict.
    """
    features = [aryl_carbenoid_sites(c) for c in cs]
    base = [
        c.metadata.get("metal_discordance_structural_count", 0)
        - c.metadata.get("metal_discordance_inner_visible_same_sign_charge_penalty", 0)
        for c in cs
    ]
    for i, c in enumerate(cs):
        found = set()
        for j, p in enumerate(cs):
            if (
                len(c.metal_states) != 1
                or len(p.metal_states) != 1
                or c.no_metal_radical_target != p.no_metal_radical_target
                or base[j] > base[i]
            ):
                continue
            a, b = c.metal_states[0], p.metal_states[0]
            if b.valence <= 0:
                continue
            if (a.idx, a.radical_num, a.valence - 2) != (
                b.idx,
                b.radical_num,
                b.valence,
            ) or c.metadata.get("nitrosyl_geometry_penalty", 0) != p.metadata.get(
                "nitrosyl_geometry_penalty", 0
            ):
                continue
            for core, supported in features[i].items():
                if (
                    not supported
                    and features[j].get(core) is True
                    and normalized_outside(c, core) == normalized_outside(p, core)
                ):
                    found.add(core)
        c.metadata[ARYL_CARBENOID_KEY] = len(found)
        for field in ("metal_discordance_structural_count", "metal_discordance_count"):
            c.metadata[field] += len(found)
