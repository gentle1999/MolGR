"""Compare validated conjugated-ligand candidates using scoped XYZ evidence.

Geometry admits a preference only in the presence of a compatible redox partner.
Outside changes must be identical, allowing bounded attached-arm resonance for
quinodiimine sites. Search intermediates and the candidate molecular graphs are
never modified here. All hydrogen atoms are explicit.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import replace
from itertools import combinations
from typing import Optional, Tuple, cast

from openbabel import openbabel as ob
from openbabel import pybel

from molgr.fallback.state import MetalCandidateState
from molgr.fallback.utils.electrons import (
    get_unpaired_electron_count,
    has_unresolved_two_electron_center,
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


PyridylSite = Tuple[Tuple[int, ...], int, int]
OutsideSignature = Tuple[
    Tuple[Tuple[int, int, int, int, bool, int], ...], Tuple[Tuple[int, int, int], ...]
]


QUINODIIMINE_KEY = "metal_quinodiimine_geometry_conflict_count"
POLYENAMINE_KEY = "metal_polyenamine_geometry_conflict_count"
MAX_VARIANTS = 32
MAX_PATH_ATOMS = 32
MAX_PATH_STATES = 4096
PYRAMIDAL_PS_KEY = "metal_pyramidal_ps_geometry_conflict_count"
PYRAMIDAL_PN_KEY = "metal_pyramidal_pn_geometry_conflict_count"
ORTHO_DIOXYGEN_KEY = "metal_ortho_dioxygen_geometry_conflict_count"
ARYL_AMIDO_NHC_KEY = "metal_aryl_amido_nhc_geometry_conflict_count"
ARYL_THIOLATE_NHC_KEY = "metal_aryl_thiolate_nhc_geometry_conflict_count"
ARYL_BENZYL_NHC_KEY = "metal_aryl_benzyl_nhc_geometry_conflict_count"
ARYL_PHENOXIDE_NHC_KEY = "metal_aryl_phenoxide_nhc_geometry_conflict_count"
AMINO_C5_KEY = "metal_amino_c5_geometry_conflict_count"


def pyramidal_pn_sites(candidate: MetalCandidateState) -> dict[tuple[int, ...], bool]:
    """Identify long P+=N / P–N(-) partners at pyramidal cyclic RPN2 sites.

    Only fully resolved, metal-visible three-coordinate P centers qualify.
    Both N neighbors are cyclic; the changed N is bonded only to two P atoms.
    The carbon substituent must be saturated: coupled pi-active carbon redox
    sites require a broader model and remain outside this local preference.
    Geometry and a complete two-electron partner are needed together; ordinary
    planar phosphazenes and unrelated P/N charge separation remain unaffected.
    """
    if candidate.no_metal_state is None:
        return {}
    mol = cast(ob.OBMol, candidate.no_metal_state.omol.OBMol)
    visible = {
        int(i)
        for i in str(candidate.metadata.get("metal_visible_atom_indices", "")).split(",")
        if i
    }
    sites: dict[tuple[int, ...], bool] = {}
    for p in mol_atoms(mol):
        if (
            p.GetAtomicNum() != 15
            or p.GetExplicitDegree() != 3
            or p.GetIdx() not in visible
            or not p.IsInRing()
        ):
            continue
        neighbors = list(atom_neighbors(p))
        if (
            sorted(a.GetAtomicNum() for a in neighbors) != [6, 7, 7]
            or not all(resolved(a) for a in [p, *neighbors])
            or any(a.GetAtomicNum() == 7 and not a.IsInRing() for a in neighbors)
        ):
            continue
        carbon = next(a for a in neighbors if a.GetAtomicNum() == 6)
        if (
            carbon.GetFormalCharge() != 0
            or carbon.GetExplicitDegree() != 4
            or carbon.GetTotalValence() != 4
            or any(b.GetBondOrder() != 1 for b in atom_bonds(carbon))
        ):
            continue
        angles = [mol.GetAngle(a, p, b) for a, b in combinations(neighbors, 2)]
        if not (all(80 < angle < 130 for angle in angles) and 300 < sum(angles) < 330):
            continue
        for n in neighbors:
            if (
                n.GetAtomicNum() != 7
                or n.GetExplicitDegree() != 2
                or not 1.64 < p.GetDistance(n) < 1.85
            ):
                continue
            if any(
                a.GetAtomicNum() != 15
                or (a.GetIdx() != p.GetIdx() and neighbor_bond_order(n, a) != 1)
                for a in atom_neighbors(n)
            ):
                continue
            if any(neighbor_bond_order(p, a) != 1 for a in neighbors if a.GetIdx() != n.GetIdx()):
                continue
            labels = (
                p.GetFormalCharge(),
                p.GetTotalValence(),
                n.GetFormalCharge(),
                n.GetTotalValence(),
                neighbor_bond_order(p, n),
            )
            if labels == (1, 4, 0, 3, 2):
                sites[(p.GetIdx(), n.GetIdx())] = False
            elif labels == (0, 3, -1, 2, 1):
                sites[(p.GetIdx(), n.GetIdx())] = True
    return sites


def pyramidal_ps_sites(candidate: MetalCandidateState) -> dict[tuple[int, ...], bool]:
    """Recognize resolved, metal-visible R2P=S / R2P–S(-) alternatives.

    A strongly pyramidal three-coordinate P supports the lone-pair-bearing
    assignment only when a complete two-electron partner exists. This is not
    a general penalty on P/S cations or on all phosphorus multiple bonds.
    """
    sites: dict[tuple[int, ...], bool] = {}
    if candidate.no_metal_state is None:
        return sites
    mol = cast(ob.OBMol, candidate.no_metal_state.omol.OBMol)
    visible = {
        int(i)
        for i in str(candidate.metadata.get("metal_visible_atom_indices", "")).split(",")
        if i
    }
    for p in mol_atoms(mol):
        if p.GetAtomicNum() != 15 or p.GetExplicitDegree() != 3 or p.GetIdx() not in visible:
            continue
        neighbors = list(atom_neighbors(p))
        if not all(resolved(a) for a in [p, *neighbors]):
            continue
        angles = [mol.GetAngle(a, p, b) for a, b in combinations(neighbors, 2)]
        if not (all(80 < angle < 130 for angle in angles) and 300 < sum(angles) < 330):
            continue
        for sulfur in neighbors:
            if sulfur.GetAtomicNum() != 16 or sulfur.GetExplicitDegree() != 1:
                continue
            if any(
                n.GetAtomicNum() != 6 or neighbor_bond_order(p, n) != 1
                for n in neighbors
                if n.GetIdx() != sulfur.GetIdx()
            ):
                continue
            labels = (
                p.GetFormalCharge(),
                p.GetTotalValence(),
                sulfur.GetFormalCharge(),
                sulfur.GetTotalValence(),
                neighbor_bond_order(p, sulfur),
            )
            site = (p.GetIdx(), sulfur.GetIdx())
            if labels == (1, 4, 0, 2, 2):
                sites[site] = False
            elif labels == (0, 3, -1, 1, 1):
                sites[site] = True
    return sites


def clear_pyramidal_ps(candidates: Sequence[MetalCandidateState]) -> None:
    """Remove prior P–S conflicts before pairing pyramidal phosphorus assignments again."""
    _clear_local_redox_pair(candidates, PYRAMIDAL_PS_KEY)


def clear_pyramidal_pn(candidates: Sequence[MetalCandidateState]) -> None:
    """Remove prior cyclic P–N conflicts before pairing phosphorus assignments again."""
    _clear_local_redox_pair(candidates, PYRAMIDAL_PN_KEY)


def _clear_local_redox_pair(candidates: Sequence[MetalCandidateState], key: str) -> None:
    """Subtract the previous site penalty before comparing a new pool of redox partners."""
    for candidate in candidates:
        previous = candidate.metadata.get(key, 0)
        for field in ("metal_discordance_structural_count", "metal_discordance_count"):
            candidate.metadata[field] = candidate.metadata.get(field, 0) - previous
        candidate.metadata[key] = 0


def annotate_pyramidal_ps(candidates: Sequence[MetalCandidateState]) -> None:
    """Count P(+)=S assignments with a charge-balanced P–S(−) partner at a pyramidal donor."""
    _annotate_local_redox_pair(candidates, pyramidal_ps_sites, PYRAMIDAL_PS_KEY)


def annotate_pyramidal_pn(candidates: Sequence[MetalCandidateState]) -> None:
    """Count cyclic P(+)=N assignments with a charge-balanced P–N(−) partner at pyramidal P."""
    _annotate_local_redox_pair(candidates, pyramidal_pn_sites, PYRAMIDAL_PN_KEY)


def _annotate_local_redox_pair(
    candidates: Sequence[MetalCandidateState],
    site_getter: Callable[[MetalCandidateState], Mapping[tuple[int, ...], bool]],
    key: str,
) -> None:
    """Compare assignments at the same ligand site when a two-unit metal charge change balances reduction."""
    _clear_local_redox_pair(candidates, key)
    features = [site_getter(c) for c in candidates]
    counts = [c.metadata.get("metal_discordance_structural_count", 0) for c in candidates]
    for c, own, count in zip(candidates, features, counts):
        conflicts = set()
        for partner, other, other_count in zip(candidates, features, counts):
            if (
                len(c.metal_states) != 1
                or len(partner.metal_states) != 1
                or other_count > count
                or c.no_metal_radical_target != partner.no_metal_radical_target
                or c.metadata.get("nitrosyl_geometry_penalty", 0)
                != partner.metadata.get("nitrosyl_geometry_penalty", 0)
            ):
                continue
            a, b = c.metal_states[0], partner.metal_states[0]
            if (a.idx, a.radical_num, a.valence + 2) != (b.idx, b.radical_num, b.valence):
                continue
            for site, reduced in own.items():
                if (
                    not reduced
                    and other.get(site) is True
                    and outside_signature(c, site) == outside_signature(partner, site)
                ):
                    conflicts.add(site)
        c.metadata[key] = len(conflicts)
        for field in ("metal_discordance_structural_count", "metal_discordance_count"):
            c.metadata[field] += len(conflicts)


def outside_signature(c: MetalCandidateState, site: Iterable[int]) -> OutsideSignature:
    """Include bonds crossing the motif boundary; unrelated changes abstain."""
    assert c.no_metal_state is not None
    m = cast(ob.OBMol, c.no_metal_state.omol.OBMol)
    s = set(site)
    atoms = tuple(
        (
            a.GetIdx(),
            a.GetAtomicNum(),
            a.GetFormalCharge(),
            get_unpaired_electron_count(a),
            has_unresolved_two_electron_center(a),
            a.GetImplicitHCount(),
        )
        for a in mol_atoms(m)
        if a.GetIdx() not in s
    )
    bonds = tuple(
        sorted(
            (
                min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                b.GetBondOrder(),
            )
            for b in mol_bonds(m)
            if b.GetBeginAtomIdx() not in s or b.GetEndAtomIdx() not in s
        )
    )
    return (atoms, bonds)


def quinodiimine_sites(c: MetalCandidateState) -> tuple[set[tuple[int, ...]], set[tuple[int, ...]]]:
    """Separate reduced C–N and neutral C=N labels on adjacent donors of an equalized six-carbon ring."""
    red: set[tuple[int, ...]] = set()
    neutral: set[tuple[int, ...]] = set()
    if c.no_metal_state is None:
        return red, neutral
    m = cast(ob.OBMol, c.no_metal_state.omol.OBMol)
    groups = [
        {int(i) for i in g.split(",") if i}
        for g in str(c.metadata.get("metal_visible_atom_groups", "")).split(";")
    ]
    for perimeter in ring_paths(m):
        ids = list(perimeter)
        if len(ids) != 6:
            continue
        ats = [cast(ob.OBAtom, m.GetAtom(i)) for i in ids]
        if any(
            a.GetAtomicNum() != 6
            or a.GetExplicitDegree() != 3
            or a.GetImplicitHCount()
            or a.GetFormalCharge() != 0
            or a.GetTotalValence() != 4
            or get_unpaired_electron_count(a)
            or has_unresolved_two_electron_center(a)
            for a in ats
        ):
            continue
        ls = [ats[i].GetDistance(ats[(i + 1) % 6]) for i in range(6)]
        if not all(1.35 < x < 1.45 for x in ls) or max(ls) - min(ls) > 0.06:
            continue
        ts = [abs(m.GetTorsion(*(ids[(i + k) % 6] for k in range(4)))) for i in range(6)]
        if not all(min(t, abs(180 - t)) < 15 for t in ts):
            continue
        for i in range(6):
            cs = [ats[i], ats[(i + 1) % 6]]
            ns = []
            for carbon in cs:
                outside = [a for a in atom_neighbors(carbon) if a.GetIdx() not in ids]
                if len(outside) != 1:
                    break
                n = outside[0]
                if (
                    n.GetAtomicNum() != 7
                    or n.GetExplicitDegree() != 2
                    or n.GetImplicitHCount()
                    or get_unpaired_electron_count(n)
                    or has_unresolved_two_electron_center(n)
                    or not 1.36 < n.GetDistance(carbon) < 1.46
                ):
                    break
                ns.append(n)
            if len(ns) != 2 or not any({n.GetIdx() for n in ns} <= g for g in groups):
                continue
            site = tuple(sorted(ids)) + tuple(sorted(n.GetIdx() for n in ns))
            labels = [
                (n.GetFormalCharge(), n.GetTotalValence(), neighbor_bond_order(n, carbon))
                for n, carbon in zip(ns, cs)
            ]
            if all((q, v) in [(0, 3), (-1, 2)] and order == 1 for q, v, order in labels):
                red.add(site)
            if labels == [(0, 3, 2), (0, 3, 2)]:
                neutral.add(site)
    return red, neutral


def clear_quinodiimine(cs: Sequence[MetalCandidateState]) -> None:
    """Subtract the ortho-quinodiimine penalty before re-evaluating alternative ring charges."""
    for c in cs:
        old = c.metadata.get(QUINODIIMINE_KEY, 0)
        for k in ["metal_discordance_structural_count", "metal_discordance_count"]:
            c.metadata[k] = c.metadata.get(k, 0) - old
        c.metadata[QUINODIIMINE_KEY] = 0


class NormalizationBoundExceeded(Exception):
    """An alternating-path search exceeded its budget before enumerating all resonance witnesses."""


def resolved(a: ob.OBAtom) -> bool:
    """Require explicit hydrogens and exclude radicals and unresolved two-electron centers."""
    return not (
        a.GetImplicitHCount()
        or get_unpaired_electron_count(a)
        or has_unresolved_two_electron_center(a)
    )


def charge_transfer_paths(m: ob.OBMol, start: int, forbidden: set[int]) -> list[tuple[int, ...]]:
    """N=C-...-X(-) <-> N(-)-C=...=X using a closed alternating path."""
    queue: deque[tuple[int, ...]] = deque([(start,)])
    paths = []
    visited = 0
    while queue:
        visited += 1
        if visited > MAX_PATH_STATES:
            raise NormalizationBoundExceeded
        path = queue.popleft()
        a = cast(ob.OBAtom, m.GetAtom(path[-1]))
        order = 2 if len(path) % 2 else 1
        for b in atom_bonds(a):
            other = cast(ob.OBAtom, b.GetNbrAtom(a))
            idx = other.GetIdx()
            if idx in forbidden or idx in path or b.GetBondOrder() != order or not resolved(other):
                continue
            z = other.GetAtomicNum()
            q = other.GetFormalCharge()
            v = other.GetTotalValence()
            if order == 1 and q == -1 and v == {6: 3, 7: 2, 8: 1}.get(z):
                if len(path) >= MAX_PATH_ATOMS:
                    raise NormalizationBoundExceeded
                paths.append(path + (idx,))
                if len(paths) > MAX_VARIANTS:
                    raise NormalizationBoundExceeded
            elif q == 0 and v == {6: 4, 7: 3, 8: 2}.get(z):
                if len(path) >= MAX_PATH_ATOMS:
                    raise NormalizationBoundExceeded
                queue.append(path + (idx,))
    return paths


def _normalized_variants(
    c: MetalCandidateState, changed_sites: Collection[tuple[int, ...]]
) -> list[ob.OBMol]:
    """Move negative charge onto the two imine nitrogens along allowed paths on ligand copies."""
    assert c.no_metal_state is not None
    core = {i for site in changed_sites for i in site}
    nitrogens = sorted({i for site in changed_sites for i in site[-2:]})
    variants = [ob.OBMol(c.no_metal_state.omol.OBMol)]
    for idx in nitrogens:
        extended = []
        for m in variants:
            n = cast(ob.OBAtom, m.GetAtom(idx))
            if n.GetFormalCharge() == -1 and n.GetTotalValence() == 2:
                extended.append(m)
            elif n.GetFormalCharge() == 0 and n.GetTotalValence() == 3:
                for path in charge_transfer_paths(m, idx, core):
                    new = ob.OBMol(m)
                    new.GetAtom(idx).SetFormalCharge(-1)
                    new.GetAtom(path[-1]).SetFormalCharge(0)
                    for i, j in zip(path[:-1], path[1:]):
                        b = cast(ob.OBBond, new.GetBond(i, j))
                        b.SetBondOrder(3 - b.GetBondOrder())
                    extended.append(new)
            if len(extended) > MAX_VARIANTS:
                raise NormalizationBoundExceeded
        variants = extended
        if not variants:
            return []
    return variants


def normalized_variants(
    c: MetalCandidateState, changed_sites: Collection[tuple[int, ...]]
) -> list[ob.OBMol]:
    # Budget exhaustion must abstain, not depend on atom/neighbor traversal order.
    """Return all completed charge-shift variants, or none when the search reaches its bound."""
    try:
        return _normalized_variants(c, changed_sites)
    except NormalizationBoundExceeded:
        return []


def normalized_outside_signatures(
    c: MetalCandidateState, changed_sites: Collection[tuple[int, ...]]
) -> set[OutsideSignature]:
    """Compare the rest of the ligand after each allowed charge shift onto the diimine site."""
    assert c.no_metal_state is not None
    core = {i for site in changed_sites for i in site}
    variants = normalized_variants(c, changed_sites)
    signatures = set()
    for m in variants:
        clone = replace(c, no_metal_state=replace(c.no_metal_state, omol=pybel.Molecule(m)))
        signatures.add(outside_signature(clone, core))
    return signatures


def annotate_quinodiimine(cs: Sequence[MetalCandidateState]) -> None:
    """Flag a diimine assignment when the metal charge balances reduction at every changed site."""
    clear_quinodiimine(cs)
    fs = [quinodiimine_sites(c) for c in cs]
    counts = [c.metadata.get("metal_discordance_structural_count", 0) for c in cs]
    cache = {}
    for c, (red, neutral), count in zip(cs, fs, counts):
        conflicts = set()
        for p, (pr, pn), pc in zip(cs, fs, counts):
            if (
                len(c.metal_states) != 1
                or len(p.metal_states) != 1
                or c.no_metal_radical_target != p.no_metal_radical_target
                or pc > count
                or c.metadata.get("nitrosyl_geometry_penalty", 0)
                != p.metadata.get("nitrosyl_geometry_penalty", 0)
            ):
                continue
            a, b = c.metal_states[0], p.metal_states[0]
            changed = neutral & pr
            if not (
                a.idx == b.idx
                and a.radical_num == b.radical_num
                and changed
                and red <= pr
                and red | neutral == pr | pn
                and b.valence - a.valence == 2 * len(changed)
            ):
                continue
            core = {i for site in changed for i in site}
            key = (id(p), tuple(sorted(changed)))
            if key not in cache:
                cache[key] = normalized_outside_signatures(p, changed)
            if outside_signature(c, core) in cache[key]:
                conflicts.update(changed)
        c.metadata[QUINODIIMINE_KEY] = len(conflicts)
        for k in ("metal_discordance_structural_count", "metal_discordance_count"):
            c.metadata[k] += len(conflicts)


def polyenamine_sites(c: MetalCandidateState) -> dict[tuple[int, ...], str]:
    """Classify planar chains with a metal-facing CH₂ end as reduced polyenamine or iminium."""
    out: dict[tuple[int, ...], str] = {}
    if c.no_metal_state is None:
        return out
    m = cast(ob.OBMol, c.no_metal_state.omol.OBMol)
    groups = [
        {int(i) for i in g.split(",") if i}
        for g in str(c.metadata.get("metal_visible_atom_groups", "")).split(";")
    ]

    def carbon(a: ob.OBAtom) -> bool:
        """Limit the conjugated chain to resolved, three-coordinate carbons outside rings."""
        return (
            a.GetAtomicNum() == 6
            and a.GetExplicitDegree() == 3
            and not a.IsInRing()
            and resolved(a)
        )

    for n in mol_atoms(m):
        if n.GetAtomicNum() != 7 or n.GetExplicitDegree() != 3 or n.IsInRing() or not resolved(n):
            continue
        neighbors = list(atom_neighbors(n))
        if any(a.GetAtomicNum() not in (1, 6) for a in neighbors):
            continue
        if any(not 0.8 < n.GetDistance(a) < 1.6 for a in neighbors):
            continue
        angles = [m.GetAngle(a, n, b) for a, b in combinations(neighbors, 2)]
        if not (all(95 < a < 145 for a in angles) and sum(angles) > 355):
            continue
        for first in neighbors:
            if not carbon(first) or not 1.30 < n.GetDistance(first) < 1.43:
                continue
            path = [n, first]
            while len(path) < 10:
                options = [
                    a
                    for a in atom_neighbors(path[-1])
                    if a.GetIdx() != path[-2].GetIdx() and carbon(a)
                ]
                if len(options) != 1:
                    break
                nxt = options[0]
                if nxt.GetIdx() in {a.GetIdx() for a in path}:
                    break
                path.append(nxt)
            if len(path) not in (4, 6, 8, 10):
                continue
            # Complete chain: its metal-facing end is CH2, not a truncated branch.
            tail = path[-1]
            if sum(a.GetAtomicNum() == 1 for a in atom_neighbors(tail)) != 2:
                continue
            ids = tuple(a.GetIdx() for a in path)
            if not any(set(ids[-2:]) <= g and ids[0] not in g for g in groups):
                continue
            lengths = [a.GetDistance(b) for a, b in zip(path[1:-1], path[2:])]
            if not all(1.30 < d < 1.50 for d in lengths):
                continue
            if not all(lengths[i + 1] - lengths[i] > 0.02 for i in range(0, len(lengths), 2)):
                continue
            if not all(105 < m.GetAngle(*path[i : i + 3]) < 145 for i in range(len(path) - 2)):
                continue
            torsions = [abs(m.GetTorsion(*ids[i : i + 4])) for i in range(len(ids) - 3)]
            if not all(math.isfinite(t) and min(t, abs(180 - t)) < 15 for t in torsions):
                continue
            orders = tuple(neighbor_bond_order(a, b) for a, b in zip(path[:-1], path[1:]))
            labels = [(a.GetFormalCharge(), a.GetTotalValence()) for a in path]
            reduced = tuple(1 if i % 2 == 0 else 2 for i in range(len(orders)))
            inverted = tuple(3 - i for i in reduced)
            if (
                orders[0] == 1
                and all(order in (1, 2) for order in orders)
                and labels[0] == (0, 3)
                and labels[1:].count((-1, 3)) == 1
                and all(label in ((0, 4), (-1, 3)) for label in labels[1:])
            ):
                out[ids] = "reduced"
            if orders == inverted and labels == [(1, 4)] + [(0, 4)] * (len(path) - 1):
                out[ids] = "iminium"
    return out


def clear_polyenamine(cs: Sequence[MetalCandidateState]) -> None:
    """Subtract the iminium penalty before searching a new pool for reduced chain partners."""
    for c in cs:
        old = c.metadata.get(POLYENAMINE_KEY, 0)
        for k in ("metal_discordance_structural_count", "metal_discordance_count"):
            c.metadata[k] = c.metadata.get(k, 0) - old
        c.metadata[POLYENAMINE_KEY] = 0


def annotate_polyenamine(cs: Sequence[MetalCandidateState]) -> None:
    """Count iminium assignments whose reduced partner preserves the graph outside the chain."""
    clear_polyenamine(cs)
    fs = [polyenamine_sites(c) for c in cs]
    counts = [c.metadata.get("metal_discordance_structural_count", 0) for c in cs]
    for c, own, count in zip(cs, fs, counts):
        conflicts = set()
        for p, other, pcount in zip(cs, fs, counts):
            if (
                len(c.metal_states) != 1
                or len(p.metal_states) != 1
                or pcount > count
                or c.no_metal_radical_target != p.no_metal_radical_target
            ):
                continue
            a, b = c.metal_states[0], p.metal_states[0]
            if a.idx != b.idx or a.radical_num != b.radical_num or b.valence != a.valence + 2:
                continue
            if c.metadata.get("nitrosyl_geometry_penalty", 0) != p.metadata.get(
                "nitrosyl_geometry_penalty", 0
            ):
                continue
            for site, state in own.items():
                if (
                    state == "iminium"
                    and other.get(site) == "reduced"
                    and outside_signature(c, site) == outside_signature(p, site)
                ):
                    conflicts.add(site)
        c.metadata[POLYENAMINE_KEY] = len(conflicts)
        for k in ("metal_discordance_structural_count", "metal_discordance_count"):
            c.metadata[k] += len(conflicts)


PAIRED_DIIMINE_KEY = "metal_paired_diimine_geometry_conflict_count"


def paired_diimine_sites(candidate: MetalCandidateState) -> dict[tuple[int, ...], int]:
    """Recognize equalized cis-diimine pairs in a saturated-bridge C10N4 ring.

    This narrow TIM motif distinguishes aggregate reduction; it does not assign
    a metal valence or choose which unit carries a localized charge pair.
    The bond-length pattern is supported by the TIM redox series reported in
    Inorg. Chem. 2010, DOI: 10.1021/ic100586h. Other ring/geometry classes abstain.
    """
    result: dict[tuple[int, ...], int] = {}
    if candidate.no_metal_state is None:
        return result
    m = cast(ob.OBMol, candidate.no_metal_state.omol.OBMol)
    groups = [
        {int(i) for i in group.split(",") if i}
        for group in str(candidate.metadata.get("metal_visible_atom_groups", "")).split(";")
    ]
    for perimeter in ring_paths(m):
        ids = list(perimeter)
        if len(ids) != 14:
            continue
        atoms = [cast(ob.OBAtom, m.GetAtom(i)) for i in ids]
        ns = {a.GetIdx() for a in atoms if a.GetAtomicNum() == 7}
        if len(ns) != 4 or any(a.GetAtomicNum() not in (6, 7) or not resolved(a) for a in atoms):
            continue
        if not any(ns <= group for group in groups):
            continue
        units = []
        for i in range(14):
            path = [ids[(i + k) % 14] for k in range(4)]
            unit = [cast(ob.OBAtom, m.GetAtom(j)) for j in path]
            if [a.GetAtomicNum() for a in unit] == [7, 6, 6, 7]:
                units.append((path, unit))
        if len(units) != 2 or len({j for path, _ in units for j in path}) != 8:
            continue
        core = tuple(sorted(j for path, _ in units for j in path))
        # The other six ring carbons form saturated three-carbon bridges.
        if any(
            a.GetAtomicNum() != 6
            or a.GetExplicitDegree() != 4
            or a.GetFormalCharge() != 0
            or a.GetTotalValence() != 4
            for a in atoms
            if a.GetIdx() not in core
        ):
            continue
        valid, lengths, charge = True, [], 0
        for path, unit in units:
            n, c, d, k = unit
            if any(a.GetExplicitDegree() != degree for a, degree in zip(unit, (2, 3, 3, 2))):
                valid = False
            if any(a.GetFormalCharge() != 0 or a.GetTotalValence() != 4 for a in (c, d)):
                valid = False
            orders = tuple(neighbor_bond_order(a, b) for a, b in zip(unit, unit[1:]))
            labels = [(a.GetFormalCharge(), a.GetTotalValence()) for a in (n, k)]
            if not (
                (orders == (2, 1, 2) and labels == [(0, 3), (0, 3)])
                or (orders == (1, 2, 1) and labels == [(-1, 2), (-1, 2)])
            ):
                valid = False
            distances = [a.GetDistance(b) for a, b in zip(unit, unit[1:])]
            if not (
                1.33 < distances[0] < 1.38
                and 1.33 < distances[2] < 1.38
                and 1.38 < distances[1] < 1.44
                and abs(distances[0] - distances[2]) < 0.025
            ):
                valid = False
            angle = abs(m.GetTorsion(*path))
            # This motif is the cis N-C-C-N chelate. A trans planar chain is
            # not evidence for the same paired-unit electronic preference.
            if not angle < 15:
                valid = False
            if not all(100 < a < 130 for a in (m.GetAngle(n, c, d), m.GetAngle(c, d, k))):
                valid = False
            lengths.append(distances)
            charge += n.GetFormalCharge() + k.GetFormalCharge()
        if not valid or any(abs(a - b) >= 0.025 for a, b in zip(lengths[0], lengths[1])):
            continue
        result[core] = charge
    return result


def clear_paired_diimine(candidates: Sequence[MetalCandidateState]) -> None:
    """Remove the previous contribution of the two cis-diimine units to metal discordance."""
    for c in candidates:
        previous = c.metadata.get(PAIRED_DIIMINE_KEY, 0)
        for key in ("metal_discordance_structural_count", "metal_discordance_count"):
            c.metadata[key] = c.metadata.get(key, 0) - previous
        c.metadata[PAIRED_DIIMINE_KEY] = 0


def _paired_diimine_group(
    c: MetalCandidateState, core: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[tuple[int, float, float, float], ...], int, int, int, float]:
    """Group TIM assignments at fixed donor geometry, metal spin and ligand radical target."""
    assert c.no_metal_state is not None
    m = cast(ob.OBMol, c.no_metal_state.omol.OBMol)
    atoms = [cast(ob.OBAtom, m.GetAtom(i)) for i in core]
    geometry = tuple((a.GetIdx(), a.GetX(), a.GetY(), a.GetZ()) for a in atoms)
    metal = c.metal_states[0]
    return (
        core,
        geometry,
        metal.idx,
        metal.radical_num,
        c.no_metal_radical_target,
        c.metadata.get("nitrosyl_geometry_penalty", 0),
    )


def annotate_paired_diimine(candidates: Sequence[MetalCandidateState]) -> None:
    """Prefer a total donor charge of −2 when matched cis-diimine units have a compensating metal state."""
    clear_paired_diimine(candidates)
    features = [paired_diimine_sites(c) for c in candidates]
    counts = [c.metadata.get("metal_discordance_structural_count", 0) for c in candidates]
    supported = set()
    for i, c in enumerate(candidates):
        if len(c.metal_states) != 1:
            continue
        for j, p in enumerate(candidates):
            if len(p.metal_states) != 1 or counts[j] > counts[i]:
                continue
            for core, charge in features[i].items():
                if charge == -2 or features[j].get(core) != -2:
                    continue
                if c.metal_states[0].valence - p.metal_states[0].valence != -2 - charge:
                    continue
                if _paired_diimine_group(c, core) == _paired_diimine_group(
                    p, core
                ) and outside_signature(c, core) == outside_signature(p, core):
                    supported.add(_paired_diimine_group(c, core))
    for c, candidate_features in zip(candidates, features):
        conflicts = (
            sum(
                charge != -2 and _paired_diimine_group(c, core) in supported
                for core, charge in candidate_features.items()
            )
            if len(c.metal_states) == 1
            else 0
        )
        c.metadata[PAIRED_DIIMINE_KEY] = conflicts
        for key in ("metal_discordance_structural_count", "metal_discordance_count"):
            c.metadata[key] += conflicts


# A pyramidal pyridyl sigma donor must approach the missing tetrahedral
# direction. Pyramidalization alone is insufficient: eta2 backbonding can
# produce the same ring distortion without the same formal redox assignment.
PYRIDYL_KEY = "metal_pyramidal_pyridyl_geometry_conflict_count"
PYRIDYL_MAX_PATH_STATES = 128
PYRIDYL_MAX_VARIANTS = 16


def pyramidal_pyridyl_centers(mol: ob.OBMol) -> Iterator[tuple[ob.OBAtom, ob.OBAtom, ob.OBAtom]]:
    """Find pyramidal para-CH centers between two elongated bonds in a six-membered pyridyl ring."""
    seen = set()
    for perimeter in ring_paths(mol):
        ids = list(perimeter)
        if len(ids) != 6:
            continue
        ats = [cast(ob.OBAtom, mol.GetAtom(i)) for i in ids]
        if sorted(a.GetAtomicNum() for a in ats) != [6, 6, 6, 6, 6, 7]:
            continue
        if any(
            a.GetExplicitDegree() != 3
            or a.GetImplicitHCount()
            or get_unpaired_electron_count(a)
            or has_unresolved_two_electron_center(a)
            for a in ats
        ):
            continue
        ni = next(i for i, a in enumerate(ats) if a.GetAtomicNum() == 7)
        ci = (ni + 3) % 6
        center = ats[ci]
        left = ats[(ci - 1) % 6]
        right = ats[(ci + 1) % 6]
        neighbors = list(atom_neighbors(center))
        external = [a for a in neighbors if a.GetIdx() not in ids]
        if (
            len(external) != 1
            or external[0].GetAtomicNum() != 1
            or neighbor_bond_order(center, external[0]) != 1
        ):
            continue
        if not all(1.44 < a.GetDistance(center) < 1.55 for a in [left, right]):
            continue
        if not all(
            1.30 < ats[(ci + sign) % 6].GetDistance(ats[(ci + 2 * sign) % 6]) < 1.43
            for sign in [-1, 1]
        ):
            continue
        angles = [mol.GetAngle(a, center, b) for a, b in combinations(neighbors, 2)]
        if not (all(80 < a < 140 for a in angles) and 340 < sum(angles) < 352):
            continue
        if center.GetIdx() in seen:
            continue
        seen.add(center.GetIdx())
        yield center, left, right


def clear_pyridyl(cs: Sequence[MetalCandidateState]) -> None:
    """Remove the pyridyl contribution before comparing charge localization around the C₅N ring."""
    for c in cs:
        previous = c.metadata.get(PYRIDYL_KEY, 0)
        for k in ["metal_discordance_structural_count", "metal_discordance_count"]:
            c.metadata[k] = c.metadata.get(k, 0) - previous
        c.metadata[PYRIDYL_KEY] = 0


def pyridyl_vacant_direction(candidate: MetalCandidateState, center: ob.OBAtom) -> bool:
    """Require the metal to lie opposite all three substituent bond vectors at the pyramidal carbon."""
    if len(candidate.metal_states) != 1:
        return False
    metal = candidate.metal_states[0]
    delta = (
        metal.position_x - center.GetX(),
        metal.position_y - center.GetY(),
        metal.position_z - center.GetZ(),
    )
    norm = sum(x * x for x in delta)
    if not math.isfinite(norm) or norm <= 0:
        return False
    neighbors = list(atom_neighbors(center))
    if len(neighbors) != 3:
        return False
    for neighbor in neighbors:
        other = (
            neighbor.GetX() - center.GetX(),
            neighbor.GetY() - center.GetY(),
            neighbor.GetZ() - center.GetZ(),
        )
        dot = sum(a * b for a, b in zip(delta, other))
        if not math.isfinite(dot) or dot >= 0:
            return False
    return True


def pyridyl_sites(candidate: MetalCandidateState) -> dict[PyridylSite, str]:
    """Identify the whole C5N ring; do not fix an anionic resonance contributor."""
    result: dict[tuple[tuple[int, ...], int, int], str] = {}
    if candidate.no_metal_state is None:
        return result
    mol = cast(ob.OBMol, candidate.no_metal_state.omol.OBMol)
    groups = [
        {int(i) for i in part.split(",") if i}
        for part in str(candidate.metadata.get("metal_visible_atom_groups", "")).split(";")
    ]
    for center, left, right in pyramidal_pyridyl_centers(mol):
        if not pyridyl_vacant_direction(candidate, center):
            continue
        if not any(center.GetIdx() in group for group in groups):
            continue
        for path in ring_paths(mol):
            ids = tuple(sorted(path))
            if len(ids) != 6 or not {center.GetIdx(), left.GetIdx(), right.GetIdx()} <= set(ids):
                continue
            atoms = [cast(ob.OBAtom, mol.GetAtom(i)) for i in ids]
            if sorted(a.GetAtomicNum() for a in atoms) != [6, 6, 6, 6, 6, 7]:
                continue
            n = next(a for a in atoms if a.GetAtomicNum() == 7)
            cs = [a for a in atoms if a.GetAtomicNum() == 6]
            if not all(resolved(a) for a in atoms):
                continue
            orders = [
                b.GetBondOrder()
                for b in mol_bonds(mol)
                if b.GetBeginAtomIdx() in ids and b.GetEndAtomIdx() in ids
            ]
            if len(orders) != 6 or any(order not in (1, 2) for order in orders):
                continue
            labels = [(a.GetFormalCharge(), a.GetTotalValence()) for a in cs]
            nlabel = (n.GetFormalCharge(), n.GetTotalValence())
            if nlabel not in [(0, 3), (1, 4)] or any(
                label not in [(0, 4), (-1, 3)] for label in labels
            ):
                continue
            key = (ids, n.GetIdx(), center.GetIdx())
            charge = sum(a.GetFormalCharge() for a in atoms)
            if charge == -1:
                result[key] = "reduced"
            elif all(label == (0, 4) for label in labels) and charge in (0, 1):
                result[key] = "oxidized"
    return result


def _pyridyl_oxidized_variants(candidate: MetalCandidateState, site: PyridylSite) -> list[ob.OBMol]:
    """Move an attached N/O cation onto ring N on copies only.

    Internal path vertices are neutral ring C atoms. A transfer alternates
    single/double bonds and terminates at resolved N(+1,v4) or O(+1,v3).
    Exceeding either bound abstains rather than retaining traversal-dependent
    partial results. The candidate is not changed.
    """
    assert candidate.no_metal_state is not None
    ring, nitrogen, center = site
    source = cast(ob.OBMol, candidate.no_metal_state.omol.OBMol)
    n = cast(ob.OBAtom, source.GetAtom(nitrogen))
    if n.GetFormalCharge() == 1 and n.GetTotalValence() == 4:
        return [ob.OBMol(source)]
    if n.GetFormalCharge() != 0 or n.GetTotalValence() != 3:
        return []
    queue: deque[tuple[int, ...]] = deque([(nitrogen,)])
    paths: list[tuple[int, ...]] = []
    visited = 0
    while queue:
        visited += 1
        if visited > PYRIDYL_MAX_PATH_STATES:
            return []
        path = queue.popleft()
        atom = cast(ob.OBAtom, source.GetAtom(path[-1]))
        order = 1 if len(path) % 2 else 2
        for bond in atom_bonds(atom):
            other = cast(ob.OBAtom, bond.GetNbrAtom(atom))
            idx = other.GetIdx()
            if idx in path or bond.GetBondOrder() != order or not resolved(other):
                continue
            if idx not in ring:
                if (
                    order == 2
                    and other.GetFormalCharge() == 1
                    and (other.GetAtomicNum(), other.GetTotalValence()) in [(7, 4), (8, 3)]
                ):
                    paths.append(path + (idx,))
                    if len(paths) > PYRIDYL_MAX_VARIANTS:
                        return []
            elif (
                other.GetAtomicNum() == 6
                and other.GetFormalCharge() == 0
                and other.GetTotalValence() == 4
            ):
                queue.append(path + (idx,))
    variants = []
    for path in paths:
        mol = ob.OBMol(source)
        mol.GetAtom(nitrogen).SetFormalCharge(1)
        mol.GetAtom(path[-1]).SetFormalCharge(0)
        for first, second in zip(path[:-1], path[1:]):
            bond = cast(ob.OBBond, mol.GetBond(first, second))
            bond.SetBondOrder(3 - bond.GetBondOrder())
        variants.append(mol)
    return variants


def _pyridyl_outside_normalizations(
    candidate: MetalCandidateState, site: PyridylSite
) -> set[OutsideSignature]:
    """Collect outside labels after moving an attached cation onto the pyridyl ring nitrogen."""
    assert candidate.no_metal_state is not None
    result = set()
    for mol in _pyridyl_oxidized_variants(candidate, site):
        copy = replace(
            candidate, no_metal_state=replace(candidate.no_metal_state, omol=pybel.Molecule(mol))
        )
        result.add(outside_signature(copy, site[0]))
    return result


def annotate_pyridyl(candidates: Sequence[MetalCandidateState]) -> None:
    """Compare ring reduction only when oxidation of the same metal accounts for the electron pair."""
    clear_pyridyl(candidates)
    features = [pyridyl_sites(c) for c in candidates]
    counts = [c.metadata.get("metal_discordance_structural_count", 0) for c in candidates]
    normalized = {}
    for index, (candidate, own, count) in enumerate(zip(candidates, features, counts)):
        conflicts = set()
        for partner, other, other_count in zip(candidates, features, counts):
            if (
                len(candidate.metal_states) != 1
                or len(partner.metal_states) != 1
                or other_count > count
            ):
                continue
            a, b = candidate.metal_states[0], partner.metal_states[0]
            if (a.idx, a.radical_num, a.valence + 2) != (b.idx, b.radical_num, b.valence):
                continue
            if (
                candidate.no_metal_radical_target != partner.no_metal_radical_target
                or candidate.metadata.get("nitrosyl_geometry_penalty", 0)
                != partner.metadata.get("nitrosyl_geometry_penalty", 0)
            ):
                continue
            for site, state in own.items():
                if state != "oxidized" or other.get(site) != "reduced":
                    continue
                key = (index, site)
                if key not in normalized:
                    normalized[key] = _pyridyl_outside_normalizations(candidate, site)
                if outside_signature(partner, site[0]) in normalized[key]:
                    conflicts.add(site)
        candidate.metadata[PYRIDYL_KEY] = len(conflicts)
        for field in ["metal_discordance_structural_count", "metal_discordance_count"]:
            candidate.metadata[field] += len(conflicts)


def ortho_dioxygen_sites(candidate: MetalCandidateState) -> dict[tuple[int, ...], bool]:
    """Identify aromatic O/O partners supported by a planar, equalized ring.

    Both adjacent C--O lengths must be 1.32--1.45 A. Terminal O sites must
    be visible to the same metal; an ether/Si--O/O--H substituent can remain
    outside that sphere. This only describes fully resolved candidates; the
    paired preference also requires unchanged outside atoms and boundary bonds.
    """
    if candidate.no_metal_state is None:
        return {}
    mol = cast(ob.OBMol, candidate.no_metal_state.omol.OBMol)
    groups = [
        {int(i) for i in group.split(",") if i}
        for group in str(candidate.metadata.get("metal_visible_atom_groups", "")).split(";")
    ]
    out = {}
    for perimeter in ring_paths(mol):
        ids = list(perimeter)
        if len(ids) != 6:
            continue
        atoms = [cast(ob.OBAtom, mol.GetAtom(i)) for i in ids]
        if any(
            a.GetAtomicNum() != 6
            or a.GetExplicitDegree() != 3
            or a.GetFormalCharge() != 0
            or a.GetTotalValence() != 4
            or not resolved(a)
            for a in atoms
        ):
            continue
        lengths = [atoms[i].GetDistance(atoms[(i + 1) % 6]) for i in range(6)]
        if not all(1.35 < d < 1.45 for d in lengths) or max(lengths) - min(lengths) > 0.06:
            continue
        torsions = [abs(mol.GetTorsion(*(ids[(i + k) % 6] for k in range(4)))) for i in range(6)]
        if not all(min(t, abs(180 - t)) < 15 for t in torsions):
            continue
        orders = [indexed_bond_order(mol, ids[i], ids[(i + 1) % 6]) for i in range(6)]
        if any(order not in (1, 2) for order in orders):
            continue
        for i in range(6):
            carbons = [atoms[i], atoms[(i + 1) % 6]]
            oxygens = []
            for carbon in carbons:
                outside = [a for a in atom_neighbors(carbon) if a.GetIdx() not in ids]
                if len(outside) != 1:
                    break
                oxygen = outside[0]
                if (
                    oxygen.GetAtomicNum() != 8
                    or oxygen.GetExplicitDegree() not in (1, 2)
                    or not resolved(oxygen)
                    or not 1.32 < carbon.GetDistance(oxygen) < 1.45
                ):
                    break
                if any(
                    neighbor_bond_order(oxygen, n) != 1
                    for n in atom_neighbors(oxygen)
                    if n.GetIdx() != carbon.GetIdx()
                ):
                    break
                oxygens.append(oxygen)
            if len(oxygens) != 2:
                continue
            terminals = {o.GetIdx() for o in oxygens if o.GetExplicitDegree() == 1}
            if not terminals or not any(terminals <= group for group in groups):
                continue
            labels = [
                (o.GetFormalCharge(), o.GetTotalValence(), neighbor_bond_order(o, c))
                for o, c in zip(oxygens, carbons)
            ]
            reduced = [(o.GetExplicitDegree() - 2, o.GetExplicitDegree(), 1) for o in oxygens]
            oxidized = [(o.GetExplicitDegree() - 1, o.GetExplicitDegree() + 1, 2) for o in oxygens]
            site = tuple(sorted(ids)) + tuple(sorted(o.GetIdx() for o in oxygens))
            if labels == reduced and all(orders[k] + orders[(k + 1) % 6] == 3 for k in range(6)):
                out[site] = True
            if labels == oxidized and sum(orders) == 8:
                out[site] = False
    return out


def clear_ortho_dioxygen(candidates: Sequence[MetalCandidateState]) -> None:
    """Remove the previous score for the adjacent oxygen donors on a shared aryl ring."""
    _clear_local_redox_pair(candidates, ORTHO_DIOXYGEN_KEY)


def annotate_ortho_dioxygen(candidates: Sequence[MetalCandidateState]) -> None:
    """Score a quinonoid assignment against a matched reduced partner on the same ortho O/O ring."""
    _annotate_local_redox_pair(candidates, ortho_dioxygen_sites, ORTHO_DIOXYGEN_KEY)


def equalized_aryl_rings(
    mol: ob.OBMol, rings: Iterable[list[int]]
) -> Iterator[tuple[list[int], list[ob.OBAtom], list[int]]]:
    """Yield resolved planar six-carbon rings and their perimeter bond orders."""
    for ids in rings:
        if len(ids) != 6:
            continue
        atoms = [cast(ob.OBAtom, mol.GetAtom(i)) for i in ids]
        if any(
            a.GetAtomicNum() != 6
            or a.GetExplicitDegree() != 3
            or a.GetFormalCharge() != 0
            or a.GetTotalValence() != 4
            or not resolved(a)
            for a in atoms
        ):
            continue
        lengths = [atoms[i].GetDistance(atoms[(i + 1) % 6]) for i in range(6)]
        if not all(1.35 < d < 1.45 for d in lengths) or max(lengths) - min(lengths) > 0.06:
            continue
        torsions = [abs(mol.GetTorsion(*(ids[(i + k) % 6] for k in range(4)))) for i in range(6)]
        if not all(min(t, abs(180 - t)) < 15 for t in torsions):
            continue
        orders = [indexed_bond_order(mol, ids[i], ids[(i + 1) % 6]) for i in range(6)]
        if any(order not in (1, 2) for order in orders):
            continue
        yield ids, atoms, orders


def _aryl_donor_nhc_sites(
    candidate: MetalCandidateState, donor_z: int
) -> dict[tuple[int, ...], bool]:
    """Compare ortho amido/phenoxide NHC arms with quinonoid partners.

    Long exocyclic C–N bonds and a planar equalized six-carbon ring
    support the reduced arm. The complete, resolved NHC ring
    belongs to the local core so either C−=N+ contributor can qualify.
    The amido N or phenoxide O and carbene C must face the same metal; the NHC nitrogen
    attached to the aryl ring need not itself be a metal donor.
    """
    if candidate.no_metal_state is None:
        return {}
    mol = cast(ob.OBMol, candidate.no_metal_state.omol.OBMol)
    groups = [
        {int(i) for i in group.split(",") if i}
        for group in str(candidate.metadata.get("metal_visible_atom_groups", "")).split(";")
    ]
    rings = [list(path) for path in ring_paths(mol)]
    out = {}
    donor_degree = {6: 3, 7: 2, 8: 1, 16: 1}[donor_z]
    length_bounds = {6: (1.43, 1.60), 7: (1.36, 1.46), 8: (1.30, 1.45), 16: (1.72, 1.85)}
    for ids, atoms, orders in equalized_aryl_rings(mol, rings):
        for i in range(6):
            carbons = [atoms[i], atoms[(i + 1) % 6]]
            ns = []
            for carbon in carbons:
                outside = [a for a in atom_neighbors(carbon) if a.GetIdx() not in ids]
                if len(outside) != 1:
                    break
                n = outside[0]
                if n.GetAtomicNum() not in (7, donor_z) or not resolved(n):
                    break
                short, long = length_bounds[n.GetAtomicNum()]
                if not short < n.GetDistance(carbon) < long:
                    break
                ns.append(n)
            if len(ns) != 2 or sorted(n.GetExplicitDegree() for n in ns) != [donor_degree, 3]:
                continue
            arm_index = 0 if ns[0].GetAtomicNum() == 7 and ns[0].GetExplicitDegree() == 3 else 1
            arm, amido = ns[arm_index], ns[1 - arm_index]
            if arm.GetAtomicNum() != 7 or amido.GetAtomicNum() != donor_z:
                continue
            ac = carbons[ns.index(arm)]
            nc = carbons[ns.index(amido)]
            if any(
                neighbor_bond_order(amido, n) != 1
                for n in atom_neighbors(amido)
                if n.GetIdx() != nc.GetIdx()
            ):
                continue
            if donor_z == 6:
                neighbors = list(atom_neighbors(amido))
                angles = [mol.GetAngle(a, amido, b) for a, b in combinations(neighbors, 2)]
                if (
                    amido.IsInRing()
                    or sum(n.GetAtomicNum() == 1 for n in neighbors) != 2
                    or not pyridyl_vacant_direction(candidate, amido)
                    or not all(90 < angle < 130 for angle in angles)
                    or not 300 < sum(angles) < 350
                ):
                    continue
            for azole in rings:
                if len(azole) != 5 or arm.GetIdx() not in azole or set(azole) & set(ids):
                    continue
                az = [cast(ob.OBAtom, mol.GetAtom(j)) for j in azole]
                if sorted(a.GetAtomicNum() for a in az) != [6, 6, 6, 7, 7] or not all(
                    resolved(a) for a in az
                ):
                    continue
                nitrogens = [a for a in az if a.GetAtomicNum() == 7]
                carbene_centers = [
                    a
                    for a in az
                    if a.GetAtomicNum() == 6
                    and a.GetExplicitDegree() == 2
                    and a.GetFormalCharge() == -1
                    and a.GetTotalValence() == 3
                    and {n.GetIdx() for n in atom_neighbors(a)} == {n.GetIdx() for n in nitrogens}
                ]
                if len(carbene_centers) != 1 or any(
                    n.GetExplicitDegree() != 3
                    or (n.GetFormalCharge(), n.GetTotalValence()) not in [(0, 3), (1, 4)]
                    for n in nitrogens
                ):
                    continue
                carbene = carbene_centers[0]
                backbone = [
                    a for a in az if a.GetAtomicNum() == 6 and a.GetIdx() != carbene.GetIdx()
                ]
                if any(a.GetFormalCharge() != 0 or a.GetTotalValence() != 4 for a in backbone):
                    continue
                bb = cast(Optional[ob.OBBond], mol.GetBond(backbone[0], backbone[1]))
                if bb is None:
                    continue
                if bb.GetBondOrder() == 2 and all(a.GetExplicitDegree() == 3 for a in backbone):
                    if not all(1.30 < az[j].GetDistance(az[(j + 1) % 5]) < 1.46 for j in range(5)):
                        continue
                    ts = [
                        abs(mol.GetTorsion(*(azole[(j + k) % 5] for k in range(4))))
                        for j in range(5)
                    ]
                    if not all(min(t, abs(180 - t)) < 15 for t in ts):
                        continue
                elif (
                    donor_z in (8, 16)
                    and bb.GetBondOrder() == 1
                    and all(
                        a.GetExplicitDegree() == 4
                        and all(b.GetBondOrder() == 1 for b in atom_bonds(a))
                        for a in backbone
                    )
                ):
                    # Saturated NHCs can pucker at the C--C backbone. Require
                    # tetrahedral backbone valence and planar N donors instead
                    # of imposing aromatic-ring planarity on this sigma bond.
                    if not 1.45 < backbone[0].GetDistance(backbone[1]) < 1.65:
                        continue
                    valid_n = True
                    for nitrogen in nitrogens:
                        neighbors = list(atom_neighbors(nitrogen))
                        angles = [
                            mol.GetAngle(a, nitrogen, b) for a, b in combinations(neighbors, 2)
                        ]
                        if not (
                            all(85 < angle < 140 for angle in angles)
                            and 350 < sum(angles) <= 360.000001
                            and 1.30 < nitrogen.GetDistance(carbene) < 1.43
                        ):
                            valid_n = False
                        for neighbor in neighbors:
                            if (
                                neighbor.GetIdx() in {a.GetIdx() for a in backbone}
                                and not 1.42 < nitrogen.GetDistance(neighbor) < 1.60
                            ):
                                valid_n = False
                    if not valid_n:
                        continue
                else:
                    continue
                if not any({amido.GetIdx(), carbene.GetIdx()} <= group for group in groups):
                    continue
                if any(
                    b.GetBondOrder() != 1
                    for a in backbone
                    for b in atom_bonds(a)
                    if b.GetNbrAtom(a).GetAtomicNum() == 7
                ):
                    continue
                aq = sum(a.GetFormalCharge() for a in az)
                q, v, bo = (
                    amido.GetFormalCharge(),
                    amido.GetTotalValence(),
                    neighbor_bond_order(amido, nc),
                )
                key = tuple(sorted(ids)) + tuple(sorted(azole)) + (amido.GetIdx(),)
                if (
                    (q, v, bo) == (-1, donor_degree, 1)
                    and neighbor_bond_order(arm, ac) == 1
                    and aq == 0
                    and all(orders[j] + orders[(j + 1) % 6] == 3 for j in range(6))
                ):
                    out[key] = True
                if (
                    (q, v, bo) == (0, donor_degree + 1, 2)
                    and neighbor_bond_order(arm, ac) == 2
                    and aq == 1
                    and sum(orders) == 8
                ):
                    out[key] = False
    return out


def aryl_amido_nhc_sites(candidate: MetalCandidateState) -> dict[tuple[int, ...], bool]:
    """Match an ortho amido nitrogen and an N-heterocyclic carbene facing the same metal."""
    return _aryl_donor_nhc_sites(candidate, 7)


def aryl_phenoxide_nhc_sites(candidate: MetalCandidateState) -> dict[tuple[int, ...], bool]:
    """Match an ortho phenoxide oxygen and an N-heterocyclic carbene facing the same metal."""
    return _aryl_donor_nhc_sites(candidate, 8)


def clear_aryl_amido_nhc(candidates: Sequence[MetalCandidateState]) -> None:
    """Subtract the previous amido–carbene chelate conflict from both metal scores."""
    _clear_local_redox_pair(candidates, ARYL_AMIDO_NHC_KEY)


def annotate_aryl_amido_nhc(candidates: Sequence[MetalCandidateState]) -> None:
    """Score the quinonoid form against the matched aromatic amido–carbene chelate."""
    _annotate_local_redox_pair(candidates, aryl_amido_nhc_sites, ARYL_AMIDO_NHC_KEY)


def clear_aryl_phenoxide_nhc(candidates: Sequence[MetalCandidateState]) -> None:
    """Subtract the previous phenoxide–carbene chelate conflict from both metal scores."""
    _clear_local_redox_pair(candidates, ARYL_PHENOXIDE_NHC_KEY)


def annotate_aryl_phenoxide_nhc(candidates: Sequence[MetalCandidateState]) -> None:
    """Score the quinonoid form against the matched aromatic phenoxide–carbene chelate."""
    _annotate_local_redox_pair(candidates, aryl_phenoxide_nhc_sites, ARYL_PHENOXIDE_NHC_KEY)


def clear_aryl_thiolate_nhc(candidates: Sequence[MetalCandidateState]) -> None:
    """Subtract the previous thiolate–carbene chelate conflict from both metal scores."""
    _clear_local_redox_pair(candidates, ARYL_THIOLATE_NHC_KEY)


def annotate_aryl_thiolate_nhc(candidates: Sequence[MetalCandidateState]) -> None:
    """Score the quinonoid form against the matched aromatic thiolate–carbene chelate."""
    _annotate_local_redox_pair(
        candidates, lambda c: _aryl_donor_nhc_sites(c, 16), ARYL_THIOLATE_NHC_KEY
    )


def clear_aryl_benzyl_nhc(candidates: Sequence[MetalCandidateState]) -> None:
    """Subtract the previous benzyl–carbene chelate conflict from both metal scores."""
    _clear_local_redox_pair(candidates, ARYL_BENZYL_NHC_KEY)


def annotate_aryl_benzyl_nhc(candidates: Sequence[MetalCandidateState]) -> None:
    """Compare an exocyclic C=C assignment with a metal-facing pyramidal benzyl anion."""
    _annotate_local_redox_pair(
        candidates, lambda c: _aryl_donor_nhc_sites(c, 6), ARYL_BENZYL_NHC_KEY
    )


def amino_c5_sites(candidate: MetalCandidateState) -> dict[tuple[int, ...], bool]:
    """Compare amino-substituted Cp/indenyl and exocyclic iminium partners.

    A planar, equalized C5 ring visible to one metal and a long exocyclic
    C–N bond support a ring anion with a neutral dialkylamino group. A
    fused benzene belongs to the local core because its Kekule bonds can
    change with the five-ring assignment. This only annotates complete
    candidates; a compatible two-electron redox partner is also required.
    """
    if candidate.no_metal_state is None:
        return {}
    mol = cast(ob.OBMol, candidate.no_metal_state.omol.OBMol)
    rings = [list(path) for path in ring_paths(mol)]
    groups = [
        {int(i) for i in group.split(",") if i}
        for group in str(candidate.metadata.get("metal_visible_atom_groups", "")).split(";")
    ]
    out = {}
    for ids in rings:
        if len(ids) != 5 or not any(set(ids) <= group for group in groups):
            continue
        atoms = [cast(ob.OBAtom, mol.GetAtom(i)) for i in ids]
        if any(
            a.GetAtomicNum() != 6
            or a.GetExplicitDegree() != 3
            or not resolved(a)
            or (a.GetFormalCharge(), a.GetTotalValence()) not in [(0, 4), (-1, 3)]
            for a in atoms
        ):
            continue
        lengths = [atoms[i].GetDistance(atoms[(i + 1) % 5]) for i in range(5)]
        if not all(1.38 < d < 1.48 for d in lengths) or max(lengths) - min(lengths) > 0.06:
            continue
        ts = [abs(mol.GetTorsion(*(ids[(i + k) % 5] for k in range(4)))) for i in range(5)]
        if not all(min(t, abs(180 - t)) < 15 for t in ts):
            continue
        core = set(ids)
        fused = [r for r in rings if set(r) & core and set(r) != set(ids)]
        if len(fused) > 1:
            continue
        if fused:
            r = fused[0]
            shared = set(r) & core
            if len(r) != 6 or len(shared) != 2 or mol.GetBond(*sorted(shared)) is None:
                continue
            aa = [cast(ob.OBAtom, mol.GetAtom(i)) for i in r]
            if any(
                a.GetAtomicNum() != 6
                or a.GetExplicitDegree() != 3
                or a.GetFormalCharge() != 0
                or a.GetTotalValence() != 4
                or not resolved(a)
                for a in aa
            ):
                continue
            ds = [aa[i].GetDistance(aa[(i + 1) % 6]) for i in range(6)]
            ts = [abs(mol.GetTorsion(*(r[(i + k) % 6] for k in range(4)))) for i in range(6)]
            if (
                not all(1.35 < d < 1.48 for d in ds)
                or max(ds) - min(ds) > 0.10
                or not all(min(t, abs(180 - t)) < 15 for t in ts)
            ):
                continue
            core.update(r)
        if any(
            b.GetBondOrder() not in (1, 2)
            for b in mol_bonds(mol)
            if b.GetBeginAtomIdx() in core and b.GetEndAtomIdx() in core
        ):
            continue
        for carbon in atoms:
            for n in atom_neighbors(carbon):
                if (
                    n.GetIdx() in core
                    or n.GetAtomicNum() != 7
                    or n.GetExplicitDegree() != 3
                    or not resolved(n)
                    or not 1.36 < n.GetDistance(carbon) < 1.46
                ):
                    continue
                rest = [a for a in atom_neighbors(n) if a.GetIdx() != carbon.GetIdx()]
                if any(
                    a.GetAtomicNum() != 6
                    or a.GetIdx() in core
                    or a.GetExplicitDegree() != 4
                    or a.GetFormalCharge() != 0
                    or a.GetTotalValence() != 4
                    or not resolved(a)
                    or neighbor_bond_order(n, a) != 1
                    for a in rest
                ):
                    continue
                neighbors = list(atom_neighbors(n))
                angles = [
                    mol.GetAngle(neighbors[i], n, neighbors[j]) for i, j in [(0, 1), (0, 2), (1, 2)]
                ]
                if not all(100 < angle < 140 for angle in angles) or sum(angles) < 355:
                    continue
                q = sum(a.GetFormalCharge() for a in atoms)
                key = tuple(sorted(core)) + (n.GetIdx(),)
                labels = (
                    n.GetFormalCharge(),
                    n.GetTotalValence(),
                    neighbor_bond_order(n, carbon),
                )
                if q == -1 and labels == (0, 3, 1):
                    out[key] = True
                if q == 0 and labels == (1, 4, 2):
                    out[key] = False
    return out


def clear_amino_c5(candidates: Sequence[MetalCandidateState]) -> None:
    """Remove the previous conflict between an exocyclic iminium and a reduced five-membered ring."""
    _clear_local_redox_pair(candidates, AMINO_C5_KEY)


def annotate_amino_c5(candidates: Sequence[MetalCandidateState]) -> None:
    """Score an exocyclic iminium against a charge-balanced aminocyclopentadienyl or indenyl partner."""
    _annotate_local_redox_pair(candidates, amino_c5_sites, AMINO_C5_KEY)
