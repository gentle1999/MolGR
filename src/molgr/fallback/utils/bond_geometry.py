"""Local bond-length windows shared by candidate generation and scoring."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from itertools import combinations
from typing import List, Optional, Tuple, cast

from openbabel import openbabel as ob

from molgr.fallback.utils.electrons import (
    get_unpaired_electron_count,
    has_unresolved_two_electron_center,
)
from molgr.fallback.utils.ob_graph import (
    atom_bonds,
    atom_neighbors,
    indexed_bond_order,
    mol_atoms,
    neighbor_bond_order,
    ring_paths,
)


FormazanArm = Tuple[int, int, int, int, int, Tuple[int, ...]]
FormazanSite = Tuple[int, List[FormazanArm], Tuple[int, ...]]


def _compact_planar_circuit(mol: ob.OBMol, ids: Sequence[int]) -> bool:
    """Screen a cyclic sequence for short perimeter bonds, nondegenerate angles and small torsions."""
    atoms = [cast(ob.OBAtom, mol.GetAtom(i)) for i in ids]
    if any(
        not 1.25 < atom.GetDistance(atoms[(i + 1) % len(ids)]) < 1.50
        or not 60.0 < mol.GetAngle(atoms[i - 1], atom, atoms[(i + 1) % len(ids)]) < 170.0
        for i, atom in enumerate(atoms)
    ):
        return False
    torsions = [
        abs(mol.GetTorsion(*(ids[(i + k) % len(ids)] for k in range(4)))) for i in range(len(ids))
    ]
    return all(min(t, abs(180.0 - t)) < 15.0 for t in torsions)


def planar_tetrapyrrole_circuits(mol: ob.OBMol) -> Iterator[tuple[int, ...]]:
    """Yield geometrically conjugated 15/16-member tetrapyrrole inner circuits.

    This recognizes a bounded search motif, not a metal oxidation state. Four
    two-coordinate N atoms must each belong to a C4N five-membered ring sharing
    three atoms with the circuit. All other circuit atoms are three-coordinate
    C; all H must be explicit. Require 1.25--1.50 A circuit bonds and local
    torsions within 15 degrees of planar. Missing/degenerate geometry abstains.
    The metal-free graph is sufficient; no metal positions are inferred here.
    """
    rings = [tuple(path) for path in ring_paths(mol)]
    for ids in rings:
        if len(ids) not in (15, 16):
            continue
        atoms = [cast(ob.OBAtom, mol.GetAtom(i)) for i in ids]
        nitrogens = [atom for atom in atoms if atom.GetAtomicNum() == 7]
        if len(nitrogens) != 4 or any(
            atom.GetAtomicNum() not in (6, 7)
            or atom.GetImplicitHCount() != 0
            or atom.GetExplicitDegree() != (2 if atom.GetAtomicNum() == 7 else 3)
            for atom in atoms
        ):
            continue
        if not _compact_planar_circuit(mol, ids):
            continue
        if all(
            any(
                len(ring) == 5
                and nitrogen.GetIdx() in ring
                and len(set(ring) & set(ids)) == 3
                and sorted(mol.GetAtom(i).GetAtomicNum() for i in ring) == [6, 6, 6, 6, 7]
                for ring in rings
            )
            for nitrogen in nitrogens
        ):
            yield ids


def planar_aza_macrocycle_circuits(mol: ob.OBMol) -> Iterator[tuple[tuple[int, ...], set[int]]]:
    """Yield a 15-member aza circuit and its four two-coordinate N donors.

    Require three C4N five-membered rings and one C4N2 six-membered ring, each
    sharing three circuit atoms. Circuit C has degree three, N degree two/three;
    all H is explicit. Use the same local length/angle/planarity windows as the
    tetrapyrrole motif. Bridging nitrogens are not all metal-facing donors.
    Recognition does not assign formal charges or count pi electrons.
    """
    rings = [tuple(path) for path in ring_paths(mol)]
    for ids in rings:
        if len(ids) != 15:
            continue
        atoms = [cast(ob.OBAtom, mol.GetAtom(i)) for i in ids]
        if any(
            atom.GetAtomicNum() not in (6, 7)
            or atom.GetImplicitHCount() != 0
            or (
                atom.GetExplicitDegree() != 3
                if atom.GetAtomicNum() == 6
                else atom.GetExplicitDegree() not in (2, 3)
            )
            for atom in atoms
        ) or not _compact_planar_circuit(mol, ids):
            continue
        pyrroles, diazines = [], []
        for ring in rings:
            if len(ring) not in (5, 6) or len(set(ring) & set(ids)) != 3:
                continue
            elements = sorted(mol.GetAtom(i).GetAtomicNum() for i in ring)
            if elements == [6, 6, 6, 6, 7]:
                pyrroles.append(ring)
            elif elements == [6, 6, 6, 6, 7, 7]:
                diazines.append(ring)
        if len(pyrroles) != 3 or len(diazines) != 1:
            continue
        donors = {
            i
            for ring in pyrroles + diazines
            for i in ring
            if i in ids
            and mol.GetAtom(i).GetAtomicNum() == 7
            and mol.GetAtom(i).GetExplicitDegree() == 2
        }
        if len(donors) == 4:
            yield ids, donors


def planar_tetraazaporphyrin_circuits(mol: ob.OBMol) -> Iterator[tuple[tuple[int, ...], set[int]]]:
    """Recognize the planar C8N8 inner circuit and four pyrrolic donors.

    Four C4N five-membered rings each share three circuit atoms. Alternating
    C/N vertices distinguish this topology from the existing C12N4 circuit;
    the four bridging nitrogens are not metal-facing donors.
    """
    rings = [tuple(path) for path in ring_paths(mol)]
    for ids in rings:
        if len(ids) != 16:
            continue
        atoms = [cast(ob.OBAtom, mol.GetAtom(i)) for i in ids]
        if sorted(a.GetAtomicNum() for a in atoms) != [6] * 8 + [7] * 8:
            continue
        if any(
            a.GetImplicitHCount()
            or a.GetExplicitDegree() != (3 if a.GetAtomicNum() == 6 else 2)
            or a.GetAtomicNum() == atoms[(i + 1) % 16].GetAtomicNum()
            for i, a in enumerate(atoms)
        ) or not _compact_planar_circuit(mol, ids):
            continue
        pyrroles = [
            ring
            for ring in rings
            if len(ring) == 5
            and len(set(ring) & set(ids)) == 3
            and sorted(mol.GetAtom(i).GetAtomicNum() for i in ring) == [6, 6, 6, 6, 7]
        ]
        donors = {
            i for ring in pyrroles for i in ring if i in ids and mol.GetAtom(i).GetAtomicNum() == 7
        }
        if len(pyrroles) == 4 and len(donors) == 4:
            yield ids, donors


def folded_eight_carbon_segments(mol: ob.OBMol) -> Iterator[tuple[int, ...]]:
    """Yield oriented C8 rings with a folded, equalized four-carbon segment.

    The first four atoms form a locally planar segment with three 1.40--1.46 A
    bonds (spread <=0.03 A). Flanking bonds are 1.43--1.55 A; the opposite
    segment has two 1.30--1.40 A bonds around a 1.40--1.50 A bond. Both hinges
    depart from planar by 25--75 degrees. This is an endpoint exploration
    motif, not an assertion that the entire ring is planar or aromatic.
    """
    seen = set()
    for perimeter in ring_paths(mol):
        ids = list(perimeter)
        if len(ids) != 8:
            continue
        atoms = [cast(ob.OBAtom, mol.GetAtom(i)) for i in ids]
        if any(
            atom.GetAtomicNum() != 6
            or atom.GetExplicitDegree() != 3
            or atom.GetImplicitHCount()
            or get_unpaired_electron_count(atom)
            or has_unresolved_two_electron_center(atom)
            for atom in atoms
        ):
            continue
        for shift in range(8):
            ordered = atoms[shift:] + atoms[:shift]
            indices = tuple(atom.GetIdx() for atom in ordered)
            lengths = [ordered[i].GetDistance(ordered[(i + 1) % 8]) for i in range(8)]
            if (
                not all(1.40 < length < 1.46 for length in lengths[:3])
                or max(lengths[:3]) - min(lengths[:3]) > 0.03
                or not all(1.43 < lengths[i] < 1.55 for i in (3, 7))
                or not all(1.30 < lengths[i] < 1.40 for i in (4, 6))
                or not 1.40 < lengths[5] < 1.50
            ):
                continue
            torsion = abs(mol.GetTorsion(*indices[:4]))
            if not min(torsion, abs(180.0 - torsion)) < 15.0:
                continue
            hinges = [
                abs(mol.GetTorsion(*(indices[(i + k) % 8] for k in range(4)))) for i in (1, 7)
            ]
            if not all(25.0 < min(t, abs(180.0 - t)) < 75.0 for t in hinges):
                continue
            if not all(
                90.0 < mol.GetAngle(ordered[i - 1], ordered[i], ordered[(i + 1) % 8]) < 150.0
                for i in range(8)
            ):
                continue
            site = tuple(sorted(indices[:4]))
            if site not in seen:
                seen.add(site)
                yield indices


def _formazan_planar_torsion(m: ob.OBMol, ids: Sequence[int]) -> bool:
    """Accept syn- and anti-coplanar segments of a formazan conjugation path."""
    torsion = abs(m.GetTorsion(*ids))
    return min(torsion, abs(180 - torsion)) < 15


def planar_phenoxy_formazan_sites(
    m: ob.OBMol, *, allow_plain_phenyl: bool = False
) -> Iterator[FormazanSite]:
    """Yield formazan sites with two phenoxy arms, or one phenoxy and one plain phenyl.

    Require two disjoint, nearly equalized planar C6 rings; terminal O at
    1.30--1.40 A; aryl C--N at 1.36--1.46 A; and a planar N--N--C--N--N
    core with short N--N and equalized C--N pairs. All H is explicit and all
    local electron labels are resolved. Return the center, two arms (inner N,
    outer N, aryl C, O-bound C, O, ring), and sorted site indices. No bond
    order or formal charge is used to infer the geometry or metal state.
    """
    rings = []
    for perimeter in ring_paths(m):
        ids = tuple(perimeter)
        ats = [cast(ob.OBAtom, m.GetAtom(i)) for i in ids]
        if len(ids) != 6 or any(
            a.GetAtomicNum() != 6 or a.GetExplicitDegree() != 3 or a.GetImplicitHCount()
            for a in ats
        ):
            continue
        ls = [ats[i].GetDistance(ats[(i + 1) % 6]) for i in range(6)]
        if not all(1.35 < x < 1.46 for x in ls) or max(ls) - min(ls) > 0.06:
            continue
        if not all(
            _formazan_planar_torsion(m, tuple(ids[(i + k) % 6] for k in range(4)))
            and 90 < m.GetAngle(ats[i - 1], ats[i], ats[(i + 1) % 6]) < 150
            for i in range(6)
        ):
            continue
        rings.append(ids)
    for center in mol_atoms(m):
        if (
            center.GetAtomicNum() != 6
            or center.GetExplicitDegree() != 3
            or center.GetImplicitHCount()
        ):
            continue
        inners = [a for a in atom_neighbors(center) if a.GetAtomicNum() == 7]
        if len(inners) != 2:
            continue
        inners.sort(key=lambda a: a.GetIdx())
        arms: list[FormazanArm] = []
        for inner in inners:
            if inner.GetExplicitDegree() != 2 or inner.GetImplicitHCount():
                break
            outer = next(a for a in atom_neighbors(inner) if a.GetIdx() != center.GetIdx())
            if (
                outer.GetAtomicNum() != 7
                or outer.GetExplicitDegree() != 2
                or outer.GetImplicitHCount()
            ):
                break
            carbon = next(a for a in atom_neighbors(outer) if a.GetIdx() != inner.GetIdx())
            if (
                not 1.24 < inner.GetDistance(outer) < 1.33
                or not 1.32 < center.GetDistance(inner) < 1.43
                or (not 1.36 < outer.GetDistance(carbon) < 1.46)
            ):
                break
            if (
                not 100 < m.GetAngle(center, inner, outer) < 140
                or not 100 < m.GetAngle(inner, outer, carbon) < 140
            ):
                break
            matches: list[FormazanArm] = []
            for ids in rings:
                if carbon.GetIdx() not in ids:
                    continue
                if allow_plain_phenyl and all(
                    neighbor.GetAtomicNum() == 1
                    or neighbor.GetIdx() in ids
                    or (i == carbon.GetIdx() and neighbor.GetIdx() == outer.GetIdx())
                    for i in ids
                    for neighbor in atom_neighbors(m.GetAtom(i))
                ):
                    matches.append((inner.GetIdx(), outer.GetIdx(), carbon.GetIdx(), 0, 0, ids))
                for adjacent in atom_neighbors(carbon):
                    if adjacent.GetIdx() not in ids:
                        continue
                    for oxygen in atom_neighbors(adjacent):
                        if (
                            oxygen.GetAtomicNum() == 8
                            and oxygen.GetExplicitDegree() == 1
                            and (not oxygen.GetImplicitHCount())
                            and (1.3 < adjacent.GetDistance(oxygen) < 1.4)
                        ):
                            matches.append(
                                (
                                    inner.GetIdx(),
                                    outer.GetIdx(),
                                    carbon.GetIdx(),
                                    adjacent.GetIdx(),
                                    oxygen.GetIdx(),
                                    ids,
                                )
                            )
            if len(matches) != 1:
                break
            arms.append(matches[0])
        if len(arms) != 2 or not any(arm[4] for arm in arms) or set(arms[0][5]) & set(arms[1][5]):
            continue
        core = (arms[0][1], arms[0][0], center.GetIdx(), arms[1][0], arms[1][1])
        if not 100 < m.GetAngle(inners[0], center, inners[1]) < 150 or not all(
            _formazan_planar_torsion(m, core[i : i + 4]) for i in range(2)
        ):
            continue
        nn = [m.GetAtom(a[0]).GetDistance(m.GetAtom(a[1])) for a in arms]
        cn = [center.GetDistance(m.GetAtom(a[0])) for a in arms]
        if abs(nn[0] - nn[1]) > 0.04 or abs(cn[0] - cn[1]) > 0.04:
            continue
        site_indices = {center.GetIdx()}
        for arm in arms:
            site_indices.update(i for i in arm[:5] if i)
            site_indices.update(arm[5])
        if any(
            get_unpaired_electron_count(m.GetAtom(i))
            or has_unresolved_two_electron_center(m.GetAtom(i))
            or m.GetAtom(i).GetImplicitHCount()
            for i in site_indices
        ):
            continue
        yield (center.GetIdx(), arms, tuple(sorted(site_indices)))


def planar_bis_phenoxy_formazan_sites(m: ob.OBMol) -> Iterator[FormazanSite]:
    """Require a terminal phenoxy oxygen on each arm of the planar formazan."""
    return planar_phenoxy_formazan_sites(m)


def equalized_three_carbon_chains(
    mol: ob.OBMol,
) -> Iterator[tuple[ob.OBAtom, ob.OBAtom, ob.OBAtom]]:
    """Yield bent, noncyclic C--C--C sites supporting a delocalized hypothesis.

    All three carbons have two explicit neighbors and no implicit H. Both chain
    bonds must be 1.35--1.50 A and differ by at most 0.05 A; each local angle
    must be 110--150 degrees. This is a scoped exploration heuristic, not a
    universal prohibition on stretched or metal-bound alkynes.
    """
    for middle in mol_atoms(mol):
        if middle.GetAtomicNum() != 6 or middle.GetExplicitDegree() != 2:
            continue
        ends = sorted(atom_neighbors(middle), key=lambda atom: atom.GetIdx())
        atoms = (ends[0], middle, ends[1])
        if any(
            atom.GetAtomicNum() != 6
            or atom.GetExplicitDegree() != 2
            or atom.GetImplicitHCount() != 0
            or atom.IsInRing()
            for atom in atoms
        ):
            continue
        lengths = [middle.GetDistance(atom) for atom in ends]
        if not all(1.35 < length < 1.50 for length in lengths):
            continue
        if abs(lengths[0] - lengths[1]) > 0.05:
            continue
        if not 110.0 < mol.GetAngle(ends[0], middle, ends[1]) < 150.0:
            continue
        if all(
            any(
                neighbor.GetIdx() != middle.GetIdx()
                and 110.0 < mol.GetAngle(neighbor, atom, middle) < 150.0
                for neighbor in atom_neighbors(atom)
            )
            for atom in ends
        ):
            yield atoms


def allows_seesaw_phosphoranide(atom: ob.OBAtom) -> bool:
    """Admit a P(-1) branch for a four-single-bond C/F or O/N seesaw.

    C/F sites require at least one axial F. O/N sites use two axial O atoms
    when oxygen is present, or two axial N atoms in a P(N)4 site.
    All four bonds must fit the corresponding length and angle windows.
    This is positive evidence for an additional
    electronic hypothesis, not a rule assigning P charge. Tetrahedral,
    ambiguous, missing and degenerate geometries do not open this branch.
    Only ligand neighbors are inspected; metal coordinates are not needed.
    """
    if atom.GetAtomicNum() != 15 or atom.GetExplicitDegree() != 4 or atom.GetImplicitHCount() != 0:
        return False
    neighbors = list(atom_neighbors(atom))
    elements = {neighbor.GetAtomicNum() for neighbor in neighbors}
    if elements <= {6, 9}:
        axial_element = 9
    elif elements <= {7, 8}:
        axial_element = 8 if 8 in elements else 7
    else:
        return False
    for neighbor in neighbors:
        if neighbor_bond_order(atom, neighbor) != 1:
            return False
        distance = atom.GetDistance(neighbor)
        if axial_element != 9:
            offset = (
                ob.GetCovalentRad(7) - ob.GetCovalentRad(8) if neighbor.GetAtomicNum() == 7 else 0.0
            )
            lower, upper = 1.55 + offset, 1.90 + offset
        else:
            lower, upper = (1.45, 1.85) if neighbor.GetAtomicNum() == 9 else (1.60, 2.05)
        if not lower < distance < upper:
            return False
    mol = cast(ob.OBMol, atom.GetParent())
    for first, second in combinations(neighbors, 2):
        axial_count = (first.GetAtomicNum() == axial_element) + (
            second.GetAtomicNum() == axial_element
        )
        if axial_count < (1 if axial_element == 9 else 2):
            continue
        if not 165.0 < mol.GetAngle(first, atom, second) <= 180.0:
            continue
        axial_ids = {first.GetIdx(), second.GetIdx()}
        equatorial = [n for n in neighbors if n.GetIdx() not in axial_ids]
        if not 80.0 < mol.GetAngle(equatorial[0], atom, equatorial[1]) < 140.0:
            continue
        if all(
            75.0 < mol.GetAngle(axial, atom, lateral) < 105.0
            for axial in (first, second)
            for lateral in equatorial
        ):
            return True
    return False


def bent_diazonium_geometry_allowed(first: ob.OBAtom, second: ob.OBAtom, order: int) -> bool:
    """Exclude a resolved C--N(+)#N with a long, bent ligand-local N--N unit.

    Require terminal neutral N, two-coordinate cationic N, a single C--N,
    N--N at 1.24--1.50 A and C--N at 1.15--1.60 A. An angle below 150 degrees
    excludes the triple-bond hypothesis, without assigning a metal oxidation
    state. Rings, open-shell/deferred labels and missing geometry abstain.
    """
    if order != 3 or any(atom.GetAtomicNum() != 7 for atom in (first, second)):
        return True
    if not 1.24 < first.GetDistance(second) < 1.50:
        return True
    if any(
        atom.IsInRing()
        or atom.GetImplicitHCount()
        or get_unpaired_electron_count(atom)
        or has_unresolved_two_electron_center(atom)
        for atom in (first, second)
    ):
        return True
    for center, terminal in ((first, second), (second, first)):
        if (
            center.GetFormalCharge() != 1
            or center.GetExplicitDegree() != 2
            or terminal.GetFormalCharge() != 0
            or terminal.GetExplicitDegree() != 1
        ):
            continue
        carbon = next(atom for atom in atom_neighbors(center) if atom.GetIdx() != terminal.GetIdx())
        if (
            carbon.GetAtomicNum() != 6
            or neighbor_bond_order(center, carbon) != 1
            or not 1.15 < center.GetDistance(carbon) < 1.60
        ):
            continue
        angle = center.GetParent().GetAngle(carbon, center, terminal)
        if 0.0 < angle < 150.0:
            return False
    return True


def bent_cnn_cumulene_geometry_allowed(first: ob.OBAtom, second: ob.OBAtom, order: int) -> bool:
    """Exclude resolved, acyclic C=N(+)=N with a strongly bent central N.

    Require a neutral three-coordinate carbon and either a substituted neutral
    outer N or a terminal N(-),
    C--N at 1.15--1.50 A and N--N at 1.20--1.50 A. Angles below 150 degrees
    contradict the cumulene assignment. Near-linear diazo units, radicals,
    deferred electron labels and missing geometry remain outside this screen.
    """
    if order != 2 or any(atom.GetAtomicNum() != 7 for atom in (first, second)):
        return True
    if not 1.20 < first.GetDistance(second) < 1.50:
        return True
    if any(
        atom.IsInRing()
        or atom.GetImplicitHCount()
        or get_unpaired_electron_count(atom)
        or has_unresolved_two_electron_center(atom)
        for atom in (first, second)
    ):
        return True
    for center, outer in ((first, second), (second, first)):
        if (
            center.GetFormalCharge() != 1
            or center.GetExplicitDegree() != 2
            or (outer.GetFormalCharge(), outer.GetExplicitDegree()) not in ((0, 2), (-1, 1))
        ):
            continue
        carbon = next(atom for atom in atom_neighbors(center) if atom.GetIdx() != outer.GetIdx())
        if (
            carbon.GetAtomicNum() != 6
            or carbon.GetFormalCharge() != 0
            or carbon.GetExplicitDegree() != 3
            or carbon.GetImplicitHCount()
            or get_unpaired_electron_count(carbon)
            or has_unresolved_two_electron_center(carbon)
            or neighbor_bond_order(center, carbon) != 2
            or not 1.15 < center.GetDistance(carbon) < 1.50
        ):
            continue
        if 1e-6 < center.GetParent().GetAngle(carbon, center, outer) < 150.0:
            return False
    return True


def short_azo_hydrazone_geometry_allowed(first: ob.OBAtom, second: ob.OBAtom, order: int) -> bool:
    """Screen C=N--N=C when short N--N and both long C--N support an azo unit.

    Inspect either proposed C=N bond so validation also sees a single N--N.
    Require resolved neutral C/N atoms, acyclic two-coordinate nitrogens and
    three-coordinate carbons with only C/H substituents. Additional heteroatom
    conjugation, including formazans, requires a larger local model. Joint lengths,
    rather than N--N alone, distinguish this endpoint from ordinary hydrazones.
    """
    if order != 2 or sorted((first.GetAtomicNum(), second.GetAtomicNum())) != [6, 7]:
        return True
    nitrogen = first if first.GetAtomicNum() == 7 else second
    for other in atom_neighbors(nitrogen):
        if (
            other.GetAtomicNum() != 7
            or neighbor_bond_order(nitrogen, other) != 1
            or not 1.20 < nitrogen.GetDistance(other) < 1.31
            or any(
                n.GetFormalCharge() != 0
                or n.GetExplicitDegree() != 2
                or n.IsInRing()
                or n.GetImplicitHCount()
                or get_unpaired_electron_count(n)
                or has_unresolved_two_electron_center(n)
                for n in (nitrogen, other)
            )
        ):
            continue
        carbons = [
            next(c for c in atom_neighbors(n) if c.GetIdx() != partner.GetIdx())
            for n, partner in ((nitrogen, other), (other, nitrogen))
        ]
        if all(
            c.GetAtomicNum() == 6
            and c.GetFormalCharge() == 0
            and c.GetExplicitDegree() == 3
            and not c.GetImplicitHCount()
            and not get_unpaired_electron_count(c)
            and not has_unresolved_two_electron_center(c)
            and all(
                neighbor.GetAtomicNum() in (1, 6)
                for neighbor in atom_neighbors(c)
                if neighbor.GetIdx() != n.GetIdx()
            )
            and neighbor_bond_order(n, c) == 2
            and 1.35 < n.GetDistance(c) < 1.50
            for n, c in zip((nitrogen, other), carbons)
        ):
            return False
    return True


def pyramidal_hydrazonium_geometry_allowed(first: ob.OBAtom, second: ob.OBAtom, order: int) -> bool:
    """Reject a long N=N+ at a strongly pyramidal dialkylamino nitrogen.

    This metal-free screen needs both length and pyramidalization. It abstains
    on rings, radicals, missing geometry and other substitution/charge states.
    A single N--N bond remains available without prescribing metal oxidation.
    """
    if order != 2 or first.GetAtomicNum() != 7 or second.GetAtomicNum() != 7:
        return True
    for donor, other in ((first, second), (second, first)):
        if (
            donor.GetFormalCharge() != 1
            or other.GetFormalCharge() != 0
            or donor.GetExplicitDegree() != 3
            or other.GetExplicitDegree() != 2
            or any(
                a.GetImplicitHCount()
                or get_unpaired_electron_count(a)
                or has_unresolved_two_electron_center(a)
                for a in (donor, other)
            )
            or not 1.38 < donor.GetDistance(other) < 1.60
            or donor.GetBond(other).IsInRing()
        ):
            continue
        neighbors = list(atom_neighbors(donor))
        carbons = [a for a in neighbors if a.GetIdx() != other.GetIdx()]
        if any(a.GetAtomicNum() != 6 or neighbor_bond_order(donor, a) != 1 for a in carbons):
            continue
        if any(not 1.20 < donor.GetDistance(a) < 1.70 for a in carbons):
            continue
        angles = [donor.GetParent().GetAngle(a, donor, b) for a, b in combinations(neighbors, 2)]
        if all(90 < angle < 125 for angle in angles) and 300 < sum(angles) < 340:
            return False
    return True


def nitrogen_nitrogen_order_allowed(first: ob.OBAtom, second: ob.OBAtom, order: int) -> bool:
    """Screen azo assignments contradicted by neighboring C--N/C--S lengths.

    Both nitrogens must have exactly two explicit neighbors and no implicit H.
    Require non-ring N--N at 1.30--1.50 A and a 1.20--1.35 A single C--N at each end.
    Ring N--N bonds abstain: this chain-local rule does not model cyclic delocalization.
    Long N--N alone is insufficient: ordinary azo groups, N--H substitution,
    and missing/degenerate geometry remain outside this local screen.
    A separate resolved, neutral C--N=N--C(=S) branch requires one short C--N
    (1.20--1.35 A) and a long terminal C=S (1.72--1.90 A). It leaves the
    N--N=C--S alternative available without prescribing its formal charges.
    """
    if not bent_diazonium_geometry_allowed(first, second, order):
        return False
    if not bent_cnn_cumulene_geometry_allowed(first, second, order):
        return False
    if not short_azo_hydrazone_geometry_allowed(first, second, order):
        return False
    if not pyramidal_hydrazonium_geometry_allowed(first, second, order):
        return False
    if (
        order != 2
        or any(
            atom.GetAtomicNum() != 7
            or atom.GetExplicitDegree() != 2
            or atom.GetImplicitHCount() != 0
            for atom in (first, second)
        )
        or not 1.30 < first.GetDistance(second) < 1.50
    ):
        return True
    bond = cast(Optional[ob.OBBond], first.GetBond(second))
    if bond is None or bond.IsInRing():
        return True
    if all(
        any(
            neighbor.GetAtomicNum() == 6
            and neighbor_bond_order(atom, neighbor) == 1
            and 1.20 < atom.GetDistance(neighbor) < 1.35
            for neighbor in atom_neighbors(atom)
        )
        for atom in (first, second)
    ):
        return False
    if any(
        atom.GetFormalCharge() != 0
        or get_unpaired_electron_count(atom)
        or has_unresolved_two_electron_center(atom)
        or not any(n.GetAtomicNum() == 6 for n in atom_neighbors(atom))
        for atom in (first, second)
    ):
        return True
    for nitrogen in (first, second):
        for carbon in atom_neighbors(nitrogen):
            if (
                carbon.GetAtomicNum() != 6
                or carbon.GetFormalCharge() != 0
                or carbon.GetExplicitDegree() != 3
                or carbon.GetImplicitHCount() != 0
                or get_unpaired_electron_count(carbon)
                or has_unresolved_two_electron_center(carbon)
                or neighbor_bond_order(nitrogen, carbon) != 1
                or not 1.20 < nitrogen.GetDistance(carbon) < 1.35
            ):
                continue
            if any(
                sulfur.GetAtomicNum() == 16
                and sulfur.GetFormalCharge() == 0
                and sulfur.GetExplicitDegree() == 1
                and sulfur.GetImplicitHCount() == 0
                and not get_unpaired_electron_count(sulfur)
                and not has_unresolved_two_electron_center(sulfur)
                and neighbor_bond_order(carbon, sulfur) == 2
                and 1.72 < carbon.GetDistance(sulfur) < 1.90
                for sulfur in atom_neighbors(carbon)
            ):
                return False
    return True


def nitronium_geometry_allowed(nitrogen: ob.OBAtom) -> bool:
    """Exclude bent, resolved closed-shell O=N(+)=O at normal N--O lengths.

    Nitrite, N substitution, open-shell/deferred labels and missing geometry
    abstain. No metal positions are inferred in the metal-free graph.
    """
    if (
        nitrogen.GetAtomicNum() != 7
        or nitrogen.GetFormalCharge() != 1
        or nitrogen.GetExplicitDegree() != 2
        or nitrogen.GetImplicitHCount() != 0
    ):
        return True
    oxygens = list(atom_neighbors(nitrogen))
    if any(
        oxygen.GetAtomicNum() != 8
        or oxygen.GetFormalCharge() != 0
        or oxygen.GetExplicitDegree() != 1
        or oxygen.GetImplicitHCount() != 0
        or neighbor_bond_order(nitrogen, oxygen) != 2
        or not 1.05 < nitrogen.GetDistance(oxygen) < 1.40
        for oxygen in oxygens
    ):
        return True
    if any(
        get_unpaired_electron_count(atom) or has_unresolved_two_electron_center(atom)
        for atom in (nitrogen, *oxygens)
    ):
        return True
    # Open Babel reports a small nonzero angle for coincident directions.
    if oxygens[0].GetDistance(oxygens[1]) < 1e-6:
        return True
    angle = nitrogen.GetParent().GetAngle(oxygens[0], nitrogen, oxygens[1])
    return not 0.0 < angle < 150


def nitrogen_oxygen_order_allowed(first: ob.OBAtom, second: ob.OBAtom, order: int) -> bool:
    """Screen bent nitronium and nitroso C--N=O with conflicting geometry.

    Require terminal O or O bound to a four-single-bond borate cap,
    two-coordinate N, a 1.30--1.50 A N--O, and a 1.20--1.34 A single C--N.
    Isolated NO, O--H, other N substituents, and missing geometry abstain.
    Charge/spin validation chooses the eventual electronic assignment;
    this screen only excludes the conflicting bond order.
    """
    if order != 2:
        return True
    nitrogen, oxygen = (first, second) if first.GetAtomicNum() == 7 else (second, first)
    if not nitronium_geometry_allowed(nitrogen):
        return False
    if (
        nitrogen.GetAtomicNum() != 7
        or oxygen.GetAtomicNum() != 8
        or nitrogen.GetExplicitDegree() != 2
        or nitrogen.GetImplicitHCount() != 0
        or oxygen.GetImplicitHCount() != 0
        or not 1.30 < nitrogen.GetDistance(oxygen) < 1.50
    ):
        return True
    borate_capped = oxygen.GetExplicitDegree() == 2 and any(
        neighbor.GetAtomicNum() == 5
        and neighbor.GetExplicitDegree() == 4
        and neighbor_bond_order(oxygen, neighbor) == 1
        and all(bond.GetBondOrder() == 1 for bond in atom_bonds(neighbor))
        for neighbor in atom_neighbors(oxygen)
    )
    if oxygen.GetExplicitDegree() != 1 and not borate_capped:
        return True
    for carbon in atom_neighbors(nitrogen):
        if carbon.GetAtomicNum() != 6 or not 1.20 < nitrogen.GetDistance(carbon) < 1.34:
            continue
        cn_order = neighbor_bond_order(nitrogen, carbon)
        if cn_order == 1:
            return False
        if (
            cn_order == 2
            and carbon.GetFormalCharge() == oxygen.GetFormalCharge() == 0
            and oxygen.GetExplicitDegree() == 1
            and nitrogen.GetFormalCharge() == 1
            and nitrogen.GetTotalValence() == 4
            and 0 < nitrogen.GetParent().GetAngle(carbon, nitrogen, oxygen) < 150
        ):
            return False
    return True


def paired_carbonyl_ketene_geometry_allowed(carbon: ob.OBAtom, oxygen: ob.OBAtom) -> bool:
    """Screen a bent O=C--C=C=O assignment with a conflicting ring carbonyl.

    Require resolved neutral labels, a terminal ketene C=O at 1.15--1.35 A,
    C=C at 1.44--1.65 A and C--C--O angle 90--150 degrees. The adjacent ring
    C--C (1.38--1.45 A) must bear a long terminal C=O (1.30--1.42 A). Bending
    alone is insufficient: other metal-bound ketenes remain outside this screen.
    """

    def resolved(atom: ob.OBAtom) -> bool:
        """Require a neutral center with explicit hydrogen counts and no unresolved electrons."""
        return not (
            atom.GetFormalCharge()
            or atom.GetImplicitHCount()
            or get_unpaired_electron_count(atom)
            or has_unresolved_two_electron_center(atom)
        )

    if (
        carbon.GetAtomicNum() != 6
        or oxygen.GetAtomicNum() != 8
        or carbon.GetExplicitDegree() != 2
        or oxygen.GetExplicitDegree() != 1
        or not resolved(carbon)
        or not resolved(oxygen)
        or not 1.15 < carbon.GetDistance(oxygen) < 1.35
    ):
        return True
    for middle in atom_neighbors(carbon):
        if (
            middle.GetAtomicNum() != 6
            or middle.GetExplicitDegree() != 3
            or not resolved(middle)
            or not middle.IsInRing()
            or neighbor_bond_order(carbon, middle) != 2
            or not 1.44 < carbon.GetDistance(middle) < 1.65
            or not 90 < carbon.GetParent().GetAngle(middle, carbon, oxygen) < 150
        ):
            continue
        for other in atom_neighbors(middle):
            if (
                other.GetIdx() == carbon.GetIdx()
                or other.GetAtomicNum() != 6
                or other.GetExplicitDegree() != 3
                or not resolved(other)
                or not other.IsInRing()
                or not middle.GetBond(other).IsInRing()
                or neighbor_bond_order(middle, other) != 1
                or not 1.38 < middle.GetDistance(other) < 1.45
            ):
                continue
            if any(
                terminal.GetAtomicNum() == 8
                and terminal.GetExplicitDegree() == 1
                and resolved(terminal)
                and neighbor_bond_order(other, terminal) == 2
                and 1.30 < other.GetDistance(terminal) < 1.42
                for terminal in atom_neighbors(other)
            ):
                return False
    return True


def silicon_carbon_order_allowed(first: ob.OBAtom, second: ob.OBAtom, order: int) -> bool:
    """Exclude long Si=C candidates with a pyramidal, carbon/hydrogen-only Si.

    This metal-free screen requires neutral, resolved, three-coordinate Si and
    C, a non-ring 1.88--2.10 A bond, and three Si angles of 80--140 degrees
    summing to less than 335 degrees. It does not assign a silyl charge or a
    metal oxidation state. Short silenes may be pyramidal, so neither length
    nor pyramidalization alone is sufficient. Heteroatom substitution, rings,
    hypercoordination, unresolved electrons, and missing geometry abstain.
    """
    if order != 2 or sorted((first.GetAtomicNum(), second.GetAtomicNum())) != [6, 14]:
        return True
    silicon, carbon = (first, second) if first.GetAtomicNum() == 14 else (second, first)
    if any(
        atom.GetFormalCharge() != 0
        or atom.GetExplicitDegree() != 3
        or atom.GetImplicitHCount()
        or get_unpaired_electron_count(atom)
        or has_unresolved_two_electron_center(atom)
        for atom in (silicon, carbon)
    ):
        return True
    if not 1.88 < silicon.GetDistance(carbon) < 2.10:
        return True
    bond = cast(Optional[ob.OBBond], silicon.GetBond(carbon))
    if bond is None or bond.IsInRing():
        return True
    neighbors = list(atom_neighbors(silicon))
    if any(atom.GetAtomicNum() not in (1, 6) for atom in neighbors):
        return True
    if any(
        atom.GetAtomicNum() not in (1, 6)
        for atom in atom_neighbors(carbon)
        if atom.GetIdx() != silicon.GetIdx()
    ):
        return True
    angles = [silicon.GetParent().GetAngle(a, silicon, b) for a, b in combinations(neighbors, 2)]
    return not (all(80 < angle < 140 for angle in angles) and sum(angles) < 335)


def twisted_ring_link_order_allowed(first: ob.OBAtom, second: ob.OBAtom, order: int) -> bool:
    """Exclude a long, strongly twisted C=C link to an equalized carbon ring.

    Both neutral, resolved carbon centers must have three explicit neighbors
    and belong to rings; the connecting bond itself must be outside a ring.
    Require a 1.47--1.65 A link, all four substituent torsions 45--90 degrees
    from planar, and a planar C6 ring with nearly equal 1.35--1.45 A bonds.
    This bounds the rule to a biaryl-like single-bond geometry. Short, planar,
    charged, open-shell, strained ring bonds and missing geometry abstain.
    """
    if order != 2 or first.GetAtomicNum() != 6 or second.GetAtomicNum() != 6:
        return True
    if any(
        atom.GetExplicitDegree() != 3
        or not atom.IsInRing()
        or atom.GetImplicitHCount()
        or atom.GetFormalCharge()
        or get_unpaired_electron_count(atom)
        or has_unresolved_two_electron_center(atom)
        for atom in (first, second)
    ):
        return True
    bond = cast(Optional[ob.OBBond], first.GetBond(second))
    if bond is None or bond.IsInRing() or not 1.47 < first.GetDistance(second) < 1.65:
        return True
    mol = cast(ob.OBMol, first.GetParent())
    groups = [
        [n for n in atom_neighbors(atom) if n.GetIdx() != other.GetIdx()]
        for atom, other in ((first, second), (second, first))
    ]
    if any(n.GetAtomicNum() != 6 for neighbors in groups for n in neighbors):
        return True
    if any(
        not 90 < mol.GetAngle(neighbors[0], atom, neighbors[1]) < 150
        for atom, neighbors in zip((first, second), groups)
    ):
        return True
    torsions = [
        abs(mol.GetTorsion(a.GetIdx(), first.GetIdx(), second.GetIdx(), b.GetIdx()))
        for a in groups[0]
        for b in groups[1]
    ]
    if not all(math.isfinite(t) and 45 < min(t, abs(180 - t)) <= 90 for t in torsions):
        return True
    for perimeter in ring_paths(mol):
        ids = list(perimeter)
        if len(ids) != 6 or (first.GetIdx() in ids) == (second.GetIdx() in ids):
            continue
        atoms = [cast(ob.OBAtom, mol.GetAtom(i)) for i in ids]
        if any(
            a.GetAtomicNum() != 6 or a.GetExplicitDegree() != 3 or a.GetImplicitHCount()
            for a in atoms
        ):
            continue
        lengths = [atoms[i].GetDistance(atoms[(i + 1) % 6]) for i in range(6)]
        torsions = [abs(mol.GetTorsion(*(ids[(i + k) % 6] for k in range(4)))) for i in range(6)]
        if (
            all(1.35 < d < 1.45 for d in lengths)
            and max(lengths) - min(lengths) < 0.06
            and all(math.isfinite(t) and min(t, abs(180 - t)) < 15 for t in torsions)
        ):
            return False
    return True


def _resolved_explicit_atom(atom: ob.OBAtom) -> bool:
    """Require explicit hydrogens and a resolved local spin assignment."""
    return not (
        atom.GetImplicitHCount()
        or get_unpaired_electron_count(atom)
        or has_unresolved_two_electron_center(atom)
    )


def _neighbor_angles(center: ob.OBAtom, neighbors: Sequence[ob.OBAtom]) -> list[float]:
    """Return the substituent–center–substituent angles in degrees."""
    mol = cast(ob.OBMol, center.GetParent())
    return [mol.GetAngle(a, center, b) for a, b in combinations(neighbors, 2)]


def _planar_iminium_witness(carbon: ob.OBAtom, nitrogen: ob.OBAtom) -> bool:
    """Look for a short N–C bond to a planar carbanion that can delocalize the iminium charge."""
    for other in atom_neighbors(nitrogen):
        if (
            other.GetIdx() == carbon.GetIdx()
            or (other.GetAtomicNum(), other.GetFormalCharge()) != (6, -1)
            or (other.GetExplicitDegree(), other.GetTotalValence()) != (3, 3)
            or not _resolved_explicit_atom(other)
            or neighbor_bond_order(nitrogen, other) != 1
            or not 1.15 < nitrogen.GetDistance(other) < 1.40
        ):
            continue
        neighbors = list(atom_neighbors(other))
        angles = _neighbor_angles(other, neighbors)
        if (
            all(not a.IsMetal() for a in neighbors)
            and all(90 < a < 150 for a in angles)
            and 345 <= sum(angles) <= 360.000001
        ):
            return True
    return False


def cationic_carbon_double_bond_allowed(carbon: ob.OBAtom, heteroatom: ob.OBAtom) -> bool:
    """Validate a long C=N+ or C=O+ against its substituents and local angles."""
    z = heteroatom.GetAtomicNum()
    if (
        carbon.GetAtomicNum() != 6
        or z not in (7, 8)
        or carbon.GetFormalCharge() != 0
        or heteroatom.GetFormalCharge() != 1
        or carbon.GetTotalValence() != 4
        or carbon.IsInRing()
        or not _resolved_explicit_atom(carbon)
        or not _resolved_explicit_atom(heteroatom)
    ):
        return True
    if z == 8 and (
        heteroatom.IsInRing()
        or heteroatom.GetExplicitDegree() != 2
        or heteroatom.GetTotalValence() != 3
    ):
        return True
    if z == 7 and (heteroatom.GetExplicitDegree() != 3 or heteroatom.GetTotalValence() != 4):
        return True
    minimum = 1.35 if z == 8 else 1.40
    if not minimum < carbon.GetDistance(heteroatom) < 1.65:
        return True
    neighbors = list(atom_neighbors(carbon))
    others = [a for a in neighbors if a.GetIdx() != heteroatom.GetIdx()]
    if len(neighbors) == 2 and z == 7:
        partner = others[0]
        if (
            partner.GetAtomicNum() != 6
            or partner.GetFormalCharge()
            or partner.GetTotalValence() != 4
            or not _resolved_explicit_atom(partner)
            or neighbor_bond_order(carbon, partner) != 2
            or not 1.15 < carbon.GetDistance(partner) < 1.60
            or not 1e-6 < carbon.GetParent().GetAngle(heteroatom, carbon, partner) < 150
        ):
            return True
    elif len(neighbors) == 3 and all(a.GetAtomicNum() in (1, 6) for a in others):
        angles = _neighbor_angles(carbon, neighbors)
        if not all(90 < a < 130 for a in angles) or not 300 < sum(angles) < 345:
            return True
    else:
        return True
    return z == 7 and _planar_iminium_witness(carbon, heteroatom)


def carbon_oxygen_order_allowed(first: ob.OBAtom, second: ob.OBAtom, order: int) -> bool:
    """Screen scoped C--O multiple-bond patterns using explicit local geometry.

    A substituted acylium triple bond requires a near-linear carbon center;
    a bent center with a 1.05--1.60 A C--O distance cannot use that candidate.
    Include short carbonyl distances: a short C--O axis does not establish
    the linear geometry required by a substituted triple-bonded carbon.
    A substituted oxygen at 1.42--1.65 A cannot form a C=O bond. Isolated CO,
    ordinary terminal carbonyls, and missing/degenerate coordinates abstain.
    The paired ring-carbonyl/ketene screen additionally requires the complete
    local motif; bending alone does not exclude a metal-bound ketene. Geometry
    does not assign formal charges or metal valences.
    """
    carbon, oxygen = (first, second) if first.GetAtomicNum() == 6 else (second, first)
    if carbon.GetAtomicNum() != 6 or oxygen.GetAtomicNum() != 8:
        return True
    if order == 2 and (
        not cationic_carbon_double_bond_allowed(carbon, oxygen)
        or not paired_carbonyl_ketene_geometry_allowed(carbon, oxygen)
    ):
        return False
    distance = carbon.GetDistance(oxygen)
    if order == 2 and oxygen.GetExplicitDegree() == 2 and 1.42 < distance < 1.65:
        return False
    if (
        order == 3
        and carbon.GetExplicitDegree() > 1
        and oxygen.GetExplicitDegree() == 1
        and 1.05 < distance < 1.60
    ):
        for neighbor in atom_neighbors(carbon):
            if neighbor.GetIdx() == oxygen.GetIdx() or carbon.GetDistance(neighbor) <= 0.0:
                continue
            angle = carbon.GetParent().GetAngle(neighbor, carbon, oxygen)
            if 0.0 < angle < 150.0:
                return False
    return True


def would_form_bent_ring_allene(center: ob.OBAtom, partner: ob.OBAtom) -> bool:
    """Reject a new ring pi bond that would put a bent C=C=C center in a ring."""
    if center.GetAtomicNum() != 6 or not center.IsInRing() or not partner.IsInRing():
        return False
    for bond in atom_bonds(center):
        neighbor = cast(ob.OBAtom, bond.GetNbrAtom(center))
        if (
            neighbor.GetIdx() != partner.GetIdx()
            and neighbor.IsInRing()
            and bond.GetBondOrder() == 2
            and 0.0 < center.GetParent().GetAngle(neighbor, center, partner) < 150.0
        ):
            return True
    return False


def is_elongated_carbonyl_pair(first: ob.OBAtom, second: ob.OBAtom) -> bool:
    """Identify stretched aldehyde/ketone C--O sites, independent of charge.

    Other carbon substituents must be H/C joined by single bonds. This excludes
    CO, CO2, esters, amides and phenoxide/enolate sites. The 1.35--1.60 A window
    is a candidate-selection heuristic, not an oxidation-state determination.
    """
    carbon, oxygen = (first, second) if first.GetAtomicNum() == 6 else (second, first)
    if (
        carbon.GetAtomicNum() != 6
        or oxygen.GetAtomicNum() != 8
        or carbon.GetExplicitDegree() + carbon.GetImplicitHCount() != 3
        or oxygen.GetExplicitDegree() != 1
        or oxygen.GetImplicitHCount() != 0
        or not 1.35 <= carbon.GetDistance(oxygen) <= 1.60
    ):
        return False
    return all(
        neighbor.GetAtomicNum() in (1, 6) and neighbor_bond_order(carbon, neighbor) == 1
        for neighbor in atom_neighbors(carbon)
        if neighbor.GetIdx() != oxygen.GetIdx()
    )


def _compact_planar_cn_six_rings(
    mol: ob.OBMol,
) -> Iterator[tuple[tuple[int, ...], list[ob.OBAtom]]]:
    """Geometry screen for the azinium preference, excluding five-membered NHCs."""
    for perimeter in ring_paths(mol):
        ids = tuple(perimeter)
        if len(ids) != 6:
            continue
        atoms = [cast(ob.OBAtom, mol.GetAtom(i)) for i in ids]
        if any(a.GetAtomicNum() not in (6, 7) for a in atoms):
            continue
        if not all(1.25 <= atoms[i].GetDistance(atoms[(i + 1) % 6]) <= 1.44 for i in range(6)):
            continue
        torsions = [abs(mol.GetTorsion(*(ids[(i + k) % 6] for k in range(4)))) for i in range(6)]
        if any(not math.isfinite(t) or min(t, 180.0 - t) > 15.0 for t in torsions):
            continue
        yield ids, atoms


def restores_azinium_ring(nitrogen: ob.OBAtom, radical: ob.OBAtom) -> bool:
    """Whether donating the N lone pair completes an alternating six-membered ring."""
    if nitrogen.GetAtomicNum() != 7 or nitrogen.GetTotalValence() != 3:
        return False
    mol = cast(ob.OBMol, nitrogen.GetParent())
    pair = {nitrogen.GetIdx(), radical.GetIdx()}
    for ids, _ in _compact_planar_cn_six_rings(mol):
        if not pair.issubset(ids):
            continue
        orders = [
            indexed_bond_order(mol, ids[i], ids[(i + 1) % 6]) + ({ids[i], ids[(i + 1) % 6]} == pair)
            for i in range(6)
        ]
        if orders in ([1, 2, 1, 2, 1, 2], [2, 1, 2, 1, 2, 1]):
            return True
    return False


def azinium_geometry_conflict_count(mol: ob.OBMol) -> int:
    """Count compact planar reduced azinium rings competing with aromatic candidates.

    A neutral tertiary ring N and a ring carbanion with two ring pi bonds form
    the narrow reduced-azinium pattern. Real reduced/nonplanar or long-bond rings
    and five-membered NHCs are outside this geometric selection preference.
    """
    count = 0
    for ids, atoms in _compact_planar_cn_six_rings(mol):
        charges = [a.GetFormalCharge() for a in atoms]
        if sorted(charges) != [-1, 0, 0, 0, 0, 0]:
            continue
        if not any(a.GetAtomicNum() == 6 and a.GetFormalCharge() == -1 for a in atoms):
            continue
        orders = [indexed_bond_order(mol, ids[i], ids[(i + 1) % 6]) for i in range(6)]
        if sorted(orders) != [1, 1, 1, 1, 2, 2]:
            continue
        if any(
            a.GetAtomicNum() == 7
            and a.GetFormalCharge() == 0
            and a.GetTotalValence() == 3
            and a.GetExplicitDegree() + a.GetImplicitHCount() == 3
            and orders[i] == orders[(i - 1) % 6] == 1
            for i, a in enumerate(atoms)
        ):
            count += 1
    return count


def is_short_terminal_carbonyl(carbon: ob.OBAtom, oxygen: ob.OBAtom) -> bool:
    """Strong C=O geometry evidence; exclude isolated CO and substituted O."""
    return (
        carbon.GetAtomicNum() == 6
        and carbon.GetExplicitDegree() >= 2
        and oxygen.GetAtomicNum() == 8
        and oxygen.GetExplicitDegree() == 1
        and 1.10 < carbon.GetDistance(oxygen) < 1.30
    )


def protect_short_carbonyl(carbon: ob.OBAtom, oxygen: ob.OBAtom, alpha: ob.OBAtom) -> bool:
    """Preserve a short carbonyl beside a terminal S or diazo carbon center.

    This does not impose a universal keto/enolate preference on other ligands.
    """
    if not is_short_terminal_carbonyl(carbon, oxygen) or alpha.GetAtomicNum() != 6:
        return False
    if (
        not carbon.IsInRing()
        and carbon.GetDistance(oxygen) < 1.26
        and 1.47 < carbon.GetDistance(alpha) < 1.65
        and 100 < carbon.GetParent().GetAngle(oxygen, carbon, alpha) < 150
        and carbon.GetFormalCharge() == oxygen.GetFormalCharge() == 0
        and alpha.GetFormalCharge() in (0, -1)
        and not any(a.GetImplicitHCount() for a in (carbon, oxygen, alpha))
    ):
        return True
    for neighbor in atom_neighbors(alpha):
        if neighbor.GetAtomicNum() == 16 and neighbor.GetExplicitDegree() == 1:
            return True
        if (
            neighbor.GetAtomicNum() == 7
            and neighbor.GetExplicitDegree() == 2
            and any(
                a.GetAtomicNum() == 7 and a.GetExplicitDegree() == 1
                for a in atom_neighbors(neighbor)
            )
        ):
            return True
    return False


def phosphorus_carbon_order_allowed(first: ob.OBAtom, second: ob.OBAtom, order: int) -> bool:
    """Screen a long P+=C at an acyclic, pyramidal three-carbon P center.

    Joint length and angle evidence is required. This is not a universal P=C
    length limit: rings, heteroatom substituents, open shells and C-=P+ charge
    separation abstain. Metal-free geometry suffices; no oxidation state or
    donor penalty is assigned here.
    """
    phosphorus, carbon = (first, second) if first.GetAtomicNum() == 15 else (second, first)
    if order != 2 or phosphorus.GetAtomicNum() != 15 or carbon.GetAtomicNum() != 6:
        return True
    if (
        phosphorus.GetFormalCharge() != 1
        or phosphorus.GetExplicitDegree() != 3
        or phosphorus.GetTotalValence() != 4
        or phosphorus.IsInRing()
        or carbon.GetFormalCharge() != 0
        or any(
            atom.GetImplicitHCount()
            or get_unpaired_electron_count(atom)
            or has_unresolved_two_electron_center(atom)
            for atom in (phosphorus, carbon)
        )
    ):
        return True
    neighbors = list(atom_neighbors(phosphorus))
    if any(atom.GetAtomicNum() != 6 for atom in neighbors):
        return True
    if not 1.78 < phosphorus.GetDistance(carbon) < 2.10:
        return True
    angles = [
        phosphorus.GetParent().GetAngle(left, phosphorus, right)
        for left, right in combinations(neighbors, 2)
    ]
    return not (all(80 < angle < 130 for angle in angles) and 300 < sum(angles) < 330)


def nitrilium_geometry_allowed(first: ob.OBAtom, second: ob.OBAtom, order: int) -> bool:
    """Abstain except for long, acyclic nitrilium triples bent at both ends."""
    carbon, nitrogen = (first, second) if first.GetAtomicNum() == 6 else (second, first)
    if order != 3 or carbon.GetAtomicNum() != 6 or nitrogen.GetAtomicNum() != 7:
        return True
    if carbon.GetFormalCharge() != 0 or nitrogen.GetFormalCharge() != 1:
        return True
    if any(
        a.GetExplicitDegree() != 2
        or a.IsInRing()
        or a.GetImplicitHCount()
        or get_unpaired_electron_count(a)
        or has_unresolved_two_electron_center(a)
        for a in (carbon, nitrogen)
    ):
        return True
    if not 1.26 < carbon.GetDistance(nitrogen) < 1.50:
        return True
    angles = []
    for center, other in ((carbon, nitrogen), (nitrogen, carbon)):
        neighbor = next(a for a in atom_neighbors(center) if a.GetIdx() != other.GetIdx())
        if neighbor.GetAtomicNum() != 6 or center.GetDistance(neighbor) <= 0:
            return True
        angles.append(center.GetParent().GetAngle(neighbor, center, other))
    return not all(0 < angle < 150 for angle in angles)


def has_conjugated_thioamide_donor(carbon: ob.OBAtom) -> bool:
    """Recognize a short C–N single bond to a neutral, closed-shell trivalent N."""
    return any(
        n.GetAtomicNum() == 7
        and n.GetFormalCharge() == 0
        and n.GetExplicitDegree() == n.GetTotalValence() == 3
        and not n.GetImplicitHCount()
        and not get_unpaired_electron_count(n)
        and not has_unresolved_two_electron_center(n)
        and neighbor_bond_order(carbon, n) == 1
        and 1.30 < carbon.GetDistance(n) < 1.42
        for n in atom_neighbors(carbon)
    )


def short_ortho_quinodiimine_sites(mol: ob.OBMol) -> Iterator[tuple[int, ...]]:
    """Yield oriented six-carbon rings followed by the two exocyclic N indices."""
    for perimeter in ring_paths(mol):
        ids = tuple(perimeter)
        if len(ids) != 6:
            continue
        atoms = [cast(ob.OBAtom, mol.GetAtom(i)) for i in ids]
        if any(
            a.GetAtomicNum() != 6 or a.GetExplicitDegree() != 3 or a.GetImplicitHCount()
            for a in atoms
        ):
            continue
        lengths = [a.GetDistance(atoms[(j + 1) % 6]) for j, a in enumerate(atoms)]
        if not all(1.34 < length < 1.50 for length in lengths):
            continue
        torsions = [abs(mol.GetTorsion(*(ids[(j + k) % 6] for k in range(4)))) for j in range(6)]
        if not all(min(t, abs(180 - t)) < 15 for t in torsions):
            continue
        for j, length in enumerate(lengths):
            if length <= 1.43:
                continue
            path = ids[j:] + ids[:j]
            edges = lengths[j:] + lengths[:j]
            if any(
                edges[k] >= 1.41 or min(edges[k - 1], edges[k + 1]) - edges[k] <= 0.015
                for k in (2, 4)
            ):
                continue
            if path[0] > path[1]:
                path = (path[1], path[0], *path[:1:-1])
            nitrogens = []
            for idx in path[:2]:
                carbon = cast(ob.OBAtom, mol.GetAtom(idx))
                n = next((a for a in atom_neighbors(carbon) if a.GetIdx() not in path), None)
                if (
                    n is None
                    or n.GetAtomicNum() != 7
                    or n.GetExplicitDegree() != 2
                    or n.GetImplicitHCount()
                    or not 1.27 < n.GetDistance(carbon) < 1.35
                ):
                    break
                nitrogens.append(n.GetIdx())
            if len(nitrogens) == 2:
                yield (*path, *nitrogens)


def short_quinodiimine_conflicts(mol: ob.OBMol) -> int:
    """Count quinodiimine sites with two short C–N bonds both assigned as reduced or polarized outward."""
    count = 0
    for site in short_ortho_quinodiimine_sites(mol):
        states = []
        for carbon_idx, n_idx in zip(site[:2], site[6:]):
            n, carbon = (
                cast(ob.OBAtom, mol.GetAtom(n_idx)),
                cast(ob.OBAtom, mol.GetAtom(carbon_idx)),
            )
            if (
                get_unpaired_electron_count(n)
                or has_unresolved_two_electron_center(n)
                or neighbor_bond_order(n, carbon) != 1
            ):
                break
            if n.GetFormalCharge() == -1 and n.GetTotalValence() == 2:
                states.append("reduced")
            elif (
                n.GetFormalCharge() == 0
                and n.GetTotalValence() == 3
                and any(
                    a.GetAtomicNum() == 6
                    and neighbor_bond_order(n, a) == 2
                    and n.GetDistance(a) - n.GetDistance(carbon) > 0.02
                    for a in atom_neighbors(n)
                )
            ):
                states.append("outward")
        count += len(states) == 2 and states[0] == states[1]
    return count
