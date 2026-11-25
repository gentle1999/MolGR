from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations, product
from typing import Dict, Iterable, Set, Tuple, cast

from openbabel import openbabel as ob
from openbabel import pybel

from molgr.config import CONFIG, OrganicTopologyConfig
from molgr.fallback.utils.consts import NON_METAL_DICT
from molgr.fallback.utils.electrons import (
    get_lone_pair_count,
    get_unpaired_electron_count,
    has_unresolved_two_electron_center,
)
from molgr.fallback.utils.ob_graph import (
    atom_bonds,
    atom_neighbors,
    mol_atoms,
    mol_bonds,
    mol_rings,
    neighbor_bond_order,
)


_AROMATIC_RING_FORMAL_CHARGE_ABS_REJECTION_THRESHOLD = 4


@dataclass(frozen=True)
class OrganicTopologyMetrics:
    """Aromatic, conjugated and hyperconjugative features used to compare ligand electronic assignments."""

    aromatic_atom_count: int
    aromatic_ring_count: int
    aromatic_stability_score: float
    conjugated_atom_count: int
    conjugated_bond_count: int
    max_conjugated_component_size: int
    conjugated_atom_indices: Tuple[int, ...]
    hyperconjugative_donor_count: int
    hyperconjugation_score: int
    # Zero-based atom sets; orthogonal cumulene pi systems remain separate.
    conjugated_atom_components: Tuple[Tuple[int, ...], ...] = ()


def _prepare_topology_working_molecule(omol: pybel.Molecule) -> pybel.Molecule:
    working = cast(pybel.Molecule, omol.clone)
    working_obmol = cast(ob.OBMol, working.OBMol)
    working_obmol.FindRingAtomsAndBonds()
    working_obmol.SetAromaticPerceived(False)
    ob.OBAromaticTyper().AssignAromaticFlags(working_obmol)
    working_obmol.SetHybridizationPerceived(False)
    for atom in mol_atoms(working_obmol):
        atom.GetHyb()
    return working


def _ring_atom_indices(ring: ob.OBRing) -> Tuple[int, ...]:
    return tuple(int(atom_idx) for atom_idx in getattr(ring, "_path", ()))


def _ring_formal_charge_sum(obmol: ob.OBMol, ring: ob.OBRing) -> int:
    charge_sum = 0
    for atom_idx in _ring_atom_indices(ring):
        atom = cast("ob.OBAtom | None", obmol.GetAtom(atom_idx))
        if atom is None:
            continue
        charge_sum += cast(int, atom.GetFormalCharge())
    return charge_sum


def _is_charge_accepted_aromatic_ring(obmol: ob.OBMol, ring: ob.OBRing) -> bool:
    if not ring.IsAromatic():
        return False
    return (
        abs(_ring_formal_charge_sum(obmol, ring))
        < _AROMATIC_RING_FORMAL_CHARGE_ABS_REJECTION_THRESHOLD
    )


def _ring_bond(obmol: ob.OBMol, begin_idx: int, end_idx: int) -> ob.OBBond | None:
    return cast("ob.OBBond | None", obmol.GetBond(begin_idx, end_idx))


def _additional_atom_pi_electrons(atom: ob.OBAtom, *, incident_to_ring_multiple_bond: bool) -> int:
    if incident_to_ring_multiple_bond:
        return 0
    atomic_num = int(atom.GetAtomicNum())
    if atomic_num == 1:
        return 0
    if int(atom.GetHyb()) != 2:
        return 0
    if int(atom.GetFormalCharge()) < 0:
        return 2
    if atomic_num != 6:
        return 2
    if _atom_has_unpaired_electrons(atom):
        return 1
    return 0


def _ring_pi_electron_count(obmol: ob.OBMol, ring: ob.OBRing) -> int | None:
    """Sum ring pi bonds and eligible atomic contributions without counting a pi-bonded atom twice."""
    ring_atom_indices = _ring_atom_indices(ring)
    if len(ring_atom_indices) < 3:
        return None

    pi_electron_count = 0
    atoms_incident_to_ring_multiple_bond: Set[int] = set()
    for offset, begin_idx in enumerate(ring_atom_indices):
        end_idx = ring_atom_indices[(offset + 1) % len(ring_atom_indices)]
        bond = _ring_bond(obmol, begin_idx, end_idx)
        if bond is None:
            return None
        if int(bond.GetBondOrder()) >= 2:
            pi_electron_count += 2
            atoms_incident_to_ring_multiple_bond.add(begin_idx)
            atoms_incident_to_ring_multiple_bond.add(end_idx)

    for atom_idx in ring_atom_indices:
        atom = cast("ob.OBAtom | None", obmol.GetAtom(atom_idx))
        if atom is None:
            continue
        pi_electron_count += _additional_atom_pi_electrons(
            atom,
            incident_to_ring_multiple_bond=atom_idx in atoms_incident_to_ring_multiple_bond,
        )
    return pi_electron_count


def _has_huckel_pi_electron_count(pi_electron_count: int | None) -> bool:
    return (
        pi_electron_count is not None
        and pi_electron_count >= 2
        and (pi_electron_count - 2) % 4 == 0
    )


def _is_huckel_accepted_aromatic_ring(obmol: ob.OBMol, ring: ob.OBRing) -> bool:
    return _is_charge_accepted_aromatic_ring(obmol, ring) and _has_huckel_pi_electron_count(
        _ring_pi_electron_count(obmol, ring)
    )


def _rings_share_fused_bond(lhs: ob.OBRing, rhs: ob.OBRing) -> bool:
    return len(set(_ring_atom_indices(lhs)).intersection(_ring_atom_indices(rhs))) >= 2


def _aromatic_ring_systems(obmol: ob.OBMol) -> list[list[ob.OBRing]]:
    sssr_rings = mol_rings(obmol)
    aromatic_rings = [ring for ring in sssr_rings if ring.IsAromatic()]
    systems: list[list[ob.OBRing]] = []
    assigned: Set[int] = set()
    for ring_index, ring in enumerate(aromatic_rings):
        if ring_index in assigned:
            continue
        assigned.add(ring_index)
        system = [ring]
        system_index = 0
        while system_index < len(system):
            system_ring = system[system_index]
            for candidate_index, candidate_ring in enumerate(aromatic_rings):
                if candidate_index in assigned or not _rings_share_fused_bond(
                    system_ring, candidate_ring
                ):
                    continue
                assigned.add(candidate_index)
                system.append(candidate_ring)
            system_index += 1
        systems.append(system)
    return systems


def _ring_system_atom_indices(rings: Iterable[ob.OBRing]) -> Set[int]:
    return {atom_idx for ring in rings for atom_idx in _ring_atom_indices(ring)}


def _is_accepted_aromatic_ring_system(obmol: ob.OBMol, rings: list[ob.OBRing]) -> bool:
    """Check isolated rings by electron count and fused aromatic systems by their net formal charge."""
    if not rings:
        return False
    if len(rings) == 1:
        return _is_huckel_accepted_aromatic_ring(obmol, rings[0])
    charge_sum = sum(
        int(atom.GetFormalCharge())
        for atom_idx in _ring_system_atom_indices(rings)
        for atom in [cast("ob.OBAtom | None", obmol.GetAtom(atom_idx))]
        if atom is not None
    )
    return abs(charge_sum) < _AROMATIC_RING_FORMAL_CHARGE_ABS_REJECTION_THRESHOLD


def _aromatic_ring_stability_weight(
    obmol: ob.OBMol,
    ring_atom_indices: Iterable[int],
    config: OrganicTopologyConfig,
) -> float:
    atoms = [cast("ob.OBAtom | None", obmol.GetAtom(atom_idx)) for atom_idx in ring_atom_indices]
    heavy_atoms = [atom for atom in atoms if atom is not None and atom.GetAtomicNum() != 1]
    if not heavy_atoms:
        return 0.0

    ring_size = len(heavy_atoms)
    hetero_count = sum(1 for atom in heavy_atoms if int(atom.GetAtomicNum()) != 6)
    charge_count = sum(1 for atom in heavy_atoms if int(atom.GetFormalCharge()) != 0)
    radical_count = sum(1 for atom in heavy_atoms if _atom_has_unpaired_electrons(atom))

    if ring_size == 6 and hetero_count == 0 and charge_count == 0 and radical_count == 0:
        return config.aromatic_stability_benzene_score

    size_factor = (
        config.aromatic_stability_ring_size_6_factor
        if ring_size == 6
        else (
            config.aromatic_stability_ring_size_5_factor
            if ring_size == 5
            else config.aromatic_stability_other_ring_size_factor
        )
    )
    hetero_factor = max(
        config.aromatic_stability_min_hetero_factor,
        1.0 - config.aromatic_stability_hetero_atom_penalty * hetero_count,
    )
    charge_factor = max(
        config.aromatic_stability_min_charge_factor,
        1.0 - config.aromatic_stability_formal_charge_penalty * charge_count,
    )
    radical_factor = max(
        config.aromatic_stability_min_radical_factor,
        1.0 - config.aromatic_stability_radical_penalty * radical_count,
    )
    return min(
        config.aromatic_stability_other_ring_max_score,
        size_factor * hetero_factor * charge_factor * radical_factor,
    )


def _atom_has_unpaired_electrons(atom: ob.OBAtom) -> bool:
    return get_unpaired_electron_count(atom) > 0


def _is_multiple_like_bond(bond: ob.OBBond) -> bool:
    return bool(bond.IsAromatic() or cast(int, bond.GetBondOrder()) >= 2)


def _is_pi_active_electron_center(atom: ob.OBAtom) -> bool:
    if atom.GetAtomicNum() == 1 or atom.IsMetal():
        return False
    lone_pair_count = get_lone_pair_count(atom)
    if (
        atom.GetFormalCharge() == 0
        and get_unpaired_electron_count(atom) == 0
        and lone_pair_count == 0
        and not has_unresolved_two_electron_center(atom)
    ):
        # Search-active pairs are not the full Lewis inventory. An ordinary
        # neutral sp2 heteroatom can supply a p pair even after reconstruction
        # has consumed every active pair (e.g. phenol, enamine, or amide).
        element_info = NON_METAL_DICT.get(int(atom.GetAtomicNum()))
        if element_info is None or atom.GetAtomicNum() == 6 or int(atom.GetHyb()) != 2:
            return False
        valence = int(atom.GetTotalValence())
        outer = int(element_info.num_outer_electrons)
        paired = outer - valence
        return outer + valence == 8 and paired >= 2 and paired % 2 == 0

    element_info = NON_METAL_DICT.get(cast(int, atom.GetAtomicNum()))
    is_under_saturated = bool(
        element_info is not None
        and element_info.default_valence > cast(int, atom.GetTotalValence())
    )
    if lone_pair_count > 0:
        return is_under_saturated or cast(int, atom.GetHyb()) in (1, 2)
    return is_under_saturated


def _attached_hydrogen_count(atom: ob.OBAtom) -> int:
    return int(atom.GetImplicitHCount()) + int(atom.ExplicitHydrogenCount())


def _is_hyperconjugative_donor(atom: ob.OBAtom) -> bool:
    return bool(
        atom.GetAtomicNum() == 6
        and atom.GetFormalCharge() == 0
        and int(atom.GetHyb()) == 3
        and get_unpaired_electron_count(atom) == 0
        and get_lone_pair_count(atom) == 0
        and not has_unresolved_two_electron_center(atom)
        and _attached_hydrogen_count(atom) > 0
    )


def _is_hyperconjugation_acceptor(
    atom: ob.OBAtom,
    *,
    incident_to_multiple_like_bond: bool,
) -> bool:
    if atom.GetAtomicNum() == 1 or atom.IsMetal():
        return False
    if incident_to_multiple_like_bond:
        return True
    if atom.GetFormalCharge() <= 0 and get_unpaired_electron_count(atom) == 0:
        return False
    element_info = NON_METAL_DICT.get(cast(int, atom.GetAtomicNum()))
    return bool(
        element_info is not None
        and element_info.default_valence > cast(int, atom.GetTotalValence())
    )


def _hyperconjugation_metrics(omol: pybel.Molecule) -> tuple[int, int]:
    obmol = cast(ob.OBMol, omol.OBMol)
    atoms_incident_to_multiple_like_bond: Set[int] = set()
    bonds: list[ob.OBBond] = []
    for bond in mol_bonds(obmol):
        bonds.append(bond)
        if not _is_multiple_like_bond(bond):
            continue
        atoms_incident_to_multiple_like_bond.add(int(bond.GetBeginAtomIdx()))
        atoms_incident_to_multiple_like_bond.add(int(bond.GetEndAtomIdx()))

    donor_atom_indices: Set[int] = set()
    score = 0
    for bond in bonds:
        if bond.IsAromatic() or int(bond.GetBondOrder()) != 1:
            continue
        begin_atom = cast(ob.OBAtom, bond.GetBeginAtom())
        end_atom = cast(ob.OBAtom, bond.GetEndAtom())
        for donor, acceptor in ((begin_atom, end_atom), (end_atom, begin_atom)):
            if not _is_hyperconjugative_donor(donor):
                continue
            if not _is_hyperconjugation_acceptor(
                acceptor,
                incident_to_multiple_like_bond=(
                    int(acceptor.GetIdx()) in atoms_incident_to_multiple_like_bond
                ),
            ):
                continue
            donor_atom_indices.add(int(donor.GetIdx()) - 1)
            score += _attached_hydrogen_count(donor)
    return len(donor_atom_indices), score


def _heavy_bond_atoms(bond: ob.OBBond) -> tuple[ob.OBAtom, ob.OBAtom] | None:
    begin_atom = cast(ob.OBAtom, bond.GetBeginAtom())
    end_atom = cast(ob.OBAtom, bond.GetEndAtom())
    if begin_atom.GetAtomicNum() == 1 or end_atom.GetAtomicNum() == 1:
        return None
    return begin_atom, end_atom


def _cumulated_multiple_bonds_by_center(
    heavy_bonds: Iterable[tuple[ob.OBBond, ob.OBAtom, ob.OBAtom]],
) -> Dict[int, Dict[int, int]]:
    multiple_bonds_by_atom: Dict[int, Dict[int, int]] = {}
    atoms_by_idx: Dict[int, ob.OBAtom] = {}
    for bond, begin_atom, end_atom in heavy_bonds:
        if bond.IsAromatic() or cast(int, bond.GetBondOrder()) < 2:
            continue
        bond_idx = cast(int, bond.GetIdx())
        begin_idx = cast(int, begin_atom.GetIdx())
        end_idx = cast(int, end_atom.GetIdx())
        atoms_by_idx[begin_idx] = begin_atom
        atoms_by_idx[end_idx] = end_atom
        multiple_bonds_by_atom.setdefault(begin_idx, {})[bond_idx] = end_idx
        multiple_bonds_by_atom.setdefault(end_idx, {})[bond_idx] = begin_idx

    return {
        atom_idx: bond_outer_atoms
        for atom_idx, bond_outer_atoms in multiple_bonds_by_atom.items()
        if cast(int, atoms_by_idx[atom_idx].GetHyb()) == 1 and len(bond_outer_atoms) >= 2
    }


def _atom_component_ids(
    heavy_bonds: Iterable[tuple[ob.OBBond, ob.OBAtom, ob.OBAtom]],
    selected_bond_indices: Set[int],
) -> Dict[int, int]:
    neighbors: Dict[int, Set[int]] = {}
    for bond, begin_atom, end_atom in heavy_bonds:
        if cast(int, bond.GetIdx()) not in selected_bond_indices:
            continue
        begin_idx = cast(int, begin_atom.GetIdx())
        end_idx = cast(int, end_atom.GetIdx())
        neighbors.setdefault(begin_idx, set()).add(end_idx)
        neighbors.setdefault(end_idx, set()).add(begin_idx)

    component_ids: Dict[int, int] = {}
    for atom_idx in neighbors:
        if atom_idx in component_ids:
            continue
        component_id = len(component_ids)
        stack = [atom_idx]
        while stack:
            current_idx = stack.pop()
            if current_idx in component_ids:
                continue
            component_ids[current_idx] = component_id
            stack.extend(neighbors.get(current_idx, set()))
    return component_ids


def _normalized_tetrahedron_volume(
    first: ob.OBAtom,
    second: ob.OBAtom,
    third: ob.OBAtom,
    fourth: ob.OBAtom,
) -> float:
    """Measure nonplanarity using tetrahedron volume normalized by the cube of its RMS edge length."""
    points = tuple(
        (float(atom.GetX()), float(atom.GetY()), float(atom.GetZ()))
        for atom in (first, second, third, fourth)
    )
    first_vector = (
        points[0][0] - points[3][0],
        points[0][1] - points[3][1],
        points[0][2] - points[3][2],
    )
    second_vector = (
        points[1][0] - points[3][0],
        points[1][1] - points[3][1],
        points[1][2] - points[3][2],
    )
    third_vector = (
        points[2][0] - points[3][0],
        points[2][1] - points[3][1],
        points[2][2] - points[3][2],
    )
    cross_product = (
        second_vector[1] * third_vector[2] - second_vector[2] * third_vector[1],
        second_vector[2] * third_vector[0] - second_vector[0] * third_vector[2],
        second_vector[0] * third_vector[1] - second_vector[1] * third_vector[0],
    )
    volume = abs(sum(lhs * rhs for lhs, rhs in zip(first_vector, cross_product))) / 6.0
    edge_squared_sum = sum(
        sum((left[axis] - right[axis]) ** 2 for axis in range(3))
        for offset, left in enumerate(points)
        for right in points[offset + 1 :]
    )
    rms_edge_cubed = (edge_squared_sum / 6.0) ** 1.5
    if rms_edge_cubed < 1e-9:
        return 0.0
    return max(0.0, min(1.0, 6.0 * (2.0**0.5) * volume / rms_edge_cubed))


def _strongly_pyramidal_pi_endpoint(atom: ob.OBAtom) -> bool:
    """A tetrahedral second-row center cannot supply a trigonal pi plane.

    Restrict this scoring guard to three explicit neighbors with resolved,
    nondegenerate geometry. It does not assign bond orders or metal charges.
    """
    if (
        atom.GetAtomicNum() not in (6, 7)
        or atom.GetExplicitDegree() != 3
        or atom.GetImplicitHCount()
    ):
        return False
    neighbors = list(atom_neighbors(atom))
    if any(atom.GetDistance(neighbor) < 1e-8 for neighbor in neighbors):
        return False
    angles = [atom.GetParent().GetAngle(a, atom, b) for a, b in combinations(neighbors, 2)]
    return all(80 < angle < 125 for angle in angles) and math.fsum(angles) < 330


def _local_pi_plane_tensor(
    center: ob.OBAtom, partner: ob.OBAtom, *, single_bonds_only: bool = True
) -> tuple[float, ...] | None:
    """Average unoriented plane normals over the neighboring bonds.

    Each local plane contains the multiple-bond axis and one substituent bond.
    Outer products remove normal-sign ambiguity; explicit H contributes just
    like other neighbors. Missing or collinear coordinates provide no plane.
    """
    axis = (
        partner.GetX() - center.GetX(),
        partner.GetY() - center.GetY(),
        partner.GetZ() - center.GetZ(),
    )
    length = math.sqrt(sum(x * x for x in axis))
    if length < 1e-12:
        return None
    axis = tuple(x / length for x in axis)
    tensors = []
    for bond in atom_bonds(center):
        if single_bonds_only and (bond.IsAromatic() or bond.GetBondOrder() != 1):
            continue
        neighbor = cast(ob.OBAtom, bond.GetNbrAtom(center))
        if neighbor.GetIdx() == partner.GetIdx():
            continue
        vector = (
            neighbor.GetX() - center.GetX(),
            neighbor.GetY() - center.GetY(),
            neighbor.GetZ() - center.GetZ(),
        )
        length = math.sqrt(sum(x * x for x in vector))
        if length < 1e-12:
            continue
        vector = tuple(x / length for x in vector)
        normal = (
            axis[1] * vector[2] - axis[2] * vector[1],
            axis[2] * vector[0] - axis[0] * vector[2],
            axis[0] * vector[1] - axis[1] * vector[0],
        )
        length = math.sqrt(sum(x * x for x in normal))
        if length < 1e-8:
            continue
        normal = tuple(x / length for x in normal)
        tensors.append(tuple(a * b for a in normal for b in normal))
    if not tensors:
        return None
    return tuple(math.fsum(tensor[i] for tensor in tensors) / len(tensors) for i in range(9))


def _alternating_double_bonds_are_geometrically_conjugated(
    obmol: ob.OBMol,
    left_inner: ob.OBAtom,
    right_inner: ob.OBAtom,
    multiple_bonds_by_atom: Dict[int, Dict[int, int]],
    min_plane_overlap: float,
) -> bool:
    """Require compatible local pi planes, independent of atom/bond ordering.

    The tensor inner product is the mean squared cosine between local normals.
    This is a geometric scoring heuristic, not a quantum overlap calculation.
    Enumerate all adjacent pi channels instead of selecting one by its index.
    Cumulated-system component boundaries are handled separately by the caller.
    """
    left_bonds = multiple_bonds_by_atom.get(int(left_inner.GetIdx()), {})
    right_bonds = multiple_bonds_by_atom.get(int(right_inner.GetIdx()), {})
    if not left_bonds or not right_bonds:
        return True
    for left_idx, right_idx in product(left_bonds.values(), right_bonds.values()):
        left_outer, right_outer = (
            cast("ob.OBAtom | None", obmol.GetAtom(left_idx)),
            cast("ob.OBAtom | None", obmol.GetAtom(right_idx)),
        )
        if left_outer is None or right_outer is None:
            continue
        if any(
            _strongly_pyramidal_pi_endpoint(atom)
            for atom in (left_outer, left_inner, right_inner, right_outer)
        ):
            continue
        planes = [
            plane
            for center, partner in (
                (left_outer, left_inner),
                (left_inner, left_outer),
                (right_inner, right_outer),
                (right_outer, right_inner),
            )
            # A triple bond has two perpendicular pi channels. A terminal
            # substituent cannot define one exclusive pi plane for that bond.
            if neighbor_bond_order(center, partner) < 3
            and (plane := _local_pi_plane_tensor(center, partner)) is not None
        ]
        # With fewer than two observable planes, geometry cannot disprove the
        # topological conjugation. Do not invent a normal for zero-length axes.
        if all(
            math.fsum(a * b for a, b in zip(left, right)) + 1e-12 >= min_plane_overlap
            for left, right in combinations(planes, 2)
        ):
            return True
    return False


def _lone_pair_is_geometrically_conjugated(
    donor: ob.OBAtom,
    acceptor: ob.OBAtom,
    volume_tolerance: float,
    min_plane_overlap: float,
) -> bool:
    """Check carbon electron centers and neutral heteroatom lone-pair donation."""
    if donor.GetAtomicNum() == 6 or (donor.GetAtomicNum() == 7 and donor.GetFormalCharge() == 0):
        # Carbon electron centers and neutral N donors use the same overlap
        # threshold as alternating double bonds. A stricter volume threshold
        # for N would reward an iminium contributor merely for moving the pair
        # into a double bond, despite identical local geometry.
        if any(_strongly_pyramidal_pi_endpoint(atom) for atom in (donor, acceptor)):
            return False
        if any(bond.GetBondOrder() >= 3 for bond in atom_bonds(acceptor)):
            return True  # A triple bond retains two perpendicular pi channels.
        left = _local_pi_plane_tensor(donor, acceptor, single_bonds_only=False)
        right = _local_pi_plane_tensor(acceptor, donor, single_bonds_only=False)
        planes = [plane for plane in (left, right) if plane is not None]

        def compatible(local_planes: list[tuple[float, ...]]) -> bool:
            return all(
                math.fsum(a * b for a, b in zip(lhs, rhs)) + 1e-12 >= min_plane_overlap
                for lhs, rhs in combinations(local_planes, 2)
            )

        if not compatible(planes):
            return False
        # The far endpoint of C=N (or any localized double bond) also sets
        # its pi plane. Omitting it would let a twisted iminium/carbanion
        # contributor bypass the same guard that rejects the neutral enamine.
        pi_partners = [
            cast(ob.OBAtom, bond.GetNbrAtom(acceptor))
            for bond in atom_bonds(acceptor)
            if not bond.IsAromatic() and bond.GetBondOrder() == 2
        ]
        if not pi_partners:
            return True
        for partner in pi_partners:
            if _strongly_pyramidal_pi_endpoint(partner):
                continue
            outer_plane = _local_pi_plane_tensor(partner, acceptor)
            if compatible(planes + ([outer_plane] if outer_plane is not None else [])):
                return True
        return False
    if donor.GetFormalCharge() != 0 or donor.GetAtomicNum() not in (7, 8, 15, 16, 33, 34):
        return True
    neighbors = [atom for atom in atom_neighbors(donor) if atom.GetIdx() != acceptor.GetIdx()]
    pi_neighbors = [
        cast(ob.OBAtom, bond.GetNbrAtom(acceptor))
        for bond in atom_bonds(acceptor)
        if _is_multiple_like_bond(bond)
    ]
    return all(
        _normalized_tetrahedron_volume(neighbor, donor, acceptor, pi_neighbor) <= volume_tolerance
        for neighbor in neighbors
        for pi_neighbor in pi_neighbors
    )


def _validated_conjugated_topology(
    omol: pybel.Molecule,
    config: OrganicTopologyConfig,
) -> tuple[Set[int], Dict[int, Dict[int, int]]]:
    obmol = cast(ob.OBMol, omol.OBMol)
    heavy_bonds: list[tuple[ob.OBBond, ob.OBAtom, ob.OBAtom]] = []
    atom_has_adjacent_multiple_like_bond: Set[int] = set()
    atom_has_adjacent_alternating_single_bond: Set[int] = set()
    aromatic_bond_indices: Set[int] = set()
    multiple_like_bond_indices: Set[int] = set()
    conjugated_bond_indices: Set[int] = set()
    nonaromatic_multiple_bonds_by_atom: Dict[int, Dict[int, int]] = {}

    for bond in mol_bonds(obmol):
        heavy_atoms = _heavy_bond_atoms(bond)
        if heavy_atoms is None:
            continue
        begin_atom, end_atom = heavy_atoms
        heavy_bonds.append((bond, begin_atom, end_atom))
        bond_idx = cast(int, bond.GetIdx())
        if not bond.IsAromatic() and cast(int, bond.GetBondOrder()) >= 2:
            begin_idx = cast(int, begin_atom.GetIdx())
            end_idx = cast(int, end_atom.GetIdx())
            nonaromatic_multiple_bonds_by_atom.setdefault(begin_idx, {})[bond_idx] = end_idx
            nonaromatic_multiple_bonds_by_atom.setdefault(end_idx, {})[bond_idx] = begin_idx
    cumulated_multiple_bonds_by_center = _cumulated_multiple_bonds_by_center(heavy_bonds)
    cumulated_multiple_bond_indices = {
        bond_idx
        for bonds_by_outer_atom in cumulated_multiple_bonds_by_center.values()
        for bond_idx in bonds_by_outer_atom
    }
    for bond, begin_atom, end_atom in heavy_bonds:
        bond_idx = cast(int, bond.GetIdx())
        if not _is_multiple_like_bond(bond):
            continue
        atom_has_adjacent_multiple_like_bond.add(begin_atom.GetIdx())
        atom_has_adjacent_multiple_like_bond.add(end_atom.GetIdx())
        if bond.IsAromatic():
            aromatic_bond_indices.add(bond_idx)
        elif bond_idx not in cumulated_multiple_bond_indices:
            multiple_like_bond_indices.add(bond_idx)

    alternating_single_bond_indices: Set[int] = set()
    for bond, begin_atom, end_atom in heavy_bonds:
        if bond.IsAromatic() or cast(int, bond.GetBondOrder()) != 1:
            continue
        begin_has_pi_bond = begin_atom.GetIdx() in atom_has_adjacent_multiple_like_bond
        end_has_pi_bond = end_atom.GetIdx() in atom_has_adjacent_multiple_like_bond
        if not (begin_has_pi_bond or end_has_pi_bond):
            continue
        if not (begin_has_pi_bond or _is_pi_active_electron_center(begin_atom)):
            continue
        if not (end_has_pi_bond or _is_pi_active_electron_center(end_atom)):
            continue
        if (
            begin_has_pi_bond
            and end_has_pi_bond
            and not (
                _alternating_double_bonds_are_geometrically_conjugated(
                    obmol,
                    begin_atom,
                    end_atom,
                    nonaromatic_multiple_bonds_by_atom,
                    config.conjugation_min_plane_overlap,
                )
            )
        ):
            continue
        if not begin_has_pi_bond and not _lone_pair_is_geometrically_conjugated(
            begin_atom,
            end_atom,
            config.conjugation_normalized_tetrahedron_volume_tolerance,
            config.conjugation_min_plane_overlap,
        ):
            continue
        if not end_has_pi_bond and not _lone_pair_is_geometrically_conjugated(
            end_atom,
            begin_atom,
            config.conjugation_normalized_tetrahedron_volume_tolerance,
            config.conjugation_min_plane_overlap,
        ):
            continue
        bond_idx = cast(int, bond.GetIdx())
        alternating_single_bond_indices.add(bond_idx)
        atom_has_adjacent_alternating_single_bond.add(begin_atom.GetIdx())
        atom_has_adjacent_alternating_single_bond.add(end_atom.GetIdx())

    conjugated_bond_indices.update(aromatic_bond_indices)
    conjugated_bond_indices.update(alternating_single_bond_indices)
    for bond, begin_atom, end_atom in heavy_bonds:
        bond_idx = cast(int, bond.GetIdx())
        if bond_idx not in multiple_like_bond_indices:
            continue
        # Aromatic ring bonds also connect an exocyclic C=O/C=S to a pi
        # system; an explicitly SINGLE neighboring bond is not required.
        if begin_atom.IsAromatic() or end_atom.IsAromatic():
            conjugated_bond_indices.add(bond_idx)
            continue
        if begin_atom.GetIdx() in atom_has_adjacent_alternating_single_bond:
            conjugated_bond_indices.add(bond_idx)
            continue
        if end_atom.GetIdx() in atom_has_adjacent_alternating_single_bond:
            conjugated_bond_indices.add(bond_idx)

    base_component_ids = _atom_component_ids(heavy_bonds, conjugated_bond_indices)
    for bonds_by_outer_atom in cumulated_multiple_bonds_by_center.values():
        outer_component_ids = [
            base_component_ids[outer_atom_idx]
            for outer_atom_idx in bonds_by_outer_atom.values()
            if outer_atom_idx in base_component_ids
        ]
        if len(outer_component_ids) != len(set(outer_component_ids)):
            continue
        for bond_idx, outer_atom_idx in bonds_by_outer_atom.items():
            if outer_atom_idx in atom_has_adjacent_alternating_single_bond:
                conjugated_bond_indices.add(bond_idx)
    return conjugated_bond_indices, cumulated_multiple_bonds_by_center


def _validated_conjugated_bond_indices(omol: pybel.Molecule) -> Set[int]:
    return _validated_conjugated_topology(omol, CONFIG.organic_topology)[0]


def is_conjugated_bond(bond: ob.OBBond) -> bool:
    parent = cast("ob.OBMol | None", bond.GetParent())
    if parent is None:
        return False
    parent_mol = _prepare_topology_working_molecule(pybel.Molecule(parent))
    return cast(int, bond.GetIdx()) in _validated_conjugated_bond_indices(parent_mol)


def compute_organic_topology_metrics(
    omol: pybel.Molecule,
    config: OrganicTopologyConfig | None = None,
) -> OrganicTopologyMetrics:
    try:
        topology_config = CONFIG.organic_topology if config is None else config
        working_omol = _prepare_topology_working_molecule(omol)
        obmol = cast(ob.OBMol, working_omol.OBMol)

        aromatic_ring_count = 0
        aromatic_stability_score = 0.0
        aromatic_atom_indices: Set[int] = set()
        for ring_system in _aromatic_ring_systems(obmol):
            if not _is_accepted_aromatic_ring_system(obmol, ring_system):
                continue
            aromatic_ring_count += len(ring_system)
            for ring in ring_system:
                ring_atom_indices = _ring_atom_indices(ring)
                aromatic_stability_score += _aromatic_ring_stability_weight(
                    obmol,
                    ring_atom_indices,
                    topology_config,
                )
                for atom_idx in ring_atom_indices:
                    atom = cast("ob.OBAtom | None", obmol.GetAtom(atom_idx))
                    if atom is None or atom.GetAtomicNum() == 1:
                        continue
                    aromatic_atom_indices.add(atom_idx)
        aromatic_atom_count = len(aromatic_atom_indices)

        conjugated_atom_indices: Set[int] = set()
        conjugated_bond_indices, cumulated_multiple_bonds_by_center = (
            _validated_conjugated_topology(working_omol, topology_config)
        )
        conjugated_bond_atoms: Dict[int, tuple[int, int]] = {}
        incident_conjugated_bonds: Dict[int, Set[int]] = {}
        for bond in mol_bonds(obmol):
            begin_atom = cast(ob.OBAtom, bond.GetBeginAtom())
            end_atom = cast(ob.OBAtom, bond.GetEndAtom())
            if begin_atom.GetAtomicNum() == 1 or end_atom.GetAtomicNum() == 1:
                continue
            if cast(int, bond.GetIdx()) not in conjugated_bond_indices:
                continue
            begin_idx = begin_atom.GetIdx() - 1
            end_idx = end_atom.GetIdx() - 1
            bond_idx = cast(int, bond.GetIdx())
            conjugated_atom_indices.add(begin_idx)
            conjugated_atom_indices.add(end_idx)
            conjugated_bond_atoms[bond_idx] = (begin_idx, end_idx)
            incident_conjugated_bonds.setdefault(begin_idx, set()).add(bond_idx)
            incident_conjugated_bonds.setdefault(end_idx, set()).add(bond_idx)

        max_conjugated_component_size = 0
        conjugated_bond_neighbors: Dict[int, Set[int]] = {}
        cumulated_bonds_by_zero_based_center = {
            center_idx - 1: set(bonds_by_outer_atom)
            for center_idx, bonds_by_outer_atom in cumulated_multiple_bonds_by_center.items()
        }
        for atom_idx, incident_bond_indices in incident_conjugated_bonds.items():
            incident = tuple(incident_bond_indices)
            cumulated_at_center = cumulated_bonds_by_zero_based_center.get(atom_idx, set())
            for offset, first_bond_idx in enumerate(incident):
                for second_bond_idx in incident[offset + 1 :]:
                    if (
                        first_bond_idx in cumulated_at_center
                        and second_bond_idx in cumulated_at_center
                    ):
                        continue
                    conjugated_bond_neighbors.setdefault(first_bond_idx, set()).add(second_bond_idx)
                    conjugated_bond_neighbors.setdefault(second_bond_idx, set()).add(first_bond_idx)

        visited_bonds: Set[int] = set()
        conjugated_atom_components: list[tuple[int, ...]] = []
        for bond_idx in conjugated_bond_indices:
            if bond_idx in visited_bonds:
                continue
            stack = [bond_idx]
            component_atom_indices: Set[int] = set()
            while stack:
                current_bond_idx = stack.pop()
                if current_bond_idx in visited_bonds:
                    continue
                visited_bonds.add(current_bond_idx)
                component_atom_indices.update(conjugated_bond_atoms[current_bond_idx])
                stack.extend(
                    neighbor_bond_idx
                    for neighbor_bond_idx in conjugated_bond_neighbors.get(current_bond_idx, set())
                    if neighbor_bond_idx not in visited_bonds
                )
            max_conjugated_component_size = max(
                max_conjugated_component_size, len(component_atom_indices)
            )
            conjugated_atom_components.append(tuple(sorted(component_atom_indices)))
        hyperconjugative_donor_count, hyperconjugation_score = _hyperconjugation_metrics(
            working_omol
        )

        return OrganicTopologyMetrics(
            aromatic_atom_count=aromatic_atom_count,
            aromatic_ring_count=aromatic_ring_count,
            aromatic_stability_score=aromatic_stability_score,
            conjugated_atom_count=len(conjugated_atom_indices),
            conjugated_bond_count=len(conjugated_bond_indices),
            max_conjugated_component_size=max_conjugated_component_size,
            conjugated_atom_indices=tuple(sorted(conjugated_atom_indices)),
            hyperconjugative_donor_count=hyperconjugative_donor_count,
            hyperconjugation_score=hyperconjugation_score,
            conjugated_atom_components=tuple(sorted(conjugated_atom_components)),
        )
    except Exception:  # noqa: BLE001
        return OrganicTopologyMetrics(
            aromatic_atom_count=0,
            aromatic_ring_count=0,
            aromatic_stability_score=0.0,
            conjugated_atom_count=0,
            conjugated_bond_count=0,
            max_conjugated_component_size=0,
            conjugated_atom_indices=(),
            hyperconjugative_donor_count=0,
            hyperconjugation_score=0,
        )


__all__ = ["OrganicTopologyMetrics", "compute_organic_topology_metrics", "is_conjugated_bond"]
