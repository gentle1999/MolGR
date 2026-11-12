"""Scoring and selection utilities for metal-aware fallback reconstruction."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import astuple, dataclass
from typing import DefaultDict, List, Optional, Set, Tuple, cast

import numpy as np
from openbabel import openbabel as ob
from openbabel import pybel

from molgr.config import CONFIG, MetalScoringConfig, MolGRConfig
from molgr.fallback.stages.fresh import assign_radical_dots
from molgr.fallback.state import (
    MetalCandidateState,
    MetalCandidateStateMachine,
    ReconstructionState,
)
from molgr.fallback.utils import consts, dataclasses
from molgr.fallback.utils.electrons import (
    get_lone_pair_count,
    get_unpaired_electron_count,
    has_unresolved_two_electron_center,
)
from molgr.fallback.utils.organic_topology import compute_organic_topology_metrics
from molgr.utils.coordination import coordination_distance_cutoff
from molgr.utils.coordination_visibility import (
    CoordinationBlockerArrays,
    Point3D,
    coordination_visibility_mask,
    empty_coordination_blocker_arrays,
)


_NEGATIVE_METAL_DISCORDANCE_PENALTY = 0.5
_MINIMUM_RING_ALLENE_ANGLE_DEGREES = 150.0
_MINIMUM_CHARGE_POLARITY_INVERSION_ELECTRONEGATIVITY_GAP = 0.3
_MAX_CHARGE_LOCALIZATION_REFERENCE_OXIDATION_STATE_DELTA = 2
_CONNECTIVITY_HASH_OFFSET = 1469598103934665603
_CONNECTIVITY_HASH_PRIME = 1099511628211
_CONNECTIVITY_HASH_MASK = (1 << 64) - 1


@dataclass(frozen=True)
class _OrganicElectronicStateMetrics:
    aromatic_atom_count: int
    aromatic_ring_count: int
    aromatic_stability_score: float
    conjugated_atom_count: int
    conjugated_bond_count: int
    max_conjugated_component_size: int
    hyperconjugative_donor_count: int
    hyperconjugation_score: int
    singlet_two_electron_center_count: int
    radical_localization_penalty: float
    charge_localization_penalty: float
    charge_localization_component_cancellation: float
    charge_localization_polarity_inversion_penalty: float


def _distance_to_metal(
    atom: ob.OBAtom,
    metal_state: dataclasses.MetalAtomPosition,
) -> float:
    dx = float(atom.GetX()) - metal_state.position_x
    dy = float(atom.GetY()) - metal_state.position_y
    dz = float(atom.GetZ()) - metal_state.position_z
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _charge_sign(charge: int) -> int:
    if charge > 0:
        return 1
    if charge < 0:
        return -1
    return 0


def _is_inner_visible_diradical_discordance_atom(atom: ob.OBAtom) -> bool:
    return int(get_unpaired_electron_count(atom)) >= 2


def _is_explicit_singlet_two_electron_center(atom: ob.OBAtom) -> bool:
    return (
        int(atom.GetAtomicNum()) in {6, 7, 15}
        and int(atom.GetFormalCharge()) == 0
        and get_unpaired_electron_count(atom) == 0
        and get_lone_pair_count(atom) == 1
        and not has_unresolved_two_electron_center(atom)
        and assign_radical_dots(atom) == 2
    )


def _bent_cumulated_ring_allene_count(obmol: ob.OBMol) -> int:
    count = 0
    for center_iter in ob.OBMolAtomIter(obmol):
        center = cast(ob.OBAtom, center_iter)
        if not bool(center.IsInRing()):
            continue
        ring_double_neighbors: list[ob.OBAtom] = []
        for bond_iter in ob.OBAtomBondIter(center):
            bond = cast(ob.OBBond, bond_iter)
            if bool(bond.IsAromatic()) or int(bond.GetBondOrder()) != 2:
                continue
            neighbor = cast(ob.OBAtom, bond.GetNbrAtom(center))
            if neighbor is not None and bool(neighbor.IsInRing()):
                ring_double_neighbors.append(neighbor)
        if len(ring_double_neighbors) < 2:
            continue
        if any(
            math.isfinite(angle := float(obmol.GetAngle(left, center, right)))
            and angle < _MINIMUM_RING_ALLENE_ANGLE_DEGREES
            for left_index, left in enumerate(ring_double_neighbors)
            for right in ring_double_neighbors[left_index + 1 :]
        ):
            count += 1
    return count


def _nearest_nonzero_metal_charge_sign_to_bond(
    begin_atom: ob.OBAtom,
    end_atom: ob.OBAtom,
    metal_states: Sequence[dataclasses.MetalAtomPosition],
) -> int:
    midpoint_x = (float(begin_atom.GetX()) + float(end_atom.GetX())) * 0.5
    midpoint_y = (float(begin_atom.GetY()) + float(end_atom.GetY())) * 0.5
    midpoint_z = (float(begin_atom.GetZ()) + float(end_atom.GetZ())) * 0.5
    best_distance_sq = float("inf")
    best_charge_sign = 0
    for metal_state in metal_states:
        metal_charge_sign = _charge_sign(int(metal_state.valence))
        if metal_charge_sign == 0:
            continue
        dx = midpoint_x - metal_state.position_x
        dy = midpoint_y - metal_state.position_y
        dz = midpoint_z - metal_state.position_z
        distance_sq = dx * dx + dy * dy + dz * dz
        if distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            best_charge_sign = metal_charge_sign
    return best_charge_sign


def _atom_coordinates(atom: ob.OBAtom) -> Point3D:
    return float(atom.GetX()), float(atom.GetY()), float(atom.GetZ())


def _build_coordination_blocker_arrays(
    obmol: ob.OBMol,
    *,
    metal_scoring_config: MetalScoringConfig,
) -> CoordinationBlockerArrays:
    blocker_indices: list[int] = []
    blocker_coordinates: list[Point3D] = []
    blocker_radii: list[float] = []
    for blocker_iter in ob.OBMolAtomIter(obmol):
        blocker = cast(ob.OBAtom, blocker_iter)
        blocker_radius = (
            metal_scoring_config.metal_access_radius_scale
            * float(ob.GetCovalentRad(int(blocker.GetAtomicNum())))
            + metal_scoring_config.metal_access_clearance_angstrom
        )
        if blocker_radius <= 0.0:
            continue
        blocker_indices.append(int(blocker.GetIdx()))
        blocker_coordinates.append(_atom_coordinates(blocker))
        blocker_radii.append(blocker_radius)
    if not blocker_indices:
        return empty_coordination_blocker_arrays()
    return (
        np.asarray(blocker_indices, dtype=np.int64),
        np.asarray(blocker_coordinates, dtype=np.float64),
        np.asarray(blocker_radii, dtype=np.float64),
    )


def _coordination_radius_cutoff(
    atom: ob.OBAtom,
    metal_state: dataclasses.MetalAtomPosition,
    *,
    metal_scoring_config: MetalScoringConfig,
) -> float:
    return coordination_distance_cutoff(
        int(metal_state.element_idx),
        int(atom.GetAtomicNum()),
        radius_scale=metal_scoring_config.metal_access_radius_scale,
        extra_tolerance_angstrom=(metal_scoring_config.metal_coordination_extra_tolerance_angstrom),
    )


def _is_inner_sphere_atom(
    atom: ob.OBAtom,
    metal_state: dataclasses.MetalAtomPosition,
    *,
    metal_scoring_config: MetalScoringConfig,
) -> bool:
    distance = _distance_to_metal(atom, metal_state)
    return (
        0.0
        < distance
        <= _coordination_radius_cutoff(
            atom,
            metal_state,
            metal_scoring_config=metal_scoring_config,
        )
    )


def _non_metal_atom_entries(obmol: ob.OBMol) -> tuple[ob.OBAtom, ...]:
    atoms: list[ob.OBAtom] = []
    for atom_iter in ob.OBMolAtomIter(obmol):
        atom = cast(ob.OBAtom, atom_iter)
        if atom.IsMetal():
            continue
        atoms.append(atom)
    return tuple(atoms)


def _component_atom_index_groups(obmol: ob.OBMol) -> tuple[tuple[int, ...], ...]:
    unseen = {int(atom.GetIdx()) for atom in ob.OBMolAtomIter(obmol)}
    components: list[tuple[int, ...]] = []
    while unseen:
        start_idx = min(unseen)
        unseen.remove(start_idx)
        stack = [start_idx]
        component = [start_idx]
        while stack:
            atom = cast(ob.OBAtom, obmol.GetAtom(stack.pop()))
            for neighbor_iter in ob.OBAtomAtomIter(atom):
                neighbor = cast(ob.OBAtom, neighbor_iter)
                neighbor_idx = int(neighbor.GetIdx())
                if neighbor_idx not in unseen:
                    continue
                unseen.remove(neighbor_idx)
                stack.append(neighbor_idx)
                component.append(neighbor_idx)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _connectivity_hash(values: Sequence[int]) -> int:
    value = _CONNECTIVITY_HASH_OFFSET
    for item in values:
        value ^= int(item) & _CONNECTIVITY_HASH_MASK
        value = (value * _CONNECTIVITY_HASH_PRIME) & _CONNECTIVITY_HASH_MASK
    return value


def _component_connectivity_signature(
    obmol: ob.OBMol,
    atom_indices: Sequence[int],
) -> tuple[int, ...]:
    labels = {
        atom_idx: int(cast(ob.OBAtom, obmol.GetAtom(atom_idx)).GetAtomicNum())
        for atom_idx in atom_indices
    }
    for _ in range(4):
        next_labels: dict[int, int] = {}
        for atom_idx in atom_indices:
            atom = cast(ob.OBAtom, obmol.GetAtom(atom_idx))
            neighbor_labels = sorted(
                labels[int(cast(ob.OBAtom, neighbor_iter).GetIdx())]
                for neighbor_iter in ob.OBAtomAtomIter(atom)
            )
            next_labels[atom_idx] = _connectivity_hash(
                (labels[atom_idx], len(neighbor_labels), *neighbor_labels)
            )
        labels = next_labels
    return tuple(sorted(labels.values()))


def _repeated_component_charge_asymmetry_count(obmol: ob.OBMol) -> int:
    charges_by_connectivity: DefaultDict[tuple[int, ...], list[int]] = defaultdict(list)
    for atom_indices in _component_atom_index_groups(obmol):
        signature = _component_connectivity_signature(obmol, atom_indices)
        charges_by_connectivity[signature].append(
            sum(
                int(cast(ob.OBAtom, obmol.GetAtom(atom_idx)).GetFormalCharge())
                for atom_idx in atom_indices
            )
        )
    return sum(
        1
        for charges in charges_by_connectivity.values()
        if len(charges) > 1 and min(charges) != max(charges)
    )


def _haptic_arene_reduction_count(
    obmol: ob.OBMol,
    visible_atoms_by_metal: Sequence[Sequence[ob.OBAtom]],
) -> int:
    """Count localized anionic carbons in haptically reduced carbon rings.

    A complete aromatic/Kekule ring is not reduced merely because its charge is
    represented on one carbon; the discordance applies only after the cyclic pi
    pattern is broken.
    """
    visible_carbon_indices_by_metal = [
        {int(atom.GetIdx()) for atom in visible_atoms if int(atom.GetAtomicNum()) == 6}
        for visible_atoms in visible_atoms_by_metal
    ]

    def is_complete_kekule_pi_ring(ring_atom_indices: tuple[int, ...]) -> bool:
        ring_atoms = [cast(ob.OBAtom, obmol.GetAtom(idx)) for idx in ring_atom_indices]
        ring_bonds: list[ob.OBBond] = []
        for offset, begin_idx in enumerate(ring_atom_indices):
            end_idx = ring_atom_indices[(offset + 1) % len(ring_atom_indices)]
            bond = cast("ob.OBBond | None", obmol.GetBond(begin_idx, end_idx))
            if bond is None or int(bond.GetBondOrder()) not in (1, 2):
                return False
            ring_bonds.append(bond)

        if all(bool(bond.IsAromatic()) for bond in ring_bonds) and all(
            bool(atom.IsAromatic()) for atom in ring_atoms
        ):
            return True

        pi_bond_indices = {
            index for index, bond in enumerate(ring_bonds) if int(bond.GetBondOrder()) == 2
        }
        if any(
            index in pi_bond_indices and (index + 1) % len(ring_atom_indices) in pi_bond_indices
            for index in range(len(ring_atom_indices))
        ):
            return False
        pi_edge_count_by_atom = [0] * len(ring_atom_indices)
        for index in pi_bond_indices:
            pi_edge_count_by_atom[index] += 1
            pi_edge_count_by_atom[(index + 1) % len(ring_atom_indices)] += 1
        if any(count > 1 for count in pi_edge_count_by_atom):
            return False

        missing_pi_atoms = [
            index for index, count in enumerate(pi_edge_count_by_atom) if count == 0
        ]
        if len(ring_atom_indices) % 2 == 0:
            return not missing_pi_atoms and len(pi_bond_indices) == len(ring_atom_indices) // 2
        return (
            len(missing_pi_atoms) == 1
            and len(pi_bond_indices) == len(ring_atom_indices) // 2
            and int(ring_atoms[missing_pi_atoms[0]].GetAtomicNum()) == 6
            and int(ring_atoms[missing_pi_atoms[0]].GetFormalCharge()) < 0
        )

    count = 0
    # OBMolRingIter can expose a dangling SWIG ring on Windows for molecules
    # without rings. GetSSSR() returns an owning vector and is stable there.
    for ring_iter in obmol.GetSSSR():
        ring = cast(ob.OBRing, ring_iter)
        ring_atom_indices = tuple(int(idx) for idx in getattr(ring, "_path", ()))
        if len(ring_atom_indices) not in {5, 6}:
            continue
        ring_atoms = [cast(ob.OBAtom, obmol.GetAtom(idx)) for idx in ring_atom_indices]
        if any(int(atom.GetAtomicNum()) != 6 for atom in ring_atoms):
            continue
        if not any(
            len(set(ring_atom_indices) & visible_indices) >= 3
            for visible_indices in visible_carbon_indices_by_metal
        ):
            continue
        if is_complete_kekule_pi_ring(ring_atom_indices):
            continue
        count += sum(int(atom.GetFormalCharge()) < 0 for atom in ring_atoms)
    return count


def _vector_norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _coordination_geometry(
    visible_atoms: Sequence[ob.OBAtom],
    metal_state: dataclasses.MetalAtomPosition,
    *,
    config: MolGRConfig | None = None,
) -> str:
    resolved_config = CONFIG if config is None else config
    geometry_config = resolved_config.metal_radical_inference
    vectors = [
        (
            float(atom.GetX()) - metal_state.position_x,
            float(atom.GetY()) - metal_state.position_y,
            float(atom.GetZ()) - metal_state.position_z,
        )
        for atom in visible_atoms
    ]
    if len(vectors) == 2:
        denominator = _vector_norm(vectors[0]) * _vector_norm(vectors[1])
        if denominator <= 1e-8:
            return "bent"
        cosine = sum(lhs * rhs for lhs, rhs in zip(vectors[0], vectors[1])) / denominator
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        return "linear" if angle >= geometry_config.linear_angle_min_degrees else "bent"
    if len(vectors) != 4:
        return "other"

    best_normal: tuple[float, float, float] | None = None
    best_norm = 0.0
    for idx, lhs in enumerate(vectors):
        for rhs in vectors[idx + 1 :]:
            normal = (
                lhs[1] * rhs[2] - lhs[2] * rhs[1],
                lhs[2] * rhs[0] - lhs[0] * rhs[2],
                lhs[0] * rhs[1] - lhs[1] * rhs[0],
            )
            normal_norm = _vector_norm(normal)
            if normal_norm > best_norm:
                best_normal = normal
                best_norm = normal_norm
    if best_normal is None or best_norm <= 1e-8:
        return "other"
    unit_normal = tuple(component / best_norm for component in best_normal)
    planarity_distance = sum(
        abs(sum(component * normal for component, normal in zip(vector, unit_normal)))
        for vector in vectors
    ) / len(vectors)
    if planarity_distance <= geometry_config.square_planar_planarity_tolerance_angstrom:
        return "square_planar"
    return "tetrahedral"


def _coordination_geometry_discordance_count(
    visible_atoms_by_metal: Sequence[Sequence[ob.OBAtom]],
    metal_states: Sequence[dataclasses.MetalAtomPosition],
    *,
    config: MolGRConfig | None = None,
) -> int:
    count = 0
    for visible_atoms, metal_state in zip(visible_atoms_by_metal, metal_states):
        geometry = _coordination_geometry(
            visible_atoms,
            metal_state,
            config=config,
        )
        if (
            metal_state.symbol in {"Pd", "Pt"}
            and metal_state.valence >= 4
            and geometry == "square_planar"
        ) or (
            metal_state.symbol in {"Ag", "Au"} and metal_state.valence >= 3 and geometry == "linear"
        ):
            count += 1
    return count


def _inner_visible_atoms_to_metal(
    atoms: Sequence[ob.OBAtom],
    metal_state: dataclasses.MetalAtomPosition,
    *,
    metal_scoring_config: MetalScoringConfig,
    blocker_arrays: CoordinationBlockerArrays,
) -> tuple[ob.OBAtom, ...]:
    if not atoms:
        return ()

    atom_indices = np.asarray([int(atom.GetIdx()) for atom in atoms], dtype=np.int64)
    atom_coordinates = np.asarray([_atom_coordinates(atom) for atom in atoms], dtype=np.float64)
    metal_coordinates = np.asarray(
        (
            float(metal_state.position_x),
            float(metal_state.position_y),
            float(metal_state.position_z),
        ),
        dtype=np.float64,
    )
    cutoff = np.asarray(
        [
            coordination_distance_cutoff(
                int(metal_state.element_idx),
                int(atom.GetAtomicNum()),
                radius_scale=metal_scoring_config.metal_access_radius_scale,
                extra_tolerance_angstrom=(
                    metal_scoring_config.metal_coordination_extra_tolerance_angstrom
                ),
            )
            for atom in atoms
        ],
        dtype=np.float64,
    )
    delta = atom_coordinates - metal_coordinates
    distance_sq = np.einsum("ij,ij->i", delta, delta)
    inner_mask = (distance_sq > 0.0) & (distance_sq <= cutoff * cutoff)
    if not np.any(inner_mask):
        return ()

    inner_indices = atom_indices[inner_mask]
    inner_coordinates = atom_coordinates[inner_mask]
    visible_mask = coordination_visibility_mask(
        inner_indices,
        inner_coordinates,
        metal_coordinates,
        blocker_arrays,
    )
    inner_atoms = [atom for atom, is_inner in zip(atoms, inner_mask) if is_inner]
    return tuple(atom for atom, is_visible in zip(inner_atoms, visible_mask) if is_visible)


def _bond_order(bond: ob.OBBond) -> int:
    if bond.IsAromatic():
        return 2
    return cast(int, bond.GetBondOrder())


def _other_bond_atom(bond: ob.OBBond, atom: ob.OBAtom) -> ob.OBAtom:
    return (
        cast(ob.OBAtom, bond.GetEndAtom())
        if cast(ob.OBAtom, bond.GetBeginAtom()).GetIdx() == atom.GetIdx()
        else cast(ob.OBAtom, bond.GetBeginAtom())
    )


def _bond_between_atoms(lhs: ob.OBAtom, rhs: ob.OBAtom) -> Optional[ob.OBBond]:
    rhs_idx = rhs.GetIdx()
    for bond_iter in ob.OBAtomBondIter(lhs):
        bond = cast(ob.OBBond, bond_iter)
        if _other_bond_atom(bond, lhs).GetIdx() == rhs_idx:
            return bond
    return None


def _has_conjugated_bridge_between_charged_carbons(
    begin_atom: ob.OBAtom,
    end_atom: ob.OBAtom,
) -> bool:
    for begin_bond_iter in ob.OBAtomBondIter(begin_atom):
        begin_bond = cast(ob.OBBond, begin_bond_iter)
        if _bond_order(begin_bond) != 1:
            continue
        begin_neighbor = _other_bond_atom(begin_bond, begin_atom)
        if begin_neighbor.GetAtomicNum() == 1:
            continue
        for middle_bond_iter in ob.OBAtomBondIter(begin_neighbor):
            middle_bond = cast(ob.OBBond, middle_bond_iter)
            if (
                int(middle_bond.GetIdx()) == int(begin_bond.GetIdx())
                or _bond_order(middle_bond) != 2
            ):
                continue
            middle_neighbor = _other_bond_atom(middle_bond, begin_neighbor)
            if middle_neighbor.GetIdx() == begin_atom.GetIdx():
                continue
            bridge_end_bond = _bond_between_atoms(middle_neighbor, end_atom)
            if bridge_end_bond is None:
                continue
            if _bond_order(bridge_end_bond) == 1:
                return True
    return False


def _inner_visible_conjugated_charged_carbon_pair_count(
    obmol: ob.OBMol,
    visible_inner_atom_indices: Set[int],
) -> int:
    charged_carbons: List[ob.OBAtom] = []
    for atom_iter in ob.OBMolAtomIter(obmol):
        atom = cast(ob.OBAtom, atom_iter)
        if int(atom.GetIdx()) not in visible_inner_atom_indices:
            continue
        if atom.GetAtomicNum() != 6 or int(atom.GetFormalCharge()) == 0:
            continue
        charged_carbons.append(atom)

    count = 0
    for index, begin_atom in enumerate(charged_carbons):
        for end_atom in charged_carbons[index + 1 :]:
            if _charge_sign(int(begin_atom.GetFormalCharge())) != _charge_sign(
                int(end_atom.GetFormalCharge())
            ):
                continue
            if _bond_between_atoms(begin_atom, end_atom) is not None:
                continue
            if _has_conjugated_bridge_between_charged_carbons(begin_atom, end_atom):
                count += 1
    return count


def _has_outer_sphere_proton(
    obmol: ob.OBMol,
    metal_states: Sequence[dataclasses.MetalAtomPosition],
    *,
    metal_scoring_config: MetalScoringConfig,
) -> bool:
    for atom_iter in ob.OBMolAtomIter(obmol):
        atom = cast(ob.OBAtom, atom_iter)
        if atom.IsMetal() or int(atom.GetAtomicNum()) != 1 or int(atom.GetFormalCharge()) <= 0:
            continue
        if any(
            _is_inner_sphere_atom(
                atom,
                metal_state,
                metal_scoring_config=metal_scoring_config,
            )
            for metal_state in metal_states
        ):
            continue
        return True
    return False


def _negative_metal_discordance_count(
    obmol: ob.OBMol,
    metal_states: Sequence[dataclasses.MetalAtomPosition],
    *,
    metal_scoring_config: MetalScoringConfig,
) -> Tuple[int, bool, bool]:
    negative_metal_count = sum(
        abs(int(metal_state.valence))
        for metal_state in metal_states
        if int(metal_state.valence) < 0
    )
    if negative_metal_count == 0:
        return 0, False, False

    has_outer_sphere_proton = _has_outer_sphere_proton(
        obmol,
        metal_states,
        metal_scoring_config=metal_scoring_config,
    )
    has_positive_metal_counterion = any(
        int(metal_state.valence) > 0 for metal_state in metal_states
    )
    if has_outer_sphere_proton or has_positive_metal_counterion:
        return 0, has_outer_sphere_proton, has_positive_metal_counterion
    return negative_metal_count, False, False


def _zero_valent_metals_with_organic_cation_count(
    obmol: ob.OBMol,
    metal_states: Sequence[dataclasses.MetalAtomPosition],
    *,
    total_charge: int,
) -> int:
    if total_charge > 0:
        return 0
    if not metal_states or any(int(metal_state.valence) != 0 for metal_state in metal_states):
        return 0
    for atom_iter in ob.OBMolAtomIter(obmol):
        atom = cast(ob.OBAtom, atom_iter)
        if atom.IsMetal() or int(atom.GetFormalCharge()) <= 0:
            continue
        if int(atom.GetAtomicNum()) > 0 and not _is_locally_charge_compensated_nonmetal_cation(
            atom
        ):
            return 1
    return 0


def _is_unsaturated_organic_cation(atom: ob.OBAtom) -> bool:
    if atom.IsMetal() or int(atom.GetFormalCharge()) <= 0:
        return False

    element_info = consts.NON_METAL_DICT.get(int(atom.GetAtomicNum()))
    if element_info is None:
        return False

    # Compare assigned outer-shell electrons with the closed-shell target.
    # Degree is not a proxy for this: one triple bond gives O+ three
    # bond-order units while still completing its octet.
    total_valence = int(atom.GetTotalValence())
    local_electron_count = (
        int(element_info.num_outer_electrons) - int(atom.GetFormalCharge()) + total_valence
    )
    shell_target = 2 if int(atom.GetAtomicNum()) == 1 else 8
    return local_electron_count < shell_target


def _has_adjacent_formal_charge_cancellation(atom: ob.OBAtom) -> bool:
    formal_charge = int(atom.GetFormalCharge())
    if formal_charge == 0:
        return False
    adjacent_charge = sum(
        int(cast(ob.OBAtom, neighbor_iter).GetFormalCharge())
        for neighbor_iter in ob.OBAtomAtomIter(atom)
        if not cast(ob.OBAtom, neighbor_iter).IsMetal()
    )
    return formal_charge + adjacent_charge == 0


def _has_adjacent_anionic_polarization_cancellation(atom: ob.OBAtom) -> bool:
    formal_charge = int(atom.GetFormalCharge())
    central_electronegativity = consts.NON_METAL_PAULING_ELECTRONEGATIVITY.get(
        int(atom.GetAtomicNum())
    )
    if formal_charge <= 0 or central_electronegativity is None:
        return False
    adjacent_negative_charge = 0
    for neighbor_iter in ob.OBAtomAtomIter(atom):
        neighbor = cast(ob.OBAtom, neighbor_iter)
        neighbor_charge = int(neighbor.GetFormalCharge())
        neighbor_electronegativity = consts.NON_METAL_PAULING_ELECTRONEGATIVITY.get(
            int(neighbor.GetAtomicNum())
        )
        if (
            neighbor_charge < 0
            and neighbor_electronegativity is not None
            and neighbor_electronegativity > central_electronegativity
        ):
            adjacent_negative_charge += abs(neighbor_charge)
    return adjacent_negative_charge >= formal_charge


def _is_locally_charge_compensated_nonmetal_cation(atom: ob.OBAtom) -> bool:
    if atom.IsMetal() or int(atom.GetFormalCharge()) <= 0:
        return False
    if _has_adjacent_formal_charge_cancellation(atom):
        return True
    element_info = consts.NON_METAL_DICT.get(int(atom.GetAtomicNum()))
    if element_info is None or int(atom.GetTotalValence()) <= int(element_info.default_valence):
        return False

    adjacent_negative_charge = 0
    adjacent_positive_charge = 0
    for neighbor_iter in ob.OBAtomAtomIter(atom):
        neighbor = cast(ob.OBAtom, neighbor_iter)
        if neighbor.IsMetal():
            continue
        neighbor_charge = int(neighbor.GetFormalCharge())
        if neighbor_charge < 0:
            adjacent_negative_charge += abs(neighbor_charge)
        elif neighbor_charge > 0:
            adjacent_positive_charge += neighbor_charge
    return adjacent_negative_charge > adjacent_positive_charge


def _unsaturated_organic_cation_discordance_count(
    obmol: ob.OBMol,
) -> int:
    for atom_iter in ob.OBMolAtomIter(obmol):
        atom = cast(ob.OBAtom, atom_iter)
        if (
            _is_unsaturated_organic_cation(atom)
            and not bool(atom.IsAromatic())
            and not _is_locally_charge_compensated_nonmetal_cation(atom)
        ):
            return 1
    return 0


def _charge_localization_penalty_for_atom(
    atom: ob.OBAtom,
    *,
    is_conjugated: bool,
) -> float:
    formal_charge = cast(int, atom.GetFormalCharge())
    if formal_charge == 0:
        return 0.0

    magnitude = float(abs(formal_charge))
    atomic_num = cast(int, atom.GetAtomicNum())
    is_aromatic = bool(atom.IsAromatic())
    radical_electrons = cast(int, get_unpaired_electron_count(atom))

    def _normalize(value: float | None, *, lower: float, upper: float, fallback: float) -> float:
        if value is None or upper <= lower:
            return fallback
        clamped = min(max(value, lower), upper)
        return (clamped - lower) / (upper - lower)

    def _approx_local_electron_count(element_info: dataclasses.ElementInfo) -> int:
        return int(element_info.num_outer_electrons) - formal_charge + int(atom.GetTotalValence())

    def _generic_penalty() -> float:
        if atomic_num == 1:
            penalty = 4.0 * magnitude
        elif formal_charge < 0:
            if atomic_num in {9, 17, 35, 53}:
                penalty = 0.2 * magnitude
            elif atomic_num in {8, 16, 34, 52}:
                penalty = 0.3 * magnitude
            elif atomic_num in {7, 15, 33, 51}:
                penalty = 0.6 * magnitude
            elif atomic_num == 6:
                penalty = (1.5 if is_conjugated or is_aromatic else 4.0) * magnitude
            else:
                penalty = (1.0 if is_conjugated or is_aromatic else 2.0) * magnitude
        else:
            if atomic_num in {7, 15, 33, 51}:
                penalty = 0.4 * magnitude
            elif atomic_num in {8, 16, 34, 52}:
                penalty = 1.0 * magnitude
            elif atomic_num == 6:
                penalty = (1.2 if is_conjugated or is_aromatic else 3.0) * magnitude
            else:
                penalty = (0.8 if is_conjugated or is_aromatic else 1.8) * magnitude
        if not is_conjugated and not is_aromatic:
            penalty += 0.5 * magnitude
        return penalty

    def _main_group_charge_penalty() -> float | None:
        element_info = consts.NON_METAL_DICT.get(atomic_num)
        if element_info is None:
            return None

        local_electron_count = _approx_local_electron_count(element_info)
        total_valence = int(atom.GetTotalValence())
        default_valence = int(element_info.default_valence)
        shell_target = 2 if atomic_num == 1 else 8
        coordination_excess = max(total_valence - default_valence, 0)
        shell_deficiency = max(shell_target - local_electron_count, 0)
        shell_surplus = max(local_electron_count - shell_target, 0)
        ionization_norm = _normalize(
            consts.NON_METAL_FIRST_IONIZATION_ENERGY_EV.get(atomic_num),
            lower=7.5,
            upper=18.0,
            fallback=0.5,
        )
        electronegativity_norm = _normalize(
            consts.NON_METAL_PAULING_ELECTRONEGATIVITY.get(atomic_num),
            lower=1.9,
            upper=4.0,
            fallback=0.5,
        )
        local_environment_penalty = 0.05 * magnitude if not (is_conjugated or is_aromatic) else 0.0

        if formal_charge > 0 and shell_deficiency > 0:
            return magnitude * (
                0.45
                + 0.55 * float(shell_deficiency)
                + 0.9 * ionization_norm
                + local_environment_penalty
            )

        if local_electron_count >= shell_target and coordination_excess > 0:
            return magnitude * (
                0.18
                + 0.12 * float(coordination_excess)
                + 0.04 * float(shell_surplus)
                + local_environment_penalty
            )

        if formal_charge < 0 and local_electron_count >= shell_target:
            return magnitude * (
                0.10 + 0.60 * (1.0 - electronegativity_norm) + local_environment_penalty
            )

        return None

    penalty = _main_group_charge_penalty()
    if penalty is None:
        penalty = _generic_penalty()
    if radical_electrons > 0 and atomic_num == 6 and not (is_conjugated or is_aromatic):
        penalty += 2.0 * float(radical_electrons)
    return penalty


def _radical_localization_penalty_for_atom(
    atom: ob.OBAtom,
    *,
    is_conjugated: bool,
) -> float:
    radical_electrons = cast(int, get_unpaired_electron_count(atom))
    # A validated neutral C/N/P two-electron deficit can be represented as an
    # explicit singlet (0 unpaired electrons, one active lone pair). It remains
    # a localized carbene-/nitrene-/phosphinidene-like state. Singlet/triplet
    # relative stability is system-dependent, so both occupations contribute
    # equally until a reliable environment-specific model is available.
    localized_electron_equivalents = float(radical_electrons) + (
        2.0 if _is_explicit_singlet_two_electron_center(atom) else 0.0
    )
    if localized_electron_equivalents <= 0.0:
        return 0.0

    magnitude = localized_electron_equivalents
    atomic_num = cast(int, atom.GetAtomicNum())
    is_aromatic = bool(atom.IsAromatic())

    if atomic_num == 1:
        return 4.0 * magnitude
    if atomic_num == 6:
        return (0.6 if is_conjugated or is_aromatic else 2.5) * magnitude
    if atomic_num in {7, 15, 33, 51}:
        return (1.0 if is_conjugated or is_aromatic else 2.5) * magnitude
    if atomic_num in {8, 16, 34, 52}:
        return (1.5 if is_conjugated or is_aromatic else 3.0) * magnitude
    return (1.2 if is_conjugated or is_aromatic else 2.5) * magnitude


def _compute_organic_electronic_state_metrics(
    omol: pybel.Molecule,
    config: MolGRConfig | None = None,
) -> _OrganicElectronicStateMetrics:
    try:
        topology_metrics = compute_organic_topology_metrics(
            omol,
            (CONFIG if config is None else config).organic_topology,
        )
        obmol = cast(ob.OBMol, omol.OBMol)
        conjugated_atom_indices = set(topology_metrics.conjugated_atom_indices)
        radical_localization_penalty = 0.0
        singlet_two_electron_center_count = 0
        unsigned_charge_localization_penalty = 0.0
        signed_charge_penalties_by_atom_idx: dict[int, float] = {}
        neighbor_indices_by_atom_idx: DefaultDict[int, list[int]] = defaultdict(list)
        atoms_by_idx: dict[int, ob.OBAtom] = {}
        for atom_iter in ob.OBMolAtomIter(obmol):
            atom = cast(ob.OBAtom, atom_iter)
            atom_idx = atom.GetIdx() - 1
            atoms_by_idx[atom_idx] = atom
            singlet_two_electron_center_count += _is_explicit_singlet_two_electron_center(atom)
            radical_localization_penalty += _radical_localization_penalty_for_atom(
                atom,
                is_conjugated=(atom_idx in conjugated_atom_indices),
            )
            atom_charge_penalty = _charge_localization_penalty_for_atom(
                atom,
                is_conjugated=(atom_idx in conjugated_atom_indices),
            )
            unsigned_charge_localization_penalty += atom_charge_penalty
            signed_charge_penalties_by_atom_idx[atom_idx] = (
                math.copysign(
                    atom_charge_penalty,
                    int(atom.GetFormalCharge()),
                )
                if atom_charge_penalty
                else 0.0
            )

        polarity_inversion_penalty = 0.0
        for bond_iter in ob.OBMolBondIter(obmol):
            bond = cast(ob.OBBond, bond_iter)
            begin_idx = int(bond.GetBeginAtomIdx()) - 1
            end_idx = int(bond.GetEndAtomIdx()) - 1
            begin_penalty = signed_charge_penalties_by_atom_idx[begin_idx]
            end_penalty = signed_charge_penalties_by_atom_idx[end_idx]
            opposite_charges = begin_penalty * end_penalty < 0.0
            allows_cancellation = True
            if (
                opposite_charges
                and int(bond.GetBondOrder()) == 1
                and not bool(bond.IsAromatic())
                and not bool(bond.IsInRing())
                and not (
                    begin_idx in conjugated_atom_indices and end_idx in conjugated_atom_indices
                )
            ):
                positive_idx = begin_idx if begin_penalty > 0.0 else end_idx
                negative_idx = end_idx if begin_penalty > 0.0 else begin_idx
                positive_electronegativity = consts.NON_METAL_PAULING_ELECTRONEGATIVITY.get(
                    int(atoms_by_idx[positive_idx].GetAtomicNum())
                )
                negative_electronegativity = consts.NON_METAL_PAULING_ELECTRONEGATIVITY.get(
                    int(atoms_by_idx[negative_idx].GetAtomicNum())
                )
                if (
                    positive_electronegativity is not None
                    and negative_electronegativity is not None
                    and positive_electronegativity
                    > negative_electronegativity
                    + _MINIMUM_CHARGE_POLARITY_INVERSION_ELECTRONEGATIVITY_GAP
                ):
                    allows_cancellation = False
                    polarity_inversion_penalty += 2.0 * min(
                        abs(begin_penalty),
                        abs(end_penalty),
                    )
            if not allows_cancellation:
                continue
            neighbor_indices_by_atom_idx[begin_idx].append(end_idx)
            neighbor_indices_by_atom_idx[end_idx].append(begin_idx)

        # Cancellation is local: neutral atoms do not bridge otherwise remote
        # charge centers in one large ligand component.
        charge_localization_penalty = 0.0
        visited_charged_atom_indices: set[int] = set()
        for root_idx, root_penalty in signed_charge_penalties_by_atom_idx.items():
            if not root_penalty or root_idx in visited_charged_atom_indices:
                continue
            component_signed_penalty = 0.0
            pending_atom_indices = [root_idx]
            visited_charged_atom_indices.add(root_idx)
            while pending_atom_indices:
                atom_idx = pending_atom_indices.pop()
                component_signed_penalty += signed_charge_penalties_by_atom_idx[atom_idx]
                for neighbor_idx in neighbor_indices_by_atom_idx[atom_idx]:
                    if (
                        neighbor_idx in visited_charged_atom_indices
                        or not signed_charge_penalties_by_atom_idx[neighbor_idx]
                    ):
                        continue
                    visited_charged_atom_indices.add(neighbor_idx)
                    pending_atom_indices.append(neighbor_idx)
            charge_localization_penalty += abs(component_signed_penalty)
        component_cancellation = max(
            0.0,
            unsigned_charge_localization_penalty - charge_localization_penalty,
        )
        charge_localization_penalty += polarity_inversion_penalty

        return _OrganicElectronicStateMetrics(
            aromatic_atom_count=topology_metrics.aromatic_atom_count,
            aromatic_ring_count=topology_metrics.aromatic_ring_count,
            aromatic_stability_score=topology_metrics.aromatic_stability_score,
            conjugated_atom_count=topology_metrics.conjugated_atom_count,
            conjugated_bond_count=topology_metrics.conjugated_bond_count,
            max_conjugated_component_size=topology_metrics.max_conjugated_component_size,
            hyperconjugative_donor_count=topology_metrics.hyperconjugative_donor_count,
            hyperconjugation_score=topology_metrics.hyperconjugation_score,
            singlet_two_electron_center_count=singlet_two_electron_center_count,
            radical_localization_penalty=radical_localization_penalty,
            charge_localization_penalty=charge_localization_penalty,
            charge_localization_component_cancellation=component_cancellation,
            charge_localization_polarity_inversion_penalty=polarity_inversion_penalty,
        )
    except Exception:  # noqa: BLE001
        return _OrganicElectronicStateMetrics(
            aromatic_atom_count=0,
            aromatic_ring_count=0,
            aromatic_stability_score=0.0,
            conjugated_atom_count=0,
            conjugated_bond_count=0,
            max_conjugated_component_size=0,
            hyperconjugative_donor_count=0,
            hyperconjugation_score=0,
            singlet_two_electron_center_count=0,
            radical_localization_penalty=float("inf"),
            charge_localization_penalty=float("inf"),
            charge_localization_component_cancellation=0.0,
            charge_localization_polarity_inversion_penalty=0.0,
        )


def _annotate_candidate_discordance_features(
    candidate: MetalCandidateState,
    *,
    config: MolGRConfig | None = None,
) -> float:
    no_metal_state = candidate.no_metal_state
    if no_metal_state is None:
        raise ValueError("MetalCandidateState requires no_metal_state before discordance scoring")
    resolved_config = CONFIG if config is None else config
    metal_scoring_config = resolved_config.metal_scoring
    obmol = cast(ob.OBMol, no_metal_state.omol.OBMol)
    blocker_arrays = _build_coordination_blocker_arrays(
        obmol,
        metal_scoring_config=metal_scoring_config,
    )
    non_metal_atoms = _non_metal_atom_entries(obmol)

    inner_visible_diradical_count = 0
    inner_visible_same_sign_charge_count = 0
    visible_inner_atom_indices: set[int] = set()
    visible_atoms_by_metal: list[tuple[ob.OBAtom, ...]] = []

    for metal_state in candidate.metal_states:
        metal_charge_sign = _charge_sign(int(metal_state.valence))
        visible_atoms = _inner_visible_atoms_to_metal(
            non_metal_atoms,
            metal_state,
            metal_scoring_config=metal_scoring_config,
            blocker_arrays=blocker_arrays,
        )
        visible_atoms_by_metal.append(visible_atoms)
        for atom in visible_atoms:
            atom_idx = int(atom.GetIdx())
            visible_inner_atom_indices.add(atom_idx)
            if _is_inner_visible_diradical_discordance_atom(atom):
                inner_visible_diradical_count += 1

            formal_charge = int(atom.GetFormalCharge())
            atom_charge_sign = _charge_sign(formal_charge)
            has_inner_same_sign_charge = (metal_charge_sign == 0 and formal_charge > 0) or (
                metal_charge_sign != 0 and atom_charge_sign == metal_charge_sign
            )
            if (
                has_inner_same_sign_charge
                and not (formal_charge > 0 and bool(atom.IsAromatic()))
                and not _has_adjacent_anionic_polarization_cancellation(atom)
            ):
                inner_visible_same_sign_charge_count += 1

    visible_singlet_two_electron_center_count = sum(
        _is_explicit_singlet_two_electron_center(cast(ob.OBAtom, obmol.GetAtom(atom_idx)))
        for atom_idx in visible_inner_atom_indices
    )
    visible_singlet_carbene_count = sum(
        int(atom.GetAtomicNum()) == 6 and _is_explicit_singlet_two_electron_center(atom)
        for atom in (
            cast(ob.OBAtom, obmol.GetAtom(atom_idx)) for atom_idx in visible_inner_atom_indices
        )
    )
    excess_visible_singlet_two_electron_center_count = max(
        0,
        visible_singlet_two_electron_center_count - len(candidate.metal_states),
    )
    bent_cumulated_ring_allene_count = _bent_cumulated_ring_allene_count(obmol)

    outer_or_invisible_adjacent_double_charge_count = 0
    outer_or_invisible_adjacent_same_sign_double_charge_count = 0
    outer_or_invisible_adjacent_opposite_sign_double_charge_count = 0
    outer_or_invisible_adjacent_unknown_metal_sign_double_charge_count = 0
    inner_visible_adjacent_carbanion_pair_count = 0
    inner_visible_conjugated_carbanion_pair_count = (
        _inner_visible_conjugated_charged_carbon_pair_count(obmol, visible_inner_atom_indices)
    )
    (
        negative_metal_count,
        negative_metal_has_outer_sphere_cation_exception,
        negative_metal_has_positive_metal_counterion_exception,
    ) = _negative_metal_discordance_count(
        obmol,
        candidate.metal_states,
        metal_scoring_config=metal_scoring_config,
    )
    zero_valent_metals_with_organic_cation_count = _zero_valent_metals_with_organic_cation_count(
        obmol,
        candidate.metal_states,
        total_charge=int(no_metal_state.total_charge)
        + sum(int(metal_state.valence) for metal_state in candidate.metal_states),
    )
    unsaturated_organic_cation_discordance_count = _unsaturated_organic_cation_discordance_count(
        obmol
    )
    for bond_iter in ob.OBMolBondIter(obmol):
        bond = cast(ob.OBBond, bond_iter)
        begin_atom = cast(ob.OBAtom, bond.GetBeginAtom())
        end_atom = cast(ob.OBAtom, bond.GetEndAtom())
        begin_charge = int(begin_atom.GetFormalCharge())
        end_charge = int(end_atom.GetFormalCharge())
        begin_charge_sign = _charge_sign(begin_charge)
        end_charge_sign = _charge_sign(end_charge)
        if begin_charge_sign == 0 or end_charge_sign == 0:
            continue
        if begin_charge_sign != end_charge_sign:
            continue

        begin_idx = int(begin_atom.GetIdx())
        end_idx = int(end_atom.GetIdx())
        pair_both_inner_visible = (
            begin_idx in visible_inner_atom_indices and end_idx in visible_inner_atom_indices
        )
        is_inner_visible_adjacent_carbanion_pair = (
            pair_both_inner_visible
            and begin_atom.GetAtomicNum() == 6
            and end_atom.GetAtomicNum() == 6
            and begin_charge_sign == end_charge_sign
        )
        is_resonance_mobile_outer_ring_carbanion_pair = (
            not pair_both_inner_visible
            and begin_idx not in visible_inner_atom_indices
            and end_idx not in visible_inner_atom_indices
            and begin_atom.GetAtomicNum() == 6
            and end_atom.GetAtomicNum() == 6
            and begin_charge < 0
            and end_charge < 0
            and bool(begin_atom.IsInRing())
            and bool(end_atom.IsInRing())
        )
        if is_inner_visible_adjacent_carbanion_pair:
            inner_visible_adjacent_carbanion_pair_count += 1
        elif not pair_both_inner_visible and not is_resonance_mobile_outer_ring_carbanion_pair:
            outer_or_invisible_adjacent_double_charge_count += 1
            metal_charge_sign = _nearest_nonzero_metal_charge_sign_to_bond(
                begin_atom,
                end_atom,
                candidate.metal_states,
            )
            if metal_charge_sign == 0:
                outer_or_invisible_adjacent_unknown_metal_sign_double_charge_count += 1
            elif begin_charge_sign == metal_charge_sign:
                outer_or_invisible_adjacent_same_sign_double_charge_count += 1
            else:
                outer_or_invisible_adjacent_opposite_sign_double_charge_count += 1

    repeated_component_charge_asymmetry_count = _repeated_component_charge_asymmetry_count(obmol)
    haptic_arene_reduction_count = _haptic_arene_reduction_count(
        obmol,
        visible_atoms_by_metal,
    )
    coordination_geometry_discordance_count = _coordination_geometry_discordance_count(
        visible_atoms_by_metal,
        candidate.metal_states,
        config=config,
    )
    negative_metal_penalty = _NEGATIVE_METAL_DISCORDANCE_PENALTY * negative_metal_count
    structural_discordance_count = (
        inner_visible_diradical_count
        + outer_or_invisible_adjacent_double_charge_count
        + inner_visible_adjacent_carbanion_pair_count
        + inner_visible_conjugated_carbanion_pair_count
        + inner_visible_same_sign_charge_count
        + excess_visible_singlet_two_electron_center_count
        + bent_cumulated_ring_allene_count
        + negative_metal_penalty
        + zero_valent_metals_with_organic_cation_count
        + unsaturated_organic_cation_discordance_count
        + repeated_component_charge_asymmetry_count
        + haptic_arene_reduction_count
        + coordination_geometry_discordance_count
    )
    discordance_count = structural_discordance_count
    candidate.metadata["metal_discordance_structural_count"] = structural_discordance_count
    candidate.metadata["metal_discordance_conjugated_atom_deficit_count"] = 0
    candidate.metadata["metal_discordance_conjugated_bond_deficit_count"] = 0
    candidate.metadata["metal_discordance_aromatic_atom_deficit_count"] = 0
    candidate.metadata["metal_discordance_aromatic_ring_deficit_count"] = 0
    candidate.metadata["metal_discordance_count"] = discordance_count
    candidate.metadata["metal_discordance_inner_visible_diradical_count"] = (
        inner_visible_diradical_count
    )
    candidate.metadata["metal_discordance_excess_visible_singlet_two_electron_center_count"] = (
        excess_visible_singlet_two_electron_center_count
    )
    candidate.metadata["organic_visible_singlet_carbene_count"] = visible_singlet_carbene_count
    candidate.metadata["metal_discordance_bent_cumulated_ring_allene_count"] = (
        bent_cumulated_ring_allene_count
    )
    candidate.metadata["metal_discordance_outer_or_invisible_adjacent_double_charge_count"] = (
        outer_or_invisible_adjacent_double_charge_count
    )
    candidate.metadata[
        "metal_discordance_outer_or_invisible_adjacent_same_sign_double_charge_count"
    ] = outer_or_invisible_adjacent_same_sign_double_charge_count
    candidate.metadata[
        "metal_discordance_outer_or_invisible_adjacent_opposite_sign_double_charge_count"
    ] = outer_or_invisible_adjacent_opposite_sign_double_charge_count
    candidate.metadata[
        "metal_discordance_outer_or_invisible_adjacent_unknown_metal_sign_double_charge_count"
    ] = outer_or_invisible_adjacent_unknown_metal_sign_double_charge_count
    candidate.metadata["metal_discordance_inner_visible_adjacent_carbanion_pair_count"] = (
        inner_visible_adjacent_carbanion_pair_count
    )
    candidate.metadata["metal_discordance_inner_visible_conjugated_carbanion_pair_count"] = (
        inner_visible_conjugated_carbanion_pair_count
    )
    candidate.metadata["metal_discordance_inner_visible_same_sign_charge_count"] = (
        inner_visible_same_sign_charge_count
    )
    candidate.metadata["metal_discordance_negative_metal_count"] = negative_metal_count
    candidate.metadata["metal_discordance_negative_metal_penalty"] = negative_metal_penalty
    candidate.metadata["metal_discordance_negative_metal_outer_sphere_cation_exception"] = (
        negative_metal_has_outer_sphere_cation_exception
    )
    candidate.metadata["metal_discordance_negative_metal_positive_metal_counterion_exception"] = (
        negative_metal_has_positive_metal_counterion_exception
    )
    candidate.metadata["metal_discordance_zero_valent_metals_with_organic_cation_count"] = (
        zero_valent_metals_with_organic_cation_count
    )
    candidate.metadata["metal_discordance_unsaturated_organic_cation_count"] = (
        unsaturated_organic_cation_discordance_count
    )
    candidate.metadata["metal_discordance_repeated_component_charge_asymmetry_count"] = (
        repeated_component_charge_asymmetry_count
    )
    candidate.metadata["metal_discordance_haptic_arene_reduction_count"] = (
        haptic_arene_reduction_count
    )
    candidate.metadata["metal_discordance_coordination_geometry_count"] = (
        coordination_geometry_discordance_count
    )
    return discordance_count


def _annotate_organic_electronic_state_consistency(
    candidate: MetalCandidateState,
    *,
    config: MolGRConfig | None = None,
) -> None:
    no_metal_state = candidate.no_metal_state
    if no_metal_state is None:
        raise ValueError("MetalCandidateState requires no_metal_state before organic-state scoring")

    resolved_config = CONFIG if config is None else config
    topology_config = resolved_config.organic_topology
    cache_key = f"organic_electronic_state_metrics:{astuple(topology_config)!r}"
    cached_metrics = cast(
        Optional[_OrganicElectronicStateMetrics],
        no_metal_state.metadata.get(cache_key),
    )
    if cached_metrics is None:
        cached_metrics = no_metal_state.get_cached_omol_value(
            cache_key,
            lambda omol: _compute_organic_electronic_state_metrics(omol, config=config),
        )
        no_metal_state.metadata[cache_key] = cached_metrics
        no_metal_state.metadata["organic_electronic_state_metrics"] = cached_metrics

    candidate.metadata["organic_aromatic_atom_count"] = cached_metrics.aromatic_atom_count
    candidate.metadata["organic_aromatic_ring_count"] = cached_metrics.aromatic_ring_count
    candidate.metadata["organic_aromatic_stability_score"] = cached_metrics.aromatic_stability_score
    candidate.metadata["organic_conjugated_atom_count"] = cached_metrics.conjugated_atom_count
    candidate.metadata["organic_conjugated_bond_count"] = cached_metrics.conjugated_bond_count
    candidate.metadata["organic_max_conjugated_component_size"] = (
        cached_metrics.max_conjugated_component_size
    )
    candidate.metadata["organic_hyperconjugative_donor_count"] = (
        cached_metrics.hyperconjugative_donor_count
    )
    candidate.metadata["organic_hyperconjugation_score"] = cached_metrics.hyperconjugation_score
    candidate.metadata["organic_singlet_two_electron_center_count"] = (
        cached_metrics.singlet_two_electron_center_count
    )
    candidate.metadata["organic_radical_localization_penalty"] = (
        cached_metrics.radical_localization_penalty
    )
    candidate.metadata["organic_charge_localization_penalty"] = (
        cached_metrics.charge_localization_penalty
    )
    candidate.metadata["organic_charge_localization_component_cancellation"] = (
        cached_metrics.charge_localization_component_cancellation
    )
    candidate.metadata["organic_charge_localization_polarity_inversion_penalty"] = (
        cached_metrics.charge_localization_polarity_inversion_penalty
    )


def _prepare_candidate_with_no_metal_state(
    candidate: MetalCandidateState,
    no_metal_state: ReconstructionState,
    *,
    config: MolGRConfig | None = None,
) -> MetalCandidateState:
    """Attach the shared no-metal reconstruction and count discordance features."""

    candidate_machine = MetalCandidateStateMachine.from_candidate_state(candidate)
    candidate_machine.set_no_metal_state("reconstruct_no_metal", no_metal_state)
    candidate_machine.annotate("score_candidate")
    scored_candidate = candidate_machine.freeze()
    if config is None:
        scored_candidate.combined_score()
    else:
        scored_candidate.combined_score(config=config)
    _annotate_candidate_discordance_features(scored_candidate, config=config)
    return scored_candidate


def _annotate_selected_candidate_metrics(
    candidate: MetalCandidateState,
    *,
    config: MolGRConfig | None = None,
) -> None:
    if candidate.score is None:
        if config is None:
            candidate.combined_score()
        else:
            candidate.combined_score(config=config)
    if "organic_aromatic_atom_count" not in candidate.metadata:
        _annotate_organic_electronic_state_consistency(candidate, config=config)


def _ensure_candidate_organic_metrics(
    candidate: MetalCandidateState,
    *,
    config: MolGRConfig | None = None,
) -> None:
    if "organic_aromatic_ring_count" not in candidate.metadata:
        _annotate_organic_electronic_state_consistency(candidate, config=config)
    if "organic_aromatic_stability_score" not in candidate.metadata:
        candidate.metadata["organic_aromatic_stability_score"] = float(
            candidate.metadata.get("organic_aromatic_ring_count", 0)
        )


def _annotate_candidate_set_discordance_features(
    candidates: Sequence[MetalCandidateState],
    *,
    config: MolGRConfig | None = None,
) -> None:
    if not candidates:
        return

    max_aromatic_atom_count = 0
    max_aromatic_ring_count = 0
    max_aromatic_stability_score = 0.0
    max_conjugated_atom_count = 0
    max_conjugated_bond_count = 0
    max_hyperconjugation_score = 0
    for candidate in candidates:
        _ensure_candidate_organic_metrics(candidate, config=config)
        max_conjugated_atom_count = max(
            max_conjugated_atom_count,
            int(candidate.metadata.get("organic_conjugated_atom_count", 0)),
        )
        max_conjugated_bond_count = max(
            max_conjugated_bond_count,
            int(candidate.metadata.get("organic_conjugated_bond_count", 0)),
        )
        max_hyperconjugation_score = max(
            max_hyperconjugation_score,
            int(candidate.metadata.get("organic_hyperconjugation_score", 0)),
        )
        max_aromatic_atom_count = max(
            max_aromatic_atom_count,
            int(candidate.metadata.get("organic_aromatic_atom_count", 0)),
        )
        max_aromatic_ring_count = max(
            max_aromatic_ring_count,
            int(candidate.metadata.get("organic_aromatic_ring_count", 0)),
        )
        max_aromatic_stability_score = max(
            max_aromatic_stability_score,
            float(candidate.metadata.get("organic_aromatic_stability_score", 0.0)),
        )

    for candidate in candidates:
        structural_discordance_count = float(
            candidate.metadata.get(
                "metal_discordance_structural_count",
                candidate.metadata.get("metal_discordance_count", 0),
            )
        )
        conjugated_atom_deficit_count = max(
            0,
            max_conjugated_atom_count
            - int(candidate.metadata.get("organic_conjugated_atom_count", 0)),
        )
        conjugated_bond_deficit_count = max(
            0,
            max_conjugated_bond_count
            - int(candidate.metadata.get("organic_conjugated_bond_count", 0)),
        )
        aromatic_atom_deficit_count = max(
            0,
            max_aromatic_atom_count - int(candidate.metadata.get("organic_aromatic_atom_count", 0)),
        )
        aromatic_ring_deficit_count = max(
            0,
            max_aromatic_ring_count - int(candidate.metadata.get("organic_aromatic_ring_count", 0)),
        )
        aromatic_stability_deficit = max(
            0.0,
            max_aromatic_stability_score
            - float(candidate.metadata.get("organic_aromatic_stability_score", 0.0)),
        )
        hyperconjugation_deficit = max(
            0,
            max_hyperconjugation_score
            - int(candidate.metadata.get("organic_hyperconjugation_score", 0)),
        )
        candidate.metadata["metal_discordance_structural_count"] = structural_discordance_count
        candidate.metadata["metal_discordance_max_conjugated_atom_count"] = (
            max_conjugated_atom_count
        )
        candidate.metadata["metal_discordance_conjugated_atom_deficit_count"] = (
            conjugated_atom_deficit_count
        )
        candidate.metadata["metal_discordance_max_conjugated_bond_count"] = (
            max_conjugated_bond_count
        )
        candidate.metadata["metal_discordance_conjugated_bond_deficit_count"] = (
            conjugated_bond_deficit_count
        )
        candidate.metadata["metal_discordance_max_aromatic_atom_count"] = max_aromatic_atom_count
        candidate.metadata["metal_discordance_aromatic_atom_deficit_count"] = (
            aromatic_atom_deficit_count
        )
        candidate.metadata["metal_discordance_max_aromatic_ring_count"] = max_aromatic_ring_count
        candidate.metadata["metal_discordance_max_aromatic_stability_score"] = (
            max_aromatic_stability_score
        )
        candidate.metadata["metal_discordance_aromatic_ring_deficit_count"] = (
            aromatic_ring_deficit_count
        )
        candidate.metadata["metal_discordance_aromatic_stability_deficit"] = (
            aromatic_stability_deficit
        )
        candidate.metadata["organic_hyperconjugation_max_score"] = max_hyperconjugation_score
        candidate.metadata["organic_hyperconjugation_deficit"] = hyperconjugation_deficit
        candidate.metadata["metal_discordance_count"] = structural_discordance_count


def select_best_candidate(
    scored_candidates: Sequence[MetalCandidateState],
    *,
    config: MolGRConfig | None = None,
) -> Optional[MetalCandidateState]:
    if not scored_candidates:
        return None

    _annotate_candidate_set_discordance_features(scored_candidates, config=config)
    min_discordance_count = min(
        float(candidate.metadata.get("metal_discordance_count", 0))
        for candidate in scored_candidates
    )
    discordance_filtered_candidates: list[MetalCandidateState] = []
    for scored_candidate in scored_candidates:
        discordance_count = float(scored_candidate.metadata.get("metal_discordance_count", 0))
        passes_discordance_filter = discordance_count == min_discordance_count
        scored_candidate.metadata["passes_metal_discordance_filter"] = passes_discordance_filter
        if passes_discordance_filter:
            discordance_filtered_candidates.append(scored_candidate)

    for scored_candidate in discordance_filtered_candidates:
        _annotate_selected_candidate_metrics(scored_candidate, config=config)

    # Keep distinct ligand radical budgets separate. Compare singlet-center
    # occupations of metal-visible carbon centers when aromatic counts are unchanged, so a
    # neutral carbene can veto unnecessary ligand charge separation. Otherwise,
    # retain occupation-specific references so aromatic anions remain competitive.
    def charge_reference_key(candidate: MetalCandidateState) -> tuple[float, int]:
        return (
            float(candidate.metadata.get("organic_charge_localization_penalty", 0.0)),
            int(candidate.metadata.get("combination_index", 0)),
        )

    def aromatic_reference_key(candidate: MetalCandidateState) -> tuple[int, int, int, int]:
        return (
            candidate.no_metal_radical_target,
            int(candidate.metadata.get("organic_singlet_two_electron_center_count", 0))
            - int(candidate.metadata.get("organic_visible_singlet_carbene_count", 0)),
            int(candidate.metadata.get("organic_aromatic_atom_count", 0)),
            int(candidate.metadata.get("organic_aromatic_ring_count", 0)),
        )

    charge_reference_by_occupation: dict[tuple[int, int], MetalCandidateState] = {}
    charge_reference_by_aromaticity: dict[tuple[int, int, int, int], MetalCandidateState] = {}
    for candidate in discordance_filtered_candidates:
        target = (
            candidate.no_metal_radical_target,
            int(candidate.metadata.get("organic_singlet_two_electron_center_count", 0)),
        )
        reference = charge_reference_by_occupation.get(target)
        key = (
            float(candidate.metadata.get("organic_charge_localization_penalty", 0.0)),
            int(candidate.metadata.get("combination_index", 0)),
        )
        if reference is None or key < (
            float(reference.metadata.get("organic_charge_localization_penalty", 0.0)),
            int(reference.metadata.get("combination_index", 0)),
        ):
            charge_reference_by_occupation[target] = candidate
        aromatic_target = aromatic_reference_key(candidate)
        aromatic_reference = charge_reference_by_aromaticity.get(aromatic_target)
        if aromatic_reference is None or key < charge_reference_key(aromatic_reference):
            charge_reference_by_aromaticity[aromatic_target] = candidate

    resolved_config = CONFIG if config is None else config
    charge_localization_margin = max(
        0.0,
        float(resolved_config.metal_scoring.charge_localization_selection_margin),
    )
    best_candidate: Optional[MetalCandidateState] = None
    best_selection_key: Optional[
        Tuple[float, int, int, int, int, int, float, float, int, float, int]
    ] = None
    for scored_candidate in discordance_filtered_candidates:
        charge_localization_reference_candidate = charge_reference_by_occupation[
            (
                scored_candidate.no_metal_radical_target,
                int(scored_candidate.metadata.get("organic_singlet_two_electron_center_count", 0)),
            )
        ]
        charge_localization_reference_candidate = min(
            charge_localization_reference_candidate,
            charge_reference_by_aromaticity[aromatic_reference_key(scored_candidate)],
            key=charge_reference_key,
        )
        minimum_charge_localization_penalty = float(
            charge_localization_reference_candidate.metadata.get(
                "organic_charge_localization_penalty", 0.0
            )
        )
        reference_valences = {
            int(state.idx): int(state.valence)
            for state in charge_localization_reference_candidate.metal_states
        }
        score_value = (
            cast(float, scored_candidate.score)
            if scored_candidate.score is not None
            else scored_candidate.combined_score(config=config)
        )
        charge_localization_penalty = float(
            scored_candidate.metadata.get("organic_charge_localization_penalty", 0.0)
        )
        charge_localization_difference = max(
            0.0,
            charge_localization_penalty - minimum_charge_localization_penalty,
        )
        reference_valence_deltas = [
            abs(int(state.valence) - reference_valences[int(state.idx)])
            for state in scored_candidate.metal_states
            if int(state.idx) in reference_valences
        ]
        reference_valence_max_delta = max(reference_valence_deltas, default=0)
        oxidation_state_jump_exceeded = (
            charge_localization_difference > 0.0
            and reference_valence_max_delta
            > _MAX_CHARGE_LOCALIZATION_REFERENCE_OXIDATION_STATE_DELTA
        )
        charge_localization_margin_exceeded = oxidation_state_jump_exceeded or (
            charge_localization_difference > 0.0
            and (
                charge_localization_difference > charge_localization_margin
                or math.isclose(
                    charge_localization_difference,
                    charge_localization_margin,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
        )
        scored_candidate.metadata["organic_charge_localization_reference_penalty"] = (
            minimum_charge_localization_penalty
        )
        scored_candidate.metadata["organic_charge_localization_selection_margin"] = (
            charge_localization_margin
        )
        scored_candidate.metadata["organic_charge_localization_margin_difference"] = (
            charge_localization_difference
        )
        scored_candidate.metadata["organic_charge_localization_margin_exceeded"] = (
            charge_localization_margin_exceeded
        )
        scored_candidate.metadata[
            "organic_charge_localization_reference_metal_valence_max_delta"
        ] = reference_valence_max_delta
        scored_candidate.metadata["organic_charge_localization_metal_valence_jump_exceeded"] = (
            oxidation_state_jump_exceeded
        )
        selection_key = (
            float(min_discordance_count),
            int(charge_localization_margin_exceeded),
            int(
                scored_candidate.metadata.get("metal_discordance_conjugated_atom_deficit_count", 0)
            ),
            int(
                scored_candidate.metadata.get("metal_discordance_conjugated_bond_deficit_count", 0)
            ),
            int(scored_candidate.metadata.get("metal_discordance_aromatic_atom_deficit_count", 0)),
            int(scored_candidate.metadata.get("metal_discordance_aromatic_ring_deficit_count", 0)),
            float(
                scored_candidate.metadata.get("metal_discordance_aromatic_stability_deficit", 0.0)
            ),
            float(scored_candidate.metadata.get("organic_radical_localization_penalty", 0.0)),
            int(scored_candidate.metadata.get("organic_hyperconjugation_deficit", 0)),
            float(score_value),
            int(scored_candidate.metadata.get("combination_index", 0)),
        )
        scored_candidate.metadata["selection_key"] = selection_key
        if best_selection_key is not None and selection_key >= best_selection_key:
            continue
        best_selection_key = selection_key
        best_candidate = scored_candidate
    return best_candidate


__all__ = [
    "_charge_localization_penalty_for_atom",
    "_annotate_candidate_set_discordance_features",
    "_radical_localization_penalty_for_atom",
    "_prepare_candidate_with_no_metal_state",
    "select_best_candidate",
]
