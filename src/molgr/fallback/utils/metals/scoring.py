"""Scoring and selection utilities for metal-aware fallback reconstruction."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import astuple, dataclass
from itertools import combinations
from typing import AbstractSet, DefaultDict, List, Optional, Set, Tuple, cast

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
from molgr.fallback.utils.bond_geometry import (
    equalized_three_carbon_chains,
    folded_eight_carbon_segments,
    has_conjugated_thioamide_donor,
    is_elongated_carbonyl_pair,
    planar_aza_macrocycle_circuits,
    planar_phenoxy_formazan_sites,
    planar_tetraazaporphyrin_circuits,
    planar_tetrapyrrole_circuits,
    short_quinodiimine_conflicts,
)
from molgr.fallback.utils.coordination_geometry import (
    build_coordination_blocker_arrays,
    inner_visible_atoms_to_metal,
    metal_donor_angle,
    terminal_nitrosyl_contacts,
)
from molgr.fallback.utils.electrons import (
    get_lone_pair_count,
    get_unpaired_electron_count,
    has_cationic_lewis_pair,
    has_unresolved_two_electron_center,
)
from molgr.fallback.utils.metals import conjugated_geometry, hydrazido_geometry, local_cc_geometry
from molgr.fallback.utils.ob_graph import (
    atom_bonds,
    atom_neighbors,
    indexed_bond_order,
    mol_atoms,
    mol_bonds,
    mol_rings,
    neighbor_bond_order,
)
from molgr.fallback.utils.organic_topology import compute_organic_topology_metrics
from molgr.utils.coordination import coordination_distance_cutoff


_NEGATIVE_METAL_DISCORDANCE_PENALTY = 0.5
# A retained donor pair reduces cation repulsion; it does not make it neutral.
_CATIONIC_DONOR_DISCORDANCE_PENALTY = 0.5
_MINIMUM_RING_ALLENE_ANGLE_DEGREES = 150.0
_MINIMUM_CHARGE_POLARITY_INVERSION_ELECTRONEGATIVITY_GAP = 0.3
_MAX_CHARGE_LOCALIZATION_REFERENCE_OXIDATION_STATE_DELTA = 2
_CATION_COMPENSATION_MAX_STATES = 4096
_CATION_COMPENSATION_MAX_ATOMS = 32
_CATION_COMPENSATION_MAX_PATHS = 32
_CONNECTIVITY_HASH_OFFSET = 1469598103934665603
_CONNECTIVITY_HASH_PRIME = 1099511628211
_CONNECTIVITY_HASH_MASK = (1 << 64) - 1


@dataclass(frozen=True)
class _OrganicElectronicStateMetrics:
    """Ligand-only aromaticity, conjugation and electron-localization terms reused across metal assignments."""

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
        int(atom.GetAtomicNum()) in {6, 7, 14, 15}
        and int(atom.GetFormalCharge()) == 0
        and get_unpaired_electron_count(atom) == 0
        and get_lone_pair_count(atom) == 1
        and not has_unresolved_two_electron_center(atom)
        and assign_radical_dots(atom) == 2
    )


def _is_neutral_carbon_cumulene_center(center: ob.OBAtom) -> bool:
    if center.GetAtomicNum() != 6 or center.GetFormalCharge() != 0:
        return False
    neighbors = [
        cast(ob.OBAtom, bond.GetNbrAtom(center))
        for bond in atom_bonds(center)
        if bond.GetBondOrder() == 2 and not bond.IsAromatic()
    ]
    return len(neighbors) == 2 and all(
        atom.GetAtomicNum() == 6 and atom.GetFormalCharge() == 0 for atom in neighbors
    )


def _bent_cumulated_ring_allene_indices(
    obmol: ob.OBMol, visible_atom_indices: AbstractSet[int] = frozenset()
) -> set[int]:
    """Find bent ring centers outside the supplied metal-visible atom set."""
    indices: set[int] = set()
    for center in mol_atoms(obmol):
        if not bool(center.IsInRing()):
            continue
        if int(center.GetIdx()) in visible_atom_indices and _is_neutral_carbon_cumulene_center(
            center
        ):
            continue
        ring_double_neighbors: list[ob.OBAtom] = []
        for bond in atom_bonds(center):
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
            indices.add(int(center.GetIdx()))
    return indices


def _bent_cumulated_ring_allene_count(
    obmol: ob.OBMol, visible_atom_indices: AbstractSet[int] = frozenset()
) -> int:
    return len(_bent_cumulated_ring_allene_indices(obmol, visible_atom_indices))


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
    for atom in mol_atoms(obmol):
        if atom.IsMetal():
            continue
        atoms.append(atom)
    return tuple(atoms)


def _component_atom_index_groups(obmol: ob.OBMol) -> tuple[tuple[int, ...], ...]:
    unseen = {int(atom.GetIdx()) for atom in mol_atoms(obmol)}
    components: list[tuple[int, ...]] = []
    while unseen:
        start_idx = min(unseen)
        unseen.remove(start_idx)
        stack = [start_idx]
        component = [start_idx]
        while stack:
            atom = cast(ob.OBAtom, obmol.GetAtom(stack.pop()))
            for neighbor in atom_neighbors(atom):
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
                labels[int(neighbor_iter.GetIdx())] for neighbor_iter in atom_neighbors(atom)
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
    for ring in mol_rings(obmol):
        ring_atom_indices = tuple(int(idx) for idx in getattr(ring, "_path", ()))
        if len(ring_atom_indices) not in {5, 6}:
            continue
        ring_atoms = [cast(ob.OBAtom, obmol.GetAtom(idx)) for idx in ring_atom_indices]
        # A tetrahedral carbon interrupts the pi system independently of charge.
        # Partially saturated rings must not inherit an arene-reduction veto.
        if any(
            int(atom.GetAtomicNum()) != 6 or atom.GetExplicitDegree() + atom.GetImplicitHCount() > 3
            for atom in ring_atoms
        ):
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


def _vector_norm(vector: Sequence[float]) -> float:
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
        if metal_state.symbol == "Au":
            # The visibility shell also includes remote C/H contacts. Assign
            # the Au geometry from the short first shell, using element radii.
            visible_atoms = tuple(
                a
                for a in visible_atoms
                if (
                    (a.GetX() - metal_state.position_x) ** 2
                    + (a.GetY() - metal_state.position_y) ** 2
                    + (a.GetZ() - metal_state.position_z) ** 2
                )
                ** 0.5
                <= coordination_distance_cutoff(
                    metal_state.element_idx,
                    a.GetAtomicNum(),
                    radius_scale=(
                        CONFIG if config is None else config
                    ).metal_scoring.metal_access_radius_scale,
                    extra_tolerance_angstrom=0.20,
                )
            )
        geometry = _coordination_geometry(
            visible_atoms,
            metal_state,
            config=config,
        )
        if (
            (
                metal_state.symbol in {"Pd", "Pt"}
                and metal_state.valence >= 4
                and geometry == "square_planar"
            )
            or (
                metal_state.symbol in {"Ag", "Au"}
                and metal_state.valence >= 3
                and geometry == "linear"
            )
            or (
                metal_state.symbol == "Au"
                and metal_state.valence == 1
                and geometry == "square_planar"
                and (
                    all(a.GetAtomicNum() == 7 for a in visible_atoms)
                    or (
                        sorted(a.GetAtomicNum() for a in visible_atoms) == [6, 6, 15, 15]
                        and all(
                            a.GetFormalCharge() == -1
                            and all(n.GetAtomicNum() == 6 for n in atom_neighbors(a))
                            for a in visible_atoms
                            if a.GetAtomicNum() == 6
                        )
                    )
                )
                # Exclude a haptic contact: four independent ligand donors only.
                and all(
                    _bond_between_atoms(a, b) is None
                    for i, a in enumerate(visible_atoms)
                    for b in visible_atoms[i + 1 :]
                )
            )
        ):
            count += 1
    return count


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
    for bond in atom_bonds(lhs):
        if _other_bond_atom(bond, lhs).GetIdx() == rhs_idx:
            return bond
    return None


def _has_conjugated_bridge_between_charged_carbons(
    begin_atom: ob.OBAtom,
    end_atom: ob.OBAtom,
) -> bool:
    for begin_bond in atom_bonds(begin_atom):
        if _bond_order(begin_bond) != 1:
            continue
        begin_neighbor = _other_bond_atom(begin_bond, begin_atom)
        if begin_neighbor.GetAtomicNum() == 1:
            continue
        for middle_bond in atom_bonds(begin_neighbor):
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


def _pyramidal_sigma_carbanion(atom: ob.OBAtom, metal: dataclasses.MetalAtomPosition) -> bool:
    """Test if a pyramidal carbon donor's vacant direction faces this metal."""
    if (
        atom.GetFormalCharge() != -1
        or atom.GetExplicitDegree() != 3
        or atom.GetImplicitHCount()
        or get_unpaired_electron_count(atom)
        or has_unresolved_two_electron_center(atom)
    ):
        return False
    neighbors = list(atom_neighbors(atom))
    vacant = [0.0, 0.0, 0.0]
    for neighbor in neighbors:
        length = atom.GetDistance(neighbor)
        if neighbor_bond_order(atom, neighbor) != 1 or not math.isfinite(length) or length <= 1e-8:
            return False
        delta = (
            atom.GetX() - neighbor.GetX(),
            atom.GetY() - neighbor.GetY(),
            atom.GetZ() - neighbor.GetZ(),
        )
        for i in range(3):
            vacant[i] += delta[i] / length
    angles = [atom.GetParent().GetAngle(a, atom, b) for a, b in combinations(neighbors, 2)]
    if not all(angle > 0 for angle in angles) or not sum(angles) < 350:
        return False
    approach = (
        metal.position_x - atom.GetX(),
        metal.position_y - atom.GetY(),
        metal.position_z - atom.GetZ(),
    )
    denominator = _vector_norm(vacant) * _vector_norm(approach)
    return (
        math.isfinite(denominator)
        and denominator > 1e-8
        and sum(a * b for a, b in zip(vacant, approach)) / denominator > math.cos(math.pi / 6)
    )


def _inner_visible_conjugated_charged_carbon_pair_count(
    obmol: ob.OBMol,
    visible_inner_atom_indices: Set[int],
    visible_atoms_by_metal: Sequence[Sequence[ob.OBAtom]],
    metal_states: Sequence[dataclasses.MetalAtomPosition],
) -> int:
    charged_carbons: List[ob.OBAtom] = []
    for atom in mol_atoms(obmol):
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
                if any(
                    {begin_atom.GetIdx(), end_atom.GetIdx()} <= {a.GetIdx() for a in visible}
                    and _pyramidal_sigma_carbanion(begin_atom, metal)
                    and _pyramidal_sigma_carbanion(end_atom, metal)
                    for visible, metal in zip(visible_atoms_by_metal, metal_states)
                ):
                    continue
                count += 1
    return count


def _has_outer_sphere_proton(
    obmol: ob.OBMol,
    metal_states: Sequence[dataclasses.MetalAtomPosition],
    *,
    metal_scoring_config: MetalScoringConfig,
) -> bool:
    for atom in mol_atoms(obmol):
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
        and not (
            metal_state.symbol == "Au"
            and sum(
                a.GetAtomicNum() == 5
                and a.GetFormalCharge() == 0
                and a.GetTotalValence() == 3
                and _is_inner_sphere_atom(a, metal_state, metal_scoring_config=metal_scoring_config)
                for a in mol_atoms(obmol)
            )
            >= 2
        )
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


def _uncompensated_organic_cation_indices(
    obmol: ob.OBMol,
    *,
    coordinated_nitrosonium_indices: AbstractSet[int] = frozenset(),
    config: MolGRConfig | None = None,
) -> set[int]:
    uncompensated: set[int] = set()
    for atom in mol_atoms(obmol):
        if atom.IsMetal() or int(atom.GetFormalCharge()) <= 0:
            continue
        if int(atom.GetIdx()) in coordinated_nitrosonium_indices:
            continue
        if int(atom.GetAtomicNum()) > 0 and not _is_locally_charge_compensated_nonmetal_cation(
            atom
        ):
            uncompensated.add(int(atom.GetIdx()))
    if not uncompensated:
        return uncompensated

    # A neutral conjugated carbanion/cation representation is not a cationic
    # ligand. Do not cancel charges across saturated bridges or against remote
    # heteroatom anions; those require their own electronic-state evidence.
    resolved = CONFIG if config is None else config
    topology = compute_organic_topology_metrics(pybel.Molecule(obmol), resolved.organic_topology)
    for component in topology.conjugated_atom_components:
        atoms = [cast(ob.OBAtom, obmol.GetAtom(index + 1)) for index in component]
        if sum(atom.GetFormalCharge() for atom in atoms) != 0:
            continue
        if any(atom.GetAtomicNum() != 6 and atom.GetFormalCharge() < 0 for atom in atoms):
            continue
        uncompensated.difference_update(atom.GetIdx() for atom in atoms)
    return uncompensated


def _cation_compensation_paths(obmol: ob.OBMol, source: int) -> list[tuple[int, ...]] | None:
    """Complete bounded alternating paths; None means the search must abstain."""
    pending: deque[tuple[int, ...]] = deque([(source,)])
    paths: list[tuple[int, ...]] = []
    visited = 0
    while pending:
        visited += 1
        if visited > _CATION_COMPENSATION_MAX_STATES:
            return None
        path = pending.popleft()
        atom = cast(ob.OBAtom, obmol.GetAtom(path[-1]))
        order = 2 if len(path) % 2 else 1
        for bond in atom_bonds(atom):
            end = cast(ob.OBAtom, bond.GetNbrAtom(atom))
            idx = end.GetIdx()
            if (
                idx in path
                or bond.GetBondOrder() != order
                or end.GetFormalCharge() != 0
                or end.GetImplicitHCount()
                or get_unpaired_electron_count(end)
                or has_unresolved_two_electron_center(end)
            ):
                continue
            if end.GetAtomicNum() == 6 and end.GetTotalValence() == 4:
                if len(path) >= _CATION_COMPENSATION_MAX_ATOMS:
                    return None
                pending.append(path + (idx,))
                continue
            info = consts.NON_METAL_DICT.get(end.GetAtomicNum())
            if (
                order != 1
                or end.GetAtomicNum() not in (7, 8, 15, 16)
                or info is None
                or info.num_outer_electrons + end.GetTotalValence() != 8
                or info.num_outer_electrons - end.GetTotalValence() < 2
                or not any(a.GetFormalCharge() < 0 for a in atom_neighbors(end))
            ):
                continue
            if len(path) >= _CATION_COMPENSATION_MAX_ATOMS:
                return None
            paths.append(path + (idx,))
            if len(paths) > _CATION_COMPENSATION_MAX_PATHS:
                return None
    return paths


def _zero_valent_metals_with_organic_cation_count(
    obmol: ob.OBMol,
    metal_states: Sequence[dataclasses.MetalAtomPosition],
    *,
    total_charge: int,
    coordinated_nitrosonium_indices: AbstractSet[int] = frozenset(),
    config: MolGRConfig | None = None,
) -> int:
    if total_charge > 0:
        return 0
    if not metal_states or any(int(metal_state.valence) != 0 for metal_state in metal_states):
        return 0
    uncompensated = _uncompensated_organic_cation_indices(
        obmol, coordinated_nitrosonium_indices=coordinated_nitrosonium_indices, config=config
    )
    if not uncompensated:
        return 0
    if len(uncompensated) != 1:
        return 1
    source = cast(ob.OBAtom, obmol.GetAtom(next(iter(uncompensated))))
    info = consts.NON_METAL_DICT.get(source.GetAtomicNum())
    if (
        info is None
        or source.GetFormalCharge() != 1
        or source.GetImplicitHCount()
        or get_unpaired_electron_count(source)
        or has_unresolved_two_electron_center(source)
        or info.num_outer_electrons - 1 + source.GetTotalValence() != 8
        or info.num_outer_electrons - 1 - source.GetTotalValence() < 0
    ):
        return 1
    # Enumerate before granting credit, so exceeding a bound cannot return an
    # atom-order-dependent partial result. The candidate itself never changes.
    paths = _cation_compensation_paths(obmol, source.GetIdx())
    if paths is None:
        return 1
    for path in paths:
        witness = ob.OBMol(obmol)
        witness.GetAtom(path[0]).SetFormalCharge(0)
        witness.GetAtom(path[-1]).SetFormalCharge(1)
        for first, second in zip(path, path[1:]):
            bond = cast(ob.OBBond, witness.GetBond(first, second))
            bond.SetBondOrder(3 - bond.GetBondOrder())
        if not _uncompensated_organic_cation_indices(
            witness,
            coordinated_nitrosonium_indices=coordinated_nitrosonium_indices,
            config=config,
        ):
            return 0
    return 1


def _oxygen_bond_geometry_discordance(
    obmol: ob.OBMol, visible_atom_indices: AbstractSet[int]
) -> tuple[int, int]:
    """Count narrowly scoped C--O and coordinated O2 geometry conflicts.

    The length windows identify strong contradictions, not general bond-order
    assignments. Intermediate O--O lengths and substituted peroxides abstain.
    Carbon monoxide, CO2 and long, activated C--O bonds are outside the C--O rule.
    """
    short_carbonyl = 0
    long_dioxygen = 0
    for bond in mol_bonds(obmol):
        first, second = cast(ob.OBAtom, bond.GetBeginAtom()), cast(ob.OBAtom, bond.GetEndAtom())
        if first.GetAtomicNum() > second.GetAtomicNum():
            first, second = second, first
        length = first.GetDistance(second)
        if (
            first.GetAtomicNum() == 6
            and second.GetAtomicNum() == 8
            and first.GetExplicitDegree() >= 2
            and not (
                first.GetExplicitDegree() == 2
                and all(a.GetAtomicNum() == 8 for a in atom_neighbors(first))
            )
            and second.GetExplicitDegree() == 1
            and second.GetImplicitHCount() == 0
            and 1.10 < length < 1.30
            and bond.GetBondOrder() == 1
            and first.GetFormalCharge() < 0
            and second.GetFormalCharge() < 0
        ):
            short_carbonyl += 1
        if (
            first.GetAtomicNum() == second.GetAtomicNum() == 8
            and first.GetExplicitDegree() == second.GetExplicitDegree() == 1
            and first.GetImplicitHCount() == second.GetImplicitHCount() == 0
            and (first.GetIdx() in visible_atom_indices or second.GetIdx() in visible_atom_indices)
            and 1.40 <= length <= 1.65
            and bond.GetBondOrder() > 1
        ):
            long_dioxygen += 1
    return short_carbonyl, long_dioxygen


def _long_carbonyl_multiple_bond_count(
    obmol: ob.OBMol, visible_atoms_by_metal: Sequence[Sequence[ob.OBAtom]]
) -> int:
    """Count stretched C=O only when C and O are visible to the same metal."""
    sites = [{int(a.GetIdx()) for a in atoms} for atoms in visible_atoms_by_metal]
    return sum(
        bond.GetBondOrder() == 2
        and is_elongated_carbonyl_pair(bond.GetBeginAtom(), bond.GetEndAtom())
        and any(
            bond.GetBeginAtomIdx() in indices and bond.GetEndAtomIdx() in indices
            for indices in sites
        )
        for bond in mol_bonds(obmol)
    )


def _sulfur_bond_geometry_discordance(
    obmol: ob.OBMol, visible_atoms_by_metal: Sequence[Sequence[ob.OBAtom]]
) -> int:
    """Penalize localized sulfur pi bonds contradicted by a chelating skeleton.

    These are local selection safeguards, not a metal oxidation-state prior.
    Isolated thiones, thioamides and charge-separated ring C=S are excluded.
    """
    sites = [{int(a.GetIdx()) for a in atoms} for atoms in visible_atoms_by_metal]
    count = 0
    for bond in mol_bonds(obmol):
        left, right = cast(ob.OBAtom, bond.GetBeginAtom()), cast(ob.OBAtom, bond.GetEndAtom())
        if (
            left.GetAtomicNum() == right.GetAtomicNum() == 6
            and bond.GetBondOrder() == 1
            and 1.25 <= left.GetDistance(right) <= 1.44
        ):
            if has_conjugated_thioamide_donor(left) and has_conjugated_thioamide_donor(right):
                continue
            ends = []
            for carbon in (left, right):
                ends.append(
                    [
                        a
                        for a in atom_neighbors(carbon)
                        if a.GetAtomicNum() in (16, 34)
                        and a.GetExplicitDegree() == 1
                        and a.GetImplicitHCount() == 0
                        and a.GetFormalCharge() == 0
                        and neighbor_bond_order(carbon, a) == 2
                        and 1.72
                        <= carbon.GetDistance(a)
                        - (ob.GetCovalentRad(a.GetAtomicNum()) - ob.GetCovalentRad(16))
                        <= 1.90
                    ]
                )
            if any(
                a.GetIdx() in indices and b.GetIdx() in indices
                for a in ends[0]
                for b in ends[1]
                for indices in sites
            ):
                count += 1
        if left.GetAtomicNum() == 16:
            left, right = right, left
        if (
            left.GetAtomicNum() == 6
            and right.GetAtomicNum() == 16
            and bond.GetBondOrder() == 2
            and left.GetFormalCharge() == 0
            and right.GetFormalCharge() > 0
            and left.GetExplicitDegree() == 3
            and left.GetImplicitHCount() == 0
            and all(a.GetAtomicNum() == 16 for a in atom_neighbors(left))
            and 1.80 <= left.GetDistance(right) <= 2.05
            and any(left.GetIdx() in indices for indices in sites)
        ):
            count += 1
    return count


def _nitrosyl_geometry_metrics(
    obmol: ob.OBMol,
    metal_states: Sequence[dataclasses.MetalAtomPosition],
    visible_atoms_by_metal: Sequence[Sequence[ob.OBAtom]],
) -> Tuple[float, Set[int], int]:
    """A restricted NO+ representation preference, not an oxidation-state theorem.

    Only isolated, hydrogen-free, terminal N-bound NO with short N--O and
    near-linear M--N--O contributes. Bent, bridging and O-bound sites abstain.
    This preference resolves ligand alternatives within each metal assignment,
    then ranks structurally admissible assignments before detached-ligand scores.
    Charge/spin constraints and the structural safeguards remain intact.
    """
    penalty = 0.0
    nitrosonium_indices: Set[int] = set()
    count = 0
    for nitrogen, oxygen, weight in terminal_nitrosyl_contacts(
        obmol, [metal.site for metal in metal_states], visible_atoms_by_metal
    ):
        charge = int(nitrogen.GetFormalCharge()) + int(oxygen.GetFormalCharge())
        penalty += weight * ((charge - 1) / 2.0) ** 2
        count += 1
        bond = cast(Optional[ob.OBBond], nitrogen.GetBond(oxygen))
        if (
            charge == 1
            and bond is not None
            and bond.GetBondOrder() == 3
            and get_unpaired_electron_count(nitrogen) == 0
            and get_unpaired_electron_count(oxygen) == 0
        ):
            nitrosonium_indices.update((int(nitrogen.GetIdx()), int(oxygen.GetIdx())))
    return penalty, nitrosonium_indices, count


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


def _is_cationic_paired_electron_donor(atom: ob.OBAtom) -> bool:
    element_info = consts.NON_METAL_DICT.get(int(atom.GetAtomicNum()))
    return element_info is not None and has_cationic_lewis_pair(
        int(element_info.num_outer_electrons),
        int(atom.GetFormalCharge()),
        int(atom.GetTotalValence()),
        get_unpaired_electron_count(atom),
        has_unresolved_two_electron_center(atom),
    )


def _has_adjacent_formal_charge_cancellation(atom: ob.OBAtom) -> bool:
    formal_charge = int(atom.GetFormalCharge())
    if formal_charge == 0:
        return False
    adjacent_charge = sum(
        int(neighbor_iter.GetFormalCharge())
        for neighbor_iter in atom_neighbors(atom)
        if not neighbor_iter.IsMetal()
    )
    return formal_charge + adjacent_charge == 0


def _nitrosonium_formal_charge_exemption(
    obmol: ob.OBMol,
    metal_states: Sequence[dataclasses.MetalAtomPosition],
    nitrosonium_indices: AbstractSet[int],
    penalized_negative_donor_indices: AbstractSet[int],
    negative_metal_penalty: float,
    visible_atom_indices: AbstractSet[int] = frozenset(),
) -> float:
    """Discount the formal metal charge explained by resolved terminal NO+ ligands."""
    if (
        len(metal_states) != 1
        or metal_states[0].valence >= 0
        or not nitrosonium_indices
        or len(nitrosonium_indices) % 2
    ):
        return 0.0
    ancillary_charge = 0
    anionic_component_indices: set[int] = set()
    has_linear_carbonyl = False
    metal = metal_states[0]
    for indices in _component_atom_index_groups(obmol):
        if not nitrosonium_indices.isdisjoint(indices):
            continue
        atoms = [cast(ob.OBAtom, obmol.GetAtom(i)) for i in indices]
        if any(
            a.GetImplicitHCount()
            or get_unpaired_electron_count(a)
            or has_unresolved_two_electron_center(a)
            for a in atoms
        ):
            return 0.0
        charge = sum(a.GetFormalCharge() for a in atoms)
        if charge > 0:
            return 0.0
        ancillary_charge += charge
        if charge < 0:
            anionic_component_indices.update(indices)
        elif len(atoms) == 2:
            carbon, oxygen = sorted(atoms, key=lambda a: a.GetAtomicNum())
            if (
                (carbon.GetAtomicNum(), oxygen.GetAtomicNum()) == (6, 8)
                and (carbon.GetFormalCharge(), oxygen.GetFormalCharge()) == (-1, 1)
                and neighbor_bond_order(carbon, oxygen) == 3
                and carbon.GetIdx() in visible_atom_indices
                and oxygen.GetIdx() not in visible_atom_indices
                and 1.05 < carbon.GetDistance(oxygen) < 1.25
            ):
                has_linear_carbonyl |= metal_donor_angle(metal.site, carbon, oxygen) > 160
    core_charge = metal.valence + len(nitrosonium_indices) // 2
    total_charge = core_charge + ancillary_charge
    if (
        total_charge
        and core_charge
        and not (ancillary_charge == 0 and (total_charge > 0 or has_linear_carbonyl))
    ):
        return 0.0
    compensated_donors = sum(
        index in anionic_component_indices
        or _has_adjacent_formal_charge_cancellation(obmol.GetAtom(index))
        for index in penalized_negative_donor_indices - nitrosonium_indices
    )
    return negative_metal_penalty + compensated_donors


def _has_adjacent_anionic_polarization_cancellation(atom: ob.OBAtom) -> bool:
    formal_charge = int(atom.GetFormalCharge())
    central_electronegativity = consts.NON_METAL_PAULING_ELECTRONEGATIVITY.get(
        int(atom.GetAtomicNum())
    )
    if formal_charge <= 0 or central_electronegativity is None:
        return False
    adjacent_negative_charge = 0
    for neighbor in atom_neighbors(atom):
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


def _has_iminium_dithiocarbonyl_compensation(nitrogen: ob.OBAtom) -> bool:
    """Recognize the local N(+)-C(=S)-S(-) path of a resolved iminium group.

    The three-coordinate N must have an external C=N bond. Terminal sulfur
    and closed-shell labels keep this distinct from remote anion cancellation.
    """
    if (
        nitrogen.GetAtomicNum() != 7
        or nitrogen.GetFormalCharge() != 1
        or nitrogen.GetExplicitDegree() != 3
        or nitrogen.GetImplicitHCount() != 0
        or nitrogen.GetTotalValence() != 4
        or get_unpaired_electron_count(nitrogen)
        or has_unresolved_two_electron_center(nitrogen)
    ):
        return False
    neighbors = list(atom_neighbors(nitrogen))
    if not any(
        atom.GetAtomicNum() == 6 and neighbor_bond_order(nitrogen, atom) == 2 for atom in neighbors
    ):
        return False
    for carbon in neighbors:
        if (
            carbon.GetAtomicNum() != 6
            or carbon.GetFormalCharge() != 0
            or carbon.GetExplicitDegree() != 3
            or carbon.GetImplicitHCount() != 0
            or get_unpaired_electron_count(carbon)
            or has_unresolved_two_electron_center(carbon)
            or neighbor_bond_order(nitrogen, carbon) != 1
        ):
            continue
        sulfurs = [a for a in atom_neighbors(carbon) if a.GetAtomicNum() == 16]
        if len(sulfurs) != 2 or any(
            atom.GetExplicitDegree() != 1
            or atom.GetImplicitHCount() != 0
            or get_unpaired_electron_count(atom)
            or has_unresolved_two_electron_center(atom)
            for atom in sulfurs
        ):
            continue
        if sorted(
            (neighbor_bond_order(carbon, atom), atom.GetFormalCharge()) for atom in sulfurs
        ) == [(1, -1), (2, 0)]:
            return True
    return False


def _has_adjacent_anionic_donor_pair(atom: ob.OBAtom) -> bool:
    """Require an actual transferable pair to compensate an electron-deficient cation."""
    if get_unpaired_electron_count(atom) or has_unresolved_two_electron_center(atom):
        return False
    for neighbor in atom_neighbors(atom):
        info = consts.NON_METAL_DICT.get(int(neighbor.GetAtomicNum()))
        if (
            info is None
            or neighbor.GetFormalCharge() >= 0
            or has_unresolved_two_electron_center(neighbor)
            or neighbor_bond_order(atom, neighbor) not in (1, 2)
        ):
            continue
        unpaired = get_unpaired_electron_count(neighbor)
        paired = (
            info.num_outer_electrons
            - neighbor.GetFormalCharge()
            - neighbor.GetTotalValence()
            - unpaired
        )
        if unpaired >= 0 and paired >= 2 and paired % 2 == 0:
            return True
    return False


def _is_locally_charge_compensated_nonmetal_cation(atom: ob.OBAtom) -> bool:
    if atom.IsMetal() or int(atom.GetFormalCharge()) <= 0:
        return False
    # Opposite formal charges do not fill a sextet when the anion has no pair
    # to transfer (for example, tetrahedral B(-) next to monovalent O(+)).
    if _is_unsaturated_organic_cation(atom) and not _has_adjacent_anionic_donor_pair(atom):
        return False
    if _has_adjacent_formal_charge_cancellation(atom):
        return True
    element_info = consts.NON_METAL_DICT.get(int(atom.GetAtomicNum()))
    if element_info is None or int(atom.GetTotalValence()) <= int(element_info.default_valence):
        return False

    adjacent_negative_charge = 0
    adjacent_positive_charge = 0
    for neighbor in atom_neighbors(atom):
        if neighbor.IsMetal():
            continue
        neighbor_charge = int(neighbor.GetFormalCharge())
        if neighbor_charge < 0:
            adjacent_negative_charge += abs(neighbor_charge)
        elif neighbor_charge > 0:
            adjacent_positive_charge += neighbor_charge
    return (
        adjacent_negative_charge > adjacent_positive_charge
        or _has_iminium_dithiocarbonyl_compensation(atom)
    )


def _unsaturated_organic_cation_discordance_count(
    obmol: ob.OBMol,
) -> int:
    for atom in mol_atoms(obmol):
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
    radical_electrons = get_unpaired_electron_count(atom)

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
    radical_electrons = get_unpaired_electron_count(atom)
    # A validated neutral C/N/Si/P two-electron deficit can be represented as an
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


def _is_closed_shell_cn_ylide_bond(bond: ob.OBBond) -> bool:
    """A resolved octet C−—N+ pair need not share a conjugated pi plane."""
    carbon, nitrogen = cast(ob.OBAtom, bond.GetBeginAtom()), cast(ob.OBAtom, bond.GetEndAtom())
    if carbon.GetAtomicNum() != 6:
        carbon, nitrogen = nitrogen, carbon
    return (
        bond.GetBondOrder() == 1
        and carbon.GetAtomicNum() == 6
        and carbon.GetFormalCharge() == -1
        and carbon.GetTotalValence() == 3
        and nitrogen.GetAtomicNum() == 7
        and nitrogen.GetFormalCharge() == 1
        and nitrogen.GetTotalValence() == 4
        and all(
            not atom.GetImplicitHCount()
            and not get_unpaired_electron_count(atom)
            and not has_unresolved_two_electron_center(atom)
            for atom in (carbon, nitrogen)
        )
    )


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
        for atom in mol_atoms(obmol):
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
        for bond in mol_bonds(obmol):
            begin_idx = int(bond.GetBeginAtomIdx()) - 1
            end_idx = int(bond.GetEndAtomIdx()) - 1
            begin_penalty = signed_charge_penalties_by_atom_idx[begin_idx]
            end_penalty = signed_charge_penalties_by_atom_idx[end_idx]
            opposite_charges = begin_penalty * end_penalty < 0.0
            allows_cancellation = True
            if (
                opposite_charges
                and not _is_closed_shell_cn_ylide_bond(bond)
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
    blocker_arrays = build_coordination_blocker_arrays(
        obmol,
        metal_scoring_config=metal_scoring_config,
    )
    non_metal_atoms = _non_metal_atom_entries(obmol)

    inner_visible_diradical_count = 0
    inner_visible_same_sign_charge_count = 0
    inner_visible_cationic_donor_count = 0
    penalized_negative_donor_indices: set[int] = set()
    visible_inner_atom_indices: set[int] = set()
    visible_atoms_by_metal: list[tuple[ob.OBAtom, ...]] = []

    for metal_state in candidate.metal_states:
        metal_charge_sign = _charge_sign(int(metal_state.valence))
        visible_atoms = inner_visible_atoms_to_metal(
            non_metal_atoms,
            metal_state.site,
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
                and not (
                    atom.GetAtomicNum() in (7, 15)
                    and get_lone_pair_count(atom) == 0
                    and any(
                        n.GetAtomicNum() == 6 and n.GetFormalCharge() < 0
                        for n in atom_neighbors(atom)
                    )
                    and _is_locally_charge_compensated_nonmetal_cation(atom)
                )
            ):
                inner_visible_same_sign_charge_count += 1
                if formal_charge < 0:
                    penalized_negative_donor_indices.add(atom_idx)
                if _is_cationic_paired_electron_donor(atom):
                    inner_visible_cationic_donor_count += 1

    candidate.metadata["metal_visible_atom_indices"] = ",".join(
        str(i) for i in sorted(visible_inner_atom_indices)
    )
    candidate.metadata["metal_visible_atom_groups"] = ";".join(
        ",".join(str(atom.GetIdx()) for atom in group) for group in visible_atoms_by_metal
    )
    candidate.metadata["metal_equalized_carbon_reduction_exemption_count"] = 0
    candidate.metadata["metal_equalized_carbon_localization_count"] = 0
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
    allene_mol = _fresh_aromatic_copy(obmol)
    initial_bent = _bent_cumulated_ring_allene_indices(allene_mol)
    bent_centers = {
        idx for idx in initial_bent if not _aromatic_allene_witness(allene_mol, idx, initial_bent)
    }
    exempt_centers = {
        index
        for index in bent_centers & visible_inner_atom_indices
        if _is_neutral_carbon_cumulene_center(allene_mol.GetAtom(index))
    }
    bent_cumulated_ring_allene_count = len(bent_centers - exempt_centers)
    candidate.metadata["metal_visible_bent_allene_indices"] = ",".join(
        str(index) for index in sorted(exempt_centers)
    )
    candidate.metadata["metal_visible_bent_allene_aromatic_alternative_count"] = 0

    outer_or_invisible_adjacent_double_charge_count = 0
    outer_or_invisible_adjacent_same_sign_double_charge_count = 0
    outer_or_invisible_adjacent_opposite_sign_double_charge_count = 0
    outer_or_invisible_adjacent_unknown_metal_sign_double_charge_count = 0
    inner_visible_adjacent_carbanion_pair_count = 0
    inner_visible_conjugated_carbanion_pair_count = (
        _inner_visible_conjugated_charged_carbon_pair_count(
            obmol, visible_inner_atom_indices, visible_atoms_by_metal, candidate.metal_states
        )
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
    nitrosyl_penalty, nitrosonium_indices, nitrosyl_count = _nitrosyl_geometry_metrics(
        obmol, candidate.metal_states, visible_atoms_by_metal
    )
    candidate.metadata["nitrosyl_geometry_penalty"] = nitrosyl_penalty
    candidate.metadata["nitrosyl_geometry_site_count"] = nitrosyl_count
    candidate.metadata["nitrosonium_redistribution_exception"] = 0
    candidate.metadata["nitrosonium_geometry_atom_indices"] = ",".join(
        str(i) for i in sorted(nitrosonium_indices)
    )
    zero_valent_metals_with_organic_cation_count = _zero_valent_metals_with_organic_cation_count(
        obmol,
        candidate.metal_states,
        total_charge=int(no_metal_state.total_charge)
        + sum(int(metal_state.valence) for metal_state in candidate.metal_states),
        coordinated_nitrosonium_indices=nitrosonium_indices,
        config=config,
    )
    unsaturated_organic_cation_discordance_count = _unsaturated_organic_cation_discordance_count(
        obmol
    )
    for bond in mol_bonds(obmol):
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
    short_carbonyl, long_dioxygen = _oxygen_bond_geometry_discordance(
        obmol, visible_inner_atom_indices
    )
    candidate.metadata["metal_discordance_short_carbonyl_reduction_count"] = short_carbonyl
    candidate.metadata["metal_discordance_long_dioxygen_multiple_bond_count"] = long_dioxygen
    long_carbonyl = _long_carbonyl_multiple_bond_count(obmol, visible_atoms_by_metal)
    candidate.metadata["metal_discordance_long_carbonyl_multiple_bond_count"] = long_carbonyl
    sulfur_geometry = _sulfur_bond_geometry_discordance(obmol, visible_atoms_by_metal)
    candidate.metadata["metal_discordance_sulfur_bond_geometry_count"] = sulfur_geometry
    short_quinodiimine = short_quinodiimine_conflicts(obmol)
    candidate.metadata["metal_discordance_short_quinodiimine_geometry_count"] = short_quinodiimine
    negative_metal_penalty = _NEGATIVE_METAL_DISCORDANCE_PENALTY * negative_metal_count
    same_sign_charge_penalty = (
        inner_visible_same_sign_charge_count
        - (1.0 - _CATIONIC_DONOR_DISCORDANCE_PENALTY) * inner_visible_cationic_donor_count
    )
    nitrosonium_formal_charge_exemption = _nitrosonium_formal_charge_exemption(
        obmol,
        candidate.metal_states,
        nitrosonium_indices,
        penalized_negative_donor_indices,
        negative_metal_penalty,
        visible_inner_atom_indices,
    )
    candidate.metadata["metal_nitrosonium_formal_charge_exemption"] = (
        nitrosonium_formal_charge_exemption
    )
    # Keep exceptional Zn formal states searchable without preferring them over
    # an otherwise valid Zn(II) allocation. This is a preference, not a ban.
    high_zinc_valence_count = sum(
        state.symbol == "Zn" and state.valence > 2 for state in candidate.metal_states
    )
    candidate.metadata["metal_discordance_high_zinc_valence_count"] = high_zinc_valence_count
    structural_discordance_count = (
        inner_visible_diradical_count
        + outer_or_invisible_adjacent_double_charge_count
        + inner_visible_adjacent_carbanion_pair_count
        + inner_visible_conjugated_carbanion_pair_count
        + same_sign_charge_penalty
        + high_zinc_valence_count
        + excess_visible_singlet_two_electron_center_count
        + bent_cumulated_ring_allene_count
        + negative_metal_penalty
        + zero_valent_metals_with_organic_cation_count
        + unsaturated_organic_cation_discordance_count
        + repeated_component_charge_asymmetry_count
        + haptic_arene_reduction_count
        + coordination_geometry_discordance_count
        + short_carbonyl
        + long_dioxygen
        + long_carbonyl
        + sulfur_geometry
        + short_quinodiimine
        - nitrosonium_formal_charge_exemption
    )
    discordance_count = structural_discordance_count
    candidate.metadata["metal_discordance_structural_count"] = structural_discordance_count
    candidate.metadata["metal_alkoxy_allylic_geometry_conflict_count"] = 0
    candidate.metadata["metal_macrocycle_electron_conflict_count"] = 0
    candidate.metadata["metal_folded_ring_geometry_conflict_count"] = 0
    candidate.metadata["metal_formazan_geometry_conflict_count"] = 0
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
    candidate.metadata["metal_discordance_inner_visible_cationic_donor_count"] = (
        inner_visible_cationic_donor_count
    )
    candidate.metadata["metal_discordance_inner_visible_same_sign_charge_penalty"] = (
        same_sign_charge_penalty
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
    credit = 0.0
    visible_indices = {
        int(i)
        for i in str(candidate.metadata.get("metal_visible_atom_indices", "")).split(",")
        if i
    }
    # Restrict this compensation to geometry-supported nitrosyl complexes.
    # Generic anionic ligands retain their established charge-localization cost.
    if not candidate.metadata.get("nitrosyl_geometry_site_count", 0):
        visible_indices.clear()
    for index in visible_indices:
        atom = cast(ob.OBAtom, no_metal_state.omol.OBMol.GetAtom(index))
        if (
            atom.GetFormalCharge() == -1
            and atom.GetAtomicNum() in (7, 8, 16)
            and atom.GetTotalValence() == {7: 2, 8: 1, 16: 1}[atom.GetAtomicNum()]
            and get_unpaired_electron_count(atom) == 0
        ):
            # A closed-shell anionic donor is not excess charge localization
            # merely because the corresponding metal is oxidized.
            credit += _charge_localization_penalty_for_atom(atom, is_conjugated=True)
    candidate.metadata["organic_coordinated_anion_charge_credit"] = credit
    candidate.metadata["organic_charge_localization_penalty"] = max(
        0.0, cached_metrics.charge_localization_penalty - credit
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


def _non_nitrosyl_signature(mol: ob.OBMol, excluded: Set[int]) -> tuple:
    atoms = tuple(
        (
            a.GetIdx(),
            a.GetAtomicNum(),
            a.GetFormalCharge(),
            a.GetImplicitHCount(),
            get_unpaired_electron_count(a),
            get_lone_pair_count(a),
            has_unresolved_two_electron_center(a),
        )
        for a in mol_atoms(mol)
        if a.GetIdx() not in excluded
    )
    bonds = tuple(
        sorted(
            (
                min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                b.GetBondOrder(),
            )
            for b in mol_bonds(mol)
            if b.GetBeginAtomIdx() not in excluded and b.GetEndAtomIdx() not in excluded
        )
    )
    return mol.NumAtoms(), atoms, bonds


def _annotate_nitrosonium_redistribution_exceptions(
    candidates: Sequence[MetalCandidateState],
) -> None:
    """Do not veto an isolated M(II)/NO- -> M(0)/NO+ redistribution.

    Other charged ligands must be identical to an existing M(II)/NO- candidate.
    This does not exempt arbitrary cation generation at zero-valent metals.
    """
    for candidate in candidates:
        previous = int(candidate.metadata.get("nitrosonium_redistribution_exception", 0))
        if previous:
            candidate.metadata["metal_discordance_structural_count"] += previous
        candidate.metadata["nitrosonium_redistribution_exception"] = 0
        if (
            len(candidate.metal_states) != 1
            or candidate.metal_states[0].valence != 0
            or candidate.metadata.get("nitrosyl_geometry_site_count") != 1
            or not candidate.metadata.get(
                "metal_discordance_zero_valent_metals_with_organic_cation_count"
            )
            or candidate.no_metal_state is None
        ):
            continue
        indices = {
            int(i)
            for i in str(candidate.metadata.get("nitrosonium_geometry_atom_indices", "")).split(",")
            if i
        }
        if len(indices) != 2:
            continue
        mol = cast(ob.OBMol, candidate.no_metal_state.omol.OBMol)
        signature = _non_nitrosyl_signature(mol, indices)
        for partner in candidates:
            if (
                len(partner.metal_states) != 1
                or partner.metal_states[0].valence != 2
                or partner.metal_states[0].idx != candidate.metal_states[0].idx
                or partner.metal_states[0].radical_num != candidate.metal_states[0].radical_num
                or partner.no_metal_radical_target != candidate.no_metal_radical_target
                or partner.no_metal_charge_target != candidate.no_metal_charge_target - 2
                or partner.metadata.get("nitrosyl_geometry_site_count") != 1
                or partner.no_metal_state is None
            ):
                continue
            other = cast(ob.OBMol, partner.no_metal_state.omol.OBMol)
            if other.NumAtoms() != mol.NumAtoms():
                continue
            atoms = [cast(ob.OBAtom, other.GetAtom(i)) for i in sorted(indices)]
            bond = cast(Optional[ob.OBBond], atoms[0].GetBond(atoms[1]))
            if (
                sorted(a.GetAtomicNum() for a in atoms) == [7, 8]
                and all(a.GetExplicitDegree() == 1 and a.GetImplicitHCount() == 0 for a in atoms)
                and all(get_unpaired_electron_count(a) == 0 for a in atoms)
                and sum(a.GetFormalCharge() for a in atoms) == -1
                and bond is not None
                and bond.GetBondOrder() == 2
                and _non_nitrosyl_signature(other, indices) == signature
            ):
                candidate.metadata["nitrosonium_redistribution_exception"] = 1
                break


def _fresh_aromatic_copy(source: ob.OBMol) -> ob.OBMol:
    # Candidate edits can leave cached flags stale. Use a copy, as the topology
    # scorer does, without mutating search seeds or their cached scores.
    mol = ob.OBMol(source)
    mol.FindRingAtomsAndBonds()
    mol.SetAromaticPerceived(False)
    ob.OBAromaticTyper().AssignAromaticFlags(mol)
    return mol


def _aromatic_allene_witness(source: ob.OBMol, center_idx: int, initial_bent: set[int]) -> bool:
    """Find an alternating charge shift that restores aromaticity at a bent allene."""

    def resolved_carbon(a: ob.OBAtom) -> bool:
        return (
            a.GetAtomicNum() == 6
            and not a.GetImplicitHCount()
            and not get_unpaired_electron_count(a)
            and not has_unresolved_two_electron_center(a)
        )

    center = cast(ob.OBAtom, source.GetAtom(center_idx))
    if not resolved_carbon(center) or center.GetFormalCharge() or center.GetTotalValence() != 4:
        return False
    paths: deque[tuple[int, ...]] = deque([(center_idx,)])
    for _ in range(4096):
        if not paths:
            break
        path = paths.popleft()
        if len(path) + 1 > 32:
            continue
        last = cast(ob.OBAtom, source.GetAtom(path[-1]))
        order = 2 if len(path) % 2 else 1
        for other in atom_neighbors(last):
            idx = other.GetIdx()
            if (
                idx in path
                or neighbor_bond_order(last, other) != order
                or not resolved_carbon(other)
            ):
                continue
            extended = (*path, idx)
            if other.GetFormalCharge() == 0 and other.GetTotalValence() == 4:
                paths.append(extended)
            elif order == 1 and other.GetFormalCharge() == -1 and other.GetTotalValence() == 3:
                trial = ob.OBMol(source)
                trial.GetAtom(center_idx).SetFormalCharge(-1)
                trial.GetAtom(idx).SetFormalCharge(0)
                for i, (left, right) in enumerate(zip(extended, extended[1:])):
                    trial.GetBond(left, right).SetBondOrder(1 + i % 2)
                trial.SetAromaticPerceived(False)
                ob.OBAromaticTyper().AssignAromaticFlags(trial)
                if trial.GetAtom(center_idx).IsAromatic():
                    remaining = _bent_cumulated_ring_allene_indices(trial)
                    if center_idx not in remaining and remaining <= initial_bent:
                        return True
    return False


def _fresh_aromatic_atom_indices(source: ob.OBMol) -> set[int]:
    mol = _fresh_aromatic_copy(source)
    return {int(atom.GetIdx()) for atom in mol_atoms(mol) if atom.IsAromatic()}


def _annotate_bent_allene_aromatic_alternatives(
    candidates: Sequence[MetalCandidateState],
) -> None:
    """Keep the angle penalty when a viable aromatic alternative exists locally.

    Metal proximity alone can stabilize a detached ligand cumulene, but must not
    excuse destroying aromaticity elsewhere in the same covalent ligand. Compare
    the same atom and its component across candidates with no greater independent
    structural discordance. Snapshot those counts
    before applying penalties so candidate order cannot affect the result.
    """
    independent_counts: list[float] = []
    for candidate in candidates:
        previous = int(
            candidate.metadata.get("metal_visible_bent_allene_aromatic_alternative_count", 0)
        )
        if previous:
            candidate.metadata["metal_discordance_structural_count"] -= previous
            candidate.metadata["metal_discordance_bent_cumulated_ring_allene_count"] -= previous
        candidate.metadata["metal_visible_bent_allene_aromatic_alternative_count"] = 0
        independent_counts.append(
            float(candidate.metadata.get("metal_discordance_structural_count", 0))
            - int(candidate.metadata.get("metal_discordance_bent_cumulated_ring_allene_count", 0))
            - int(candidate.metadata.get("nitrosonium_redistribution_exception", 0))
        )
    for candidate, independent_count in zip(candidates, independent_counts):
        indices = {
            int(index)
            for index in str(candidate.metadata.get("metal_visible_bent_allene_indices", "")).split(
                ","
            )
            if index
        }
        if not indices:
            continue
        assert candidate.no_metal_state is not None
        source = cast(ob.OBMol, candidate.no_metal_state.omol.OBMol)
        components: dict[int, set[int]] = {}
        for index in indices:
            component: set[int] = set()
            pending = [index]
            while pending:
                current = pending.pop()
                if current in component:
                    continue
                component.add(current)
                pending.extend(
                    int(atom.GetIdx())
                    for atom in atom_neighbors(source.GetAtom(current))
                    if int(atom.GetIdx()) not in component
                )
            components[index] = component
        own_aromatic = _fresh_aromatic_atom_indices(source)
        protected: set[int] = set()
        for partner, partner_count in zip(candidates, independent_counts):
            if partner.no_metal_state is None or partner_count > independent_count:
                continue
            aromatic = _fresh_aromatic_atom_indices(partner.no_metal_state.omol.OBMol)
            for index, component in components.items():
                if index in aromatic or len(aromatic & component) > len(own_aromatic & component):
                    protected.add(index)
        count = len(protected)
        if count:
            candidate.metadata["metal_discordance_structural_count"] += count
            candidate.metadata["metal_discordance_bent_cumulated_ring_allene_count"] += count
        candidate.metadata["metal_visible_bent_allene_aromatic_alternative_count"] = count


def _equalized_carbon_electronic_sites(
    candidate: MetalCandidateState,
) -> tuple[dict[tuple[int, ...], tuple[int, ...]], dict[tuple[int, ...], tuple[int, ...]]]:
    """Compare a localized alkyne and a reduced C=C pair in the same metal-visible C₃ chain."""
    reduced: dict[tuple[int, ...], tuple[int, ...]] = {}
    localized: dict[tuple[int, ...], tuple[int, ...]] = {}
    if candidate.no_metal_state is None:
        return reduced, localized
    groups = [
        {int(i) for i in group.split(",") if i}
        for group in str(candidate.metadata.get("metal_visible_atom_groups", "")).split(";")
    ]
    mol = cast(ob.OBMol, candidate.no_metal_state.omol.OBMol)
    for left, middle, right in equalized_three_carbon_chains(mol):
        site = tuple(sorted(atom.GetIdx() for atom in (left, middle, right)))
        if not any(set(site).issubset(group) for group in groups):
            continue
        for first, other in ((left, right), (right, left)):
            if neighbor_bond_order(
                middle, other
            ) != 1 or not _is_explicit_singlet_two_electron_center(other):
                continue
            pair = tuple(sorted((first.GetIdx(), middle.GetIdx())))
            order = neighbor_bond_order(first, middle)
            if order == 2 and first.GetFormalCharge() == middle.GetFormalCharge() == -1:
                reduced[site] = pair
            if order == 3 and first.GetFormalCharge() == middle.GetFormalCharge() == 0:
                localized[site] = pair
    return reduced, localized


def _annotate_equalized_carbon_alternatives(candidates: Sequence[MetalCandidateState]) -> None:
    """Admit a reduced pair at an equalized three-carbon metal-bound site.

    Exempt its adjacent C(-1)=C(-1) pair from the generic charge conflict. Only
    when a validated reduced alternative exists at that same site, count the
    competing localized alkyne/carbene as a geometric conflict. All three atoms
    must be visible to one metal. Ring alkynes and isolated eta2 alkynes are not
    covered. Restore previous adjustments so repeated annotation is idempotent.
    """
    features = [_equalized_carbon_electronic_sites(candidate) for candidate in candidates]
    supported = {site for reduced, _ in features for site in reduced}
    for candidate, (reduced, localized) in zip(candidates, features):
        previous = int(
            candidate.metadata.get("metal_equalized_carbon_localization_count", 0)
        ) - int(candidate.metadata.get("metal_equalized_carbon_reduction_exemption_count", 0))
        exemptions = len(set(reduced.values()))
        conflicts = len({pair for site, pair in localized.items() if site in supported})
        candidate.metadata["metal_discordance_structural_count"] = (
            float(
                candidate.metadata.get(
                    "metal_discordance_structural_count",
                    candidate.metadata.get("metal_discordance_count", 0),
                )
            )
            + conflicts
            - exemptions
            - previous
        )
        candidate.metadata["metal_equalized_carbon_reduction_exemption_count"] = exemptions
        candidate.metadata["metal_equalized_carbon_localization_count"] = conflicts


def _alkoxy_allylic_electronic_sites(
    candidate: MetalCandidateState,
) -> tuple[set[tuple[int, ...]], set[tuple[int, ...]]]:
    """Find nonaromatic alkoxy/oxonium alternatives at one metal-visible C3 site."""
    reduced: set[tuple[int, ...]] = set()
    inverted: set[tuple[int, ...]] = set()
    if candidate.no_metal_state is None:
        return reduced, inverted
    groups = [
        {int(i) for i in group.split(",") if i}
        for group in str(candidate.metadata.get("metal_visible_atom_groups", "")).split(";")
    ]
    mol = cast(ob.OBMol, candidate.no_metal_state.omol.OBMol)
    aromatic = _fresh_aromatic_atom_indices(mol)
    for middle in mol_atoms(mol):
        if middle.GetAtomicNum() != 6:
            continue
        neighbors = [a for a in atom_neighbors(middle) if a.GetAtomicNum() == 6]
        for left, right in combinations(neighbors, 2):
            atoms = (left, middle, right)
            if any(
                a.GetIdx() in aromatic
                or a.GetExplicitDegree() != 3
                or a.GetImplicitHCount() != 0
                or get_unpaired_electron_count(a)
                for a in atoms
            ):
                continue
            if not any({a.GetIdx() for a in atoms} <= group for group in groups):
                continue
            first, last = (
                (left, right)
                if left.GetDistance(middle) > right.GetDistance(middle)
                else (right, left)
            )
            longer, shorter = first.GetDistance(middle), last.GetDistance(middle)
            if not (1.44 < longer < 1.55 and 1.30 < shorter < 1.43 and longer - shorter > 0.04):
                continue
            orders = (neighbor_bond_order(first, middle), neighbor_bond_order(middle, last))
            charges = tuple(a.GetFormalCharge() for a in (first, middle, last))
            for oxygen in atom_neighbors(last):
                if (
                    oxygen.GetAtomicNum() != 8
                    or oxygen.GetExplicitDegree() != 2
                    or oxygen.GetImplicitHCount() != 0
                    or get_unpaired_electron_count(oxygen)
                    or not 1.30 < oxygen.GetDistance(last) < 1.42
                ):
                    continue
                caps = [a for a in atom_neighbors(oxygen) if a.GetIdx() != last.GetIdx()]
                if (
                    len(caps) != 1
                    or caps[0].GetAtomicNum() != 6
                    or caps[0].GetExplicitDegree() != 4
                    or neighbor_bond_order(oxygen, caps[0]) != 1
                ):
                    continue
                site = tuple(a.GetIdx() for a in (first, middle, last, oxygen, caps[0]))
                co_order = neighbor_bond_order(last, oxygen)
                if (
                    orders == (1, 2)
                    and charges == (-1, 0, 0)
                    and oxygen.GetFormalCharge() == 0
                    and co_order == 1
                ):
                    reduced.add(site)
                if (
                    orders == (2, 1)
                    and charges == (0, 0, 0)
                    and oxygen.GetFormalCharge() == 1
                    and co_order == 2
                ):
                    inverted.add(site)
    return reduced, inverted


def _clear_alkoxy_allylic_geometry_conflicts(candidates: Sequence[MetalCandidateState]) -> None:
    for candidate in candidates:
        previous = int(candidate.metadata.get("metal_alkoxy_allylic_geometry_conflict_count", 0))
        for key in ("metal_discordance_structural_count", "metal_discordance_count"):
            candidate.metadata[key] = float(candidate.metadata.get(key, 0)) - previous
        candidate.metadata["metal_alkoxy_allylic_geometry_conflict_count"] = 0


def _annotate_alkoxy_allylic_alternatives(candidates: Sequence[MetalCandidateState]) -> None:
    """Prefer a supported reduced ether over its inverted oxonium assignment.

    Require an existing +2 metal alternative at the same site, with the same
    radical budgets and no larger independent structural conflict count.
    Snapshot counts before updating any candidate so order cannot affect eligibility.
    """
    _clear_alkoxy_allylic_geometry_conflicts(candidates)
    features = [_alkoxy_allylic_electronic_sites(c) for c in candidates]
    counts = [float(c.metadata.get("metal_discordance_structural_count", 0)) for c in candidates]
    for candidate, (_, inverted), own_count in zip(candidates, features, counts):
        supported = set()
        for partner, (reduced, _), partner_count in zip(candidates, features, counts):
            if (
                len(candidate.metal_states) != 1
                or len(partner.metal_states) != 1
                or partner.no_metal_radical_target != candidate.no_metal_radical_target
                or partner_count > own_count
            ):
                continue
            metal, other = candidate.metal_states[0], partner.metal_states[0]
            if (
                metal.idx == other.idx
                and other.valence == metal.valence + 2
                and other.radical_num == metal.radical_num
            ):
                supported.update(inverted & reduced)
        count = len(supported)
        candidate.metadata["metal_alkoxy_allylic_geometry_conflict_count"] = count
        candidate.metadata["metal_discordance_structural_count"] += count
        candidate.metadata["metal_discordance_count"] += count


def _formazan_electronic_sites(c: MetalCandidateState) -> dict[tuple[int, ...], str]:
    """Distinguish the reduced and oxidized allocations at a phenoxy formazan site."""
    out: dict[tuple[int, ...], str] = {}
    if c.no_metal_state is None:
        return out
    m = cast(ob.OBMol, c.no_metal_state.omol.OBMol)
    groups = [
        {int(i) for i in g.split(",") if i}
        for g in str(c.metadata.get("metal_visible_atom_groups", "")).split(";")
    ]
    for center, arms, site in planar_phenoxy_formazan_sites(m, allow_plain_phenyl=True):
        if not any({a[i] for a in arms for i in (1, 4) if a[i]} <= g for g in groups):
            continue
        if any(
            m.GetAtom(i).GetFormalCharge() != 0 or m.GetAtom(i).GetTotalValence() != 4
            for a in arms
            for i in a[5]
        ):
            continue
        nitrogens = [cast(ob.OBAtom, m.GetAtom(i)) for arm in arms for i in arm[:2]]
        central = cast(ob.OBAtom, m.GetAtom(center))
        if (central.GetFormalCharge(), central.GetTotalValence()) not in ((0, 4), (-1, 3)):
            continue
        if any(
            (n.GetFormalCharge(), n.GetTotalValence()) not in ((0, 3), (-1, 2)) for n in nitrogens
        ):
            continue
        core_charge = central.GetFormalCharge() + sum(n.GetFormalCharge() for n in nitrogens)
        states = []
        for inner, outer, carbon, adjacent, oxygen, _ring in arms:
            orders = tuple(
                indexed_bond_order(m, i, j)
                for (i, j) in [
                    (center, inner),
                    (inner, outer),
                    (outer, carbon),
                ]
            ) + (indexed_bond_order(m, adjacent, oxygen) if oxygen else 1,)
            o = cast(ob.OBAtom, m.GetAtom(oxygen)) if oxygen else None
            # The anionic NNCNN core admits both C- and N- contributors.
            # Its total charge and resolved valences determine the permitted
            # alternating pi bonds; do not require a particular contributor.
            if (
                orders[:2] in ((1, 2), (2, 1), (1, 1))
                and orders[2:] == (1, 1)
                and (o is None or (o.GetFormalCharge(), o.GetTotalValence()) == (-1, 1))
            ):
                states.append("reduced")
            elif (
                o is not None
                and orders == (2, 1, 2, 2)
                and (o.GetFormalCharge(), o.GetTotalValence()) == (0, 2)
            ):
                states.append("oxidized")
            else:
                states.append("other")
        if states == ["reduced", "reduced"] and core_charge == -1:
            out[site] = "reduced"
        if sorted(states) == ["oxidized", "reduced"] and core_charge == 0:
            out[site] = "oxidized"
    return out


def _clear_formazan_geometry_conflicts(cs: Sequence[MetalCandidateState]) -> None:
    for c in cs:
        previous = c.metadata.get("metal_formazan_geometry_conflict_count", 0)
        for key in ["metal_discordance_structural_count", "metal_discordance_count"]:
            c.metadata[key] = c.metadata.get(key, 0) - previous
        c.metadata["metal_formazan_geometry_conflict_count"] = 0


def _annotate_formazan_alternatives(cs: Sequence[MetalCandidateState]) -> None:
    """Prefer a complete reduced site only against an existing +2-metal partner.

    Keep the same metal/spins/NO penalty, no worse independent conflicts and
    identical outside labels/bonds. Geometry alone cannot assign Co(III),
    lower a global charge penalty, or exclude an intermediate search state.
    """
    _clear_formazan_geometry_conflicts(cs)
    features = [_formazan_electronic_sites(c) for c in cs]
    counts = [c.metadata.get("metal_discordance_structural_count", 0) for c in cs]
    for c, sites, count in zip(cs, features, counts):
        conflicts = set()
        for p, psites, pcount in zip(cs, features, counts):
            if (
                len(c.metal_states) != 1
                or len(p.metal_states) != 1
                or c.no_metal_radical_target != p.no_metal_radical_target
                or (pcount > count)
                or (
                    c.metadata.get("nitrosyl_geometry_penalty", 0)
                    != p.metadata.get("nitrosyl_geometry_penalty", 0)
                )
            ):
                continue
            (a, b) = (c.metal_states[0], p.metal_states[0])
            if a.idx != b.idx or a.radical_num != b.radical_num or b.valence - a.valence != 2:
                continue
            for site, state in sites.items():
                if (
                    state == "oxidized"
                    and psites.get(site) == "reduced"
                    and (
                        conjugated_geometry.outside_signature(c, site)
                        == conjugated_geometry.outside_signature(p, site)
                    )
                ):
                    conflicts.add(site)
        c.metadata["metal_formazan_geometry_conflict_count"] = len(conflicts)
        for key in ["metal_discordance_structural_count", "metal_discordance_count"]:
            c.metadata[key] += len(conflicts)


def _folded_ring_electronic_sites(
    candidate: MetalCandidateState,
) -> tuple[set[tuple[int, ...]], set[tuple[int, ...]]]:
    """Split folded C₈ segments by their charge and bond-order patterns for paired redox comparison."""
    reduced: set[tuple[int, ...]] = set()
    neutral: set[tuple[int, ...]] = set()
    if candidate.no_metal_state is None:
        return reduced, neutral
    mol = cast(ob.OBMol, candidate.no_metal_state.omol.OBMol)
    groups = [
        {int(i) for i in group.split(",") if i}
        for group in str(candidate.metadata.get("metal_visible_atom_groups", "")).split(";")
    ]
    for ids in folded_eight_carbon_segments(mol):
        site = tuple(sorted(ids[:4]))
        if not any(set(site) <= group for group in groups):
            continue
        atoms = [cast(ob.OBAtom, mol.GetAtom(i)) for i in ids]
        if any(
            (atom.GetFormalCharge(), atom.GetTotalValence()) not in ((0, 4), (-1, 3))
            for atom in atoms
        ):
            continue
        orders = [neighbor_bond_order(atoms[i], atoms[(i + 1) % 8]) for i in range(8)]
        charges = [atom.GetFormalCharge() for atom in atoms]
        if orders[3:] != [1, 2, 1, 2, 1] or any(charges[4:]):
            continue
        if orders[:3] == [1, 2, 1] and charges[:4] == [-1, 0, 0, -1]:
            reduced.add(site)
        if orders[:3] == [2, 1, 2] and not any(charges):
            neutral.add(site)
    return reduced, neutral


def _clear_folded_ring_geometry_conflicts(candidates: Sequence[MetalCandidateState]) -> None:
    for candidate in candidates:
        previous = int(candidate.metadata.get("metal_folded_ring_geometry_conflict_count", 0))
        for key in ("metal_discordance_structural_count", "metal_discordance_count"):
            candidate.metadata[key] = float(candidate.metadata.get(key, 0)) - previous
        candidate.metadata["metal_folded_ring_geometry_conflict_count"] = 0


def _annotate_folded_ring_alternatives(candidates: Sequence[MetalCandidateState]) -> None:
    """Compare existing local C4 reductions at a fixed metal/site/spin budget.

    Each recognized segment transfers two electrons. Multiple independent
    segments may change together only when the metal formal charge balances
    their sum; unchanged sites, NO geometry and independent conflicts are kept.
    """
    _clear_folded_ring_geometry_conflicts(candidates)
    features = [_folded_ring_electronic_sites(candidate) for candidate in candidates]
    counts = [float(c.metadata.get("metal_discordance_structural_count", 0)) for c in candidates]
    for candidate, (reduced, neutral), count in zip(candidates, features, counts):
        conflicts = set()
        for partner, (partner_reduced, partner_neutral), partner_count in zip(
            candidates, features, counts
        ):
            if (
                len(candidate.metal_states) != 1
                or len(partner.metal_states) != 1
                or candidate.no_metal_radical_target != partner.no_metal_radical_target
                or partner_count > count
                or candidate.metadata.get("nitrosyl_geometry_penalty", 0)
                != partner.metadata.get("nitrosyl_geometry_penalty", 0)
            ):
                continue
            metal, other = candidate.metal_states[0], partner.metal_states[0]
            changed = neutral & partner_reduced
            if (
                metal.idx == other.idx
                and metal.radical_num == other.radical_num
                and reduced <= partner_reduced
                and reduced | neutral == partner_reduced | partner_neutral
                and changed
                and other.valence - metal.valence == 2 * len(changed)
            ):
                conflicts.update(changed)
        count = len(conflicts)
        candidate.metadata["metal_folded_ring_geometry_conflict_count"] = count
        for key in ("metal_discordance_structural_count", "metal_discordance_count"):
            candidate.metadata[key] += count


def _metal_displaced_from_n4_plane(
    mol: ob.OBMol, donors: set[int], metal: dataclasses.MetalAtomPosition
) -> bool:
    """Check all four donor triples of a recognized C8N8 circuit."""
    site = np.array((metal.position_x, metal.position_y, metal.position_z))
    atoms = [cast(ob.OBAtom, mol.GetAtom(i)) for i in sorted(donors)]
    points = [np.array((atom.GetX(), atom.GetY(), atom.GetZ())) for atom in atoms]
    for a, b, c in combinations(points, 3):
        normal = np.cross(b - a, c - a)
        scale = np.linalg.norm(normal)
        if not (scale > 0.01 and abs(np.dot(site - a, normal)) > 0.80 * scale):
            return False
    return True


def _macrocycle_electron_sites(candidate: MetalCandidateState) -> dict[tuple[int, ...], int]:
    """Count inner-circuit pi electrons only for resolved C/N valence labels."""
    sites: dict[tuple[int, ...], int] = {}
    if candidate.no_metal_state is None:
        return sites
    mol = cast(ob.OBMol, candidate.no_metal_state.omol.OBMol)
    groups = [
        {int(i) for i in group.split(",") if i}
        for group in str(candidate.metadata.get("metal_visible_atom_groups", "")).split(";")
    ]
    circuits = [
        (ids, {i for i in ids if mol.GetAtom(i).GetAtomicNum() == 7})
        for ids in planar_tetrapyrrole_circuits(mol)
    ]
    circuits.extend(planar_aza_macrocycle_circuits(mol))
    for ids, donors in planar_tetraazaporphyrin_circuits(mol):
        if len(candidate.metal_states) == 1 and _metal_displaced_from_n4_plane(
            mol, donors, candidate.metal_states[0]
        ):
            continue
        circuits.append((ids, donors))
    for ids, donors in circuits:
        atoms = [cast(ob.OBAtom, mol.GetAtom(i)) for i in ids]
        if any(
            get_unpaired_electron_count(atom)
            or has_unresolved_two_electron_center(atom)
            or (atom.GetFormalCharge(), atom.GetTotalValence())
            not in (((0, 4), (-1, 3)) if atom.GetAtomicNum() == 6 else ((0, 3), (-1, 2)))
            for atom in atoms
        ):
            continue
        if any(donors <= group for group in groups):
            # A resolved neutral, three-coordinate N supplies its p lone pair
            # (two electrons); the other admitted neutral sites supply one.
            electrons = sum(
                2 if atom.GetAtomicNum() == 7 and atom.GetExplicitDegree() == 3 else 1
                for atom in atoms
            )
            sites[tuple(sorted(ids))] = electrons - sum(atom.GetFormalCharge() for atom in atoms)
    return sites


def _clear_macrocycle_electron_conflicts(candidates: Sequence[MetalCandidateState]) -> None:
    for candidate in candidates:
        previous = int(candidate.metadata.get("metal_macrocycle_electron_conflict_count", 0))
        for key in ("metal_discordance_structural_count", "metal_discordance_count"):
            candidate.metadata[key] = float(candidate.metadata.get(key, 0)) - previous
        candidate.metadata["metal_macrocycle_electron_conflict_count"] = 0


def _annotate_macrocycle_alternatives(candidates: Sequence[MetalCandidateState]) -> None:
    """Prefer an existing 4n+2 partner for a planar, closed-shell 4n circuit.

    This bounded inner-circuit model is not a general aromaticity test. Require
    the same metal/site/spins, unchanged NO geometry penalty, a two-electron
    transfer, and no larger independent structural conflict count. Never invent
    a metal state or reject a lone candidate on electron count alone.
    """
    _clear_macrocycle_electron_conflicts(candidates)
    features = [_macrocycle_electron_sites(candidate) for candidate in candidates]
    counts = [float(c.metadata.get("metal_discordance_structural_count", 0)) for c in candidates]
    for candidate, sites, count in zip(candidates, features, counts):
        conflicts = set()
        for partner, partner_sites, partner_count in zip(candidates, features, counts):
            if (
                len(candidate.metal_states) != 1
                or len(partner.metal_states) != 1
                or candidate.no_metal_radical_target != partner.no_metal_radical_target
                or partner_count > count
                or candidate.metadata.get("nitrosyl_geometry_penalty", 0)
                != partner.metadata.get("nitrosyl_geometry_penalty", 0)
            ):
                continue
            metal, other = candidate.metal_states[0], partner.metal_states[0]
            if metal.idx != other.idx or metal.radical_num != other.radical_num:
                continue
            for site, electrons in sites.items():
                other_electrons = partner_sites.get(site)
                if (
                    other_electrons is not None
                    and electrons % 4 == 0
                    and other_electrons % 4 == 2
                    and abs(other_electrons - electrons) == 2
                    and other.valence - metal.valence == other_electrons - electrons
                ):
                    conflicts.add(site)
        count = len(conflicts)
        candidate.metadata["metal_macrocycle_electron_conflict_count"] = count
        candidate.metadata["metal_discordance_structural_count"] += count
        candidate.metadata["metal_discordance_count"] += count


def _annotate_candidate_set_discordance_features(
    candidates: Sequence[MetalCandidateState],
    *,
    config: MolGRConfig | None = None,
) -> None:
    if not candidates:
        return

    conjugated_geometry.clear_pyramidal_ps(candidates)
    conjugated_geometry.clear_pyramidal_pn(candidates)
    conjugated_geometry.clear_ortho_dioxygen(candidates)
    conjugated_geometry.clear_aryl_amido_nhc(candidates)
    conjugated_geometry.clear_aryl_phenoxide_nhc(candidates)
    conjugated_geometry.clear_aryl_thiolate_nhc(candidates)
    conjugated_geometry.clear_aryl_benzyl_nhc(candidates)
    conjugated_geometry.clear_amino_c5(candidates)
    local_cc_geometry.clear(candidates)
    local_cc_geometry.clear_aryl_carbenoid(candidates)
    conjugated_geometry.clear_pyridyl(candidates)
    conjugated_geometry.clear_paired_diimine(candidates)
    conjugated_geometry.clear_polyenamine(candidates)
    conjugated_geometry.clear_quinodiimine(candidates)
    _clear_formazan_geometry_conflicts(candidates)
    _clear_folded_ring_geometry_conflicts(candidates)
    hydrazido_geometry.clear(candidates)
    _clear_macrocycle_electron_conflicts(candidates)
    _clear_alkoxy_allylic_geometry_conflicts(candidates)
    _annotate_equalized_carbon_alternatives(candidates)
    _annotate_nitrosonium_redistribution_exceptions(candidates)
    _annotate_bent_allene_aromatic_alternatives(candidates)
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
        structural_discordance_count -= float(
            candidate.metadata.get("nitrosonium_redistribution_exception", 0)
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

    _annotate_alkoxy_allylic_alternatives(candidates)
    _annotate_macrocycle_alternatives(candidates)
    _annotate_folded_ring_alternatives(candidates)
    hydrazido_geometry.annotate(candidates)
    _annotate_formazan_alternatives(candidates)
    conjugated_geometry.annotate_quinodiimine(candidates)
    conjugated_geometry.annotate_polyenamine(candidates)
    conjugated_geometry.annotate_paired_diimine(candidates)
    conjugated_geometry.annotate_pyridyl(candidates)
    local_cc_geometry.annotate(candidates)
    local_cc_geometry.annotate_aryl_carbenoid(candidates)
    conjugated_geometry.annotate_pyramidal_ps(candidates)
    conjugated_geometry.annotate_pyramidal_pn(candidates)
    conjugated_geometry.annotate_ortho_dioxygen(candidates)
    conjugated_geometry.annotate_aryl_amido_nhc(candidates)
    conjugated_geometry.annotate_aryl_phenoxide_nhc(candidates)
    conjugated_geometry.annotate_aryl_thiolate_nhc(candidates)
    conjugated_geometry.annotate_aryl_benzyl_nhc(candidates)
    conjugated_geometry.annotate_amino_c5(candidates)


def select_ligand_alternatives(
    candidates: Sequence[MetalCandidateState], *, config: MolGRConfig | None = None
) -> list[MetalCandidateState]:
    """Resolve ligand contributors before comparing different metal assignments.

    Preserve NO geometry, local geometry and minimum charge separation. Within
    those bounds, prefer a contributor without an unsupported unsaturated cation
    before isolated-ligand aromaticity can hide that contributor. Other structural
    penalties break exact isolated-ligand score ties before atom-index ordering.
    """
    from molgr.fallback.utils.no_metals.selection import (
        _no_metal_candidate_graph_tie_break_key,
        _no_metal_candidate_selection_key,
    )

    groups: dict[int, list[MetalCandidateState]] = {}
    for candidate in candidates:
        groups.setdefault(int(candidate.metadata["combination_index"]), []).append(candidate)
    retained = []
    for group in groups.values():

        def contributor_key(
            candidate: MetalCandidateState,
        ) -> tuple[float, tuple[int, ...], int, int, tuple[float, ...], float, tuple[int, ...]]:
            state = cast(ReconstructionState, candidate.no_metal_state)
            organic_key = _no_metal_candidate_selection_key(state, config=config)
            return (
                float(candidate.metadata.get("nitrosyl_geometry_penalty", 0.0)),
                organic_key[:3],
                int(candidate.metadata.get("metal_discordance_sulfur_bond_geometry_count", 0)),
                int(
                    candidate.metadata.get("metal_discordance_unsaturated_organic_cation_count", 0)
                ),
                organic_key,
                float(candidate.metadata.get("metal_discordance_structural_count", 0)),
                _no_metal_candidate_graph_tie_break_key(state),
            )

        retained.append(min(group, key=contributor_key))
    return retained


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
        Tuple[float, int, int, int, int, int, float, float, int, float, int, float, int]
    ] = None
    minimum_nitrosyl_penalty = min(
        float(c.metadata.get("nitrosyl_geometry_penalty", 0.0))
        for c in discordance_filtered_candidates
    )
    for scored_candidate in discordance_filtered_candidates:
        if (
            float(scored_candidate.metadata.get("nitrosyl_geometry_penalty", 0.0))
            > minimum_nitrosyl_penalty
        ):
            continue
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
            scored_candidate.score
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
            float(scored_candidate.metadata.get("nitrosyl_geometry_penalty", 0.0)),
            # Prefer ordinary Zn states when all preceding electronic criteria tie.
            int(scored_candidate.metadata.get("metal_discordance_high_zinc_valence_count", 0)),
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
