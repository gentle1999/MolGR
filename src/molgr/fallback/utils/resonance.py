"""Resonance helpers shared by fallback pipeline orchestration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Protocol, Sequence, Tuple, cast

from openbabel import openbabel as ob
from openbabel import pybel

from molgr.fallback.stages.clean import (
    clean_1_4_radicals,
    clean_1_6_radicals,
    clean_neighbor_radicals,
    clean_possible_1_3_dipole,
    clean_resonances,
)
from molgr.fallback.stages.eliminate import (
    eliminate_1_3_dipole_postive,
    eliminate_negative_charges,
    eliminate_positive_charges,
    eliminate_possible_cp_like_radical_anion,
)
from molgr.fallback.state import OmolStateMachine
from molgr.fallback.utils import consts
from molgr.fallback.utils.electrons import (
    get_lone_pair_count,
    get_unpaired_electron_count,
    has_unresolved_two_electron_center,
    set_unpaired_electron_count,
    set_unresolved_two_electron_center,
)
from molgr.fallback.utils.ob_graph import atom_bonds, atom_neighbors, indexed_atoms, indexed_bonds

from . import smarts


ResonanceAtomKey = Tuple[int, int, int, int, bool, bool]
ResonanceBondKey = Tuple[int, int, int, bool]
ResonanceStateKey = Tuple[Tuple[ResonanceAtomKey, ...], Tuple[ResonanceBondKey, ...]]
ProcessedResonanceKey = str
ResonanceBondIndexMap = Dict[Tuple[int, int], int]
_UFF_LITE_BOND_STRAIN_SCORE_WEIGHT = 1.0
_UFF_LITE_ANGLE_STRAIN_SCORE_WEIGHT = 0.35
_UFF_LITE_RADICAL_SCORE_WEIGHT = 1.0
_UFF_LITE_CONJUGATION_SCORE_WEIGHT = 0.25
_KCAL_TO_KJ = 4.184

_BondOrderOverrides = Dict[Tuple[int, int], int]
_UffLiteGainMetrics = Tuple[float, float, float, float]


@dataclass(frozen=True)
class ResonanceTraversalContext:
    """Read-only state exposed to traversal policies for the current expansion step."""

    root_omol: pybel.Molecule
    current_omol: pybel.Molecule
    current_state_key: ResonanceStateKey
    depth: int
    max_depth: int


@dataclass(frozen=True)
class ResonanceTraversalMove:
    """A one-step radical resonance move and the incremental key of its output state."""

    path: Tuple[int, int, int]
    next_state_key: ResonanceStateKey


@dataclass(frozen=True)
class ResonanceSearchNode:
    """A resonance state yielded during traversal."""

    omol: pybel.Molecule
    state_key: ResonanceStateKey
    depth: int
    discrepancy: int = 0


@dataclass(frozen=True)
class _IndexedResonanceTraversalMove:
    idxs: Tuple[int, int, int]
    next_state_key: ResonanceStateKey


class ResonanceTraversalPolicy(Protocol):
    def __call__(
        self,
        context: ResonanceTraversalContext,
        moves: Sequence[ResonanceTraversalMove],
    ) -> Optional[Sequence[ResonanceTraversalMove]]: ...


class LimitedDiscrepancyResonanceTraversalPolicy(ResonanceTraversalPolicy, Protocol):
    max_discrepancy: int


@dataclass(frozen=True)
class _LimitedDiscrepancyUffLiteGainTraversalPolicy:
    max_discrepancy: int

    def __call__(
        self,
        context: ResonanceTraversalContext,
        moves: Sequence[ResonanceTraversalMove],
    ) -> Optional[Sequence[ResonanceTraversalMove]]:
        if len(moves) < 2:
            return moves
        ordered_moves = _order_uff_lite_gain_moves(context.current_omol, moves)
        return [move for move, _score in ordered_moves]


@dataclass(frozen=True)
class _LimitedDiscrepancyInputOrderTraversalPolicy:
    max_discrepancy: int

    def __call__(
        self,
        context: ResonanceTraversalContext,
        moves: Sequence[ResonanceTraversalMove],
    ) -> Optional[Sequence[ResonanceTraversalMove]]:
        return list(moves)


_LIMITED_DISCREPANCY_POLICY_TYPES = (
    _LimitedDiscrepancyUffLiteGainTraversalPolicy,
    _LimitedDiscrepancyInputOrderTraversalPolicy,
)


def process_resonance(
    resonance: pybel.Molecule,
    charge: int,
    *,
    total_charge: int = 0,
    total_radical_electrons: int = 0,
) -> tuple[pybel.Molecule, int, bool]:
    """Normalize one resonance candidate using only classified electron states.

    Radical-to-charge rules consume real unpaired electrons, adjacent-radical
    cleanup pairs them into bonds, and closed-shell resonance rules may consume an
    explicitly active lone pair. A pure unresolved two-electron center is normally
    carried to validation, but a remaining charge deficit of magnitude at least
    two may consume it directly as ``+2`` or ``-2``. This function never converts
    that marker to radicals by parity.
    """

    machine = OmolStateMachine(
        resonance,
        charge,
        total_charge=total_charge,
        total_radical_electrons=total_radical_electrons,
    )
    return _run_process_resonance(machine, emit_summary=False)


def _run_process_resonance(
    machine: OmolStateMachine,
    *,
    emit_summary: bool,
) -> tuple[pybel.Molecule, int, bool]:
    """Run process_resonance stages on an existing machine for nested tracing."""

    hit = machine.run_omol_charge_stage(
        "process_resonance_eliminate_1_3_dipole_postive",
        eliminate_1_3_dipole_postive,
    )
    hit = (
        machine.run_omol_charge_stage(
            "process_resonance_eliminate_possible_cp_like_radical_anion",
            eliminate_possible_cp_like_radical_anion,
            machine.total_radical_electrons,
            machine.metadata.get("metal_contact_distances", ()),
        )
        or hit
    )
    hit = (
        machine.run_omol_stage(
            "process_resonance_clean_possible_1_3_dipole",
            clean_possible_1_3_dipole,
            machine.given_charge,
            machine.total_radical_electrons,
        )
        or hit
    )
    hit = (
        machine.run_omol_stage(
            "process_resonance_clean_1_4_radicals",
            clean_1_4_radicals,
            machine.given_charge,
            machine.total_radical_electrons,
        )
        or hit
    )
    hit = (
        machine.run_omol_stage(
            "process_resonance_clean_neighbor_radicals",
            clean_neighbor_radicals,
            machine.given_charge,
            machine.total_radical_electrons,
        )
        or hit
    )
    hit = (
        machine.run_omol_stage(
            "process_resonance_clean_1_6_radicals",
            clean_1_6_radicals,
            machine.given_charge,
            machine.total_radical_electrons,
        )
        or hit
    )
    hit = (
        machine.run_omol_charge_stage(
            "process_resonance_eliminate_positive_charges_1",
            eliminate_positive_charges,
            machine.total_radical_electrons,
        )
        or hit
    )
    hit = (
        machine.run_omol_charge_stage(
            "process_resonance_eliminate_negative_charges",
            eliminate_negative_charges,
            machine.total_radical_electrons,
            machine.metadata.get("metal_contact_distances", ()),
        )
        or hit
    )
    hit = (
        machine.run_omol_charge_stage(
            "process_resonance_eliminate_positive_charges_2",
            eliminate_positive_charges,
            machine.total_radical_electrons,
        )
        or hit
    )
    hit = (
        machine.run_omol_stage(
            "process_resonance_clean_resonances",
            clean_resonances,
        )
        or hit
    )
    if emit_summary:
        machine.annotate("full_resonance_normalization")
    return machine.omol, machine.given_charge, hit


def build_resonance_state_key(omol: pybel.Molecule) -> ResonanceStateKey:
    """Build the structural key used to deduplicate raw resonance states."""

    obmol = cast(ob.OBMol, omol.OBMol)
    atom_keys: List[ResonanceAtomKey] = []
    for atom in indexed_atoms(obmol):
        atom_keys.append(
            (
                atom.GetAtomicNum(),
                atom.GetFormalCharge(),
                get_unpaired_electron_count(atom),
                get_lone_pair_count(atom),
                has_unresolved_two_electron_center(atom),
                bool(atom.IsAromatic()),
            )
        )

    bond_keys: List[ResonanceBondKey] = []
    for bond in indexed_bonds(obmol):
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if begin_idx > end_idx:
            begin_idx, end_idx = end_idx, begin_idx
        bond_keys.append((begin_idx, end_idx, bond.GetBondOrder(), bool(bond.IsAromatic())))
    return tuple(atom_keys), tuple(bond_keys)


def build_processed_resonance_key(omol: pybel.Molecule) -> ProcessedResonanceKey:
    """Build the dedup key after `process_resonance` has normalized the candidate."""

    atom_keys, bond_keys = build_resonance_state_key(omol)
    atom_part = "".join(
        f"{atomic_num},{formal_charge},{radical_num},{lone_pair_count},"
        f"{int(unresolved_center)},{int(is_aromatic)};"
        for atomic_num, formal_charge, radical_num, lone_pair_count, unresolved_center, is_aromatic in atom_keys
    )
    bond_part = "".join(
        f"{begin_idx},{end_idx},{bond_order},{int(is_aromatic)};"
        for begin_idx, end_idx, bond_order, is_aromatic in bond_keys
    )
    return f"A{len(atom_keys)}:{atom_part}|B{len(bond_keys)}:{bond_part}"


def _build_resonance_search_context(
    omol: pybel.Molecule,
) -> Tuple[ResonanceStateKey, ResonanceBondIndexMap]:
    """Index the bond entries that radical shifts can update in the initial state key."""

    key = build_resonance_state_key(omol)
    bond_index_map = {(begin, end): i for i, (begin, end, _, _) in enumerate(key[1])}
    return key, bond_index_map


def _increment_resonance_state_key(
    state_key: ResonanceStateKey,
    bond_index_map: ResonanceBondIndexMap,
    idxs: Tuple[int, int, int],
    *,
    target_unresolved: bool = False,
) -> ResonanceStateKey:
    atom_keys, bond_keys = state_key
    atom_key_list = list(atom_keys)
    bond_key_list = list(bond_keys)

    atom1_idx = idxs[0] - 1
    atom3_idx = idxs[2] - 1

    atom1 = atom_key_list[atom1_idx]
    atom3 = atom_key_list[atom3_idx]
    atom_key_list[atom1_idx] = (atom1[0], atom1[1], atom1[2] - 1, atom1[3], atom1[4], atom1[5])
    atom_key_list[atom3_idx] = (
        atom3[0],
        atom3[1],
        0 if target_unresolved else atom3[2] + 1,
        atom3[3],
        target_unresolved or atom3[4],
        atom3[5],
    )

    bond1_pair = (min(idxs[0], idxs[1]), max(idxs[0], idxs[1]))
    bond2_pair = (min(idxs[1], idxs[2]), max(idxs[1], idxs[2]))
    bond1_idx = bond_index_map[bond1_pair]
    bond2_idx = bond_index_map[bond2_pair]
    bond1 = bond_key_list[bond1_idx]
    bond2 = bond_key_list[bond2_idx]
    bond_key_list[bond1_idx] = (bond1[0], bond1[1], bond1[2] + 1, bond1[3])
    bond_key_list[bond2_idx] = (bond2[0], bond2[1], bond2[2] - 1, bond2[3])

    return tuple(atom_key_list), tuple(bond_key_list)


def _enumerate_one_step_resonance_moves(
    omol: pybel.Molecule,
    state_key: ResonanceStateKey,
    bond_index_map: ResonanceBondIndexMap,
) -> List[_IndexedResonanceTraversalMove]:
    """Enumerate one-electron radical moves across an adjacent pi bond.

    The target may carry zero or one unpaired electron. A zero-radical target
    follows the ordinary ``r1-r0 -> r0-r1`` route. If a one-radical target becomes
    a neutral C/N/Si/P two-electron deficit after the bond shift, ``r1-r1`` becomes
    ``r0-unresolved`` so validation can choose its singlet or triplet occupancy.
    """

    obmol = cast(ob.OBMol, omol.OBMol)
    res: List[Tuple[int, int, int]] = list(smarts.RESONANCE_ONE_STEP.findall(omol))
    result: List[_IndexedResonanceTraversalMove] = []
    for idxs in res:
        atom1 = cast(ob.OBAtom, obmol.GetAtom(idxs[0]))
        if get_unpaired_electron_count(atom1) != 1:
            continue
        atom3 = cast(ob.OBAtom, obmol.GetAtom(idxs[2]))
        target_radicals = get_unpaired_electron_count(atom3)
        if target_radicals > 1 or has_unresolved_two_electron_center(atom3):
            continue
        bond1 = cast(ob.OBBond, obmol.GetBond(idxs[0], idxs[1]))
        bond2 = cast(ob.OBBond, obmol.GetBond(idxs[1], idxs[2]))
        if bond1.GetBondOrder() <= 2 and bond2.GetBondOrder() >= 2:
            post_target_valence = atom3.GetTotalValence() - 1
            target_unresolved = (
                target_radicals == 1
                and atom3.GetAtomicNum() in (6, 7, 14, 15)
                and atom3.GetFormalCharge() == 0
                and get_lone_pair_count(atom3) == 0
                and max(
                    0,
                    ob.GetTypicalValence(
                        atom3.GetAtomicNum(),
                        post_target_valence,
                        atom3.GetFormalCharge(),
                    )
                    - post_target_valence,
                )
                == 2
            )
            result.append(
                _IndexedResonanceTraversalMove(
                    idxs=idxs,
                    next_state_key=_increment_resonance_state_key(
                        state_key,
                        bond_index_map,
                        idxs,
                        target_unresolved=target_unresolved,
                    ),
                )
            )
    return result


def _materialize_one_step_resonance(
    omol: pybel.Molecule,
    idxs: Tuple[int, int, int],
) -> pybel.Molecule:
    """Move one real unpaired electron and the coupled pi-bond pattern.

    One unpaired electron is removed from the source and added to the target while
    the intervening single/multiple bond orders swap by one. A resulting neutral
    C/N/Si/P two-electron deficit is represented by an unresolved marker instead of
    being forced to two unpaired electrons.
    """

    new_omol = cast(pybel.Molecule, omol.clone)
    new_obmol = cast(ob.OBMol, new_omol.OBMol)
    atom1_clone = cast(ob.OBAtom, new_obmol.GetAtom(idxs[0]))
    atom3_clone = cast(ob.OBAtom, new_obmol.GetAtom(idxs[2]))
    bond1_clone = cast(ob.OBBond, new_obmol.GetBond(idxs[0], idxs[1]))
    bond2_clone = cast(ob.OBBond, new_obmol.GetBond(idxs[1], idxs[2]))
    bond1_clone.SetBondOrder(bond1_clone.GetBondOrder() + 1)
    bond2_clone.SetBondOrder(bond2_clone.GetBondOrder() - 1)
    set_unpaired_electron_count(atom1_clone, get_unpaired_electron_count(atom1_clone) - 1)
    target_unpaired = get_unpaired_electron_count(atom3_clone)
    target_valence = atom3_clone.GetTotalValence()
    target_unresolved = (
        target_unpaired == 1
        and atom3_clone.GetAtomicNum() in (6, 7, 14, 15)
        and atom3_clone.GetFormalCharge() == 0
        and get_lone_pair_count(atom3_clone) == 0
        and max(
            0,
            ob.GetTypicalValence(
                atom3_clone.GetAtomicNum(),
                target_valence,
                atom3_clone.GetFormalCharge(),
            )
            - target_valence,
        )
        == 2
    )
    if target_unresolved:
        set_unpaired_electron_count(atom3_clone, 0)
        set_unresolved_two_electron_center(atom3_clone, True)
    else:
        set_unpaired_electron_count(atom3_clone, target_unpaired + 1)
    return new_omol


def _bond_pair(atom_idx_a: int, atom_idx_b: int) -> Tuple[int, int]:
    if atom_idx_a <= atom_idx_b:
        return atom_idx_a, atom_idx_b
    return atom_idx_b, atom_idx_a


def _get_bond_order_with_overrides(
    bond: ob.OBBond,
    bond_order_overrides: _BondOrderOverrides,
) -> int:
    override = bond_order_overrides.get(
        _bond_pair(
            cast(int, bond.GetBeginAtomIdx()),
            cast(int, bond.GetEndAtomIdx()),
        )
    )
    if override is not None:
        return override
    return cast(int, bond.GetBondOrder())


def _normalized_bond_order_with_overrides(
    bond: ob.OBBond,
    bond_order_overrides: _BondOrderOverrides,
) -> float:
    override = bond_order_overrides.get(
        _bond_pair(
            cast(int, bond.GetBeginAtomIdx()),
            cast(int, bond.GetEndAtomIdx()),
        )
    )
    if override is not None:
        return float(max(override, 1))
    if bond.IsAromatic():
        return 1.5
    if bond.IsAmide():
        return 1.41
    return float(max(cast(int, bond.GetBondOrder()), 1))


def _clamped_electronegativity(atomic_num: int) -> float:
    return max(float(ob.GetElectroNeg(atomic_num)), 0.25)


def _clamped_covalent_radius(atomic_num: int) -> float:
    value = float(ob.GetCovalentRad(atomic_num))
    return value if value > 0.0 else 0.75


@lru_cache(maxsize=1024)
def _estimate_uff_equilibrium_bond_length(
    atomic_num_a: int,
    atomic_num_b: int,
    bond_order: float,
) -> float:
    ri = _clamped_covalent_radius(atomic_num_a)
    rj = _clamped_covalent_radius(atomic_num_b)
    chi_i = _clamped_electronegativity(atomic_num_a)
    chi_j = _clamped_electronegativity(atomic_num_b)
    safe_bond_order = max(bond_order, 0.25)
    rbo = -0.1332 * (ri + rj) * math.log(safe_bond_order)
    ren = (
        ri
        * rj
        * math.pow(math.sqrt(chi_i) - math.sqrt(chi_j), 2.0)
        / max(
            chi_i * ri + chi_j * rj,
            1e-9,
        )
    )
    return max(ri + rj + rbo - ren, 0.4)


def _estimate_uff_bond_force_constant(equilibrium_distance: float) -> float:
    return (0.5 * _KCAL_TO_KJ * 664.12) / max(equilibrium_distance**3, 1e-6)


def _uff_lite_bond_stretch_energy(
    bond: ob.OBBond,
    bond_order_overrides: _BondOrderOverrides,
) -> float:
    begin_atom = cast(ob.OBAtom, bond.GetBeginAtom())
    end_atom = cast(ob.OBAtom, bond.GetEndAtom())
    if begin_atom is None or end_atom is None:
        return 0.0
    bond_order = _normalized_bond_order_with_overrides(bond, bond_order_overrides)
    r0 = _estimate_uff_equilibrium_bond_length(
        cast(int, begin_atom.GetAtomicNum()),
        cast(int, end_atom.GetAtomicNum()),
        bond_order,
    )
    dx = float(begin_atom.GetX()) - float(end_atom.GetX())
    dy = float(begin_atom.GetY()) - float(end_atom.GetY())
    dz = float(begin_atom.GetZ()) - float(end_atom.GetZ())
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    delta = distance - r0
    return _estimate_uff_bond_force_constant(r0) * delta * delta


def _local_bond_stretch_energy(
    obmol: ob.OBMol,
    affected_bonds: Sequence[Tuple[int, int]],
    bond_order_overrides: _BondOrderOverrides,
) -> float:
    total = 0.0
    for left_idx, right_idx in affected_bonds:
        bond = cast(ob.OBBond, obmol.GetBond(left_idx, right_idx))
        if bond is not None:
            total += _uff_lite_bond_stretch_energy(bond, bond_order_overrides)
    return total


def _ideal_angle_radians_for_atom(
    atom: ob.OBAtom,
    bond_order_overrides: _BondOrderOverrides,
) -> float:
    heavy_degree = 0
    multiple_bond_count = 0
    max_bond_order = 1.0
    for bond in atom_bonds(atom):
        other = cast(ob.OBAtom, bond.GetNbrAtom(atom))
        if other is not None and cast(int, other.GetAtomicNum()) != 1:
            heavy_degree += 1
        bond_order = _normalized_bond_order_with_overrides(bond, bond_order_overrides)
        max_bond_order = max(max_bond_order, bond_order)
        if bond_order >= 1.5:
            multiple_bond_count += 1

    if max_bond_order >= 2.5 or multiple_bond_count >= 2:
        return math.pi
    if max_bond_order >= 1.5 or heavy_degree == 3:
        return 2.0 * math.pi / 3.0
    return math.radians(109.5)


def _uff_lite_angle_energy(
    obmol: ob.OBMol,
    left_atom: ob.OBAtom,
    center_atom: ob.OBAtom,
    right_atom: ob.OBAtom,
    bond_order_overrides: _BondOrderOverrides,
    ideal_angle: float,
) -> float:
    left_bond = cast(ob.OBBond, obmol.GetBond(left_atom.GetIdx(), center_atom.GetIdx()))
    right_bond = cast(ob.OBBond, obmol.GetBond(center_atom.GetIdx(), right_atom.GetIdx()))
    if left_bond is None or right_bond is None:
        return 0.0

    theta_degrees = float(obmol.GetAngle(left_atom, center_atom, right_atom))
    if not math.isfinite(theta_degrees):
        return 0.0

    theta = math.radians(theta_degrees)
    left_order = _normalized_bond_order_with_overrides(left_bond, bond_order_overrides)
    right_order = _normalized_bond_order_with_overrides(right_bond, bond_order_overrides)
    left_r0 = _estimate_uff_equilibrium_bond_length(
        cast(int, left_atom.GetAtomicNum()),
        cast(int, center_atom.GetAtomicNum()),
        left_order,
    )
    right_r0 = _estimate_uff_equilibrium_bond_length(
        cast(int, center_atom.GetAtomicNum()),
        cast(int, right_atom.GetAtomicNum()),
        right_order,
    )
    stiffness = (
        25.0
        * (1.0 + 0.15 * max(0.0, left_order + right_order - 2.0))
        / max(
            math.sqrt(left_r0 * right_r0),
            0.5,
        )
    )
    cos_delta = math.cos(theta) - math.cos(ideal_angle)
    return stiffness * cos_delta * cos_delta


def _local_angle_strain_energy(
    obmol: ob.OBMol,
    affected_center_indices: Sequence[int],
    bond_order_overrides: _BondOrderOverrides,
) -> float:
    total = 0.0
    for center_idx in affected_center_indices:
        center_atom = cast(ob.OBAtom, obmol.GetAtom(center_idx))
        if center_atom is None:
            continue
        neighbors = list(atom_neighbors(center_atom))
        if len(neighbors) < 2:
            continue
        ideal_angle = _ideal_angle_radians_for_atom(center_atom, bond_order_overrides)
        for left_index in range(len(neighbors)):
            for right_index in range(left_index + 1, len(neighbors)):
                total += _uff_lite_angle_energy(
                    obmol,
                    neighbors[left_index],
                    center_atom,
                    neighbors[right_index],
                    bond_order_overrides,
                    ideal_angle,
                )
    return total


def _radical_penalty_for_atom(atomic_num: int, radical_num: int, heavy_degree: int) -> float:
    if radical_num <= 0:
        return 0.0
    if atomic_num in consts.HETEROATOM:
        return float(radical_num) * 10.0
    return float(radical_num) * max(0.0, 3.0 - float(heavy_degree)) * 1.5


def _conjugated_bond_kind(
    bond: ob.OBBond,
    bond_order_overrides: _BondOrderOverrides,
) -> int:
    if bond.IsAromatic():
        return 2
    bond_order = _get_bond_order_with_overrides(bond, bond_order_overrides)
    if bond_order <= 0:
        return 0
    if bond_order == 1:
        return 1
    return 2


def _estimate_radical_conjugation_size(
    obmol: ob.OBMol,
    radical_atom_idx: int,
    bond_order_overrides: Optional[_BondOrderOverrides] = None,
) -> int:
    overrides = bond_order_overrides or {}
    visited_states = {(radical_atom_idx, 0)}
    visited_atoms = {radical_atom_idx}
    frontier = [(radical_atom_idx, 0)]

    while frontier:
        atom_idx, previous_bond_kind = frontier.pop()
        atom = cast(ob.OBAtom, obmol.GetAtom(atom_idx))
        if atom is None:
            continue
        for bond in atom_bonds(atom):
            bond_kind = _conjugated_bond_kind(bond, overrides)
            if bond_kind == 0:
                continue
            if previous_bond_kind != 0 and bond_kind == previous_bond_kind:
                continue
            neighbor = cast(ob.OBAtom, bond.GetNbrAtom(atom))
            neighbor_idx = cast(int, neighbor.GetIdx())
            state = (neighbor_idx, bond_kind)
            if state in visited_states:
                continue
            visited_states.add(state)
            visited_atoms.add(neighbor_idx)
            frontier.append(state)

    return len(visited_atoms)


def _compute_uff_lite_gain_resonance_metrics(
    omol: pybel.Molecule,
    move_path: Tuple[int, int, int],
) -> _UffLiteGainMetrics:
    obmol = cast(ob.OBMol, omol.OBMol)
    old_radical_idx, center_idx, new_radical_idx = move_path

    bond_old_center = cast(ob.OBBond, obmol.GetBond(old_radical_idx, center_idx))
    bond_center_new = cast(ob.OBBond, obmol.GetBond(center_idx, new_radical_idx))
    old_atom = cast(ob.OBAtom, obmol.GetAtom(old_radical_idx))
    new_atom = cast(ob.OBAtom, obmol.GetAtom(new_radical_idx))
    if bond_old_center is None or bond_center_new is None or old_atom is None or new_atom is None:
        return 0.0, 0.0, 0.0, 0.0

    bond_order_overrides = {
        _bond_pair(old_radical_idx, center_idx): cast(int, bond_old_center.GetBondOrder()) + 1,
        _bond_pair(center_idx, new_radical_idx): cast(int, bond_center_new.GetBondOrder()) - 1,
    }
    # Match native std::set accumulation order. Roundoff in otherwise tied
    # move scores can change which branches fit the discrepancy budget.
    affected_bonds = sorted(
        (
            _bond_pair(old_radical_idx, center_idx),
            _bond_pair(center_idx, new_radical_idx),
        )
    )
    affected_angle_centers = sorted((old_radical_idx, center_idx, new_radical_idx))

    bond_strain_gain = _local_bond_stretch_energy(
        obmol,
        affected_bonds,
        {},
    ) - _local_bond_stretch_energy(obmol, affected_bonds, bond_order_overrides)
    angle_strain_gain = _local_angle_strain_energy(
        obmol,
        affected_angle_centers,
        {},
    ) - _local_angle_strain_energy(obmol, affected_angle_centers, bond_order_overrides)
    radical_penalty_before = _radical_penalty_for_atom(
        cast(int, old_atom.GetAtomicNum()),
        get_unpaired_electron_count(old_atom),
        cast(int, old_atom.GetHvyDegree()),
    ) + _radical_penalty_for_atom(
        cast(int, new_atom.GetAtomicNum()),
        get_unpaired_electron_count(new_atom),
        cast(int, new_atom.GetHvyDegree()),
    )
    radical_penalty_after = _radical_penalty_for_atom(
        cast(int, old_atom.GetAtomicNum()),
        get_unpaired_electron_count(old_atom) - 1,
        cast(int, old_atom.GetHvyDegree()),
    ) + _radical_penalty_for_atom(
        cast(int, new_atom.GetAtomicNum()),
        get_unpaired_electron_count(new_atom) + 1,
        cast(int, new_atom.GetHvyDegree()),
    )
    radical_penalty_gain = radical_penalty_before - radical_penalty_after
    conjugation_gain = float(
        _estimate_radical_conjugation_size(obmol, new_radical_idx, bond_order_overrides)
        - _estimate_radical_conjugation_size(obmol, old_radical_idx)
    )
    return (
        bond_strain_gain,
        angle_strain_gain,
        radical_penalty_gain,
        conjugation_gain,
    )


def _uff_lite_gain_move_score(metrics: _UffLiteGainMetrics) -> float:
    return -(
        metrics[0] * _UFF_LITE_BOND_STRAIN_SCORE_WEIGHT
        + metrics[1] * _UFF_LITE_ANGLE_STRAIN_SCORE_WEIGHT
        + metrics[2] * _UFF_LITE_RADICAL_SCORE_WEIGHT
        + metrics[3] * _UFF_LITE_CONJUGATION_SCORE_WEIGHT
    )


def _order_uff_lite_gain_moves(
    omol: pybel.Molecule,
    moves: Sequence[ResonanceTraversalMove],
) -> List[Tuple[ResonanceTraversalMove, float]]:
    ordered_moves: List[Tuple[ResonanceTraversalMove, float]] = []
    for move in moves:
        ordered_moves.append(
            (
                move,
                _uff_lite_gain_move_score(
                    _compute_uff_lite_gain_resonance_metrics(omol, move.path)
                ),
            )
        )
    ordered_moves.sort(key=lambda item: (item[1], item[0].path))
    return ordered_moves


def make_limited_discrepancy_uff_lite_gain_traversal_policy(
    *,
    max_discrepancy: int = 2,
) -> ResonanceTraversalPolicy:
    """Prefer larger UFF-lite improvements, with bounded discrepancy search."""

    return _LimitedDiscrepancyUffLiteGainTraversalPolicy(
        max_discrepancy=max(0, max_discrepancy),
    )


def make_limited_discrepancy_input_order_traversal_policy(
    *,
    max_discrepancy: int = 2,
) -> ResonanceTraversalPolicy:
    """Use raw resonance move order, with bounded discrepancy search."""

    return _LimitedDiscrepancyInputOrderTraversalPolicy(
        max_discrepancy=max(0, max_discrepancy),
    )


def _apply_resonance_traversal_policy(
    context: ResonanceTraversalContext,
    moves: Sequence[_IndexedResonanceTraversalMove],
    traversal_policy: Optional[ResonanceTraversalPolicy],
) -> Sequence[_IndexedResonanceTraversalMove]:
    if traversal_policy is None or not moves:
        return moves

    public_moves = tuple(
        ResonanceTraversalMove(
            path=move.idxs,
            next_state_key=move.next_state_key,
        )
        for move in moves
    )
    selected_moves = traversal_policy(context, public_moves)
    if selected_moves is None:
        return moves

    move_by_key = {(move.idxs, move.next_state_key): move for move in moves}
    selected_indexed_moves: List[_IndexedResonanceTraversalMove] = []
    seen_selected_keys = set()
    for move in selected_moves:
        move_key = (move.path, move.next_state_key)
        if move_key in seen_selected_keys:
            continue
        indexed_move = move_by_key.get(move_key)
        if indexed_move is None:
            raise ValueError(
                "resonance traversal policy returned a move outside the candidate frontier"
            )
        seen_selected_keys.add(move_key)
        selected_indexed_moves.append(indexed_move)
    return selected_indexed_moves


__all__ = [
    "ProcessedResonanceKey",
    "ResonanceBondIndexMap",
    "ResonanceSearchNode",
    "ResonanceStateKey",
    "ResonanceTraversalContext",
    "ResonanceTraversalMove",
    "ResonanceTraversalPolicy",
    "LimitedDiscrepancyResonanceTraversalPolicy",
    "build_processed_resonance_key",
    "build_resonance_state_key",
    "make_limited_discrepancy_input_order_traversal_policy",
    "make_limited_discrepancy_uff_lite_gain_traversal_policy",
    "process_resonance",
]
