"""Resonance recovery helpers for no-metal fallback reconstruction."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, cast

from openbabel import openbabel as ob
from openbabel import pybel

from molgr.config import CONFIG, MolGRConfig
from molgr.fallback.pipeline.resonance import ResonanceTraversalPolicy, walk_radical_resonances
from molgr.fallback.stages.clean import clean_carbene_neighbor_unsaturated
from molgr.fallback.stages.fresh import assign_charge_radical_for_atom
from molgr.fallback.stages.preprocess import validate_omol
from molgr.fallback.state import (
    TRACE_NODE_METADATA_KEY,
    OmolStateMachine,
    ReconstructionState,
)
from molgr.fallback.utils import consts
from molgr.fallback.utils import resonance as resonance_utils
from molgr.fallback.utils.electrons import (
    get_lone_pair_count,
    get_unpaired_electron_count,
    has_unresolved_two_electron_center,
    set_lone_pair_count,
    set_unpaired_electron_count,
    set_unresolved_two_electron_center,
)
from molgr.fallback.utils.ob_graph import mol_atoms, mol_bonds

from .selection import _annotate_no_metal_candidate_topology, _score_reconstruction_candidate


_OmolStage = Callable[[pybel.Molecule], Tuple[pybel.Molecule, bool]]
_OmolChargeStage = Callable[
    [pybel.Molecule, int],
    Tuple[pybel.Molecule, int, bool],
]
_MAX_BORON_NEIGHBOR_SEEDS = 256


def _default_resonance_traversal_policy(
    config: MolGRConfig | None = None,
) -> ResonanceTraversalPolicy:
    resolved_config = CONFIG if config is None else config
    max_discrepancy = max(0, int(resolved_config.resonance.limited_discrepancy_max_discrepancy))
    traversal_score = resolved_config.resonance.traversal_score
    if traversal_score == "uff_lite_gain":
        return resonance_utils.make_limited_discrepancy_uff_lite_gain_traversal_policy(
            max_discrepancy=max_discrepancy,
        )
    if traversal_score == "input_order":
        return resonance_utils.make_limited_discrepancy_input_order_traversal_policy(
            max_discrepancy=max_discrepancy,
        )
    raise ValueError(
        "Unsupported resonance traversal_score. Expected 'uff_lite_gain' or 'input_order'."
    )


def _resonance_max_depth(config: MolGRConfig | None = None) -> int:
    resolved_config = CONFIG if config is None else config
    return max(0, int(resolved_config.resonance.max_depth))


def _clone_omol(omol: pybel.Molecule) -> pybel.Molecule:
    return pybel.Molecule(ob.OBMol(cast(ob.OBMol, omol.OBMol)))


@dataclass
class _RawResonanceCandidate:
    """A ligand graph before validation, with indices back to its seed and traversal order."""

    seed: ReconstructionState
    seed_index: int
    resonance_index: int
    raw_index: int
    omol: pybel.Molecule


@dataclass
class _ResonanceSearchSession:
    """Track raw and processed ligand states across seeds so shared resonances can reuse search work."""

    seen_raw_states: set[tuple[object, int]] = field(default_factory=set)
    labels_by_state: dict[tuple[object, int], list[tuple[int, int]]] = field(default_factory=dict)
    seen_processed_states: set[tuple[str, int]] = field(default_factory=set)
    next_raw_index: int = 0


def _candidate_key(state: ReconstructionState) -> tuple[object, int]:
    return resonance_utils.build_resonance_state_key(state.omol), state.given_charge


def _index_unique_states(
    states: Iterable[ReconstructionState],
) -> dict[tuple[object, int], ReconstructionState]:
    """Keep the first seed for each resonance graph and remaining charge, in input order."""

    unique: dict[tuple[object, int], ReconstructionState] = {}
    for state in states:
        unique.setdefault(_candidate_key(state), state)
    return unique


def build_resonance_seed_pool(
    neighbor_seeds: Sequence[ReconstructionState],
) -> list[ReconstructionState]:
    """Add optional budgeted electronic states and resonance normalizations."""

    pool = expand_boron_neighbor_radical_seeds(neighbor_seeds)
    stages = (
        ("relocate_carbene_radical_for_resonance", clean_carbene_neighbor_unsaturated, False),
    )
    for phase, stage, uses_charge_budget in stages:
        pool = expand_resonance_seed_pool_stage(
            pool,
            phase=phase,
            stage=stage,
            uses_charge_budget=uses_charge_budget,
        )
    return pool


def _boron_neighbor_radical_bonds(omol: pybel.Molecule) -> list[tuple[int, int]]:
    pairs = []
    for bond in mol_bonds(cast(ob.OBMol, omol.OBMol)):
        if bond.GetBondOrder() != 2 or bond.IsAromatic():
            continue
        boron, neighbor = cast(ob.OBAtom, bond.GetBeginAtom()), cast(ob.OBAtom, bond.GetEndAtom())
        if boron.GetAtomicNum() != 5 or boron.GetFormalCharge() != -1:
            boron, neighbor = neighbor, boron
        if (
            boron.GetAtomicNum() == 5
            and neighbor.GetAtomicNum() in consts.NON_METAL_DICT
            and boron.GetFormalCharge() == -1
            and neighbor.GetFormalCharge() == 0
            and boron.GetTotalValence() == 4
            and all(
                get_unpaired_electron_count(atom) == 0
                and get_lone_pair_count(atom) == 0
                and not has_unresolved_two_electron_center(atom)
                for atom in (boron, neighbor)
            )
        ):
            pairs.append((boron.GetIdx(), neighbor.GetIdx()))
    return sorted(pairs)


def _reduce_boron_neighbor_bond(
    omol: pybel.Molecule,
    given_charge: int,
    total_radical_electrons: int,
    boron_idx: int,
    neighbor_idx: int,
) -> tuple[pybel.Molecule, int, bool]:
    """Turn B-=X into B- radical-X+ using one positive-charge/radical unit."""
    mol = cast(ob.OBMol, omol.OBMol)
    if (
        given_charge <= 0
        or sum(get_unpaired_electron_count(a) for a in mol_atoms(mol)) >= total_radical_electrons
        or (boron_idx, neighbor_idx) not in _boron_neighbor_radical_bonds(omol)
    ):
        return omol, given_charge, False
    boron, neighbor = (
        cast(ob.OBAtom, mol.GetAtom(boron_idx)),
        cast(ob.OBAtom, mol.GetAtom(neighbor_idx)),
    )
    bond = cast(ob.OBBond, mol.GetBond(boron_idx, neighbor_idx))

    def restore() -> None:
        bond.SetBondOrder(2)
        for atom, charge in ((boron, -1), (neighbor, 0)):
            atom.SetFormalCharge(charge)
            set_unpaired_electron_count(atom, 0)
            set_lone_pair_count(atom, 0)
            set_unresolved_two_electron_center(atom, False)

    # Require a model-consistent starting state, not merely missing labels.
    if assign_charge_radical_for_atom(neighbor):
        restore()
        return omol, given_charge, False
    bond.SetBondOrder(1)
    neighbor.SetFormalCharge(1)
    assign_charge_radical_for_atom(boron)
    assign_charge_radical_for_atom(neighbor)
    # Other elements can create extra active electrons or normalize charge.
    # This action is only allowed to spend one charge and one radical slot.
    if any(
        atom.GetFormalCharge() != charge
        or get_unpaired_electron_count(atom) != radicals
        or get_lone_pair_count(atom) != 0
        or has_unresolved_two_electron_center(atom)
        for atom, charge, radicals in ((boron, -1, 1), (neighbor, 1, 0))
    ):
        restore()
        return omol, given_charge, False
    return omol, given_charge - 1, True


def expand_boron_neighbor_radical_seeds(
    states: Sequence[ReconstructionState],
) -> list[ReconstructionState]:
    """Keep original seeds and enumerate budgeted B=X reductions before traversal.

    Each eligible bond gets its own branch; multiple centers can be adjusted in
    combination while both budgets allow it. Bound the pool as in neighbor seed
    enumeration, retaining all original seeds even when already over the cap.
    """
    unique = _index_unique_states(states)
    pool = list(unique.values())
    if not any(s.given_charge > 0 and s.total_radical_electrons > 0 for s in pool):
        return pool
    seen = set(unique)
    next_index = 0
    while next_index < len(pool) and len(pool) < _MAX_BORON_NEIGHBOR_SEEDS:
        state = pool[next_index]
        next_index += 1
        if state.given_charge <= 0 or state.total_radical_electrons <= 0:
            continue
        if (
            sum(
                get_unpaired_electron_count(atom)
                for atom in mol_atoms(cast(ob.OBMol, state.omol.OBMol))
            )
            >= state.total_radical_electrons
        ):
            continue
        for boron_idx, neighbor_idx in _boron_neighbor_radical_bonds(state.omol):
            machine = OmolStateMachine.from_reconstruction_state(state).branch(
                "branch_boron_neighbor_radical_seed", omol=_clone_omol(state.omol)
            )
            if not machine.run_omol_charge_stage(
                "reduce_boron_neighbor_bond_for_radical",
                _reduce_boron_neighbor_bond,
                state.total_radical_electrons,
                boron_idx,
                neighbor_idx,
            ):
                continue
            candidate = machine.freeze_like(state)
            key = _candidate_key(candidate)
            if key not in seen:
                seen.add(key)
                pool.append(candidate)
            if len(pool) >= _MAX_BORON_NEIGHBOR_SEEDS:
                break
    return pool


def expand_resonance_seed_pool_stage(
    states: Sequence[ReconstructionState],
    *,
    phase: str,
    stage: _OmolStage | _OmolChargeStage,
    uses_charge_budget: bool,
) -> list[ReconstructionState]:
    """Retain every input seed and add the changed result of one optional stage."""

    unique = _index_unique_states(states)
    # Only the cloned graph is normalized; the original seed keys remain valid.
    for state in list(unique.values()):
        machine = OmolStateMachine.from_reconstruction_state(state).branch(
            None,
            omol=_clone_omol(state.omol),
        )
        if uses_charge_budget:
            hit = machine.run_omol_charge_stage(
                phase,
                cast(_OmolChargeStage, stage),
            )
        else:
            hit = machine.run_omol_stage(phase, cast(_OmolStage, stage))
        if hit:
            candidate = machine.freeze_like(state)
            unique.setdefault(_candidate_key(candidate), candidate)
    return list(unique.values())


def _register_traversal_label(
    labels_by_state: dict[tuple[object, int], list[tuple[int, int]]],
    state_key: tuple[object, int],
    label: tuple[int, int],
) -> bool:
    labels = labels_by_state.setdefault(state_key, [])
    if any(depth <= label[0] and discrepancy <= label[1] for depth, discrepancy in labels):
        return False
    labels[:] = [
        (depth, discrepancy)
        for depth, discrepancy in labels
        if not (label[0] <= depth and label[1] <= discrepancy)
    ]
    labels.append(label)
    return True


def search_resonance_candidates(
    states: Sequence[ReconstructionState],
    *,
    resonance_traversal_policy: Optional[ResonanceTraversalPolicy] = None,
    config: MolGRConfig | None = None,
    session: _ResonanceSearchSession | None = None,
    walk_stage: Callable[..., None] | None = None,
    full_normalization_stage: _OmolChargeStage | None = None,
    validation_stage: Callable[[pybel.Molecule, int, int], bool] | None = None,
    score_stage: Callable[..., float] | None = None,
    topology_annotation_stage: Callable[..., object] | None = None,
) -> list[ReconstructionState]:
    """Search radical resonance states, then resolve deferred centers at validation.

    Traversal moves only real unpaired electrons; active lone pairs and unresolved
    markers remain part of the deduplication key. Each normalized candidate keeps
    unresolved centers until ``validate_omol`` assigns enough centers as triplets
    to satisfy the global radical budget and assigns the remainder as singlets.
    Impossible budgets are rejected without partially resolving a candidate.
    """

    walk_stage = walk_radical_resonances if walk_stage is None else walk_stage
    full_normalization_stage = (
        resonance_utils.process_resonance
        if full_normalization_stage is None
        else full_normalization_stage
    )
    validation_stage = validate_omol if validation_stage is None else validation_stage
    score_stage = _score_reconstruction_candidate if score_stage is None else score_stage
    topology_annotation_stage = (
        _annotate_no_metal_candidate_topology
        if topology_annotation_stage is None
        else topology_annotation_stage
    )
    search_session = _ResonanceSearchSession() if session is None else session
    raw_candidates: list[_RawResonanceCandidate] = []
    use_dominance_pruning = resonance_traversal_policy is None or isinstance(
        resonance_traversal_policy,
        resonance_utils._LIMITED_DISCREPANCY_POLICY_TYPES,
    )
    resonance_max_depth = _resonance_max_depth(config)
    for seed_index, state in enumerate(states):
        resonance_index = 0

        def collect(
            node: resonance_utils.ResonanceSearchNode,
            state: ReconstructionState = state,
            seed_index: int = seed_index,
        ) -> bool:
            nonlocal resonance_index
            raw_key = (node.state_key, state.given_charge)
            should_expand = True
            if use_dominance_pruning and node.depth < resonance_max_depth:
                should_expand = _register_traversal_label(
                    search_session.labels_by_state,
                    raw_key,
                    (node.depth, node.discrepancy),
                )
            if raw_key not in search_session.seen_raw_states:
                search_session.seen_raw_states.add(raw_key)
                raw_candidates.append(
                    _RawResonanceCandidate(
                        seed=state,
                        seed_index=seed_index,
                        resonance_index=resonance_index,
                        raw_index=search_session.next_raw_index,
                        omol=node.omol,
                    )
                )
                search_session.next_raw_index += 1
            resonance_index += 1
            return should_expand

        walk_stage(
            state.omol,
            max_depth=resonance_max_depth,
            traversal_policy=resonance_traversal_policy,
            visit=collect,
        )

    candidates: list[ReconstructionState] = []
    for raw in raw_candidates:
        machine = OmolStateMachine.from_reconstruction_state(raw.seed).branch(
            "branch_resonance_candidate",
            omol=_clone_omol(raw.omol),
            metadata={
                "resonance_seed_index": raw.seed_index,
                "resonance_index": raw.resonance_index,
                "resonance_raw_index": raw.raw_index,
                "resonance_normalization": "full_resonance_normalization",
            },
        )
        if full_normalization_stage is resonance_utils.process_resonance:
            resonance_utils._run_process_resonance(machine, emit_summary=True)
        else:
            machine.run_omol_charge_stage(
                "full_resonance_normalization",
                full_normalization_stage,
            )
        processed_key = (
            resonance_utils.build_processed_resonance_key(machine.omol),
            machine.given_charge,
        )
        if processed_key in search_session.seen_processed_states:
            machine.trace_checkpoint(
                "discard_duplicate_processed_resonance_candidate",
                reason="processed_state_already_seen",
            )
            continue
        search_session.seen_processed_states.add(processed_key)
        unresolved_indices = [
            int(atom.GetIdx())
            for atom in mol_atoms(cast(ob.OBMol, machine.omol.OBMol))
            if has_unresolved_two_electron_center(atom)
        ]
        if not validation_stage(
            machine.omol,
            raw.seed.total_charge,
            raw.seed.total_radical_electrons,
        ):
            machine.trace_checkpoint(
                "reject_no_metal_candidate_validation",
                reason="charge_or_radical_target_mismatch",
            )
            continue
        if unresolved_indices:
            triplet_center_count = sum(
                get_unpaired_electron_count(machine.omol.OBMol.GetAtom(atom_idx)) == 2
                for atom_idx in unresolved_indices
            )
            machine.annotate(
                "resolve_unresolved_two_electron_centers_at_validation",
                unresolved_two_electron_singlet_centers=(
                    len(unresolved_indices) - triplet_center_count
                ),
                unresolved_two_electron_triplet_centers=triplet_center_count,
            )
        machine.annotate(
            "validate_no_metal_candidate",
            resonance_seed_index=raw.seed_index,
            resonance_index=raw.resonance_index,
            resonance_raw_index=raw.raw_index,
            resonance_normalization="full_resonance_normalization",
        )
        candidate = machine.freeze_like(raw.seed)
        try:
            score_stage(candidate, config=config)
        except ValueError:
            machine.trace_checkpoint(
                "reject_no_metal_candidate_scoring",
                reason="force_field_scoring_failed",
            )
            continue
        topology_annotation_stage(candidate, config=config)
        machine.trace_checkpoint(
            "accept_no_metal_candidate",
            score=candidate.metadata.get("score"),
        )
        if TRACE_NODE_METADATA_KEY in machine.metadata:
            candidate.metadata[TRACE_NODE_METADATA_KEY] = machine.metadata[TRACE_NODE_METADATA_KEY]
        candidates.append(candidate)

    return candidates


__all__ = [
    "ResonanceTraversalPolicy",
    "_default_resonance_traversal_policy",
    "build_resonance_seed_pool",
    "expand_boron_neighbor_radical_seeds",
    "expand_resonance_seed_pool_stage",
    "search_resonance_candidates",
]
