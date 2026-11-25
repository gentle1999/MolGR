"""Neighbor-radical seed enumeration for no-metal reconstruction."""

from __future__ import annotations

from typing import Literal, cast

from openbabel import openbabel as ob
from openbabel import pybel

from molgr.fallback.stages.fresh import assign_charge_radical_for_atom
from molgr.fallback.state import OmolStateMachine, ReconstructionState
from molgr.fallback.utils import resonance as resonance_utils
from molgr.fallback.utils.bond_geometry import (
    is_elongated_carbonyl_pair,
    would_form_bent_ring_allene,
)
from molgr.fallback.utils.electrons import (
    get_unpaired_electron_count,
    set_unpaired_electron_count,
)
from molgr.fallback.utils.ob_graph import atom_neighbors, mol_bonds


NeighborRadicalMode = Literal["bond_order", "charge_separation"]
_MAX_NEIGHBOR_RADICAL_SEEDS = 256
_CHARGE_SEPARATION_DISCREPANCY_KEY = "neighbor_radical_charge_separation_discrepancy"


def has_substituted_n_o_radical_pair(omol: pybel.Molecule) -> bool:
    """Keep a charge-aware alternative to closing an organic N--O radical pair.

    A terminal oxygen attached to a substituted nitrogen can become oximate
    after resonance/charge assignment. Closing N=O first can make that state
    unreachable. Isolated NO and O-H groups are outside this seed alternative.
    """
    for bond in mol_bonds(cast(ob.OBMol, omol.OBMol)):
        nitrogen = cast(ob.OBAtom, bond.GetBeginAtom())
        oxygen = cast(ob.OBAtom, bond.GetEndAtom())
        if nitrogen.GetAtomicNum() == 8:
            nitrogen, oxygen = oxygen, nitrogen
        if (
            nitrogen.GetAtomicNum() == 7
            and oxygen.GetAtomicNum() == 8
            and bond.GetBondOrder() == 1
            and oxygen.GetExplicitDegree() == 1
            and oxygen.GetImplicitHCount() == 0
            and get_unpaired_electron_count(nitrogen) > 0
            and get_unpaired_electron_count(oxygen) > 0
            and any(
                atom.GetAtomicNum() != 1 and atom.GetIdx() != oxygen.GetIdx()
                for atom in atom_neighbors(nitrogen)
            )
        ):
            return True
    return False


def _clone_omol(omol: pybel.Molecule) -> pybel.Molecule:
    return pybel.Molecule(ob.OBMol(cast(ob.OBMol, omol.OBMol)))


def neighbor_radical_bond_pairs(omol: pybel.Molecule) -> list[tuple[int, int]]:
    """Defer elongated O2 and carbonyl pairs to charge/spin-aware normalization."""

    pairs: list[tuple[int, int]] = []
    for bond in mol_bonds(cast(ob.OBMol, omol.OBMol)):
        begin_atom = cast(ob.OBAtom, bond.GetBeginAtom())
        end_atom = cast(ob.OBAtom, bond.GetEndAtom())
        if (
            get_unpaired_electron_count(begin_atom) <= 0
            or get_unpaired_electron_count(end_atom) <= 0
        ):
            continue
        # Keep elongated, isolated O2 radicals for the charge/spin-aware
        # normalization. Resolving every pair here would offer only O=O or
        # O+/O-, losing the O-/O- candidate before charge assignment can run.
        if (
            begin_atom.GetAtomicNum() == end_atom.GetAtomicNum() == 8
            and begin_atom.GetExplicitDegree() == end_atom.GetExplicitDegree() == 1
            and begin_atom.GetImplicitHCount() == end_atom.GetImplicitHCount() == 0
            and bond.GetBondOrder() == 1
            and 1.40 <= begin_atom.GetDistance(end_atom) <= 1.65
        ):
            continue
        if bond.GetBondOrder() == 1 and is_elongated_carbonyl_pair(begin_atom, end_atom):
            # Preserve the reduced carbonyl hypothesis instead of irreversibly
            # closing C=O before the available charge can be assigned.
            continue
        if bond.GetBondOrder() == 1 and (
            would_form_bent_ring_allene(begin_atom, end_atom)
            or would_form_bent_ring_allene(end_atom, begin_atom)
        ):
            continue
        pairs.append(tuple(sorted((begin_atom.GetIdx(), end_atom.GetIdx()))))
    return sorted(set(pairs))


def _resolve_neighbor_radical_pair(
    omol: pybel.Molecule,
    begin_idx: int,
    end_idx: int,
    mode: NeighborRadicalMode,
    positive_atom_idx: int | None,
) -> tuple[pybel.Molecule, bool]:
    """Resolve adjacent real radicals by bonding or formal charge separation.

    Both modes consume ``min(unpaired_left, unpaired_right)`` real unpaired
    electrons from each endpoint. ``bond_order`` converts each pair into one bond
    increment and rebuilds endpoint labels. ``charge_separation`` converts the
    same count into ``+n/-n`` formal charges without using lone pairs or unresolved
    centers. The ``n > 1`` charge branch is a documented high-spin heuristic.
    Preconditions are validated before any electron field is changed.
    """

    obmol = cast(ob.OBMol, omol.OBMol)
    begin_atom = cast(ob.OBAtom, obmol.GetAtom(begin_idx))
    end_atom = cast(ob.OBAtom, obmol.GetAtom(end_idx))
    if begin_atom is None or end_atom is None:
        return omol, False

    spin_to_consume = min(
        get_unpaired_electron_count(begin_atom),
        get_unpaired_electron_count(end_atom),
    )
    if spin_to_consume <= 0:
        return omol, False

    bond = None
    if mode == "bond_order":
        bond = cast(ob.OBBond, obmol.GetBond(begin_idx, end_idx))
        if bond is None:
            return omol, False
        spin_to_consume = min(spin_to_consume, 3 - bond.GetBondOrder())
        if spin_to_consume <= 0:
            return omol, False
    elif positive_atom_idx not in (begin_idx, end_idx):
        raise ValueError("positive_atom_idx must identify one endpoint of the radical bond")

    set_unpaired_electron_count(
        begin_atom, get_unpaired_electron_count(begin_atom) - spin_to_consume
    )
    set_unpaired_electron_count(end_atom, get_unpaired_electron_count(end_atom) - spin_to_consume)
    if mode == "bond_order":
        assert bond is not None
        bond.SetBondOrder(bond.GetBondOrder() + spin_to_consume)
        assign_charge_radical_for_atom(begin_atom)
        assign_charge_radical_for_atom(end_atom)
        return omol, True

    negative_atom = end_atom if positive_atom_idx == begin_idx else begin_atom
    positive_atom = begin_atom if positive_atom_idx == begin_idx else end_atom
    positive_atom.SetFormalCharge(positive_atom.GetFormalCharge() + spin_to_consume)
    negative_atom.SetFormalCharge(negative_atom.GetFormalCharge() - spin_to_consume)
    return omol, True


def _state_key(state: ReconstructionState) -> tuple[object, int]:
    return resonance_utils.build_resonance_state_key(state.omol), state.given_charge


def _enumeration_key(
    state: ReconstructionState,
    *,
    track_discrepancy: bool,
) -> tuple[object, ...]:
    state_key = _state_key(state)
    if not track_discrepancy:
        return state_key
    return (*state_key, int(state.metadata.get(_CHARGE_SEPARATION_DISCREPANCY_KEY, 0)))


def _resolved_seed(
    state: ReconstructionState,
    *,
    begin_idx: int,
    end_idx: int,
    mode: NeighborRadicalMode,
    positive_atom_idx: int | None,
) -> ReconstructionState | None:
    machine = OmolStateMachine.from_reconstruction_state(state).branch(
        None,
        omol=_clone_omol(state.omol),
    )
    phase = (
        "resolve_neighbor_radicals_by_bond_order"
        if mode == "bond_order"
        else "resolve_neighbor_radicals_by_charge_separation"
    )
    hit = machine.run_omol_stage(
        phase,
        _resolve_neighbor_radical_pair,
        begin_idx,
        end_idx,
        mode,
        positive_atom_idx,
    )
    if not hit:
        return None

    action = (
        f"bond_order:{begin_idx}-{end_idx}"
        if mode == "bond_order"
        else f"charge_separation:{positive_atom_idx}+"
        f"{end_idx if positive_atom_idx == begin_idx else begin_idx}-"
    )
    actions = tuple(state.metadata.get("neighbor_radical_actions", ())) + (action,)
    machine.metadata["neighbor_radical_actions"] = actions
    machine.metadata["neighbor_radical_resolution"] = mode
    discrepancy = int(state.metadata.get(_CHARGE_SEPARATION_DISCREPANCY_KEY, 0))
    if mode == "charge_separation":
        discrepancy += 1
    machine.metadata[_CHARGE_SEPARATION_DISCREPANCY_KEY] = discrepancy
    if positive_atom_idx is None:
        machine.metadata.pop("positive_atom_idx", None)
    else:
        machine.metadata["positive_atom_idx"] = positive_atom_idx
    return machine.freeze_like(state)


def enumerate_neighbor_radical_seeds(
    state: ReconstructionState,
    *,
    exact_discrepancy: int | None = None,
) -> list[ReconstructionState]:
    """Resolve eligible radical pairs before charge/spin-aware normalization."""

    if exact_discrepancy is not None and exact_discrepancy < 0:
        return []

    if not neighbor_radical_bond_pairs(state.omol):
        machine = OmolStateMachine.from_reconstruction_state(state)
        machine.annotate("neighbor_radicals_not_present", neighbor_radical_resolution="none")
        unchanged = machine.freeze_like(state)
        unchanged.metadata[_CHARGE_SEPARATION_DISCREPANCY_KEY] = 0
        if exact_discrepancy not in (None, 0):
            return []
        return [unchanged]

    pending = [state]
    finished: dict[tuple[object, int], ReconstructionState] = {}
    expanded: set[tuple[object, ...]] = set()
    while pending and len(expanded) < _MAX_NEIGHBOR_RADICAL_SEEDS:
        current = pending.pop()
        current_key = _state_key(current)
        enumeration_key = _enumeration_key(
            current,
            track_discrepancy=exact_discrepancy is not None,
        )
        if enumeration_key in expanded:
            continue
        expanded.add(enumeration_key)
        current_discrepancy = int(current.metadata.get(_CHARGE_SEPARATION_DISCREPANCY_KEY, 0))
        if exact_discrepancy is not None and current_discrepancy > exact_discrepancy:
            continue
        pairs = neighbor_radical_bond_pairs(current.omol)
        if not pairs:
            if exact_discrepancy is None or current_discrepancy == exact_discrepancy:
                finished.setdefault(current_key, current)
            continue

        for begin_idx, end_idx in pairs:
            actions = (
                ("bond_order", None),
                ("charge_separation", begin_idx),
                ("charge_separation", end_idx),
            )
            for mode, positive_atom_idx in actions:
                candidate = _resolved_seed(
                    current,
                    begin_idx=begin_idx,
                    end_idx=end_idx,
                    mode=cast(NeighborRadicalMode, mode),
                    positive_atom_idx=positive_atom_idx,
                )
                if candidate is not None:
                    pending.append(candidate)

    return sorted(
        finished.values(),
        key=lambda candidate: tuple(candidate.metadata.get("neighbor_radical_actions", ())),
    )


__all__ = [
    "enumerate_neighbor_radical_seeds",
    "has_substituted_n_o_radical_pair",
    "neighbor_radical_bond_pairs",
]
