"""Search-space utilities for metal-aware fallback reconstruction."""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Optional, Tuple, Union, cast

from typing_extensions import TypeAlias

from molgr.config import CONFIG, MolGRConfig
from molgr.fallback.state import MetalCandidateState, MetalCandidateStateMachine
from molgr.fallback.utils import consts, dataclasses


_ValenceBoundsKey = Tuple[Tuple[str, int, int], ...]
_MetalStateChoice = Tuple[dataclasses.MetalAtomPosition, ...]
_MetalStateOptionInput: TypeAlias = Union[dataclasses.MetalAtomPosition, _MetalStateChoice]
_MetalStateChoiceGroup = Tuple[_MetalStateChoice, ...]
_MetalStateSearchLayers = Tuple[Tuple[_MetalStateChoiceGroup, ...], ...]
_ChargeGroupedAssignments = Dict[
    int,
    List[Tuple[_ValenceBoundsKey, List["_PartialMetalAssignment"]]],
]
_RadicalBucketLevel = Tuple[int, _ChargeGroupedAssignments]
_RadicalPrefixReachability = Tuple[Tuple[int, ...], Tuple[_RadicalBucketLevel, ...]]


@dataclass(frozen=True)
class _PartialMetalAssignment:
    metal_states: Tuple[dataclasses.MetalAtomPosition, ...]
    total_metal_charge: int
    total_metal_radicals: int
    metal_assignment_rank: float
    valence_bounds: _ValenceBoundsKey
    order: int


def _metal_state_assignment_penalty(metal_state: dataclasses.MetalAtomPosition) -> float:
    penalty = 0.0
    if metal_state.valence <= 0:
        penalty += 10.0 * max(abs(metal_state.valence), 1)

    prior_list = consts.METAL_VALENCE_AVAILABLE_PRIOR.get(metal_state.symbol, [])
    minor_list = consts.METAL_VALENCE_AVAILABLE_MINOR.get(metal_state.symbol, [])
    if metal_state.valence not in prior_list:
        if metal_state.valence in minor_list:
            penalty += 10.0
        else:
            penalty += 20.0
    return penalty


def _metal_state_choice_sort_key(
    metal_state_choice: _MetalStateChoice,
) -> tuple[float, int, tuple[tuple[str, int, int, int], ...]]:
    return (
        sum(_metal_state_assignment_penalty(metal_state) for metal_state in metal_state_choice),
        sum(int(metal_state.radical_num) for metal_state in metal_state_choice),
        tuple(
            (
                metal_state.symbol,
                int(metal_state.valence),
                int(metal_state.radical_num),
                int(metal_state.idx),
            )
            for metal_state in metal_state_choice
        ),
    )


def _build_unified_same_element_state_options(
    grouped_state_options: Sequence[Sequence[dataclasses.MetalAtomPosition]],
) -> _MetalStateChoiceGroup | None:
    signature_maps: list[dict[tuple[int, int], dataclasses.MetalAtomPosition]] = []
    shared_signatures: set[tuple[int, int]] | None = None
    for state_options in grouped_state_options:
        signature_map: dict[tuple[int, int], dataclasses.MetalAtomPosition] = {}
        for state in state_options:
            signature_map.setdefault((int(state.valence), int(state.radical_num)), state)
        if not signature_map:
            return None
        signature_maps.append(signature_map)
        signatures = set(signature_map)
        shared_signatures = (
            signatures if shared_signatures is None else shared_signatures & signatures
        )
    if not shared_signatures:
        return None

    return tuple(
        tuple(signature_map[signature] for signature_map in signature_maps)
        for signature in sorted(
            shared_signatures,
            key=lambda item: _metal_state_choice_sort_key(
                tuple(signature_map[item] for signature_map in signature_maps)
            ),
        )
    )


def _build_metal_state_search_groups(
    available_valence_radical_states: Sequence[Sequence[dataclasses.MetalAtomPosition]],
    *,
    config: MolGRConfig | None = None,
) -> Tuple[_MetalStateChoiceGroup, ...]:
    resolved_config = CONFIG if config is None else config
    metal_scoring_config = resolved_config.metal_scoring
    unify_threshold = int(metal_scoring_config.same_element_multimetal_unify_threshold)

    grouped_indices_by_symbol: DefaultDict[str, list[int]] = defaultdict(list)
    for idx, state_options in enumerate(available_valence_radical_states):
        if not state_options:
            continue
        grouped_indices_by_symbol[state_options[0].symbol].append(idx)

    search_groups: list[_MetalStateChoiceGroup] = []
    skipped_indices: set[int] = set()
    for idx, state_options in enumerate(available_valence_radical_states):
        if idx in skipped_indices:
            continue
        if not state_options:
            search_groups.append(())
            continue

        symbol = state_options[0].symbol
        grouped_indices = grouped_indices_by_symbol[symbol]
        if (
            unify_threshold >= 0
            and len(grouped_indices) > unify_threshold
            and idx == grouped_indices[0]
        ):
            unified_state_options = _build_unified_same_element_state_options(
                tuple(
                    available_valence_radical_states[grouped_idx] for grouped_idx in grouped_indices
                )
            )
            if unified_state_options is not None:
                search_groups.append(unified_state_options)
                skipped_indices.update(grouped_indices[1:])
                continue

        search_groups.append(tuple((state,) for state in state_options))

    return tuple(search_groups)


def _build_layered_metal_state_search_groups(
    available_state_search_groups: Sequence[Sequence[_MetalStateChoice]],
    total_radical_electrons: int,
    *,
    config: MolGRConfig | None = None,
) -> _MetalStateSearchLayers:
    radical_budget = max(0, total_radical_electrons)
    normalized_groups = tuple(
        tuple(
            state_choice
            for state_choice in group
            if sum(int(state.radical_num) for state in state_choice) <= radical_budget
        )
        for group in available_state_search_groups
    )
    if total_radical_electrons <= 0 or len(normalized_groups) < 2:
        return (normalized_groups,)

    resolved_config = CONFIG if config is None else config
    metal_scoring_config = resolved_config.metal_scoring
    penalty_window = float(metal_scoring_config.open_shell_multimetal_state_penalty_window)
    min_state_options = max(1, int(metal_scoring_config.open_shell_multimetal_min_state_options))

    ranked_group_entries: list[tuple[tuple[_MetalStateChoice, float], ...]] = []
    group_thresholds: list[tuple[float, ...]] = []
    layer_count = 1
    for state_search_group in normalized_groups:
        candidate_state_options = tuple(state_search_group)
        ranked_state_entries = tuple(
            (
                state_choice,
                sum(_metal_state_assignment_penalty(metal_state) for metal_state in state_choice),
            )
            for state_choice in sorted(candidate_state_options, key=_metal_state_choice_sort_key)
        )
        ranked_group_entries.append(ranked_state_entries)
        if not ranked_state_entries:
            group_thresholds.append((0.0,))
            continue

        penalties = tuple(penalty for _, penalty in ranked_state_entries)
        if len(ranked_state_entries) <= min_state_options or penalty_window < 0.0:
            group_thresholds.append((penalties[-1],))
            continue

        initial_limit = penalties[min(len(penalties), min_state_options) - 1]
        thresholds: list[float] = [initial_limit]
        unique_penalties = sorted(set(penalties))
        current_limit = initial_limit
        max_penalty = unique_penalties[-1]
        while current_limit < max_penalty:
            if penalty_window == 0.0:
                next_limit = next(
                    penalty for penalty in unique_penalties if penalty > current_limit
                )
            else:
                candidate_limits = [
                    penalty
                    for penalty in unique_penalties
                    if current_limit < penalty <= current_limit + penalty_window
                ]
                if candidate_limits:
                    next_limit = candidate_limits[-1]
                else:
                    next_limit = next(
                        penalty for penalty in unique_penalties if penalty > current_limit
                    )
            thresholds.append(next_limit)
            current_limit = next_limit
        group_thresholds.append(tuple(thresholds))
        layer_count = max(layer_count, len(thresholds))

    layers: list[Tuple[_MetalStateChoiceGroup, ...]] = []
    previous_layer: Tuple[_MetalStateChoiceGroup, ...] | None = None
    for layer_idx in range(layer_count):
        layer_groups: list[_MetalStateChoiceGroup] = []
        for ranked_state_entries, threshold_values in zip(ranked_group_entries, group_thresholds):
            threshold = threshold_values[min(layer_idx, len(threshold_values) - 1)]
            layer_groups.append(
                tuple(
                    state_choice
                    for state_choice, penalty in ranked_state_entries
                    if penalty <= threshold
                )
            )
        layer = tuple(layer_groups)
        if previous_layer is not None and layer == previous_layer:
            continue
        layers.append(layer)
        previous_layer = layer

    return tuple(layers) if layers else (normalized_groups,)


def _update_valence_bounds(
    bounds: _ValenceBoundsKey,
    metal_state: dataclasses.MetalAtomPosition,
    max_mixed_valence_spread: Optional[int],
) -> Optional[_ValenceBoundsKey]:
    if max_mixed_valence_spread is None or max_mixed_valence_spread < 0:
        return ()

    updated_bounds = list(bounds)
    for idx, (symbol, lower, upper) in enumerate(updated_bounds):
        if symbol != metal_state.symbol:
            continue
        next_lower = min(lower, metal_state.valence)
        next_upper = max(upper, metal_state.valence)
        if next_upper - next_lower > max_mixed_valence_spread:
            return None
        updated_bounds[idx] = (symbol, next_lower, next_upper)
        break
    else:
        updated_bounds.append((metal_state.symbol, metal_state.valence, metal_state.valence))
        updated_bounds.sort(key=lambda item: item[0])

    return tuple(updated_bounds)


def _trim_partial_assignments(
    entries: List[_PartialMetalAssignment],
    max_assignments_per_target: int,
) -> List[_PartialMetalAssignment]:
    limit = max(1, max_assignments_per_target)
    entries.sort(key=lambda entry: (entry.metal_assignment_rank, entry.order))
    return entries[:limit]


def _resolve_search_limits(
    total_radical_electrons: int,
    *,
    config: MolGRConfig | None = None,
) -> tuple[Optional[int], int, int]:
    resolved_config = CONFIG if config is None else config
    metal_scoring_config = resolved_config.metal_scoring
    return (
        metal_scoring_config.max_mixed_valence_spread,
        max(0, total_radical_electrons),
        metal_scoring_config.max_assignments_per_target,
    )


def _enumerate_partial_assignment_frontier(
    available_valence_radical_states: Sequence[Sequence[_MetalStateChoice]],
    total_radical_electrons: int,
    *,
    config: MolGRConfig | None = None,
) -> Dict[Tuple[int, int, _ValenceBoundsKey], List[_PartialMetalAssignment]]:
    """Enumerate one half of the metal search space while pruning dominated prefixes."""

    max_mixed_valence_spread, max_total_metal_radicals, max_assignments_per_state = (
        _resolve_search_limits(total_radical_electrons, config=config)
    )
    partial_assignments: Dict[Tuple[int, int, _ValenceBoundsKey], List[_PartialMetalAssignment]] = {
        (0, 0, ()): [
            _PartialMetalAssignment(
                metal_states=(),
                total_metal_charge=0,
                total_metal_radicals=0,
                metal_assignment_rank=0.0,
                valence_bounds=(),
                order=0,
            )
        ]
    }
    next_order = 1

    for metal_state_options in available_valence_radical_states:
        next_partial_assignments: Dict[
            Tuple[int, int, _ValenceBoundsKey], List[_PartialMetalAssignment]
        ] = defaultdict(list)
        # Match the native std::map traversal before assigning tie-break orders.
        # Insertion order changes which equally ranked mixed-metal state wins.
        for frontier_key in sorted(partial_assignments):
            entries = partial_assignments[frontier_key]
            for entry in entries:
                for metal_state_choice in metal_state_options:
                    next_total_metal_charge = entry.total_metal_charge
                    next_total_metal_radicals = entry.total_metal_radicals
                    next_metal_assignment_rank = entry.metal_assignment_rank
                    next_valence_bounds: Optional[_ValenceBoundsKey] = entry.valence_bounds
                    for metal_state in metal_state_choice:
                        next_total_metal_charge += int(metal_state.valence)
                        next_total_metal_radicals += int(metal_state.radical_num)
                        next_metal_assignment_rank += _metal_state_assignment_penalty(metal_state)
                        next_valence_bounds = _update_valence_bounds(
                            cast(_ValenceBoundsKey, next_valence_bounds),
                            metal_state,
                            max_mixed_valence_spread,
                        )
                        if next_valence_bounds is None:
                            break
                    if next_valence_bounds is None:
                        continue
                    if (
                        max_total_metal_radicals is not None
                        and next_total_metal_radicals > max_total_metal_radicals
                    ):
                        continue

                    next_entry = _PartialMetalAssignment(
                        metal_states=entry.metal_states + metal_state_choice,
                        total_metal_charge=next_total_metal_charge,
                        total_metal_radicals=next_total_metal_radicals,
                        metal_assignment_rank=next_metal_assignment_rank,
                        valence_bounds=cast(_ValenceBoundsKey, next_valence_bounds),
                        order=next_order,
                    )
                    next_order += 1
                    next_partial_assignments[
                        (
                            next_entry.total_metal_charge,
                            next_entry.total_metal_radicals,
                            next_entry.valence_bounds,
                        )
                    ].append(next_entry)

        partial_assignments = {
            key: _trim_partial_assignments(entries, max_assignments_per_state)
            for key, entries in next_partial_assignments.items()
        }
        if not partial_assignments:
            break

    return partial_assignments


def _merge_valence_bounds(
    left_bounds: _ValenceBoundsKey,
    right_bounds: _ValenceBoundsKey,
    max_mixed_valence_spread: Optional[int],
) -> Optional[_ValenceBoundsKey]:
    if max_mixed_valence_spread is None or max_mixed_valence_spread < 0:
        return ()

    merged_bounds: Dict[str, Tuple[int, int]] = {
        symbol: (lower, upper) for symbol, lower, upper in left_bounds
    }
    for symbol, lower, upper in right_bounds:
        previous_bounds = merged_bounds.get(symbol)
        if previous_bounds is None:
            merged_bounds[symbol] = (lower, upper)
            continue
        next_lower = min(previous_bounds[0], lower)
        next_upper = max(previous_bounds[1], upper)
        if next_upper - next_lower > max_mixed_valence_spread:
            return None
        merged_bounds[symbol] = (next_lower, next_upper)

    return tuple(sorted((symbol, lower, upper) for symbol, (lower, upper) in merged_bounds.items()))


def _bucket_partial_assignments_by_charge_radicals(
    frontier: Dict[Tuple[int, int, _ValenceBoundsKey], List[_PartialMetalAssignment]],
) -> Dict[int, _ChargeGroupedAssignments]:
    bucket_index: DefaultDict[
        int,
        DefaultDict[int, List[Tuple[_ValenceBoundsKey, List[_PartialMetalAssignment]]]],
    ] = defaultdict(lambda: defaultdict(list))
    for charge, radicals, valence_bounds in sorted(frontier):
        entries = frontier[(charge, radicals, valence_bounds)]
        bucket_index[radicals][charge].append((valence_bounds, entries))

    return {
        radicals: {charge: charge_groups[charge] for charge in sorted(charge_groups)}
        for radicals, charge_groups in sorted(bucket_index.items())
    }


def _build_radical_prefix_reachability(
    bucket_index: Dict[int, _ChargeGroupedAssignments],
) -> _RadicalPrefixReachability:
    radical_levels = tuple(sorted(bucket_index))
    return radical_levels, tuple((radicals, bucket_index[radicals]) for radicals in radical_levels)


def _iter_reachable_radical_buckets(
    prefix_reachability: _RadicalPrefixReachability,
    max_radicals: int,
) -> Sequence[_RadicalBucketLevel]:
    radical_levels, bucket_levels = prefix_reachability
    prefix_end = bisect_right(radical_levels, max_radicals)
    if prefix_end <= 0:
        return ()
    return bucket_levels[:prefix_end]


def _combine_partial_assignment_frontiers(
    left_frontier: Dict[Tuple[int, int, _ValenceBoundsKey], List[_PartialMetalAssignment]],
    right_frontier: Dict[Tuple[int, int, _ValenceBoundsKey], List[_PartialMetalAssignment]],
    total_charge: int,
    total_radical_electrons: int,
    *,
    config: MolGRConfig | None = None,
) -> Dict[Tuple[int, int], List[_PartialMetalAssignment]]:
    """Join the left/right DP frontiers by compatible charge and radical buckets."""

    max_mixed_valence_spread, max_total_metal_radicals, max_assignments_per_target = (
        _resolve_search_limits(total_radical_electrons, config=config)
    )
    grouped_entries: Dict[Tuple[int, int], List[_PartialMetalAssignment]] = defaultdict(list)
    trim_trigger = max(1, max_assignments_per_target) * 4
    left_bucket_index = _bucket_partial_assignments_by_charge_radicals(left_frontier)
    right_bucket_index = _bucket_partial_assignments_by_charge_radicals(right_frontier)
    right_prefix_reachability = _build_radical_prefix_reachability(right_bucket_index)
    right_order_stride = (
        max(
            (entry.order for entries in right_frontier.values() for entry in entries),
            default=0,
        )
        + 1
    )
    max_combined_metal_radicals = max_total_metal_radicals

    for left_radicals, left_charge_groups in left_bucket_index.items():
        max_right_radicals = max_combined_metal_radicals - left_radicals
        if max_right_radicals < 0:
            continue

        for _right_radicals, right_charge_groups in _iter_reachable_radical_buckets(
            right_prefix_reachability,
            max_right_radicals,
        ):
            total_metal_radicals = left_radicals + _right_radicals
            target_radicals = total_radical_electrons - total_metal_radicals
            for left_charge, left_valence_groups in left_charge_groups.items():
                for right_charge, right_valence_groups in right_charge_groups.items():
                    target = (total_charge - (left_charge + right_charge), target_radicals)
                    bucket = grouped_entries[target]
                    initial_bucket_size = len(bucket)
                    for left_bounds, left_entries in left_valence_groups:
                        for right_bounds, right_entries in right_valence_groups:
                            merged_bounds = _merge_valence_bounds(
                                left_bounds,
                                right_bounds,
                                max_mixed_valence_spread,
                            )
                            if merged_bounds is None:
                                continue

                            for left_entry in left_entries:
                                for right_entry in right_entries:
                                    bucket.append(
                                        _PartialMetalAssignment(
                                            metal_states=left_entry.metal_states
                                            + right_entry.metal_states,
                                            total_metal_charge=left_entry.total_metal_charge
                                            + right_entry.total_metal_charge,
                                            total_metal_radicals=left_entry.total_metal_radicals
                                            + right_entry.total_metal_radicals,
                                            metal_assignment_rank=left_entry.metal_assignment_rank
                                            + right_entry.metal_assignment_rank,
                                            valence_bounds=merged_bounds,
                                            order=(
                                                left_entry.order * right_order_stride
                                                + right_entry.order
                                            ),
                                        )
                                    )

                    if len(bucket) == initial_bucket_size:
                        if initial_bucket_size == 0:
                            grouped_entries.pop(target, None)
                        continue

                    if len(bucket) > trim_trigger:
                        grouped_entries[target] = _trim_partial_assignments(
                            bucket,
                            max_assignments_per_target,
                        )

    return {
        target: _trim_partial_assignments(entries, max_assignments_per_target)
        for target, entries in grouped_entries.items()
    }


def _group_candidates_by_target_dp(
    base_phase_history: Sequence[str],
    available_valence_radical_states: Sequence[Sequence[_MetalStateOptionInput]],
    total_charge: int,
    total_radical_electrons: int,
    *,
    config: MolGRConfig | None = None,
) -> Dict[Tuple[int, int], List[MetalCandidateState]]:
    """Build target buckets directly, without enumerating the full metal Cartesian product."""

    normalized_state_groups: Tuple[_MetalStateChoiceGroup, ...] = tuple(
        tuple(
            (state_option,)
            if isinstance(state_option, dataclasses.MetalAtomPosition)
            else state_option
            for state_option in state_options
        )
        for state_options in available_valence_radical_states
    )
    split_index = len(normalized_state_groups) // 2
    left_frontier = _enumerate_partial_assignment_frontier(
        normalized_state_groups[:split_index],
        total_radical_electrons,
        config=config,
    )
    right_frontier = _enumerate_partial_assignment_frontier(
        normalized_state_groups[split_index:],
        total_radical_electrons,
        config=config,
    )
    if not left_frontier or not right_frontier:
        return {}

    grouped_entries = _combine_partial_assignment_frontiers(
        left_frontier,
        right_frontier,
        total_charge,
        total_radical_electrons,
        config=config,
    )
    grouped_candidates: Dict[Tuple[int, int], List[MetalCandidateState]] = {}
    combination_index = 0
    for target, entries in sorted(grouped_entries.items()):
        bucket: List[MetalCandidateState] = []
        for entry in entries:
            candidate_machine = MetalCandidateStateMachine(
                entry.metal_states,
                target[0],
                target[1],
                phase_history=base_phase_history,
                metadata={"combination_index": combination_index},
            )
            candidate_machine.annotate("enumerate_metal_combination")
            candidate_machine.annotate("reconstruct_no_metal_candidate")
            candidate_machine.annotate(
                "rank_metal_assignment_for_target",
                metal_assignment_rank=entry.metal_assignment_rank,
            )
            bucket.append(candidate_machine.freeze())
            combination_index += 1
        grouped_candidates[target] = bucket
    return grouped_candidates


__all__ = [
    "_PartialMetalAssignment",
    "_build_layered_metal_state_search_groups",
    "_build_metal_state_search_groups",
    "_build_radical_prefix_reachability",
    "_combine_partial_assignment_frontiers",
    "_enumerate_partial_assignment_frontier",
    "_group_candidates_by_target_dp",
    "_iter_reachable_radical_buckets",
    "_merge_valence_bounds",
    "_metal_state_assignment_penalty",
]
