# pyright: reportMissingImports=false

from __future__ import annotations

from collections import deque
from dataclasses import replace

import pytest


pytest.importorskip("openbabel")

from openbabel import openbabel as ob
from openbabel import pybel

from molgr.config import MolGRConfig
from molgr.fallback.pipeline import resonance as resonance_module
from molgr.fallback.pipeline.resonance import (
    ResonanceTraversalContext,
    ResonanceTraversalMove,
    build_processed_resonance_key,
    build_resonance_state_key,
    get_radical_resonances,
    process_resonance,
)
from molgr.fallback.stages import clean as clean_module
from molgr.fallback.state import (
    OmolStateMachine,
    OmolTraceRecorder,
    ReconstructionState,
    trace_omol_state_machine,
)
from molgr.fallback.utils import resonance as resonance_utils_module
from molgr.fallback.utils.electrons import (
    get_unpaired_electron_count,
    has_unresolved_two_electron_center,
    set_unpaired_electron_count,
    set_unresolved_two_electron_center,
)
from molgr.fallback.utils.force_field import selection_force_field_energy
from molgr.fallback.utils.no_metals import resonance as no_metal_resonance_module


def _make_seed(smiles: str, radical_atom_indices: tuple[int, ...]) -> pybel.Molecule:
    mol = pybel.readstring("smi", smiles)
    for idx in radical_atom_indices:
        set_unpaired_electron_count(mol.OBMol.GetAtom(idx), 1)
    return mol


def _make_radical_carbonyl_resonance_seed(*, target_radical: bool) -> pybel.Molecule:
    obmol = ob.OBMol()
    for atomic_num in (8, 6, 6, 6):
        atom = obmol.NewAtom()
        atom.SetAtomicNum(atomic_num)
    obmol.AddBond(1, 2, 1)
    obmol.AddBond(2, 3, 2)
    obmol.AddBond(3, 4, 1)
    set_unpaired_electron_count(obmol.GetAtom(1), 1)
    if target_radical:
        set_unpaired_electron_count(obmol.GetAtom(3), 1)
    return pybel.Molecule(obmol)


@pytest.mark.parametrize("target_radical", [False, True])
@pytest.mark.parametrize("target_atomic_num", [6, 14])
def test_radical_resonance_regenerates_deferred_two_electron_target_for_python_and_cpp(
    target_radical: bool,
    target_atomic_num: int,
) -> None:
    seed = _make_radical_carbonyl_resonance_seed(target_radical=target_radical)
    seed.OBMol.GetAtom(3).SetAtomicNum(target_atomic_num)
    python_states = get_radical_resonances(seed, max_depth=1)

    assert len(python_states) == 2
    python_target = python_states[1].OBMol.GetAtom(3)
    if target_radical:
        assert get_unpaired_electron_count(python_target) == 0
        assert has_unresolved_two_electron_center(python_target)
    else:
        assert get_unpaired_electron_count(python_target) == 1
        assert not has_unresolved_two_electron_center(python_target)

    from molgr import _core  # type: ignore

    cpp_seed = _make_radical_carbonyl_resonance_seed(target_radical=target_radical)
    cpp_seed.OBMol.GetAtom(3).SetAtomicNum(target_atomic_num)
    cpp_ptrs = _core.dev.pipeline.resonance.get_radical_resonances_ptr(
        int(getattr(cpp_seed.OBMol, "this", cpp_seed.OBMol))
    )
    try:
        assert len(cpp_ptrs) == 2
        cpp_target = _core.utils.extract_molecule_data(cpp_ptrs[1]).atoms[2]
        assert cpp_target.radical_num == get_unpaired_electron_count(python_target)
        assert cpp_target.unresolved_two_electron_center is has_unresolved_two_electron_center(
            python_target
        )
    finally:
        for ptr in cpp_ptrs:
            _core.free_obmol_ptr(ptr)


def test_default_resonance_traversal_score_names_uff_lite_gain() -> None:
    config = MolGRConfig()

    assert config.resonance.traversal_score == "uff_lite_gain"


def test_no_metal_default_resonance_policy_accepts_uff_lite_gain() -> None:
    base_config = MolGRConfig()
    config = replace(
        base_config,
        resonance=replace(
            base_config.resonance,
            traversal_score="uff_lite_gain",
        ),
    )

    policy = no_metal_resonance_module._default_resonance_traversal_policy(config)

    assert type(policy).__name__ == "_LimitedDiscrepancyUffLiteGainTraversalPolicy"
    assert policy.max_discrepancy == config.resonance.limited_discrepancy_max_discrepancy


def _naive_resonance_state_keys(seed: pybel.Molecule, max_depth: int = 2) -> set[tuple]:
    root_key, bond_index_map = resonance_utils_module._build_resonance_search_context(seed)
    seen = {root_key}
    frontier = deque([(seed, root_key, 0)])
    while frontier:
        current, current_key, depth = frontier.popleft()
        if depth >= max_depth:
            continue
        for move in resonance_utils_module._enumerate_one_step_resonance_moves(
            current,
            current_key,
            bond_index_map,
        ):
            next_resonance = resonance_utils_module._materialize_one_step_resonance(
                current, move.idxs
            )
            next_key = move.next_state_key
            if next_key in seen:
                continue
            seen.add(next_key)
            frontier.append((next_resonance, next_key, depth + 1))
    return seen


def test_build_resonance_state_key_matches_clone_and_changes_with_state() -> None:
    seed = _make_seed("C=CC=C", (2,))
    clone = seed.clone
    moved = get_radical_resonances(seed, max_depth=1)[1]

    assert build_resonance_state_key(seed) == build_resonance_state_key(clone)
    assert build_resonance_state_key(seed) != build_resonance_state_key(moved)


def test_build_processed_resonance_key_matches_clone_and_changes_with_state() -> None:
    seed = _make_seed("C=CC=C", (2,))
    clone = seed.clone
    moved = get_radical_resonances(seed, max_depth=1)[1]

    assert build_processed_resonance_key(seed) == build_processed_resonance_key(clone)
    assert build_processed_resonance_key(seed) != build_processed_resonance_key(moved)


def test_resonance_keys_include_unresolved_two_electron_center_marker() -> None:
    seed = pybel.readstring("smi", "CC")
    marked = seed.clone
    set_unresolved_two_electron_center(marked.OBMol.GetAtom(1), True)

    assert build_resonance_state_key(seed) != build_resonance_state_key(marked)
    assert build_processed_resonance_key(seed) != build_processed_resonance_key(marked)


def test_build_processed_resonance_key_avoids_openbabel_format_serialization(
    monkeypatch,
) -> None:
    def fail_write(self, format: str = "smi", filename=None, overwrite=False, opt=None):
        raise AssertionError(f"processed resonance key should not call OpenBabel write({format!r})")

    monkeypatch.setattr(pybel.Molecule, "write", fail_write)

    seed = _make_seed("C=CC=C", (2,))
    key = build_processed_resonance_key(seed)

    assert key.startswith("A4:")
    assert "|B3:" in key


def test_get_radical_resonances_avoids_smiles_serialization_for_dedup(monkeypatch) -> None:
    original_write = pybel.Molecule.write

    def patched_write(self, format: str = "smi", filename=None, overwrite=False, opt=None):
        if format == "smi":
            raise AssertionError("SMILES serialization should not be used for resonance dedup")
        return original_write(self, format=format, filename=filename, overwrite=overwrite, opt=opt)

    monkeypatch.setattr(pybel.Molecule, "write", patched_write)

    seed = _make_seed("C=CC=C", (2,))
    resonances = get_radical_resonances(seed)

    assert len(resonances) > 1
    assert len({build_resonance_state_key(mol) for mol in resonances}) == len(resonances)


def test_incremental_resonance_keys_match_rebuilt_keys() -> None:
    seed = _make_seed("C=CC=C", (2,))
    root_key, bond_index_map = resonance_utils_module._build_resonance_search_context(seed)

    for move in resonance_utils_module._enumerate_one_step_resonance_moves(
        seed,
        root_key,
        bond_index_map,
    ):
        moved = resonance_utils_module._materialize_one_step_resonance(seed, move.idxs)
        incremental_key = move.next_state_key
        assert incremental_key == build_resonance_state_key(moved)


def test_get_radical_resonances_builds_root_key_only_once(monkeypatch) -> None:
    original = resonance_module.build_resonance_state_key
    calls = []

    def wrapped(omol):
        calls.append(1)
        return original(omol)

    monkeypatch.setattr(resonance_module, "build_resonance_state_key", wrapped)
    seed = _make_seed("C=CC=C", (2,))
    resonances = get_radical_resonances(seed)

    assert len(resonances) > 1
    assert len(calls) == 0


def test_get_radical_resonances_matches_naive_bfs_on_independent_components() -> None:
    seed = _make_seed("C=CC=C.C=CC=C", (2, 6))

    resonances = get_radical_resonances(seed, max_depth=2)
    actual_keys = {build_resonance_state_key(mol) for mol in resonances}
    expected_keys = _naive_resonance_state_keys(seed, max_depth=2)

    assert actual_keys == expected_keys


def test_get_radical_resonances_accepts_traversal_policy_to_prune_directions() -> None:
    seed = _make_seed("C=CC=C.C=CC=C", (2, 6))
    calls = []

    def traversal_policy(
        context: ResonanceTraversalContext,
        moves: tuple[ResonanceTraversalMove, ...],
    ) -> list[ResonanceTraversalMove]:
        calls.append((context.depth, tuple(move.path for move in moves)))
        if context.depth == 0:
            return [move for move in moves if move.path == (2, 3, 4)]
        return []

    resonances = get_radical_resonances(
        seed,
        max_depth=2,
        traversal_policy=traversal_policy,
    )

    assert len(resonances) == 2
    assert calls[0] == (0, ((2, 3, 4), (6, 7, 8)))
    assert calls[1][0] == 1
    assert (6, 7, 8) in calls[1][1]


def test_clean_resonances_returns_false_when_no_rule_hits(monkeypatch) -> None:
    ordered_rules = (11, 0, 1, 2, 3, 4, 9, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18)
    calls = []

    def make_stage(rule_id):
        def stage(omol):
            calls.append(rule_id)
            return omol, False

        return stage

    for rule_id in set(ordered_rules):
        monkeypatch.setattr(clean_module, f"clean_resonances_{rule_id}", make_stage(rule_id))

    omol = pybel.readstring("smi", "CC")
    result, hit = clean_module.clean_resonances(omol)

    assert result is omol
    assert hit is False
    assert calls == list(ordered_rules)


def test_clean_resonances_accumulates_stage_hits_without_global_signature(monkeypatch) -> None:
    ordered_rules = (11, 0, 1, 2, 3, 4, 9, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18)
    calls = []

    def make_stage(rule_id):
        def stage(omol):
            calls.append(rule_id)
            return omol, rule_id in {3, 10}

        return stage

    for rule_id in set(ordered_rules):
        monkeypatch.setattr(clean_module, f"clean_resonances_{rule_id}", make_stage(rule_id))

    omol = pybel.readstring("smi", "CC")
    result, hit = clean_module.clean_resonances(omol)

    assert result is omol
    assert hit is True
    assert calls == list(ordered_rules)


def test_get_radical_resonances_limited_discrepancy_policy_prefers_low_discrepancy_paths(
    monkeypatch,
) -> None:
    seed = _make_seed("C=CC=C", (2,))
    state_key_a = (((6, 0, 0, False),), ((1, 2, 1, False),))
    state_key_b = (((6, 0, 0, False),), ((1, 2, 2, False),))
    state_key_c = (((6, 0, 0, False),), ((1, 2, 3, False),))
    state_key_a1 = (((6, 0, 0, False),), ((1, 2, 4, False),))
    state_key_b1 = (((6, 0, 0, False),), ((1, 2, 5, False),))
    move_a = resonance_utils_module._IndexedResonanceTraversalMove(  # type: ignore[attr-defined]
        idxs=(1, 2, 3),
        next_state_key=state_key_a,
    )
    move_b = resonance_utils_module._IndexedResonanceTraversalMove(  # type: ignore[attr-defined]
        idxs=(4, 5, 6),
        next_state_key=state_key_b,
    )
    move_c = resonance_utils_module._IndexedResonanceTraversalMove(  # type: ignore[attr-defined]
        idxs=(7, 8, 9),
        next_state_key=state_key_c,
    )
    child_move_a = resonance_utils_module._IndexedResonanceTraversalMove(  # type: ignore[attr-defined]
        idxs=(10, 11, 12),
        next_state_key=state_key_a1,
    )
    child_move_b = resonance_utils_module._IndexedResonanceTraversalMove(  # type: ignore[attr-defined]
        idxs=(20, 21, 22),
        next_state_key=state_key_b1,
    )
    policy = resonance_module.make_limited_discrepancy_uff_lite_gain_traversal_policy(
        max_discrepancy=1
    )

    scores_by_state_and_move = {
        ("seed", (1, 2, 3)): 1.0,
        ("seed", (4, 5, 6)): 2.0,
        ("seed", (7, 8, 9)): 3.0,
        ("state-a", (10, 11, 12)): 1.0,
        ("state-b", (20, 21, 22)): 1.0,
    }

    def fake_order_uff_lite_gain_moves(omol, moves):
        label = "seed" if omol is seed else omol
        ordered_moves = [(move, scores_by_state_and_move[(label, move.path)]) for move in moves]
        ordered_moves.sort(key=lambda item: (item[1], item[0].path))
        return ordered_moves

    def fake_enumerate(omol, state_key, bond_index_map):
        if omol is seed:
            return [move_c, move_b, move_a]
        if omol == "state-a":
            return [child_move_a]
        if omol == "state-b":
            return [child_move_b]
        return []

    def fake_materialize(omol, idxs):
        if omol is seed:
            return {
                (1, 2, 3): "state-a",
                (4, 5, 6): "state-b",
                (7, 8, 9): "state-c",
            }[idxs]
        if omol == "state-a":
            return "state-a1"
        if omol == "state-b":
            return "state-b1"
        return f"terminal-{idxs}"

    monkeypatch.setattr(
        resonance_utils_module,
        "_order_uff_lite_gain_moves",
        fake_order_uff_lite_gain_moves,
    )
    monkeypatch.setattr(
        resonance_utils_module, "_enumerate_one_step_resonance_moves", fake_enumerate
    )
    monkeypatch.setattr(resonance_utils_module, "_materialize_one_step_resonance", fake_materialize)

    resonances = get_radical_resonances(seed, max_depth=2, traversal_policy=policy)

    assert resonances[0] is seed
    assert resonances[1:] == ["state-a", "state-a1", "state-b", "state-b1"]


def test_uff_lite_limited_discrepancy_prunes_high_rank_resonance_branch(
    monkeypatch,
) -> None:
    seed = _make_seed("C=CC=C", (2,))
    state_key_a = (((6, 0, 0, False),), ((1, 2, 1, False),))
    state_key_b = (((6, 0, 0, False),), ((1, 2, 2, False),))
    state_key_c = (((6, 0, 0, False),), ((1, 2, 3, False),))
    move_a = resonance_utils_module._IndexedResonanceTraversalMove(  # type: ignore[attr-defined]
        idxs=(1, 2, 3),
        next_state_key=state_key_a,
    )
    move_b = resonance_utils_module._IndexedResonanceTraversalMove(  # type: ignore[attr-defined]
        idxs=(4, 5, 6),
        next_state_key=state_key_b,
    )
    move_c = resonance_utils_module._IndexedResonanceTraversalMove(  # type: ignore[attr-defined]
        idxs=(7, 8, 9),
        next_state_key=state_key_c,
    )
    policy = resonance_module.make_limited_discrepancy_uff_lite_gain_traversal_policy(
        max_discrepancy=1
    )

    monkeypatch.setattr(
        resonance_utils_module,
        "_order_uff_lite_gain_moves",
        lambda omol, moves: sorted(
            [
                (
                    move,
                    {
                        (1, 2, 3): 1.0,
                        (4, 5, 6): 2.0,
                        (7, 8, 9): 3.0,
                    }[move.path],
                )
                for move in moves
            ],
            key=lambda item: (item[1], item[0].path),
        ),
    )
    monkeypatch.setattr(
        resonance_utils_module,
        "_enumerate_one_step_resonance_moves",
        lambda omol, state_key, bond_index_map: [move_c, move_b, move_a] if omol is seed else [],
    )
    monkeypatch.setattr(
        resonance_utils_module,
        "_materialize_one_step_resonance",
        lambda omol, idxs: f"state-{idxs[0]}",
    )

    resonances = get_radical_resonances(seed, max_depth=1, traversal_policy=policy)

    assert resonances == [seed, "state-1", "state-4"]


def test_limited_discrepancy_search_prefers_lower_cost_duplicate_state(monkeypatch) -> None:
    seed = _make_seed("C=CC=C", (2,))
    shared_state_key = (((6, 0, 0, False),), ((1, 2, 1, False),))
    state_key_a = (((6, 0, 0, False),), ((1, 2, 2, False),))
    move_a = resonance_utils_module._IndexedResonanceTraversalMove(  # type: ignore[attr-defined]
        idxs=(1, 2, 3),
        next_state_key=state_key_a,
    )
    move_b = resonance_utils_module._IndexedResonanceTraversalMove(  # type: ignore[attr-defined]
        idxs=(4, 5, 6),
        next_state_key=shared_state_key,
    )
    child_move_a = resonance_utils_module._IndexedResonanceTraversalMove(  # type: ignore[attr-defined]
        idxs=(10, 11, 12),
        next_state_key=shared_state_key,
    )
    policy = resonance_module.make_limited_discrepancy_uff_lite_gain_traversal_policy(
        max_discrepancy=1
    )

    scores_by_state_and_move = {
        ("seed", (1, 2, 3)): 1.0,
        ("seed", (4, 5, 6)): 2.0,
        ("state-a", (10, 11, 12)): 1.0,
    }

    def fake_order_uff_lite_gain_moves(omol, moves):
        label = "seed" if omol is seed else omol
        ordered_moves = [(move, scores_by_state_and_move[(label, move.path)]) for move in moves]
        ordered_moves.sort(key=lambda item: (item[1], item[0].path))
        return ordered_moves

    def fake_enumerate(omol, state_key, bond_index_map):
        if omol is seed:
            return [move_b, move_a]
        if omol == "state-a":
            return [child_move_a]
        return []

    def fake_materialize(omol, idxs):
        if omol is seed and idxs == (1, 2, 3):
            return "state-a"
        return "shared-state"

    monkeypatch.setattr(
        resonance_utils_module,
        "_order_uff_lite_gain_moves",
        fake_order_uff_lite_gain_moves,
    )
    monkeypatch.setattr(
        resonance_utils_module, "_enumerate_one_step_resonance_moves", fake_enumerate
    )
    monkeypatch.setattr(resonance_utils_module, "_materialize_one_step_resonance", fake_materialize)

    resonances = get_radical_resonances(seed, max_depth=2, traversal_policy=policy)

    assert resonances[0] is seed
    assert resonances[1:] == ["state-a", "shared-state"]


def test_process_resonance_matches_equivalent_clone_states_without_cache() -> None:
    seed = _make_seed("C=CC=C", (2,))
    first_resonance, first_charge, _ = process_resonance(seed, 0)
    second_resonance, second_charge, _ = process_resonance(seed.clone, 0)

    assert first_charge == second_charge
    assert build_resonance_state_key(first_resonance) == build_resonance_state_key(second_resonance)


def test_omol_state_machine_caches_resonance_key_until_omol_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = _make_seed("C=CC=C", (2,))
    machine = OmolStateMachine(seed)
    calls = 0
    original_builder = resonance_module.build_resonance_state_key

    def tracking_builder(omol):
        nonlocal calls
        calls += 1
        return original_builder(omol)

    monkeypatch.setattr(resonance_module, "build_resonance_state_key", tracking_builder)

    first_key = machine.get_cached_omol_value(
        "resonance_state_key",
        resonance_module.build_resonance_state_key,
    )
    second_key = machine.get_cached_omol_value(
        "resonance_state_key",
        resonance_module.build_resonance_state_key,
    )
    assert first_key == second_key
    assert calls == 1

    machine.run_omol_stage("clone", lambda current_omol: (current_omol.clone, True))
    third_key = machine.get_cached_omol_value(
        "resonance_state_key",
        resonance_module.build_resonance_state_key,
    )
    assert third_key == first_key
    assert calls == 2


def test_omol_trace_recorder_follows_state_machine_branches() -> None:
    machine = OmolStateMachine(
        _make_seed("CC", ()),
        total_charge=-1,
        total_radical_electrons=2,
    )
    recorder = OmolTraceRecorder()

    with trace_omol_state_machine(recorder):
        machine.annotate("root")
        child = machine.branch("child")
        child.run_omol_stage("child_stage", lambda omol: (omol, False))

    assert [record["phase"] for record in recorder.records.values()] == [
        "root",
        "child",
        "child_stage",
    ]
    assert recorder.records[1]["parent_id"] == 0
    assert recorder.records[2]["parent_id"] == 1
    assert recorder.records[2]["total_charge"] == -1
    assert recorder.records[2]["total_radical_electrons"] == 2

    machine.annotate("outside_trace")
    assert len(recorder.records) == 3


def test_omol_state_machine_preserves_global_targets_across_branches() -> None:
    state = ReconstructionState(
        _make_seed("CC", ()),
        given_charge=1,
        total_charge=-2,
        total_radical_electrons=3,
    )

    machine = OmolStateMachine.from_reconstruction_state(state)
    child = machine.branch("child")
    frozen = child.freeze()

    assert machine.total_charge == child.total_charge == -2
    assert machine.total_radical_electrons == child.total_radical_electrons == 3
    assert frozen.total_charge == -2
    assert frozen.total_radical_electrons == 3


def test_search_resonance_candidates_returns_valid_candidates() -> None:
    seed = _make_seed("C=CC=C", (2,))
    state = ReconstructionState(
        seed,
        given_charge=0,
        total_charge=0,
        total_radical_electrons=1,
    )
    candidates = no_metal_resonance_module.search_resonance_candidates([state])
    assert candidates
    scores = [selection_force_field_energy(candidate.omol) for candidate in candidates]
    assert min(scores) == sorted(scores)[0]


def test_search_resonance_candidates_chains_full_normalization_and_validation() -> None:
    seed = _make_seed("C", ())
    state = ReconstructionState(
        seed,
        given_charge=0,
        total_charge=-1,
        total_radical_electrons=0,
    )
    calls: list[tuple[str, int]] = []

    def walk_root(omol, *, visit, **kwargs):
        del kwargs
        visit(
            resonance_utils_module.ResonanceSearchNode(
                omol,
                build_resonance_state_key(omol),
                0,
                0,
            )
        )

    def full_normalization(omol, given_charge):
        calls.append(("full", omol.OBMol.GetAtom(1).GetFormalCharge()))
        return omol, given_charge + 1, True

    def validate(omol, total_charge, total_radical_electrons):
        del total_charge, total_radical_electrons
        formal_charge = omol.OBMol.GetAtom(1).GetFormalCharge()
        calls.append(("validate", formal_charge))
        return formal_charge == 0

    recorder = OmolTraceRecorder()
    with trace_omol_state_machine(recorder):
        candidates = no_metal_resonance_module.search_resonance_candidates(
            [state],
            walk_stage=walk_root,
            full_normalization_stage=full_normalization,
            validation_stage=validate,
            score_stage=lambda candidate, **kwargs: 0.0,
            topology_annotation_stage=lambda candidate, **kwargs: None,
        )

    assert len(candidates) == 1
    assert calls == [("full", 0), ("validate", 0)]
    assert candidates[0].given_charge == 1
    assert candidates[0].metadata["resonance_normalization"] == "full_resonance_normalization"

    phases = [record["phase"] for record in recorder.records.values()]
    branch_id = phases.index("branch_resonance_candidate")
    full_id = phases.index("full_resonance_normalization")
    assert recorder.records[full_id]["parent_id"] == branch_id


def test_search_resonance_candidates_traces_every_process_resonance_step() -> None:
    seed = _make_seed("C", ())
    state = ReconstructionState(seed, given_charge=0, total_charge=0, total_radical_electrons=0)

    def walk_root(omol, *, visit, **kwargs):
        del kwargs
        visit(
            resonance_utils_module.ResonanceSearchNode(
                omol,
                build_resonance_state_key(omol),
                0,
                0,
            )
        )

    recorder = OmolTraceRecorder()
    with trace_omol_state_machine(recorder):
        no_metal_resonance_module.search_resonance_candidates(
            [state],
            walk_stage=walk_root,
            validation_stage=lambda *args: True,
            score_stage=lambda candidate, **kwargs: 0.0,
            topology_annotation_stage=lambda candidate, **kwargs: None,
        )

    process_phases = [
        "process_resonance_eliminate_1_3_dipole_postive",
        "process_resonance_eliminate_possible_cp_like_radical_anion",
        "process_resonance_clean_possible_1_3_dipole",
        "process_resonance_clean_1_4_radicals",
        "process_resonance_clean_neighbor_radicals",
        "process_resonance_clean_1_6_radicals",
        "process_resonance_eliminate_positive_charges_1",
        "process_resonance_eliminate_negative_charges",
        "process_resonance_eliminate_positive_charges_2",
        "process_resonance_clean_resonances",
    ]
    phases = [record["phase"] for record in recorder.records.values()]
    branch_id = phases.index("branch_resonance_candidate")
    process_ids = [phases.index(phase) for phase in process_phases]
    full_id = phases.index("full_resonance_normalization")

    assert recorder.records[process_ids[0]]["parent_id"] == branch_id
    assert all(
        recorder.records[current]["parent_id"] == previous
        for previous, current in zip(process_ids, process_ids[1:])
    )
    assert recorder.records[full_id]["parent_id"] == process_ids[-1]


def test_search_resonance_candidates_without_shared_region_scoring() -> None:
    seed = _make_seed("C=CC=C", (2,))
    state = ReconstructionState(
        seed,
        given_charge=0,
        total_charge=0,
        total_radical_electrons=1,
    )
    candidates = no_metal_resonance_module.search_resonance_candidates([state])

    assert candidates


def test_search_resonance_candidates_dedups_processed_states(monkeypatch) -> None:
    seed = _make_seed("C=CC=C", (2,))
    state = ReconstructionState(
        seed,
        given_charge=0,
        total_charge=0,
        total_radical_electrons=1,
    )

    resonance_a = seed.clone
    resonance_b = seed.clone
    validate_calls = 0
    force_field_score_calls = 0

    def walk_resonances(omol, *, visit, **kwargs):
        del omol, kwargs
        for resonance in (resonance_a, resonance_b):
            visit(
                resonance_utils_module.ResonanceSearchNode(
                    resonance,
                    (id(resonance),),
                    0,
                    0,
                )
            )

    monkeypatch.setattr(
        no_metal_resonance_module,
        "walk_radical_resonances",
        walk_resonances,
    )
    monkeypatch.setattr(
        resonance_utils_module,
        "build_resonance_state_key",
        lambda omol: (id(omol),),
    )
    monkeypatch.setattr(
        resonance_utils_module,
        "build_processed_resonance_key",
        lambda omol: "same-processed-state",
    )

    def validate(*args, **kwargs):
        nonlocal validate_calls
        validate_calls += 1
        return True

    def full_score(self):
        nonlocal force_field_score_calls
        force_field_score_calls += 1
        return 1.0

    monkeypatch.setattr(no_metal_resonance_module, "validate_omol", validate)
    monkeypatch.setattr(ReconstructionState, "full_score", full_score)

    candidates = no_metal_resonance_module.search_resonance_candidates([state])

    assert len(candidates) == 1
    assert validate_calls == 1
    assert force_field_score_calls == 1


def test_search_resonance_candidates_forwards_traversal_policy(monkeypatch) -> None:
    seed = _make_seed("C=CC=C", (2,))
    state = ReconstructionState(
        seed,
        given_charge=0,
        total_charge=0,
        total_radical_electrons=1,
    )
    policy = object()
    recorded_policy = None

    def fake_walk_radical_resonances(
        omol,
        *,
        traversal_policy=None,
        max_depth=2,
        visit=None,
    ):
        del omol, max_depth, visit
        nonlocal recorded_policy
        recorded_policy = traversal_policy

    monkeypatch.setattr(
        no_metal_resonance_module,
        "walk_radical_resonances",
        fake_walk_radical_resonances,
    )

    candidates = no_metal_resonance_module.search_resonance_candidates(
        [state],
        resonance_traversal_policy=policy,
    )

    assert candidates == []
    assert recorded_policy is policy


def test_resonance_pipeline_no_longer_exposes_incumbent_bound_helpers() -> None:
    from molgr.fallback.pipeline import reconstruct_without_metals as no_metal_module

    assert not hasattr(
        resonance_module,
        "estimate_remaining_resonance_score_improvement_upper_bound",
    )
    assert not hasattr(no_metal_module, "_recover_resonance_candidates_with_incumbent_bound")
    assert not hasattr(no_metal_module, "_recover_resonance_candidates")
