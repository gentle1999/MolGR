"""Search completeness and early-exit regressions for the evaluator."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from rdkit import Chem

from molgr.utils import _legacy_equivalence as eq


FLAGS = Chem.ResonanceFlags.UNCONSTRAINED_CATIONS | Chem.ResonanceFlags.UNCONSTRAINED_ANIONS


def prepared_key(smiles):
    source = eq._prepare_resonance_source(Chem.MolFromSmiles(smiles), use_chirality=False)
    return eq._resonance_form_smiles(source, False, stereo_prepared=True)


@pytest.fixture(autouse=True)
def clear_resonance_cache():
    eq._cached_resonance_smiles.cache_clear()
    yield
    eq._cached_resonance_smiles.cache_clear()


def test_cached_bfs_expands_newly_discovered_states(monkeypatch):
    # RDKit enumeration must not hide a broken custom two-step closure.
    monkeypatch.setattr(eq, "ResonanceMolSupplier", lambda *args, **kwargs: ())
    target = prepared_key("[NH-]C=CC=O")
    expanded = []
    original = eq._iter_charge_shift_resonance_forms

    def tracked(form, **kwargs):
        expanded.append(eq._resonance_form_smiles(form, False))
        yield from original(form, **kwargs)

    monkeypatch.setattr(eq, "_iter_charge_shift_resonance_forms", tracked)
    result = eq._cached_resonance_smiles(
        "[O-]C=CC=N",
        use_chirality=False,
        max_resonance=100,
        resonance_flags=int(FLAGS),
    )

    assert target in result
    assert len(expanded) == len(set(expanded))
    assert not result.limit_reached


def test_duplicate_supplier_forms_do_not_hide_truncation(monkeypatch):
    monkeypatch.setattr(
        eq,
        "ResonanceMolSupplier",
        lambda *args, **kwargs: (Chem.MolFromSmiles("CC") for _ in range(3)),
    )
    result = eq._cached_resonance_smiles(
        "CC",
        use_chirality=False,
        max_resonance=3,
        resonance_flags=int(FLAGS),
    )

    assert len(result) == 1
    assert result.supplier_limit_reached
    assert result.limit_reached


def test_early_hit_does_not_consume_remaining_neighbors(monkeypatch):
    target = prepared_key("O=C[CH-]C=N")
    original = eq._iter_charge_shift_resonance_forms

    def only_first_neighbor(form, **kwargs):
        yield next(original(form, **kwargs))
        raise AssertionError("Search continued after the target witness")

    monkeypatch.setattr(eq, "_iter_charge_shift_resonance_forms", only_first_neighbor)
    result = eq._local_charge_shift_search(
        Chem.MolFromSmiles("[O-]C=CC=N"),
        use_chirality=False,
        max_resonance=100,
        targets=frozenset((target,)),
    )

    assert result.target_found
    assert len(result) == 2


def test_partial_target_cache_does_not_replace_full_closure(monkeypatch):
    monkeypatch.setattr(eq, "ResonanceMolSupplier", lambda *args, **kwargs: ())
    source = "[O-]C=CC=N"
    source_key = prepared_key(source)
    kwargs = {"use_chirality": False, "max_resonance": 100, "resonance_flags": int(FLAGS)}

    partial = eq._cached_resonance_smiles(source, target_smiles=source_key, **kwargs)
    full = eq._cached_resonance_smiles(source, **kwargs)

    assert partial.target_found
    assert len(partial) == 1
    assert len(full) > 1
    assert not full.target_found
    assert not full.limit_reached


def test_raw_search_limit_survives_later_unbounded_miss(monkeypatch):
    monkeypatch.setattr(
        eq,
        "_bidirectional_charge_shift_search",
        lambda *args, **kwargs: (eq._ResonanceForms(local_limit_reached=True),) * 2,
    )
    monkeypatch.setattr(
        eq,
        "_paired_resonance_search",
        lambda *args, **kwargs: (eq._ResonanceForms(),) * 2,
    )
    result = eq._resonance_match(
        Chem.MolFromSmiles("[O-]C=CC=N"),
        Chem.MolFromSmiles("[NH-]C=CC=O"),
        use_chirality=False,
        max_resonance=1,
        resonance_flags=FLAGS,
    )

    assert not result.matched
    assert result.local_limit_reached
    assert not result.exhaustive
    assert result.stop_reason == "limit_reached"


def test_successful_early_search_is_not_reported_as_exhaustive():
    result = eq.evaluate_equivalence(
        Chem.MolFromSmiles("[O-]C=CC=N"),
        Chem.MolFromSmiles("[NH-]C=CC=O"),
        use_chirality=False,
        max_resonance=100,
    )

    assert result.decision == eq.EquivalenceDecision.EQUIVALENT
    assert result.bounded_search.attempted
    assert result.bounded_search.stop_reason == "target_found"
    assert result.bounded_search.exhaustive is False


def test_compatible_first_mapping_avoids_explicit_hydrogen_permutations(monkeypatch):
    mol = Chem.MolFromSmiles("CCCCCCCCCC")
    for atom in mol.GetAtoms():
        atom.SetIntProp("MOLGR_LONE_PAIR_COUNT", 0)

    def unexpected_alternatives(*args, **kwargs):
        raise AssertionError("Compatible first mapping should finish the search")

    monkeypatch.setattr(Chem.Mol, "GetSubstructMatches", unexpected_alternatives)
    result = eq.evaluate_equivalence(mol, Chem.Mol(mol), use_chirality=False)

    assert result.decision == eq.EquivalenceDecision.EQUIVALENT
    coverage = result.invariants["electronic_properties"].mol1
    assert coverage["mappings_examined"] == 1
    assert coverage["mapping_limit_reached"] is False


def test_first_mapping_conflict_still_checks_symmetric_alternative():
    left, right = (Chem.MolFromSmiles("[Cu+2].[Cu+2]") for _ in range(2))
    for mol, spins in ((left, (0, 1)), (right, (1, 0))):
        for atom, spin in zip(mol.GetAtoms(), spins):
            atom.SetIntProp("MOLGR_METAL_UNPAIRED_ELECTRONS", spin)

    result = eq.evaluate_equivalence(left, right, use_chirality=False)

    assert result.decision == eq.EquivalenceDecision.EQUIVALENT
    assert result.invariants["electronic_properties"].mol1["mappings_examined"] == 2


def test_failed_initial_sanitization_keeps_existing_retry(monkeypatch):
    original = eq._sanitize_if_possible
    attempts = []

    def fail_once(mol, **kwargs):
        attempts.append(mol)
        if len(attempts) == 1:
            return False
        return original(mol, **kwargs)

    monkeypatch.setattr(eq, "_sanitize_if_possible", fail_once)
    result = eq._standardize_metal_bonds(Chem.MolFromSmiles("CC"))

    assert len(attempts) == 2
    assert eq._canon_smiles(result, False) == eq._canon_smiles(
        Chem.AddHs(Chem.MolFromSmiles("CC")), False
    )


@pytest.mark.parametrize("annotated", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_coordination_and_electronic_proof_require_one_shared_mapping(annotated, reverse):
    left = Chem.MolFromSmiles("[Cu+]<-[O]CC[O-]")
    right = Chem.MolFromSmiles("[O]CC[O-]->[Cu+]")
    if annotated:
        for mol in (left, right):
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 29:
                    atom.SetIntProp("MOLGR_METAL_UNPAIRED_ELECTRONS", 0)
    if reverse:
        left, right = right, left

    assert eq.evaluate_equivalence(left, right, use_chirality=False).equivalent
    strict = eq.evaluate_equivalence(
        left,
        right,
        use_chirality=False,
        ignore_metal_coordination_bonds=False,
    )

    # Topology permits swapping O endpoints; localized charge/radical identity
    # requires keeping them fixed. Neither mapping proves both constraints.
    assert strict.invariants["metal_coordination"].status == eq.InvariantStatus.PASSED
    assert strict.decision == eq.EquivalenceDecision.INCONCLUSIVE
    assert strict.invariants["electronic_properties"].mol1["mappings_examined"] > 0


def test_fixed_mapping_identity_needs_no_canonical_roundtrip(monkeypatch):
    left = Chem.AddHs(Chem.MolFromSmiles("[13CH3]CO"))
    order = list(reversed(range(left.GetNumAtoms())))
    right = Chem.RenumberAtoms(left, order)

    def unexpected_canonicalization(*args, **kwargs):
        raise AssertionError("Fixed mapped identity is already a graph proof")

    monkeypatch.setattr(eq, "_canon_smiles", unexpected_canonicalization)
    assert eq._mapped_electronic_structure_match(
        left,
        right,
        tuple(order),
        use_chirality=False,
        max_resonance=100,
        resonance_flags=FLAGS,
    ) == (True, False)


@pytest.mark.parametrize(
    ("first", "second", "chirality"),
    [
        ("[13CH4]", "C", False),
        ("[CH3]", "[CH3-]", False),
        ("N->[Cu]", "N<-[Cu]", False),
        ("F[C@](Cl)(Br)I", "F[C@@](Cl)(Br)I", True),
    ],
)
def test_fixed_mapping_identity_retains_isotope_electron_direction_and_stereo_constraints(
    first,
    second,
    chirality,
):
    left, right = Chem.MolFromSmiles(first), Chem.MolFromSmiles(second)
    assert left is not None and right is not None
    assert not eq._mapped_graph_identity(
        left,
        right,
        tuple(range(left.GetNumAtoms())),
        use_chirality=chirality,
    )


def test_same_proposal_from_two_paths_reuses_validated_molecule(monkeypatch):
    source = eq._prepare_resonance_source(
        Chem.MolFromSmiles("[O-]C=C.[O-]C=C"), use_chirality=False
    )
    context = eq._ChargeShiftContext(source, use_chirality=False)
    sanitize = eq._sanitize_if_possible
    calls = []

    def tracked(mol, **kwargs):
        calls.append(mol)
        return sanitize(mol, **kwargs)

    monkeypatch.setattr(eq, "_sanitize_if_possible", tracked)
    children = tuple(eq._iter_charge_shift_resonance_forms(source, context=context))
    assert len(children) == 2
    grandchildren = [
        tuple(eq._iter_charge_shift_resonance_forms(child, context=context)) for child in children
    ]

    def both_shifted(mol):
        return sum(a.GetFormalCharge() == -1 and a.GetAtomicNum() == 6 for a in mol.GetAtoms()) == 2

    first = next(m for m in grandchildren[0] if both_shifted(m))
    second = next(m for m in grandchildren[1] if both_shifted(m))
    assert first is second
    assert len(calls) == 4  # Six paths reach only four distinct proposals.
    assert context.key(first) == context.key(second)


def test_partial_neighbor_iteration_is_not_cached_as_complete():
    source = eq._prepare_resonance_source(
        Chem.MolFromSmiles("[O-]C=C.[O-]C=C"), use_chirality=False
    )
    context = eq._ChargeShiftContext(source, use_chirality=False)
    partial = eq._iter_charge_shift_resonance_forms(source, context=context)
    first = next(partial)
    partial.close()
    assert source not in context.finished
    complete = tuple(eq._iter_charge_shift_resonance_forms(source, context=context))
    assert len(complete) == 2
    assert first is complete[0]


def test_completed_expansion_is_reused_across_search_stages(monkeypatch):
    source = eq._prepare_resonance_source(Chem.MolFromSmiles("[O-]C=CC=N"), use_chirality=False)
    context = eq._ChargeShiftContext(source, use_chirality=False)
    first = tuple(eq._iter_charge_shift_resonance_forms(source, context=context))
    clone = Chem.Mol(source)

    def unexpected_validation(*args, **kwargs):
        raise AssertionError("Identical state was already fully expanded")

    monkeypatch.setattr(eq, "_sanitize_if_possible", unexpected_validation)
    second = tuple(eq._iter_charge_shift_resonance_forms(clone, context=context))
    assert second == first
    assert all(a is b for a, b in zip(first, second))


def test_invalid_migration_is_not_a_witness_or_a_complete_negative(monkeypatch):
    left, right = Chem.MolFromSmiles("[O-]C=CC=N"), Chem.MolFromSmiles("[NH-]C=CC=O")
    sanitize = eq._sanitize_if_possible

    def reject_local_candidate(mol, **kwargs):
        if "annotated_indices" in kwargs:
            return False
        return sanitize(mol, **kwargs)

    monkeypatch.setattr(eq, "_sanitize_if_possible", reject_local_candidate)
    monkeypatch.setattr(eq, "ResonanceMolSupplier", lambda *args, **kwargs: ())
    result = eq.evaluate_equivalence(left, right, use_chirality=False)
    assert result.decision == eq.EquivalenceDecision.INCONCLUSIVE
    assert result.bounded_search.validation_failed
    assert result.bounded_search.exhaustive is False
    assert result.bounded_search.stop_reason == "validation_failed"


def test_valid_witness_survives_an_independent_invalid_candidate(monkeypatch):
    source = eq._prepare_resonance_source(
        Chem.MolFromSmiles("[O-]C=C.[O-]C=C"), use_chirality=False
    )
    target = eq._resonance_form_smiles(next(eq._iter_charge_shift_resonance_forms(source)), False)
    sanitize = eq._sanitize_if_possible
    count = 0

    def fail_first(mol, **kwargs):
        nonlocal count
        count += 1
        return False if count == 1 else sanitize(mol, **kwargs)

    monkeypatch.setattr(eq, "_sanitize_if_possible", fail_first)
    result = eq._local_charge_shift_search(
        source, use_chirality=False, max_resonance=100, targets=frozenset((target,))
    )
    assert result.target_found
    assert result.validation_failed


def test_resonance_search_uses_live_graphs_without_smiles_parsing(monkeypatch):
    left = Chem.MolFromSmiles("CC1=[N+]=C(OC=[N-])[N]N1")
    right = Chem.MolFromSmiles("Cc1nc(OC=[N])n[nH]1")
    supplier = eq.ResonanceMolSupplier
    calls = []

    def tracked(*args, **kwargs):
        calls.append(1)
        return supplier(*args, **kwargs)

    def unexpected_parse(*args, **kwargs):
        raise AssertionError("The search already has both molecular graphs")

    monkeypatch.setattr(eq, "ResonanceMolSupplier", tracked)
    monkeypatch.setattr(Chem, "MolFromSmiles", unexpected_parse)
    result = eq._resonance_match(
        left, right, use_chirality=False, max_resonance=100, resonance_flags=FLAGS
    )
    assert result.matched
    assert calls


def test_paired_supplier_search_stops_at_first_common_witness(monkeypatch):
    left, right = Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("COC")
    calls = []

    def supplier(source, **kwargs):
        calls.append(1)
        yield Chem.MolFromSmiles("C")
        raise AssertionError("Supplier consumed past the shared witness")

    monkeypatch.setattr(eq, "ResonanceMolSupplier", supplier)
    contexts = tuple(eq._ChargeShiftContext(m, use_chirality=False) for m in (left, right))
    a, b = eq._paired_resonance_search(
        (left, right),
        contexts=contexts,
        use_chirality=False,
        max_resonance=100,
        resonance_flags=int(FLAGS),
    )
    assert set(a) & set(b)
    assert a.target_found and b.target_found
    assert len(calls) == 2


def test_cache_eviction_does_not_drop_neighbors_or_become_a_search_limit():
    source = eq._prepare_resonance_source(
        Chem.MolFromSmiles("[O-]C=C.[O-]C=C"), use_chirality=False
    )
    context = eq._ChargeShiftContext(source, use_chirality=False, cache_limit=1)
    expected = tuple(eq._iter_charge_shift_resonance_forms(source))
    for _ in range(2):
        actual = tuple(eq._iter_charge_shift_resonance_forms(source, context=context))
        assert len(actual) == len(expected) == 2
        assert {eq._resonance_form_smiles(m, False) for m in actual} == {
            eq._resonance_form_smiles(m, False) for m in expected
        }
        assert all(
            len(cache) <= 1
            for cache in (
                context.states,
                context.proposals,
                context.keys,
                context.finished,
                context.transitions,
            )
        )
        assert context.transition_edges <= 1
        assert not context.validation_failed


def test_failed_validation_is_retryable_without_clearing_helper_cache(monkeypatch):
    monkeypatch.setattr(eq, "ResonanceMolSupplier", lambda *args, **kwargs: ())
    kwargs = {"use_chirality": False, "max_resonance": 100, "resonance_flags": int(FLAGS)}
    with monkeypatch.context() as failed:
        failed.setattr(eq, "_sanitize_if_possible", lambda *args, **kwargs: False)
        with pytest.raises(ValueError, match="validation failed"):
            eq._cached_resonance_smiles("[O-]C=CC=N", **kwargs)
    recovered = eq._cached_resonance_smiles("[O-]C=CC=N", **kwargs)
    assert not recovered.validation_failed
    assert prepared_key("[NH-]C=CC=O") in recovered


def run_frontier_graph(monkeypatch, edges, *, budget=100, roots=("A", "B")):
    """Exercise scheduling/completeness independently of RDKit enumeration."""
    expanded = []

    def neighbors(state, **kwargs):
        expanded.append(state)
        yield from edges.get(state, ())

    monkeypatch.setattr(eq, "_iter_charge_shift_resonance_forms", neighbors)
    contexts = tuple(
        SimpleNamespace(key=lambda state: state, validation_failed=False) for _ in roots
    )
    results = eq._bidirectional_charge_shift_search(
        tuple([(root, root)] for root in roots), contexts=contexts, local_budget=budget
    )
    return results, expanded


def test_smaller_bfs_frontier_finds_witness_without_expanding_wide_branch(monkeypatch):
    edges = {"A": [f"L{i}" for i in range(6)], "B": ["R1"], "R1": ["R2"], "R2": ["L3"]}
    (left, right), expanded = run_frontier_graph(monkeypatch, edges)

    assert expanded == ["A", "B", "R1", "R2"]
    assert left.target_found and right.target_found
    assert left.witness_path == ("A", "L3")
    assert right.witness_path == ("B", "R1", "R2", "L3")
    for path in (left.witness_path, right.witness_path):
        assert all(b in edges[a] for a, b in zip(path, path[1:]))


def test_empty_frontier_does_not_end_asymmetric_common_witness_search(monkeypatch):
    (left, right), expanded = run_frontier_graph(monkeypatch, {"B": ["C"], "C": ["D"], "D": ["A"]})

    assert expanded == ["A", "B", "C", "D"]
    assert left.target_found and right.target_found
    assert left.witness_path == ("A",)
    assert right.witness_path == ("B", "C", "D", "A")


def test_capped_frontier_remains_available_as_other_sides_target(monkeypatch):
    (left, right), expanded = run_frontier_graph(
        monkeypatch, {"A": ["P", "Q"], "B": ["P"]}, budget=1
    )

    assert expanded == ["A", "B"]
    assert left.local_limit_reached
    assert left.target_found and right.target_found
    assert left.witness_path == ("A", "P")
    assert right.witness_path == ("B", "P")


def test_cycle_and_exhausted_disjoint_frontiers_are_visited_once(monkeypatch):
    (left, right), expanded = run_frontier_graph(
        monkeypatch, {"A": ["C", "C"], "C": ["A"], "B": ["D"], "D": ["B"]}
    )

    assert len(expanded) == len(set(expanded)) == 4
    assert not left.target_found and not right.target_found
    assert not left.limit_reached and not right.limit_reached
    assert not left.witness_path and not right.witness_path


def test_bounded_miss_cannot_claim_exhaustive_closures(monkeypatch):
    (left, right), _ = run_frontier_graph(
        monkeypatch, {"A": ["C"], "C": ["E"], "B": ["D"], "D": ["E"]}, budget=1
    )

    assert not left.target_found and not right.target_found
    assert left.local_limit_reached and right.local_limit_reached


def test_common_witness_stops_mid_neighbor_generator(monkeypatch):
    contexts = tuple(
        SimpleNamespace(key=lambda state: state, validation_failed=False) for _ in range(2)
    )

    def neighbors(state, **kwargs):
        yield "B"
        raise AssertionError("Consumed another neighbor after finding the target")

    monkeypatch.setattr(eq, "_iter_charge_shift_resonance_forms", neighbors)
    left, right = eq._bidirectional_charge_shift_search(
        ([("A", "A")], [("B", "B")]), contexts=contexts, local_budget=1
    )
    assert left.witness_path == ("A", "B")
    assert right.witness_path == ("B",)


@pytest.mark.parametrize("reverse", [False, True])
def test_bidirectional_chemical_witness_paths_replay_validated_moves(monkeypatch, reverse):
    mols = [Chem.MolFromSmiles(s) for s in ("[O-]C=CC=CC=N", "O=CC=CC=C[NH-]")]
    if reverse:
        mols.reverse()
    # Deliberately scramble one side's atom indices. A raw state vector cannot
    # establish the meeting; isotope/map-aware canonical keys must still agree.
    mols[1] = Chem.RenumberAtoms(mols[1], list(reversed(range(mols[1].GetNumAtoms()))))
    sources = [eq._prepare_resonance_source(m, use_chirality=False) for m in mols]
    contexts = tuple(eq._ChargeShiftContext(m, use_chirality=False) for m in sources)
    original = eq._iter_charge_shift_resonance_forms
    actual_edges = set()

    def tracked(form, *, context, **kwargs):
        for child in original(form, context=context, **kwargs):
            actual_edges.add((context.key(form), context.key(child)))
            yield child

    monkeypatch.setattr(eq, "_iter_charge_shift_resonance_forms", tracked)
    left, right = eq._bidirectional_charge_shift_search(
        tuple([(m, c.key(m))] for m, c in zip(sources, contexts)),
        contexts=contexts,
        local_budget=100,
    )
    assert left.target_found and right.target_found
    assert left.witness_path[-1] == right.witness_path[-1]
    assert sum(len(path) - 1 for path in (left.witness_path, right.witness_path)) == 3
    for path in (left.witness_path, right.witness_path):
        assert all(edge in actual_edges for edge in zip(path, path[1:]))


def test_supplier_roots_keep_local_paths_and_truncation_metadata(monkeypatch):
    edges = {"X": ["M"], "Y": ["N"], "N": ["M"]}
    monkeypatch.setattr(
        eq, "_iter_charge_shift_resonance_forms", lambda state, **kwargs: iter(edges.get(state, ()))
    )
    contexts = tuple(
        SimpleNamespace(key=lambda state: state, validation_failed=False) for _ in range(2)
    )
    left, right = eq._bidirectional_charge_shift_search(
        ([("A", "A"), ("X", "X")], [("B", "B"), ("Y", "Y")]),
        contexts=contexts,
        local_budget=10,
        supplier_limited=(True, False),
    )

    assert left.witness_path == ("X", "M")
    assert right.witness_path == ("Y", "N", "M")
    assert left.supplier_limit_reached and not right.supplier_limit_reached
    assert left.target_found and right.target_found


def test_exact_local_budget_can_exhaust_without_omitting_any_state(monkeypatch):
    (left, right), _ = run_frontier_graph(monkeypatch, {"A": ["C"], "B": ["D"]}, budget=1)
    assert not left.target_found and not right.target_found
    assert not left.local_limit_reached and not right.local_limit_reached


def test_duplicate_supplier_seed_does_not_replace_prepared_source(monkeypatch):
    source, duplicate, other = (Chem.MolFromSmiles("C") for _ in range(3))
    expanded = []

    def neighbors(mol, **kwargs):
        expanded.append(mol)
        return iter(())

    monkeypatch.setattr(eq, "_iter_charge_shift_resonance_forms", neighbors)
    contexts = tuple(
        SimpleNamespace(key=lambda state: state, validation_failed=False) for _ in range(2)
    )
    eq._bidirectional_charge_shift_search(
        ([(source, "A"), (duplicate, "A")], [(other, "B")]),
        contexts=contexts,
        local_budget=10,
    )
    assert expanded == [source, other]


def test_direct_local_edge_avoids_preparing_second_search_end(monkeypatch):
    left = Chem.AddHs(Chem.MolFromSmiles("[O-]C=C"))
    right = Chem.AddHs(Chem.MolFromSmiles("O=C[CH2-]"))
    prepare = eq._prepare_resonance_source
    calls = []

    def tracked(mol, **kwargs):
        calls.append(mol)
        return prepare(mol, **kwargs)

    monkeypatch.setattr(eq, "_prepare_resonance_source", tracked)
    result = eq._resonance_match(
        left, right, use_chirality=False, max_resonance=100, resonance_flags=FLAGS
    )
    assert result.matched
    assert len(calls) == 1
    assert result.mol2_count == 0
    assert len(result.local_witness_paths[0]) == 2
    assert result.local_witness_paths[0][-1] == result.hit_smiles
