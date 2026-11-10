"""Factored charge migration must construct a validated whole-graph proof."""

from __future__ import annotations

import pytest
from rdkit import Chem

from molgr.utils import _legacy_equivalence as eq


FLAGS = Chem.ResonanceFlags.UNCONSTRAINED_CATIONS | Chem.ResonanceFlags.UNCONSTRAINED_ANIONS


def independent_sites(count):
    left = ".".join(f"[O-:{3 * i + 1}][CH:{3 * i + 2}]=[CH2:{3 * i + 3}]" for i in range(count))
    right = ".".join(f"[O:{3 * i + 1}]=[CH:{3 * i + 2}][CH2-:{3 * i + 3}]" for i in range(count))
    return Chem.MolFromSmiles(left), Chem.MolFromSmiles(right)


def connected_sites():
    return tuple(
        Chem.MolFromSmiles(s)
        for s in (
            "[O-:1][CH:2]=[CH:3][CH2:4][CH:5]=[CH:6][O-:7]",
            "[O:1]=[CH:2][CH-:3][CH2:4][CH-:5][CH:6]=[O:7]",
        )
    )


def witness(left, right, budget=100, use_chirality=False):
    return eq._regional_charge_shift_witness(
        left, right, use_chirality=use_chirality, max_resonance=budget
    )


@pytest.mark.parametrize("hops", [1, 2])
def test_unbranched_migration_needs_no_regional_alignment(monkeypatch, hops):
    left = Chem.MolFromSmiles("[O-:1][CH:2]=[CH:3][CH:4]=[NH:5]")
    right = Chem.MolFromSmiles(
        "[O:1]=[CH:2][CH-:3][CH:4]=[NH:5]" if hops == 1 else "[O:1]=[CH:2][CH:3]=[CH:4][NH-:5]"
    )
    calls = []

    def unexpected(*args, **kwargs):
        calls.append(args)
        raise AssertionError("A simple chain must not pay for regional alignment")

    monkeypatch.setattr(eq, "_regional_charge_shift_witness", unexpected)
    result = eq._resonance_match(
        left, right, use_chirality=False, max_resonance=100, resonance_flags=FLAGS
    )
    assert result.matched
    assert result.regional_witness is None
    assert not calls


@pytest.mark.parametrize("permuted", [False, True])
def test_existing_correspondence_needs_no_topology_copy(monkeypatch, permuted):
    left, right = (Chem.AddHs(m) for m in connected_sites())
    if permuted:
        right = Chem.RenumberAtoms(right, list(reversed(range(right.GetNumAtoms()))))

    def unexpected(*args, **kwargs):
        raise AssertionError("Verified indices/maps need no topology copy or isomorphism search")

    monkeypatch.setattr(Chem, "RWMol", unexpected)
    mapping = eq._charge_shift_atom_mapping(left, right)
    assert len(mapping) == left.GetNumAtoms()
    assert all(
        left.GetAtomWithIdx(i).GetAtomMapNum() == right.GetAtomWithIdx(j).GetAtomMapNum()
        for i, j in enumerate(mapping)
    )
    assert all(
        right.GetBondBetweenAtoms(mapping[b.GetBeginAtomIdx()], mapping[b.GetEndAtomIdx()])
        is not None
        for b in left.GetBonds()
    )


def test_equal_index_labels_do_not_imply_equal_topology():
    # Six CH2 atoms and six bonds on either side, but incompatible connectivity.
    assert not eq._charge_shift_atom_mapping(
        Chem.MolFromSmiles("C1CCCCC1"), Chem.MolFromSmiles("C1CC1.C1CC1")
    )


def test_independent_sites_do_not_enumerate_their_cartesian_product(monkeypatch):
    left, right = independent_sites(8)
    calls = []
    sanitize = eq._sanitize_if_possible

    def tracked(mol, **kwargs):
        calls.append(mol)
        return sanitize(mol, **kwargs)

    def unexpected(*args, **kwargs):
        raise AssertionError("Regional expansion needs neither canonicalization nor SMILES parsing")

    monkeypatch.setattr(eq, "_sanitize_if_possible", tracked)
    monkeypatch.setattr(Chem, "MolFromSmiles", unexpected)
    monkeypatch.setattr(Chem, "MolToSmiles", unexpected)
    result = witness(left, right, budget=9)

    assert result is not None
    assert result.region_count == 8
    assert result.state_count == 9
    assert len(result.steps) == len(calls) == 8


@pytest.mark.parametrize("reverse", [False, True])
def test_regions_inside_one_connected_molecule_and_permuted_atoms(reverse):
    left, right = connected_sites()
    if reverse:
        left, right = right, left
    right = Chem.RenumberAtoms(right, list(reversed(range(right.GetNumAtoms()))))
    assert len(Chem.GetMolFrags(left)) == 1
    result = witness(left, right)
    assert result is not None
    assert result.region_count == 2
    assert result.state_count == 3
    assert len(result.steps) == 2
    assert all(3 not in step for step in result.steps)  # Saturated bridge is frozen.


def test_unchanged_spectator_region_is_not_expanded():
    left, right = connected_sites()
    spectator = Chem.MolFromSmiles("[O-:8][CH:9]=[CH2:10]")
    result = witness(Chem.CombineMols(left, spectator), Chem.CombineMols(right, spectator))
    assert result is not None
    assert result.region_count == 2
    assert len(result.steps) == 2
    assert all(max(step) < left.GetNumAtoms() for step in result.steps)


def test_overlapping_moves_form_one_region_and_record_every_step():
    left = Chem.MolFromSmiles("[O-:1][CH:2]=[CH:3][CH:4]=[NH:5]")
    right = Chem.MolFromSmiles("[O:1]=[CH:2][CH:3]=[CH:4][NH-:5]")
    result = witness(left, right)
    assert result is not None
    assert result.region_count == 1
    assert result.steps == ((0, 1, 2), (2, 3, 4))
    assert result.distance_pruned > 0


def test_molecular_shortest_distance_is_not_a_resonance_step_upper_bound():
    source = Chem.MolFromSmiles("[CH-]1C=CC=CC=CC1")
    for atom in source.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    source = eq._prepare_resonance_source(source, use_chirality=False)
    target = source
    for terminal in (2, 4, 6):
        target = next(
            m
            for m in eq._iter_charge_shift_resonance_forms(target)
            if m.GetAtomWithIdx(terminal).GetFormalCharge() == -1
        )
    # The short two-bond route crosses a saturated carbon and cannot transfer
    # the anion. All six changed bonds on the longer arc must remain searchable.
    assert Chem.GetDistanceMatrix(source)[0, 6] == 2
    result = witness(source, target)
    assert result is not None
    assert len(result.steps) == 3
    assert result.steps == ((0, 1, 2), (2, 3, 4), (4, 5, 6))


def test_region_budget_is_shared_and_not_multiplied_by_region_count():
    left, right = independent_sites(8)
    assert witness(left, right, budget=8) is None
    assert witness(left, right, budget=9).state_count == 9


@pytest.mark.parametrize("failure", ["miss", "exception"])
def test_restricted_search_failure_falls_back_to_general_search(monkeypatch, failure):
    left, right = connected_sites()

    def fail(*args, **kwargs):
        if failure == "exception":
            raise ValueError("temporary projection failure")
        return None

    monkeypatch.setattr(eq, "_search_charge_shift_region", fail)
    result = eq._resonance_match(
        left, right, use_chirality=False, max_resonance=100, resonance_flags=FLAGS
    )
    assert result.matched
    assert result.regional_witness is None


def test_region_timeout_still_propagates(monkeypatch):
    def fail(*args, **kwargs):
        raise TimeoutError("regional timeout")

    monkeypatch.setattr(eq, "_search_charge_shift_region", fail)
    with pytest.raises(TimeoutError, match="regional timeout"):
        eq._resonance_match(
            *connected_sites(), use_chirality=False, max_resonance=100, resonance_flags=FLAGS
        )


def test_invalid_abstract_path_cannot_become_a_positive_proof(monkeypatch):
    left, right = connected_sites()
    monkeypatch.setattr(eq, "_sanitize_if_possible", lambda *args, **kwargs: False)
    assert witness(left, right) is None


def test_sanitizer_change_outside_region_invalidates_factorization(monkeypatch):
    left, right = connected_sites()
    sanitize = eq._sanitize_if_possible

    def alter_boundary(mol, **kwargs):
        valid = sanitize(mol, **kwargs)
        mol.GetAtomWithIdx(3).SetFormalCharge(1)
        return valid

    monkeypatch.setattr(eq, "_sanitize_if_possible", alter_boundary)
    assert witness(left, right) is None


@pytest.mark.parametrize("label", ["isotope", "atom_map", "hydrogen", "radical"])
def test_alignment_preserves_immutable_labels(label):
    left, right = connected_sites()
    atom = right.GetAtomWithIdx(3)
    if label == "isotope":
        atom.SetIsotope(13)
    elif label == "atom_map":
        atom.SetAtomMapNum(99)
    elif label == "radical":
        atom.SetNumRadicalElectrons(1)
    else:
        atom.SetNumExplicitHs(1)
        atom.SetNoImplicit(True)
    assert witness(left, right) is None


def test_source_annotations_and_graphs_are_not_modified():
    left, right = connected_sites()
    for mol in (left, right):
        for atom in mol.GetAtoms():
            atom.SetIntProp("MOLGR_LONE_PAIR_COUNT", atom.GetIdx() % 3)
    before = [eq._ChargeShiftContext(m, use_chirality=True).state(m) for m in (left, right)]
    result = witness(left, right)
    assert result is not None
    assert before == [eq._ChargeShiftContext(m, use_chirality=True).state(m) for m in (left, right)]
    assert [
        result.molecule.GetAtomWithIdx(i).GetIntProp("MOLGR_LONE_PAIR_COUNT")
        for i in range(left.GetNumAtoms())
    ] == [i % 3 for i in range(left.GetNumAtoms())]


def test_defined_stereo_uses_existing_complete_path():
    left, right = connected_sites()
    spectator = Chem.MolFromSmiles("F[C@](Cl)(Br)I")
    left, right = Chem.CombineMols(left, spectator), Chem.CombineMols(right, spectator)
    assert witness(left, right, use_chirality=True) is None
    result = eq._resonance_match(
        left, right, use_chirality=True, max_resonance=100, resonance_flags=FLAGS
    )
    assert result.matched


def test_two_ended_2k_corridor_marks_outward_branches():
    # 0--1--2--3--4 is the target path; 2--5--6 is an outward branch.
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6)]
    adjacency = [[] for _ in range(7)]
    for index, (a, b) in enumerate(edges):
        adjacency[a].append((index, b))
        adjacency[b].append((index, a))
    source, target = (-1, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, -1, 0, 0)
    allowed, distances = eq._charge_shift_corridor(adjacency, source, target, max_steps=2)
    assert allowed == frozenset(range(5))
    assert distances[0] == 4
    assert eq._charge_shift_corridor(adjacency, source, target, max_steps=3)[0] == frozenset(
        range(6)
    )


def test_2k_envelope_allows_a_within_budget_outward_step():
    # Synthetic transition graph: toggle a cycle, then transfer to the target.
    # The first move increases target distance from 2 to 4, but the entire
    # witness uses only three migrations. Monotone-distance pruning loses it.
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (4, 5)]
    adjacency = [[] for _ in range(6)]
    for index, (a, b) in enumerate(edges):
        adjacency[a].append((index, b))
        adjacency[b].append((index, a))
    source = ((-1, 0, 0, 0, 0, 0), (1, 2, 1, 2, 1, 2))
    target = ((0, 0, 0, 0, 0, -1), (2, 1, 2, 1, 2, 1))
    # These deliberately permissive valence bounds isolate the search theorem
    # from periodic-table/sanitization constraints, checked by separate tests.
    valences = [(4, 6)] * 6
    args = (tuple(range(6)), tuple(range(6)), source, target, adjacency, valences, 100)
    assert eq._search_charge_shift_region(*args, max_steps=2) is None
    found = eq._search_charge_shift_region(*args, max_steps=3)
    assert found is not None
    assert tuple(step[:3] for step in found[0]) == ((0, 1, 2), (2, 3, 0), (0, 4, 5))


def test_regional_metadata_separates_projected_states_from_validated_forms():
    # Distinguishable sites retain two successors after canonical deduplication.
    left = Chem.MolFromSmiles("[O-]C=CCC=C[NH-]")
    right = Chem.MolFromSmiles("O=C[CH-]C[CH-]C=N")
    result = eq.evaluate_equivalence(left, right, use_chirality=False, max_resonance=100)
    assert result.equivalent
    assert result.bounded_search.regional_region_count == 2
    assert result.bounded_search.regional_state_count == 3
    assert result.bounded_search.mol1_count == 3
    assert result.bounded_search.stop_reason == "target_found"
