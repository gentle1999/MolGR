from __future__ import annotations

from dataclasses import replace

import pytest
from openbabel import openbabel as ob
from openbabel import pybel

from molgr.config import MolGRConfig
from molgr.fallback.pipeline import reconstruct_without_metals
from molgr.fallback.state import ReconstructionState
from molgr.fallback.utils.electrons import (
    get_lone_pair_count,
    get_unpaired_electron_count,
    has_unresolved_two_electron_center,
    set_lone_pair_count,
    set_unpaired_electron_count,
    set_unresolved_two_electron_center,
)
from molgr.fallback.utils.no_metals import neighbor_radicals
from molgr.fallback.utils.no_metals import resonance as no_metal_resonance
from molgr.fallback.utils.no_metals import selection as no_metal_selection


def _oxygen_radical_pair() -> pybel.Molecule:
    obmol = ob.OBMol()
    obmol.BeginModify()
    for _ in range(2):
        atom = obmol.NewAtom()
        atom.SetAtomicNum(8)
        atom.SetFormalCharge(0)
        set_unpaired_electron_count(atom, 1)
    obmol.AddBond(1, 2, 1)
    obmol.EndModify()
    return pybel.Molecule(obmol)


def _state(omol: pybel.Molecule, *, source: str = "test") -> ReconstructionState:
    return ReconstructionState(
        omol=omol,
        given_charge=0,
        total_charge=0,
        total_radical_electrons=0,
        phase_history=("prepared_no_metal_seed",),
        metadata={"source": source},
    )


def test_ligand_topology_reuse_respects_configuration_and_graph_revision() -> None:
    benzene = pybel.readstring("smi", "c1ccccc1")
    benzene.addh()
    state = _state(benzene)
    annotate = no_metal_selection._annotate_no_metal_candidate_topology
    config = MolGRConfig()
    original = annotate(state, config=config)
    assert annotate(state, config=MolGRConfig()) is original

    config.organic_topology.aromatic_stability_benzene_score = 0.25
    rescored = annotate(state, config=config)
    assert rescored.aromatic_stability_score == pytest.approx(0.25)
    assert original.aromatic_stability_score == pytest.approx(1.0)

    set_unpaired_electron_count(benzene.OBMol.GetAtom(1), 1)
    revised = replace(state, omol_revision=state.omol_revision + 1)
    assert annotate(revised, config=config) == no_metal_selection.compute_organic_topology_metrics(
        benzene, config.organic_topology
    )
    assert annotate(revised, config=config) is not rescored


def _boron_neighbor_seed(
    given_charge: int,
    target: int,
    existing_radicals: int = 0,
    *,
    element: int = 6,
    valence: int = 4,
    reverse_bond: bool = False,
):
    mol = ob.OBMol()
    pairs = []
    for _ in range(2):
        boron = mol.NewAtom()
        boron.SetAtomicNum(5)
        boron.SetFormalCharge(-1)
        neighbor = mol.NewAtom()
        neighbor.SetAtomicNum(element)
        begin, end = boron.GetIdx(), neighbor.GetIdx()
        mol.AddBond(end if reverse_bond else begin, begin if reverse_bond else end, 2)
        pairs.append((boron.GetIdx(), neighbor.GetIdx()))
        for endpoint, ligand, count in ((boron, 9, 2), (neighbor, 1, valence - 2)):
            for _ in range(count):
                ligand_atom = mol.NewAtom()
                ligand_atom.SetAtomicNum(ligand)
                mol.AddBond(endpoint.GetIdx(), ligand_atom.GetIdx(), 1)
    if existing_radicals:
        carbon = mol.NewAtom()
        carbon.SetAtomicNum(6)
        set_unpaired_electron_count(carbon, 1)
        for _ in range(3):
            hydrogen = mol.NewAtom()
            hydrogen.SetAtomicNum(1)
            mol.AddBond(carbon.GetIdx(), hydrogen.GetIdx(), 1)
    return ReconstructionState(
        omol=pybel.Molecule(mol),
        given_charge=given_charge,
        total_charge=-2 + given_charge,
        total_radical_electrons=target,
    ), pairs


@pytest.mark.parametrize(
    "charge,target,existing_radicals,expected_changes",
    [
        (2, 2, 0, {(), (0,), (1,), (0, 1)}),
        (1, 2, 0, {(), (0,), (1,)}),
        (2, 1, 0, {(), (0,), (1,)}),
        (0, 1, 0, {()}),
        (-1, 1, 0, {()}),
        (2, 0, 0, {()}),
        (1, 1, 1, {()}),
        (1, 2, 1, {(), (0,), (1,)}),
    ],
)
def test_boron_neighbor_seed_expansion_preserves_original_and_obeys_budgets(
    charge, target, existing_radicals, expected_changes
):
    original, pairs = _boron_neighbor_seed(charge, target, existing_radicals)
    seeds = no_metal_resonance.expand_boron_neighbor_radical_seeds([original])
    assert seeds[0] is original
    assert original.given_charge == charge
    assert all(original.omol.OBMol.GetBond(b, c).GetBondOrder() == 2 for b, c in pairs)
    seen_changes = set()
    for seed in seeds:
        mol = seed.omol.OBMol
        changed = tuple(
            i for i, (b, c) in enumerate(pairs) if mol.GetBond(b, c).GetBondOrder() == 1
        )
        seen_changes.add(changed)
        assert seed.given_charge == charge - len(changed)
        assert mol.NumAtoms() == original.omol.OBMol.NumAtoms()
        assert mol.NumBonds() == original.omol.OBMol.NumBonds()
        assert sum(a.GetFormalCharge() for a in ob.OBMolAtomIter(mol)) == -2 + len(changed)
        assert sum(get_unpaired_electron_count(a) for a in ob.OBMolAtomIter(mol)) == (
            existing_radicals + len(changed)
        )
        for i, (b, c) in enumerate(pairs):
            assert mol.GetAtom(b).GetFormalCharge() == -1
            assert mol.GetAtom(c).GetFormalCharge() == int(i in changed)
            assert get_unpaired_electron_count(mol.GetAtom(b)) == int(i in changed)
            assert get_unpaired_electron_count(mol.GetAtom(c)) == 0
    assert seen_changes == expected_changes
    assert len(seeds) == len(expected_changes)
    # Running the expansion again must not duplicate already generated states.
    assert len(no_metal_resonance.expand_boron_neighbor_radical_seeds(seeds)) == len(seeds)


@pytest.mark.parametrize(
    "ineligible_state",
    ["neutral_boron", "charged_carbon", "single_bond", "radical", "lone_pair", "unresolved"],
)
def test_boron_neighbor_expansion_requires_closed_shell_anionic_double_bond(ineligible_state):
    original, pairs = _boron_neighbor_seed(2, 2)
    mol = original.omol.OBMol
    for b, c in pairs:
        boron, carbon = mol.GetAtom(b), mol.GetAtom(c)
        if ineligible_state == "neutral_boron":
            boron.SetFormalCharge(0)
        elif ineligible_state == "charged_carbon":
            carbon.SetFormalCharge(1)
        elif ineligible_state == "single_bond":
            mol.GetBond(b, c).SetBondOrder(1)
        elif ineligible_state == "radical":
            set_unpaired_electron_count(boron, 1)
        elif ineligible_state == "lone_pair":
            set_lone_pair_count(carbon, 1)
        else:
            set_unresolved_two_electron_center(carbon, True)
    assert no_metal_resonance.expand_boron_neighbor_radical_seeds([original]) == [original]


@pytest.mark.parametrize(
    "element,valence,accepted",
    [
        (6, 4, True),
        (14, 4, True),
        (32, 4, True),
        (5, 3, True),
        (7, 3, False),
        (8, 2, False),
        (15, 3, False),
        (16, 2, False),
        (16, 4, False),
        (6, 3, False),
        (26, 4, False),
    ],
)
@pytest.mark.parametrize("reverse_bond", [False, True])
def test_boron_neighbor_reduction_uses_electronic_state_not_element_identity(
    element, valence, accepted, reverse_bond
):
    original, pairs = _boron_neighbor_seed(
        1, 1, element=element, valence=valence, reverse_bond=reverse_bond
    )
    before = no_metal_resonance._candidate_key(original)
    seeds = no_metal_resonance.expand_boron_neighbor_radical_seeds([original])
    assert no_metal_resonance._candidate_key(original) == before
    assert len(seeds) == (3 if accepted else 1)
    for seed in seeds[1:]:
        assert seed.given_charge == 0
        mol = seed.omol.OBMol
        assert sum(a.GetFormalCharge() for a in ob.OBMolAtomIter(mol)) == -1
        assert sum(get_unpaired_electron_count(a) for a in ob.OBMolAtomIter(mol)) == 1
        for b, x in pairs:
            changed = mol.GetBond(b, x).GetBondOrder() == 1
            assert mol.GetAtom(b).GetFormalCharge() == -1
            assert get_unpaired_electron_count(mol.GetAtom(b)) == int(changed)
            assert mol.GetAtom(x).GetFormalCharge() == int(changed)
            assert get_unpaired_electron_count(mol.GetAtom(x)) == 0
            assert get_lone_pair_count(mol.GetAtom(x)) == 0
    if not accepted:
        # A failed local probe must also leave the caller's graph/electrons intact.
        _, charge, changed = no_metal_resonance._reduce_boron_neighbor_bond(
            original.omol, 1, 1, *pairs[0]
        )
        assert not changed
        assert charge == 1
        assert no_metal_resonance._candidate_key(original) == before


def _charge_separated_resonance(*, negative_atom_idx: int) -> ReconstructionState:
    omol = pybel.readstring("smi", "O=CO")
    obmol = omol.OBMol
    terminal_indices = (1, 3)
    positive_atom_idx = next(idx for idx in terminal_indices if idx != negative_atom_idx)
    obmol.GetAtom(negative_atom_idx).SetFormalCharge(-1)
    obmol.GetAtom(positive_atom_idx).SetFormalCharge(1)
    for bond in ob.OBMolBondIter(obmol):
        terminal_idx = (
            bond.GetEndAtomIdx() if bond.GetBeginAtomIdx() == 2 else bond.GetBeginAtomIdx()
        )
        bond.SetBondOrder(1 if terminal_idx == negative_atom_idx else 2)
    return _state(omol, source=f"negative-{negative_atom_idx}")


def test_no_metal_exact_score_ties_use_graph_key_not_enumeration_order(monkeypatch) -> None:
    lower_key = _charge_separated_resonance(negative_atom_idx=1)
    higher_key = _charge_separated_resonance(negative_atom_idx=3)
    candidates = [higher_key, lower_key]

    monkeypatch.setattr(
        reconstruct_without_metals,
        "_run_linear_preparation",
        lambda state: state,
    )
    monkeypatch.setattr(
        reconstruct_without_metals.neighbor_radicals,
        "enumerate_neighbor_radical_seeds",
        lambda state, *, exact_discrepancy=None: [state] if exact_discrepancy == 0 else [],
    )
    monkeypatch.setattr(
        reconstruct_without_metals.no_metal_resonance,
        "search_resonance_candidates",
        lambda states, **kwargs: candidates,
    )
    monkeypatch.setattr(
        reconstruct_without_metals.selection,
        "_no_metal_candidate_selection_key",
        lambda candidate, **kwargs: (0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0.0),
    )

    expected_key = no_metal_selection._no_metal_candidate_graph_tie_break_key(lower_key)
    assert expected_key < no_metal_selection._no_metal_candidate_graph_tie_break_key(higher_key)

    first = reconstruct_without_metals._run_no_metal_pipeline_from_state(lower_key)
    candidates.reverse()
    second = reconstruct_without_metals._run_no_metal_pipeline_from_state(lower_key)

    assert first is not None
    assert second is not None
    assert first.metadata["source"] == second.metadata["source"] == "negative-1"


def test_neighbor_radical_enumeration_returns_seeds_before_recovery() -> None:
    seeds = neighbor_radicals.enumerate_neighbor_radical_seeds(_state(_oxygen_radical_pair()))

    assert [seed.metadata["neighbor_radical_resolution"] for seed in seeds] == [
        "bond_order",
        "charge_separation",
        "charge_separation",
    ]
    assert [seed.metadata.get("positive_atom_idx") for seed in seeds] == [None, 1, 2]
    assert all("break_deformed_ene" not in seed.phase_history for seed in seeds)
    assert all("break_one_bond" not in seed.phase_history for seed in seeds)
    assert all("fresh_omol_charge_radical_final" not in seed.phase_history for seed in seeds)


def test_neighbor_radical_enumeration_supports_exact_charge_separation_layers() -> None:
    state = _state(_oxygen_radical_pair())

    layer_zero = neighbor_radicals.enumerate_neighbor_radical_seeds(
        state,
        exact_discrepancy=0,
    )
    layer_one = neighbor_radicals.enumerate_neighbor_radical_seeds(
        state,
        exact_discrepancy=1,
    )
    layer_two = neighbor_radicals.enumerate_neighbor_radical_seeds(
        state,
        exact_discrepancy=2,
    )

    assert [seed.metadata["neighbor_radical_resolution"] for seed in layer_zero] == ["bond_order"]
    assert [seed.metadata.get("positive_atom_idx") for seed in layer_one] == [1, 2]
    assert layer_two == []


def test_neighbor_radical_resolution_validates_before_consuming_electrons() -> None:
    omol = _oxygen_radical_pair()
    before = tuple(get_unpaired_electron_count(atom.OBAtom) for atom in omol)

    with pytest.raises(ValueError, match="positive_atom_idx"):
        neighbor_radicals._resolve_neighbor_radical_pair(
            omol,
            1,
            2,
            "charge_separation",
            3,
        )

    assert tuple(get_unpaired_electron_count(atom.OBAtom) for atom in omol) == before


def test_resonance_seed_pool_preserves_unresolved_two_electron_center() -> None:
    omol = pybel.readstring("smi", "[C]([H])[H]")
    center = omol.OBMol.GetAtom(1)
    set_unpaired_electron_count(center, 0)
    set_unresolved_two_electron_center(center, True)

    seeds = no_metal_resonance.build_resonance_seed_pool([_state(omol)])

    assert len(seeds) == 1
    center = seeds[0].omol.OBMol.GetAtom(1)
    assert (get_unpaired_electron_count(center), get_lone_pair_count(center)) == (0, 0)
    assert has_unresolved_two_electron_center(center)
    assert "assign_unresolved_two_electron_centers_for_resonance" not in seeds[0].phase_history


def test_resonance_seed_pool_preserves_charge_separated_oxygen_state(monkeypatch) -> None:
    obmol = ob.OBMol()
    oxygen = obmol.NewAtom()
    oxygen.SetAtomicNum(8)
    oxygen.SetFormalCharge(1)
    set_unpaired_electron_count(oxygen, 0)
    carbon = obmol.NewAtom()
    carbon.SetAtomicNum(6)
    carbon.SetFormalCharge(-1)
    set_unpaired_electron_count(carbon, 0)
    obmol.AddBond(1, 2, 1)

    state = _state(pybel.Molecule(obmol), source="synthetic-charge-separation")
    seeds = no_metal_resonance.build_resonance_seed_pool([state])

    assert seeds
    assert all(get_unpaired_electron_count(seed.omol.OBMol.GetAtom(1)) == 0 for seed in seeds)
    assert all(
        "refresh_electronic_labels_for_resonance" not in seed.phase_history for seed in seeds
    )

    monkeypatch.setattr(
        no_metal_resonance,
        "search_resonance_candidates",
        lambda states, **kwargs: [],
    )
    layer = reconstruct_without_metals._expand_resonance_layer(
        [state],
        traversal_policy=no_metal_resonance._default_resonance_traversal_policy(),
        config=None,
    )

    assert layer.resonance_seeds
    assert all(
        get_unpaired_electron_count(seed.omol.OBMol.GetAtom(1)) == 0
        for seed in layer.resonance_seeds
    )
    assert all(
        "refresh_electronic_labels_for_resonance" not in seed.phase_history
        for seed in layer.resonance_seeds
    )


def test_resonance_seed_pool_does_not_expand_multiple_unresolved_centers() -> None:
    omol = pybel.readstring("smi", "[C]([H])[H].[C]([H])[H]")
    for atom_idx in (1, 4):
        center = omol.OBMol.GetAtom(atom_idx)
        set_unpaired_electron_count(center, 0)
        set_unresolved_two_electron_center(center, True)

    seeds = no_metal_resonance.build_resonance_seed_pool([_state(omol)])

    assert len(seeds) == 1
    assert all(
        has_unresolved_two_electron_center(seeds[0].omol.OBMol.GetAtom(atom_idx))
        for atom_idx in (1, 4)
    )


def test_resonance_candidate_validation_resolves_new_unresolved_center(
    monkeypatch,
) -> None:
    omol = pybel.readstring("smi", "[C]([H])[H]")
    center = omol.OBMol.GetAtom(1)
    set_unpaired_electron_count(center, 0)
    set_unresolved_two_electron_center(center, True)
    state = _state(omol)

    def walk_root(root, *, visit, **kwargs):
        del kwargs
        visit(
            no_metal_resonance.resonance_utils.ResonanceSearchNode(
                root,
                no_metal_resonance.resonance_utils.build_resonance_state_key(root),
                0,
                0,
            )
        )

    monkeypatch.setattr(no_metal_resonance, "walk_radical_resonances", walk_root)
    monkeypatch.setattr(
        no_metal_resonance,
        "_score_reconstruction_candidate",
        lambda candidate, **kwargs: 0.0,
    )
    monkeypatch.setattr(
        no_metal_resonance,
        "_annotate_no_metal_candidate_topology",
        lambda candidate, **kwargs: None,
    )

    candidates = no_metal_resonance.search_resonance_candidates([state])

    assert len(candidates) == 1
    center = candidates[0].omol.OBMol.GetAtom(1)
    assert (get_unpaired_electron_count(center), get_lone_pair_count(center)) == (0, 1)
    assert not has_unresolved_two_electron_center(center)
    assert "resolve_unresolved_two_electron_centers_at_validation" in candidates[0].phase_history


def test_no_metal_pipeline_stops_after_first_valid_discrepancy_layer(monkeypatch) -> None:
    seed_state = _state(pybel.readstring("smi", "CC"), source="input")
    prepared_state = _state(pybel.readstring("smi", "CC"), source="prepared")
    branch_seeds = [
        _state(pybel.readstring("smi", "CC"), source="branch-0"),
        _state(pybel.readstring("smi", "C=C"), source="branch-1"),
    ]
    winner = branch_seeds[1]
    recorded: list[tuple[ReconstructionState, ...]] = []

    monkeypatch.setattr(
        reconstruct_without_metals,
        "_run_linear_preparation",
        lambda state: prepared_state,
    )
    monkeypatch.setattr(
        reconstruct_without_metals.neighbor_radicals,
        "enumerate_neighbor_radical_seeds",
        lambda state, *, exact_discrepancy=None: branch_seeds if exact_discrepancy == 0 else [],
    )

    def search(states, **kwargs):
        del kwargs
        recorded.append(tuple(states))
        return [winner]

    monkeypatch.setattr(
        reconstruct_without_metals.no_metal_resonance,
        "search_resonance_candidates",
        search,
    )
    monkeypatch.setattr(
        reconstruct_without_metals.selection,
        "_score_reconstruction_candidate",
        lambda candidate, **kwargs: 0.0,
    )
    monkeypatch.setattr(
        reconstruct_without_metals.selection,
        "_no_metal_candidate_selection_key",
        lambda candidate, **kwargs: (0.0, 0, 0.0, 0.0, 0.0, 0.0),
    )

    result = reconstruct_without_metals._run_no_metal_pipeline_from_state(seed_state)

    assert result is not None
    assert result.omol is winner.omol
    assert result.metadata["source"] == "branch-1"
    assert result.phase_history[-1] == "select_best_no_metal_candidate"
    assert recorded == [tuple(branch_seeds)]


def test_no_metal_pipeline_widens_with_one_shared_resonance_session(monkeypatch) -> None:
    seed_state = _state(pybel.readstring("smi", "CC"), source="input")
    prepared_state = _state(pybel.readstring("smi", "CC"), source="prepared")
    layer_zero = _state(pybel.readstring("smi", "CC"), source="layer-0")
    layer_one = _state(pybel.readstring("smi", "C=C"), source="layer-1")
    calls: list[tuple[int, object]] = []

    monkeypatch.setattr(
        reconstruct_without_metals,
        "_run_linear_preparation",
        lambda state: prepared_state,
    )

    def enumerate_layer(state, *, exact_discrepancy=None):
        del state
        return [layer_zero] if exact_discrepancy == 0 else [layer_one]

    monkeypatch.setattr(
        reconstruct_without_metals.neighbor_radicals,
        "enumerate_neighbor_radical_seeds",
        enumerate_layer,
    )

    def search(states, **kwargs):
        calls.append((len(states), kwargs["session"]))
        return [] if states[0] is layer_zero else [layer_one]

    monkeypatch.setattr(
        reconstruct_without_metals.no_metal_resonance,
        "search_resonance_candidates",
        search,
    )
    monkeypatch.setattr(
        reconstruct_without_metals.selection,
        "_no_metal_candidate_selection_key",
        lambda candidate, **kwargs: (0.0, 0, 0.0, 0.0, 0.0, 0, 0.0),
    )

    result = reconstruct_without_metals._run_no_metal_pipeline_from_state(seed_state)

    assert result is not None
    assert result.metadata["source"] == "layer-1"
    assert len(calls) == 2
    assert calls[0][1] is calls[1][1]


def test_resonance_candidate_deduplication_spans_all_seeds(monkeypatch) -> None:
    first = _state(pybel.readstring("smi", "[CH3]"), source="first")
    second = _state(pybel.readstring("smi", "[CH3]"), source="second")

    def walk_resonances(omol, *, visit, **kwargs):
        del kwargs
        resonance = omol.clone
        visit(
            no_metal_resonance.resonance_utils.ResonanceSearchNode(
                resonance,
                no_metal_resonance.resonance_utils.build_resonance_state_key(resonance),
                0,
                0,
            )
        )

    monkeypatch.setattr(
        no_metal_resonance,
        "walk_radical_resonances",
        walk_resonances,
    )
    monkeypatch.setattr(
        no_metal_resonance.resonance_utils,
        "process_resonance",
        lambda omol, remaining_charge: (omol, remaining_charge, False),
    )
    monkeypatch.setattr(no_metal_resonance, "validate_omol", lambda *args: True)
    monkeypatch.setattr(
        no_metal_resonance,
        "_score_reconstruction_candidate",
        lambda candidate, **kwargs: 0.0,
    )
    monkeypatch.setattr(
        no_metal_resonance,
        "_annotate_no_metal_candidate_topology",
        lambda candidate, **kwargs: None,
    )
    monkeypatch.setattr(ReconstructionState, "score", lambda self, *args, **kwargs: 0.0)

    candidates = no_metal_resonance.search_resonance_candidates([first, second])

    assert len(candidates) == 1


def test_resonance_candidate_deduplication_spans_session_calls(monkeypatch) -> None:
    state = _state(pybel.readstring("smi", "[CH3]"))

    def walk_resonances(omol, *, visit, **kwargs):
        del kwargs
        visit(
            no_metal_resonance.resonance_utils.ResonanceSearchNode(
                omol.clone,
                no_metal_resonance.resonance_utils.build_resonance_state_key(omol),
                0,
                0,
            )
        )

    monkeypatch.setattr(no_metal_resonance, "walk_radical_resonances", walk_resonances)
    monkeypatch.setattr(no_metal_resonance, "validate_omol", lambda *args: True)
    monkeypatch.setattr(
        no_metal_resonance,
        "_score_reconstruction_candidate",
        lambda candidate, **kwargs: 0.0,
    )
    monkeypatch.setattr(
        no_metal_resonance,
        "_annotate_no_metal_candidate_topology",
        lambda candidate, **kwargs: None,
    )
    session = no_metal_resonance._ResonanceSearchSession()

    first = no_metal_resonance.search_resonance_candidates([state], session=session)
    second = no_metal_resonance.search_resonance_candidates([state], session=session)

    assert len(first) == 1
    assert second == []


def test_resonance_traversal_labels_prune_only_dominated_paths() -> None:
    state_key = (("state",), 0)
    labels = {}

    assert no_metal_resonance._register_traversal_label(labels, state_key, (2, 1))
    assert not no_metal_resonance._register_traversal_label(labels, state_key, (2, 2))
    assert no_metal_resonance._register_traversal_label(labels, state_key, (1, 2))
    assert no_metal_resonance._register_traversal_label(labels, state_key, (1, 0))
    assert labels[state_key] == [(1, 0)]
