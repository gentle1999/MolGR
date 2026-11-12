# pyright: reportMissingImports=false

from __future__ import annotations

import csv
import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest


pytest.importorskip("openbabel")

from openbabel import pybel
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDistGeom

from molgr.config import MolGRConfig
from molgr.fallback.pipeline import reconstruct_with_metals as with_metals_module
from molgr.fallback.pipeline import reconstruct_without_metals as without_metals_module
from molgr.fallback.pipeline.reconstruct_with_metals import xyz2omol
from molgr.fallback.state import (
    MetalCandidateState,
    MetalPreparationState,
    ReconstructionState,
    make_metal_candidate_state,
)
from molgr.fallback.utils.dataclasses import MetalAtomPosition
from molgr.fallback.utils.electrons import set_lone_pair_count, set_unpaired_electron_count
from molgr.fallback.utils.force_field import (
    build_force_field_score_key,
    organic_force_field_energy,
)
from molgr.fallback.utils.metals import preparation as metal_preparation_module
from molgr.fallback.utils.metals import scoring as metal_scoring_module
from molgr.fallback.utils.metals import search as metal_search_module
from molgr.fallback.utils.metals.preparation import prepare_metal_state
from molgr.fallback.utils.metals.search import _group_candidates_by_target_dp


RDLogger.DisableLog("rdApp.*")  # type: ignore


_MOLFILE_CASES_SPEC = importlib.util.spec_from_file_location(
    "molgr_cases_molfile",
    Path("scripts/molgr_cases_molfile.py").resolve(),
)
assert _MOLFILE_CASES_SPEC is not None
_MOLFILE_CASES_MODULE = importlib.util.module_from_spec(_MOLFILE_CASES_SPEC)
assert _MOLFILE_CASES_SPEC.loader is not None
_MOLFILE_CASES_SPEC.loader.exec_module(_MOLFILE_CASES_MODULE)
load_molfile_cases = _MOLFILE_CASES_MODULE.load_molfile_cases


def _total_charge_and_radicals(mol: Chem.Mol) -> tuple[int, int]:
    charge = 0
    radicals = 0
    for atom in mol.GetAtoms():  # pyright: ignore[reportCallIssue]
        charge += int(atom.GetFormalCharge())
        radicals += int(atom.GetNumRadicalElectrons())
    return charge, radicals


def _seed_case(smiles: str) -> tuple[str, int, int]:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    mol_h = Chem.AddHs(mol)
    embed_code = rdDistGeom.EmbedMolecule(mol_h)  # pyright: ignore[reportCallIssue]
    assert int(embed_code) == 0
    total_charge, total_radical_electrons = _total_charge_and_radicals(mol_h)
    return Chem.MolToXYZBlock(mol_h), total_charge, total_radical_electrons


def _patch_no_metal_seed_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
    no_metal_state: ReconstructionState,
    *,
    parse_calls: dict[str, int] | None = None,
) -> None:
    def fake_seed_omol_from_xyz(*args, **kwargs):
        del args, kwargs
        if parse_calls is not None:
            parse_calls["count"] += 1
        return no_metal_state.omol

    monkeypatch.setattr(
        with_metals_module.no_metal_preparation,
        "_seed_omol_from_xyz",
        fake_seed_omol_from_xyz,
    )
    monkeypatch.setattr(
        without_metals_module,
        "_seed_omol_to_omol_no_metal_state",
        lambda *args, **kwargs: no_metal_state,
    )


def _load_parity_cases() -> list[object]:
    csv_path = Path(__file__).with_name("test_cases.csv")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    cases: list[object] = []
    for idx in [1, 2, 5, 10]:
        xyz_block, total_charge, total_radical_electrons = _seed_case(rows[idx - 1]["smiles"])
        cases.append(
            pytest.param(
                xyz_block,
                total_charge,
                total_radical_electrons,
                id=f"curated-{idx}",
            )
        )
    return cases


@pytest.mark.parametrize(
    ("xyz_block", "total_charge", "total_radical_electrons"),
    _load_parity_cases(),
)
def test_fallback_with_metals_reconstructs_curated_cases(
    xyz_block: str,
    total_charge: int,
    total_radical_electrons: int,
) -> None:
    result = xyz2omol(xyz_block, total_charge, total_radical_electrons)
    assert result is not None


def test_fallback_with_metals_tracks_top_level_phases() -> None:
    state = prepare_metal_state(
        """3
LiCO
Li 0.0 0.0 0.0
C 2.0 0.0 0.0
O 3.2 0.0 0.0
""",
        0,
        0,
    )

    assert state is not None
    assert state.phase_history[:4] == (
        "read_xyz",
        "build_metal_state_options",
        "remove_metal_atoms",
        "serialize_no_metal_xyz",
    )
    assert state.metadata["metal_atom_count"] == 1


def test_prepare_metal_state_preserves_metal_free_xyz_block() -> None:
    xyz_block = """2
NO
N 0.0 0.0 0.0
O 1.2 0.0 0.0
"""

    state = prepare_metal_state(xyz_block, 0, 1)

    assert state.no_metal_xyz_block == xyz_block
    assert state.available_valence_radical_states == ()
    assert state.phase_history == (
        "read_xyz",
        "build_metal_state_options",
        "preserve_no_metal_xyz",
    )
    assert state.metadata["metal_atom_count"] == 0


def test_xyz2omol_state_prunes_open_shell_multimetal_state_space_for_monnmo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = load_molfile_cases(Path("tests/data/sdf/MoNNMo.sdf"), limit=1)[0]
    raw_state = prepare_metal_state(
        case["xyz_block"],
        case["total_charge"],
        case["total_radical_electrons"],
    )
    captured: dict[str, object] = {}
    original_group = metal_search_module._group_candidates_by_target_dp

    def wrapped_group(*args, **kwargs):
        available_states = args[1]
        captured["option_counts"] = [len(options) for options in available_states]
        return original_group(*args, **kwargs)

    monkeypatch.setattr(metal_search_module, "_group_candidates_by_target_dp", wrapped_group)

    result = with_metals_module.xyz2omol_state(
        case["xyz_block"],
        total_charge=case["total_charge"],
        total_radical_electrons=case["total_radical_electrons"],
    )

    assert result is not None
    assert [len(options) for options in raw_state.available_valence_radical_states] == [17, 17]
    assert captured["option_counts"] == [7, 7]
    # The closed-shell ligand target (-6, 0) must remain competitive with (-4, 2):
    # removing ligand charges by creating radicals must not force Mo(II).
    assert [(state.symbol, state.valence, state.radical_num) for state in result.metal_states] == [
        ("Mo", 3, 1),
        ("Mo", 3, 1),
    ]


def test_build_metal_state_search_groups_unifies_same_element_groups_beyond_threshold() -> None:
    raw_state_options = (
        (
            MetalAtomPosition(1, "Mo", 42, 3, 1, 0.0, 0.0, 0.0),
            MetalAtomPosition(1, "Mo", 42, 4, 0, 0.0, 0.0, 0.0),
        ),
        (
            MetalAtomPosition(2, "Mo", 42, 3, 1, 1.0, 0.0, 0.0),
            MetalAtomPosition(2, "Mo", 42, 4, 0, 1.0, 0.0, 0.0),
        ),
        (
            MetalAtomPosition(3, "Mo", 42, 3, 1, 2.0, 0.0, 0.0),
            MetalAtomPosition(3, "Mo", 42, 4, 0, 2.0, 0.0, 0.0),
        ),
        (
            MetalAtomPosition(4, "Mo", 42, 3, 1, 3.0, 0.0, 0.0),
            MetalAtomPosition(4, "Mo", 42, 4, 0, 3.0, 0.0, 0.0),
        ),
    )

    grouped_state_options = metal_search_module._build_metal_state_search_groups(raw_state_options)

    assert len(grouped_state_options) == 1
    assert len(grouped_state_options[0]) == 2
    assert [
        [(state.idx, state.valence, state.radical_num) for state in state_choice]
        for state_choice in grouped_state_options[0]
    ] == [
        [(1, 4, 0), (2, 4, 0), (3, 4, 0), (4, 4, 0)],
        [(1, 3, 1), (2, 3, 1), (3, 3, 1), (4, 3, 1)],
    ]


def test_xyz2omol_state_layered_expansion_retries_when_first_layer_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    low_penalty_left = MetalAtomPosition(1, "Mo", 42, 3, 1, 0.0, 0.0, 0.0)
    expanded_left = MetalAtomPosition(1, "Mo", 42, 2, 0, 0.0, 0.0, 0.0)
    low_penalty_right = MetalAtomPosition(2, "Mo", 42, 3, 1, 5.0, 0.0, 0.0)
    expanded_right = MetalAtomPosition(2, "Mo", 42, 2, 0, 5.0, 0.0, 0.0)
    prepared_state = MetalPreparationState(
        no_metal_xyz_block="ignored",
        available_valence_radical_states=(
            (low_penalty_left, expanded_left),
            (low_penalty_right, expanded_right),
        ),
        total_charge=0,
        total_radical_electrons=2,
    )
    fallback_candidate = make_metal_candidate_state(
        (),
        (expanded_left, expanded_right),
        -4,
        2,
        combination_index=0,
    )
    no_metal = pybel.readstring(
        "xyz",
        """2
CO
C 0.0 0.0 0.0
O 1.2 0.0 0.0
""",
    )
    no_metal_state = ReconstructionState(
        omol=no_metal,
        given_charge=0,
        total_charge=-4,
        total_radical_electrons=2,
    )
    layer_calls: list[list[int]] = []

    monkeypatch.setattr(
        metal_preparation_module,
        "prepare_metal_state",
        lambda *args, **kwargs: prepared_state,
    )

    def fake_group(*args, **kwargs):
        del kwargs
        available_state_groups = args[1]
        layer_calls.append([len(options) for options in available_state_groups])
        if layer_calls == [[1, 1]]:
            return {}
        return {
            (
                fallback_candidate.no_metal_charge_target,
                fallback_candidate.no_metal_radical_target,
            ): [fallback_candidate]
        }

    monkeypatch.setattr(metal_search_module, "_group_candidates_by_target_dp", fake_group)
    parse_calls = {"count": 0}
    _patch_no_metal_seed_reconstruction(
        monkeypatch,
        no_metal_state,
        parse_calls=parse_calls,
    )

    def fake_prepare_candidate(candidate, no_metal_state, *, config=None):
        del config
        candidate.no_metal_state = no_metal_state
        candidate.score = 1.0
        candidate.metadata["score"] = 1.0
        candidate.metadata["metal_assignment_rank"] = 0.0
        candidate.metadata["organic_aromatic_atom_count"] = 6
        candidate.metadata["organic_aromatic_ring_count"] = 1
        candidate.metadata["organic_conjugated_atom_count"] = 6
        candidate.metadata["organic_conjugated_bond_count"] = 6
        candidate.metadata["organic_max_conjugated_component_size"] = 6
        candidate.metadata["organic_radical_localization_penalty"] = 0.0
        candidate.metadata["organic_charge_localization_penalty"] = 0.0
        candidate.combined_omol = {
            "valences": tuple(state.valence for state in candidate.metal_states)
        }
        return candidate

    monkeypatch.setattr(
        metal_scoring_module,
        "_prepare_candidate_with_no_metal_state",
        fake_prepare_candidate,
    )
    layered_config = replace(
        MolGRConfig(),
        metal_scoring=replace(
            MolGRConfig().metal_scoring,
            open_shell_multimetal_state_penalty_window=0.0,
            open_shell_multimetal_min_state_options=1,
        ),
    )

    result = with_metals_module.xyz2omol_state(
        "ignored",
        total_charge=0,
        total_radical_electrons=2,
        config=layered_config,
    )

    assert result is not None
    assert layer_calls == [[1, 1], [2, 2]]
    assert parse_calls == {"count": 1}
    assert result.metadata["search_layer_index"] == 1
    assert result.combined_omol == {"valences": (2, 2)}


def test_xyz2omol_state_reuses_clean_no_metal_seed_across_target_buckets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_metal_state = MetalAtomPosition(1, "Li", 3, 1, 0, 0.0, 0.0, 0.0)
    second_metal_state = MetalAtomPosition(1, "Li", 3, 2, 0, 0.0, 0.0, 0.0)
    first_candidate = make_metal_candidate_state(
        (),
        (first_metal_state,),
        -1,
        0,
        combination_index=0,
    )
    second_candidate = make_metal_candidate_state(
        (),
        (second_metal_state,),
        -2,
        0,
        combination_index=1,
    )
    no_metal_seed = pybel.readstring(
        "xyz",
        """2
CO
C 0.0 0.0 0.0
O 1.2 0.0 0.0
""",
    )
    prepared_state = MetalPreparationState(
        no_metal_xyz_block="clean-seed-xyz",
        available_valence_radical_states=((first_metal_state, second_metal_state),),
        total_charge=0,
        total_radical_electrons=0,
    )
    no_metal_state = ReconstructionState(
        omol=no_metal_seed,
        given_charge=0,
        total_charge=-1,
        total_radical_electrons=0,
    )

    monkeypatch.setattr(
        metal_preparation_module,
        "prepare_metal_state",
        lambda *args, **kwargs: prepared_state,
    )
    monkeypatch.setattr(
        metal_search_module,
        "_group_candidates_by_target_dp",
        lambda *args, **kwargs: {
            (
                first_candidate.no_metal_charge_target,
                first_candidate.no_metal_radical_target,
            ): [first_candidate],
            (
                second_candidate.no_metal_charge_target,
                second_candidate.no_metal_radical_target,
            ): [second_candidate],
        },
    )
    parse_calls = {"count": 0}

    def fake_seed_omol_from_xyz(xyz_block):
        assert xyz_block == "clean-seed-xyz"
        parse_calls["count"] += 1
        return no_metal_seed

    monkeypatch.setattr(
        with_metals_module.no_metal_preparation,
        "_seed_omol_from_xyz",
        fake_seed_omol_from_xyz,
    )

    captured_seeds: list[object] = []

    def fake_seed_to_state(seed_omol, *args, **kwargs):
        del args, kwargs
        captured_seeds.append(seed_omol)
        return no_metal_state

    monkeypatch.setattr(
        without_metals_module,
        "_seed_omol_to_omol_no_metal_state",
        fake_seed_to_state,
    )

    def fake_prepare_candidate(candidate, no_metal_state, *, config=None):
        del config
        candidate.no_metal_state = no_metal_state
        candidate.score = 1.0
        candidate.metadata["score"] = 1.0
        candidate.metadata["metal_assignment_rank"] = 0.0
        candidate.metadata["organic_aromatic_atom_count"] = 0
        candidate.metadata["organic_aromatic_ring_count"] = 0
        candidate.metadata["organic_conjugated_atom_count"] = 0
        candidate.metadata["organic_conjugated_bond_count"] = 0
        candidate.metadata["organic_max_conjugated_component_size"] = 0
        candidate.metadata["organic_radical_localization_penalty"] = 0.0
        candidate.metadata["organic_charge_localization_penalty"] = 0.0
        candidate.combined_omol = {"used_seed": True}
        return candidate

    monkeypatch.setattr(
        metal_scoring_module,
        "_prepare_candidate_with_no_metal_state",
        fake_prepare_candidate,
    )

    result = with_metals_module.xyz2omol_state("ignored", 0, 0)

    assert result is not None
    assert parse_calls == {"count": 1}
    assert captured_seeds == [no_metal_seed, no_metal_seed]
    assert result.combined_omol == {"used_seed": True}


def test_xyz2omol_state_reads_target_group_pruning_limits_from_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_state = MetalPreparationState(
        no_metal_xyz_block="ignored",
        available_valence_radical_states=(
            (
                MetalAtomPosition(1, "Mo", 42, 3, 1, 0.0, 0.0, 0.0),
                MetalAtomPosition(1, "Mo", 42, 4, 0, 0.0, 0.0, 0.0),
            ),
        ),
        total_charge=0,
        total_radical_electrons=2,
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        metal_preparation_module,
        "prepare_metal_state",
        lambda *args, **kwargs: prepared_state,
    )

    def fake_group(*args, **kwargs):
        del args
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(metal_search_module, "_group_candidates_by_target_dp", fake_group)

    custom_config = replace(
        MolGRConfig(),
        metal_scoring=replace(
            MolGRConfig().metal_scoring,
            max_mixed_valence_spread=5,
            max_assignments_per_target=7,
        ),
    )

    result = with_metals_module.xyz2omol_state(
        "ignored",
        total_charge=0,
        total_radical_electrons=2,
        config=custom_config,
    )

    assert result is None
    assert captured == {"config": custom_config}


def test_group_candidates_by_target_dp_caps_candidates_per_target() -> None:
    state_options = tuple(
        (
            MetalAtomPosition(idx, "Fe", 26, 0, 0, float(idx), 0.0, 0.0),
            MetalAtomPosition(idx, "Fe", 26, 2, 0, float(idx), 0.0, 0.0),
        )
        for idx in range(1, 5)
    )
    custom_config = replace(
        MolGRConfig(),
        metal_scoring=replace(
            MolGRConfig().metal_scoring,
            max_mixed_valence_spread=None,
            max_assignments_per_target=1,
        ),
    )

    grouped = _group_candidates_by_target_dp(
        (),
        state_options,
        total_charge=0,
        total_radical_electrons=0,
        config=custom_config,
    )

    assert sorted(grouped) == [(-8, 0), (-6, 0), (-4, 0), (-2, 0), (0, 0)]
    assert sum(len(bucket) for bucket in grouped.values()) == 5
    assert all(len(bucket) == 1 for bucket in grouped.values())


def test_group_candidates_by_target_dp_assigns_combination_indices_in_cpp_target_order() -> None:
    state_options = (
        (
            MetalAtomPosition(1, "Fe", 26, 0, 0, 0.0, 0.0, 0.0),
            MetalAtomPosition(1, "Fe", 26, 2, 0, 0.0, 0.0, 0.0),
        ),
        (
            MetalAtomPosition(2, "Fe", 26, 0, 0, 1.0, 0.0, 0.0),
            MetalAtomPosition(2, "Fe", 26, 2, 0, 1.0, 0.0, 0.0),
        ),
    )
    custom_config = replace(
        MolGRConfig(),
        metal_scoring=replace(
            MolGRConfig().metal_scoring,
            max_mixed_valence_spread=None,
            max_assignments_per_target=10,
        ),
    )

    grouped = _group_candidates_by_target_dp(
        (),
        state_options,
        total_charge=0,
        total_radical_electrons=0,
        config=custom_config,
    )

    assert list(grouped) == [(-4, 0), (-2, 0), (0, 0)]
    assert [
        (
            target,
            [candidate.metadata["combination_index"] for candidate in candidates],
        )
        for target, candidates in grouped.items()
    ] == [
        ((-4, 0), [0]),
        ((-2, 0), [1, 2]),
        ((0, 0), [3]),
    ]


def test_group_candidates_by_target_dp_uses_meet_in_the_middle_split(monkeypatch) -> None:
    state_options = tuple(
        (MetalAtomPosition(idx, "Fe", 26, 2, 0, float(idx), 0.0, 0.0),) for idx in range(1, 5)
    )
    calls: list[int] = []
    original_frontier = metal_search_module._enumerate_partial_assignment_frontier

    def tracking_frontier(*args, **kwargs):
        calls.append(len(args[0]))
        return original_frontier(*args, **kwargs)

    monkeypatch.setattr(
        metal_search_module,
        "_enumerate_partial_assignment_frontier",
        tracking_frontier,
    )
    custom_config = replace(
        MolGRConfig(),
        metal_scoring=replace(
            MolGRConfig().metal_scoring,
            max_mixed_valence_spread=3,
            max_assignments_per_target=1,
        ),
    )

    _group_candidates_by_target_dp(
        (),
        state_options,
        total_charge=0,
        total_radical_electrons=0,
        config=custom_config,
    )

    assert calls == [2, 2]


def test_group_candidates_by_target_dp_preserves_cross_half_valence_spread_pruning() -> None:
    state_options = (
        (MetalAtomPosition(1, "Fe", 26, 2, 0, 1.0, 0.0, 0.0),),
        (MetalAtomPosition(2, "Fe", 26, 6, 0, 2.0, 0.0, 0.0),),
    )
    custom_config = replace(
        MolGRConfig(),
        metal_scoring=replace(
            MolGRConfig().metal_scoring,
            max_mixed_valence_spread=3,
            max_assignments_per_target=1,
        ),
    )

    grouped = _group_candidates_by_target_dp(
        (),
        state_options,
        total_charge=0,
        total_radical_electrons=0,
        config=custom_config,
    )

    assert grouped == {}


def test_combine_partial_assignment_frontiers_rejects_metal_spin_over_global_budget(
    monkeypatch,
) -> None:
    partial_assignment = metal_search_module._PartialMetalAssignment
    left_frontier = {
        (0, 1, ()): [
            partial_assignment(
                metal_states=(),
                total_metal_charge=0,
                total_metal_radicals=1,
                metal_assignment_rank=0.0,
                valence_bounds=(),
                order=0,
            )
        ]
    }
    right_frontier = {
        (1, 0, ()): [
            partial_assignment(
                metal_states=(),
                total_metal_charge=1,
                total_metal_radicals=0,
                metal_assignment_rank=0.0,
                valence_bounds=(),
                order=1,
            )
        ],
        (2, 3, ()): [
            partial_assignment(
                metal_states=(),
                total_metal_charge=2,
                total_metal_radicals=3,
                metal_assignment_rank=0.0,
                valence_bounds=(),
                order=2,
            )
        ],
    }
    calls = {"count": 0}

    def tracking_merge(left_bounds, right_bounds, max_mixed_valence_spread):
        calls["count"] += 1
        return ()

    monkeypatch.setattr(metal_search_module, "_merge_valence_bounds", tracking_merge)
    custom_config = replace(
        MolGRConfig(),
        metal_scoring=replace(
            MolGRConfig().metal_scoring,
            max_mixed_valence_spread=3,
            max_assignments_per_target=1,
        ),
    )

    grouped = metal_search_module._combine_partial_assignment_frontiers(
        left_frontier,
        right_frontier,
        total_charge=0,
        total_radical_electrons=1,
        config=custom_config,
    )

    assert calls["count"] == 1
    assert sorted(grouped) == [(-1, 0)]


def test_group_candidates_by_target_dp_rejects_single_metal_spin_over_global_budget() -> None:
    grouped = _group_candidates_by_target_dp(
        (),
        ((MetalAtomPosition(1, "Fe", 26, 2, 1, 0.0, 0.0, 0.0),),),
        total_charge=0,
        total_radical_electrons=0,
    )

    assert grouped == {}


def test_layered_search_removes_over_budget_states_for_single_metal_singlet() -> None:
    closed_shell = MetalAtomPosition(1, "Fe", 26, 2, 0, 0.0, 0.0, 0.0)
    open_shell = MetalAtomPosition(1, "Fe", 26, 2, 1, 0.0, 0.0, 0.0)

    layers = metal_search_module._build_layered_metal_state_search_groups(
        (((closed_shell,), (open_shell,)),),
        total_radical_electrons=0,
    )

    assert len(layers) == 1
    assert layers[0][0] == ((closed_shell,),)


def test_final_metal_candidate_validation_rejects_excess_radicals() -> None:
    candidate = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Fe", 26, 2, 1, 0.0, 0.0, 0.0),),
        no_metal_charge_target=-2,
        no_metal_radical_target=0,
        combination_index=0,
    )

    assert with_metals_module._candidate_matches_global_electronic_state(candidate, 0, 1)
    assert not with_metals_module._candidate_matches_global_electronic_state(candidate, 0, 0)


def test_iter_reachable_radical_buckets_uses_prefix_cutoff() -> None:
    prefix_reachability = metal_search_module._build_radical_prefix_reachability(
        {
            0: {1: []},
            2: {3: []},
            5: {7: []},
        }
    )

    reachable = metal_search_module._iter_reachable_radical_buckets(prefix_reachability, 3)

    assert [radicals for radicals, _ in reachable] == [0, 2]


def test_xyz2omol_state_uses_target_bucket_dp(monkeypatch) -> None:
    calls = {"count": 0}

    original_dp = metal_search_module._group_candidates_by_target_dp

    def tracking_dp(*args, **kwargs):
        calls["count"] += 1
        return original_dp(*args, **kwargs)

    monkeypatch.setattr(metal_search_module, "_group_candidates_by_target_dp", tracking_dp)

    with_metals_module.xyz2omol_state(
        """3
LiCO
Li 0.0 0.0 0.0
C 2.0 0.0 0.0
O 3.2 0.0 0.0
""",
        0,
        0,
    )

    assert calls["count"] == 1


def test_prepare_candidate_with_no_metal_state_uses_force_field_energy(monkeypatch) -> None:
    no_metal = pybel.readstring(
        "xyz",
        """2
CO
C 20.0 0.0 0.0
O 21.2 0.0 0.0
""",
    )
    no_metal_state = ReconstructionState(
        omol=no_metal,
        given_charge=0,
        total_charge=0,
        total_radical_electrons=0,
        metadata={
            "organic_core_score": organic_force_field_energy(no_metal),
            "force_field_score_key": build_force_field_score_key(no_metal),
        },
    )
    candidate = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Li", 3, 1, 0, 0.0, 0.0, 0.0),),
        -1,
        0,
        combination_index=0,
    )
    calls = {"combined_score": 0}

    def fake_combined_score(self):
        calls["combined_score"] += 1
        self.score = 1.25
        self.metadata["score"] = 1.25
        return 1.25

    monkeypatch.setattr(MetalCandidateState, "combined_score", fake_combined_score)

    scored = metal_scoring_module._prepare_candidate_with_no_metal_state(candidate, no_metal_state)

    assert calls == {"combined_score": 1}
    assert scored.score == 1.25
    assert scored.metadata["score"] == 1.25


def test_prepare_candidate_with_no_metal_state_reuses_candidate_cache(monkeypatch) -> None:
    no_metal = pybel.readstring(
        "xyz",
        """2
CO
C 2.0 0.0 0.0
O 3.2 0.0 0.0
""",
    )
    no_metal_state = ReconstructionState(
        omol=no_metal,
        given_charge=0,
        total_charge=0,
        total_radical_electrons=0,
        metadata={
            "organic_core_score": organic_force_field_energy(no_metal),
            "force_field_score_key": build_force_field_score_key(no_metal),
        },
    )
    candidate = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Li", 3, 1, 0, 0.0, 0.0, 0.0),),
        -1,
        0,
        combination_index=0,
    )
    calls = {"combined_score": 0}

    def fake_combined_score(self):
        if self.score is not None:
            return self.score
        calls["combined_score"] += 1
        self.score = 1.25
        self.metadata["score"] = 1.25
        return 1.25

    monkeypatch.setattr(MetalCandidateState, "combined_score", fake_combined_score)

    first = metal_scoring_module._prepare_candidate_with_no_metal_state(candidate, no_metal_state)
    second = metal_scoring_module._prepare_candidate_with_no_metal_state(first, no_metal_state)

    assert calls == {"combined_score": 1}
    assert second.score == 1.25


def test_xyz2omol_state_uses_aromatic_diagnostics_before_organic_score(
    monkeypatch,
) -> None:
    no_metal = pybel.readstring(
        "xyz",
        """2
CO
C 2.0 0.0 0.0
O 3.2 0.0 0.0
""",
    )
    no_metal_state = ReconstructionState(
        omol=no_metal,
        given_charge=0,
        total_charge=0,
        total_radical_electrons=0,
        metadata={
            "organic_core_score": organic_force_field_energy(no_metal),
            "force_field_score_key": build_force_field_score_key(no_metal),
        },
    )
    candidate_a = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Li", 3, 1, 0, 0.0, 0.0, 0.0),),
        -1,
        0,
        combination_index=0,
    )
    candidate_b = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Li", 3, 2, 0, 0.0, 0.0, 0.0),),
        -2,
        0,
        combination_index=1,
    )

    monkeypatch.setattr(
        metal_search_module,
        "_group_candidates_by_target_dp",
        lambda *args, **kwargs: {
            (candidate_a.no_metal_charge_target, candidate_a.no_metal_radical_target): [
                candidate_a
            ],
            (candidate_b.no_metal_charge_target, candidate_b.no_metal_radical_target): [
                candidate_b
            ],
        },
    )
    _patch_no_metal_seed_reconstruction(monkeypatch, no_metal_state)
    calls = {"count": 0}

    def fake_prepare_candidate(candidate, no_metal_state, *, config=None):
        del config
        calls["count"] += 1
        candidate.no_metal_state = no_metal_state
        valence = candidate.metal_states[0].valence
        candidate.score = 1.9 if valence == 1 else 1.0
        candidate.metadata["score"] = candidate.score
        candidate.metadata["metal_assignment_rank"] = 0.0
        if valence == 1:
            candidate.metadata["organic_aromatic_atom_count"] = 6
            candidate.metadata["organic_aromatic_ring_count"] = 1
            candidate.metadata["organic_conjugated_atom_count"] = 6
            candidate.metadata["organic_conjugated_bond_count"] = 6
            candidate.metadata["organic_max_conjugated_component_size"] = 6
            candidate.metadata["organic_radical_localization_penalty"] = 0.0
            candidate.metadata["organic_charge_localization_penalty"] = 0.0
        else:
            candidate.metadata["organic_aromatic_atom_count"] = 0
            candidate.metadata["organic_aromatic_ring_count"] = 0
            candidate.metadata["organic_conjugated_atom_count"] = 2
            candidate.metadata["organic_conjugated_bond_count"] = 1
            candidate.metadata["organic_max_conjugated_component_size"] = 2
            candidate.metadata["organic_radical_localization_penalty"] = 3.0
            candidate.metadata["organic_charge_localization_penalty"] = 4.0
        candidate.combined_omol = {"valence": valence}
        return candidate

    monkeypatch.setattr(
        metal_scoring_module,
        "_prepare_candidate_with_no_metal_state",
        fake_prepare_candidate,
    )
    result = with_metals_module.xyz2omol_state(
        """3
LiCO
Li 0.0 0.0 0.0
C 2.0 0.0 0.0
O 3.2 0.0 0.0
""",
        0,
        0,
    )

    assert result is not None
    assert result.score == 1.9
    assert calls == {"count": 2}
    assert "organic_electronic_state_key" not in result.metadata
    assert result.combined_omol == {"valence": 1}


def test_xyz2omol_state_uses_electronic_state_before_organic_score(
    monkeypatch,
) -> None:
    no_metal = pybel.readstring(
        "xyz",
        """2
CO
C 2.0 0.0 0.0
O 3.2 0.0 0.0
""",
    )
    no_metal_state = ReconstructionState(
        omol=no_metal,
        given_charge=0,
        total_charge=0,
        total_radical_electrons=0,
        metadata={
            "organic_core_score": organic_force_field_energy(no_metal),
            "force_field_score_key": build_force_field_score_key(no_metal),
        },
    )
    lexicographically_better_but_worse_overall = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Li", 3, 2, 0, 0.0, 0.0, 0.0),),
        -2,
        0,
        combination_index=1,
    )
    more_balanced = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Li", 3, 1, 0, 0.0, 0.0, 0.0),),
        -1,
        0,
        combination_index=0,
    )

    monkeypatch.setattr(
        metal_search_module,
        "_group_candidates_by_target_dp",
        lambda *args, **kwargs: {
            (
                lexicographically_better_but_worse_overall.no_metal_charge_target,
                lexicographically_better_but_worse_overall.no_metal_radical_target,
            ): [lexicographically_better_but_worse_overall],
            (more_balanced.no_metal_charge_target, more_balanced.no_metal_radical_target): [
                more_balanced
            ],
        },
    )
    _patch_no_metal_seed_reconstruction(monkeypatch, no_metal_state)

    def fake_prepare_candidate(candidate, no_metal_state, *, config=None):
        del config
        candidate.no_metal_state = no_metal_state
        valence = candidate.metal_states[0].valence
        candidate.score = 1.2 if valence == 1 else 1.0
        candidate.metadata["score"] = candidate.score
        if valence == 1:
            candidate.metadata["organic_aromatic_atom_count"] = 6
            candidate.metadata["organic_aromatic_ring_count"] = 1
            candidate.metadata["organic_conjugated_atom_count"] = 6
            candidate.metadata["organic_conjugated_bond_count"] = 6
            candidate.metadata["organic_max_conjugated_component_size"] = 6
            candidate.metadata["organic_radical_localization_penalty"] = 0.0
            candidate.metadata["organic_charge_localization_penalty"] = 1.0
            candidate.metadata["metal_assignment_rank"] = 0.0
        else:
            candidate.metadata["organic_aromatic_atom_count"] = 6
            candidate.metadata["organic_aromatic_ring_count"] = 1
            candidate.metadata["organic_conjugated_atom_count"] = 6
            candidate.metadata["organic_conjugated_bond_count"] = 6
            candidate.metadata["organic_max_conjugated_component_size"] = 6
            candidate.metadata["organic_radical_localization_penalty"] = 3.0
            candidate.metadata["organic_charge_localization_penalty"] = 1.2
            candidate.metadata["metal_assignment_rank"] = 10.0
        candidate.combined_omol = {"valence": valence}
        return candidate

    monkeypatch.setattr(
        metal_scoring_module,
        "_prepare_candidate_with_no_metal_state",
        fake_prepare_candidate,
    )
    result = with_metals_module.xyz2omol_state(
        """3
LiCO
Li 0.0 0.0 0.0
C 2.0 0.0 0.0
O 3.2 0.0 0.0
""",
        0,
        0,
    )

    assert result is not None
    assert result.combined_omol == {"valence": 1}
    assert result.score == 1.2
    assert "weighted_selection_score" not in result.metadata


def test_xyz2omol_state_uses_aromatic_ring_loss_before_force_field(
    monkeypatch,
) -> None:
    no_metal = pybel.readstring(
        "xyz",
        """2
CO
C 2.0 0.0 0.0
O 3.2 0.0 0.0
""",
    )
    no_metal_state = ReconstructionState(
        omol=no_metal,
        given_charge=0,
        total_charge=0,
        total_radical_electrons=0,
        metadata={
            "organic_core_score": organic_force_field_energy(no_metal),
            "force_field_score_key": build_force_field_score_key(no_metal),
        },
    )
    attractive_but_too_high = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Li", 3, 1, 0, 0.0, 0.0, 0.0),),
        -1,
        0,
        combination_index=0,
    )
    fallback = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Li", 3, 2, 0, 0.0, 0.0, 0.0),),
        -2,
        0,
        combination_index=1,
    )

    monkeypatch.setattr(
        metal_search_module,
        "_group_candidates_by_target_dp",
        lambda *args, **kwargs: {
            (
                attractive_but_too_high.no_metal_charge_target,
                attractive_but_too_high.no_metal_radical_target,
            ): [attractive_but_too_high],
            (fallback.no_metal_charge_target, fallback.no_metal_radical_target): [fallback],
        },
    )
    _patch_no_metal_seed_reconstruction(monkeypatch, no_metal_state)

    def fake_prepare_candidate(candidate, no_metal_state, *, config=None):
        del config
        candidate.no_metal_state = no_metal_state
        valence = candidate.metal_states[0].valence
        candidate.score = 2.1 if valence == 1 else 1.0
        candidate.metadata["score"] = candidate.score
        candidate.metadata["metal_assignment_rank"] = 0.0
        if valence == 1:
            candidate.metadata["organic_aromatic_atom_count"] = 6
            candidate.metadata["organic_aromatic_ring_count"] = 1
            candidate.metadata["organic_conjugated_atom_count"] = 6
            candidate.metadata["organic_conjugated_bond_count"] = 6
            candidate.metadata["organic_max_conjugated_component_size"] = 6
            candidate.metadata["organic_radical_localization_penalty"] = 0.0
            candidate.metadata["organic_charge_localization_penalty"] = 0.0
        else:
            candidate.metadata["organic_aromatic_atom_count"] = 0
            candidate.metadata["organic_aromatic_ring_count"] = 0
            candidate.metadata["organic_conjugated_atom_count"] = 2
            candidate.metadata["organic_conjugated_bond_count"] = 1
            candidate.metadata["organic_max_conjugated_component_size"] = 2
            candidate.metadata["organic_radical_localization_penalty"] = 3.0
            candidate.metadata["organic_charge_localization_penalty"] = 4.0
        candidate.combined_omol = {"valence": valence}
        return candidate

    monkeypatch.setattr(
        metal_scoring_module,
        "_prepare_candidate_with_no_metal_state",
        fake_prepare_candidate,
    )
    result = with_metals_module.xyz2omol_state(
        """3
LiCO
Li 0.0 0.0 0.0
C 2.0 0.0 0.0
O 3.2 0.0 0.0
""",
        0,
        0,
    )

    assert result is not None
    assert result.score == 2.1
    assert "passes_organic_force_field_guard" not in result.metadata
    assert result.metadata["metal_discordance_structural_count"] == 0
    assert result.metadata["metal_discordance_aromatic_ring_deficit_count"] == 0
    assert result.combined_omol == {"valence": 1}


def test_xyz2omol_state_uses_combination_index_to_break_organic_score_ties(
    monkeypatch,
) -> None:
    no_metal = pybel.readstring(
        "xyz",
        """2
CO
C 2.0 0.0 0.0
O 3.2 0.0 0.0
""",
    )
    no_metal_state = ReconstructionState(
        omol=no_metal,
        given_charge=0,
        total_charge=0,
        total_radical_electrons=0,
        metadata={
            "organic_core_score": organic_force_field_energy(no_metal),
            "force_field_score_key": build_force_field_score_key(no_metal),
        },
    )
    preferred = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Li", 3, 1, 0, 0.0, 0.0, 0.0),),
        -1,
        0,
        combination_index=0,
    )
    preferred.metadata["metal_assignment_rank"] = 0.0
    fallback = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Li", 3, 2, 0, 0.0, 0.0, 0.0),),
        -2,
        0,
        combination_index=1,
    )
    fallback.metadata["metal_assignment_rank"] = -10.0

    monkeypatch.setattr(
        metal_search_module,
        "_group_candidates_by_target_dp",
        lambda *args, **kwargs: {
            (preferred.no_metal_charge_target, preferred.no_metal_radical_target): [preferred],
            (fallback.no_metal_charge_target, fallback.no_metal_radical_target): [fallback],
        },
    )
    _patch_no_metal_seed_reconstruction(monkeypatch, no_metal_state)

    def fake_combined_score(self):
        valence = self.metal_states[0].valence
        self.combined_omol = {"valence": valence}
        self.score = 1.0
        self.metadata["score"] = 1.0
        return self.score

    monkeypatch.setattr(MetalCandidateState, "combined_score", fake_combined_score)

    result = with_metals_module.xyz2omol_state(
        """3
LiCO
Li 0.0 0.0 0.0
C 2.0 0.0 0.0
O 3.2 0.0 0.0
""",
        0,
        0,
    )

    assert result is not None
    assert result.score == 1.0
    assert result.combined_omol == {"valence": 1}


def test_select_best_candidate_uses_aromatic_deficit_before_force_field() -> None:
    preferred = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Li", 3, 1, 0, 0.0, 0.0, 0.0),),
        -1,
        0,
        combination_index=0,
    )
    preferred.score = 2.0
    preferred.metadata.update(
        {
            "score": 2.0,
            "metal_discordance_count": 0,
            "metal_discordance_structural_count": 0,
            "organic_aromatic_atom_count": 6,
            "organic_aromatic_ring_count": 2,
            "organic_aromatic_stability_score": 1.6,
            "organic_conjugated_atom_count": 6,
            "organic_conjugated_bond_count": 6,
            "organic_max_conjugated_component_size": 6,
            "organic_radical_localization_penalty": 0.0,
            "organic_charge_localization_penalty": 0.0,
        }
    )
    fallback = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Li", 3, 2, 0, 0.0, 0.0, 0.0),),
        -2,
        0,
        combination_index=1,
    )
    fallback.score = 1.0
    fallback.metadata.update(
        {
            "score": 1.0,
            "metal_discordance_count": 0,
            "metal_discordance_structural_count": 0,
            "organic_aromatic_atom_count": 0,
            "organic_aromatic_ring_count": 0,
            "organic_aromatic_stability_score": 0.0,
            "organic_conjugated_atom_count": 2,
            "organic_conjugated_bond_count": 1,
            "organic_max_conjugated_component_size": 2,
            "organic_radical_localization_penalty": 0.0,
            "organic_charge_localization_penalty": 0.0,
        }
    )

    result = metal_scoring_module.select_best_candidate([preferred, fallback])

    assert result is preferred
    assert preferred.metadata["metal_discordance_structural_count"] == 0
    assert preferred.metadata["metal_discordance_aromatic_ring_deficit_count"] == 0
    assert preferred.metadata["metal_discordance_aromatic_stability_deficit"] == pytest.approx(0.0)
    assert preferred.metadata["metal_discordance_count"] == 0
    assert preferred.metadata["metal_discordance_max_aromatic_ring_count"] == 2
    assert preferred.metadata["metal_discordance_max_aromatic_stability_score"] == pytest.approx(
        1.6
    )
    assert fallback.metadata["metal_discordance_structural_count"] == 0
    assert fallback.metadata["metal_discordance_aromatic_ring_deficit_count"] == 2
    assert fallback.metadata["metal_discordance_aromatic_stability_deficit"] == pytest.approx(1.6)
    assert fallback.metadata["metal_discordance_count"] == pytest.approx(0.0)
    assert fallback.metadata["metal_discordance_max_aromatic_ring_count"] == 2
    assert fallback.metadata["metal_discordance_max_aromatic_stability_score"] == pytest.approx(1.6)


def test_select_best_candidate_preserves_conjugated_atoms_and_bonds_before_aromatic_gain() -> None:
    preferred = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Ir", 77, 3, 0, 0.0, 0.0, 0.0),),
        -3,
        0,
        combination_index=0,
    )
    over_reduced = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Ir", 77, 5, 0, 0.0, 0.0, 0.0),),
        -5,
        0,
        combination_index=1,
    )
    for candidate, topology, score in (
        (preferred, (11, 2, 1.7056, 12, 17, 19), 422.0),
        (over_reduced, (15, 3, 2.7056, 10, 15, 16), 400.0),
    ):
        aromatic_atoms, aromatic_rings, stability, max_component, atoms, bonds = topology
        candidate.score = score
        candidate.metadata.update(
            {
                "score": score,
                "metal_discordance_count": 0.0,
                "metal_discordance_structural_count": 0.0,
                "organic_aromatic_atom_count": aromatic_atoms,
                "organic_aromatic_ring_count": aromatic_rings,
                "organic_aromatic_stability_score": stability,
                "organic_max_conjugated_component_size": max_component,
                "organic_conjugated_atom_count": atoms,
                "organic_conjugated_bond_count": bonds,
                "organic_radical_localization_penalty": 0.0,
                "organic_charge_localization_penalty": 0.0,
            }
        )

    result = metal_scoring_module.select_best_candidate([over_reduced, preferred])

    assert result is preferred
    assert preferred.metadata["selection_key"][:5] == (0.0, 0, 0, 0, 4)
    assert preferred.metadata["metal_discordance_aromatic_atom_deficit_count"] == 4
    assert over_reduced.metadata["metal_discordance_conjugated_atom_deficit_count"] == 2
    assert over_reduced.metadata["metal_discordance_conjugated_bond_deficit_count"] == 3


def test_select_best_candidate_uses_aromatic_deficit_to_break_score_ties() -> None:
    over_oxidized = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Cr", 24, 6, 0, 0.0, 0.0, 0.0),),
        -6,
        0,
        combination_index=0,
    )
    neutral = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Cr", 24, 0, 0, 0.0, 0.0, 0.0),),
        0,
        0,
        combination_index=1,
    )
    for candidate, aromatic_ring_count, aromatic_stability_score in (
        (over_oxidized, 0, 0.0),
        (neutral, 2, 2.0),
    ):
        candidate.score = 10.0
        candidate.metadata.update(
            {
                "score": 10.0,
                "metal_discordance_count": 0.0,
                "metal_discordance_structural_count": 0.0,
                "organic_aromatic_atom_count": 0,
                "organic_aromatic_ring_count": aromatic_ring_count,
                "organic_aromatic_stability_score": aromatic_stability_score,
                "organic_conjugated_atom_count": 0,
                "organic_conjugated_bond_count": 0,
                "organic_max_conjugated_component_size": 0,
                "organic_radical_localization_penalty": 0.0,
                "organic_charge_localization_penalty": 0.0,
            }
        )

    result = metal_scoring_module.select_best_candidate([over_oxidized, neutral])

    assert result is neutral
    assert neutral.metadata["selection_key"][:7] == (0.0, 0, 0, 0, 0, 0, 0.0)


def test_select_best_candidate_uses_charge_localization_to_break_remaining_ties() -> None:
    over_oxidized = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Mn", 25, 5, 0, 0.0, 0.0, 0.0),),
        -4,
        0,
        combination_index=0,
    )
    preferred = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Mn", 25, 1, 0, 0.0, 0.0, 0.0),),
        0,
        0,
        combination_index=1,
    )
    for candidate, charge_localization_penalty in (
        (over_oxidized, 4.8),
        (preferred, 2.7),
    ):
        candidate.score = 10.0
        candidate.metadata.update(
            {
                "score": 10.0,
                "metal_discordance_count": 0.0,
                "metal_discordance_structural_count": 0.0,
                "organic_aromatic_atom_count": 6,
                "organic_aromatic_ring_count": 1,
                "organic_aromatic_stability_score": 1.0,
                "organic_conjugated_atom_count": 6,
                "organic_conjugated_bond_count": 6,
                "organic_max_conjugated_component_size": 6,
                "organic_radical_localization_penalty": 0.0,
                "organic_charge_localization_penalty": charge_localization_penalty,
            }
        )

    result = metal_scoring_module.select_best_candidate([over_oxidized, preferred])

    assert result is preferred
    assert preferred.metadata["selection_key"][-4:] == (0, 0, 10.0, 1)
    assert preferred.metadata["organic_charge_localization_margin_exceeded"] is False
    assert over_oxidized.metadata["organic_charge_localization_margin_exceeded"] is True


def test_select_best_candidate_uses_charge_localization_before_conjugation() -> None:
    lower_charge_penalty = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Ru", 44, 6, 0, 0.0, 0.0, 0.0),),
        -6,
        0,
        combination_index=0,
    )
    more_conjugated = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Ru", 44, 2, 0, 0.0, 0.0, 0.0),),
        -2,
        0,
        combination_index=1,
    )
    for candidate, conjugated_atom_count, charge_penalty in (
        (lower_charge_penalty, 5, 0.0),
        (more_conjugated, 6, 1.0),
    ):
        candidate.score = 10.0
        candidate.metadata.update(
            {
                "score": 10.0,
                "metal_discordance_count": 0.0,
                "metal_discordance_structural_count": 0.0,
                "organic_aromatic_atom_count": 0,
                "organic_aromatic_ring_count": 0,
                "organic_aromatic_stability_score": 0.0,
                "organic_conjugated_atom_count": conjugated_atom_count,
                "organic_conjugated_bond_count": conjugated_atom_count - 1,
                "organic_max_conjugated_component_size": conjugated_atom_count,
                "organic_radical_localization_penalty": 0.0,
                "organic_charge_localization_penalty": charge_penalty,
            }
        )

    result = metal_scoring_module.select_best_candidate([more_conjugated, lower_charge_penalty])

    assert result is lower_charge_penalty
    assert lower_charge_penalty.metadata["selection_key"][:4] == (0.0, 0, 1, 1)
    assert more_conjugated.metadata["selection_key"][:4] == (0.0, 1, 0, 0)


def test_select_best_candidate_uses_hyperconjugation_after_charge_localization_margin() -> None:
    more_hyperconjugated = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Zn", 30, 2, 0, 0.0, 0.0, 0.0),),
        -2,
        0,
        combination_index=0,
    )
    less_hyperconjugated = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Zn", 30, 2, 0, 0.0, 0.0, 0.0),),
        -2,
        0,
        combination_index=1,
    )
    for candidate, hyperconjugation_score, force_field_score in (
        (more_hyperconjugated, 6, 20.0),
        (less_hyperconjugated, 2, 10.0),
    ):
        candidate.score = force_field_score
        candidate.metadata.update(
            {
                "score": force_field_score,
                "metal_discordance_count": 0.0,
                "metal_discordance_structural_count": 0.0,
                "organic_aromatic_atom_count": 0,
                "organic_aromatic_ring_count": 0,
                "organic_aromatic_stability_score": 0.0,
                "organic_conjugated_atom_count": 0,
                "organic_conjugated_bond_count": 0,
                "organic_max_conjugated_component_size": 0,
                "organic_hyperconjugation_score": hyperconjugation_score,
                "organic_radical_localization_penalty": 0.0,
                "organic_charge_localization_penalty": 0.0,
            }
        )

    result = metal_scoring_module.select_best_candidate(
        [less_hyperconjugated, more_hyperconjugated]
    )

    assert result is more_hyperconjugated
    assert more_hyperconjugated.metadata["organic_hyperconjugation_deficit"] == 0
    assert less_hyperconjugated.metadata["organic_hyperconjugation_deficit"] == 4
    assert more_hyperconjugated.metadata["selection_key"][-4:] == (0, 0, 20.0, 0)
    assert less_hyperconjugated.metadata["selection_key"][-4:] == (0, 4, 10.0, 1)


@pytest.mark.parametrize(
    ("charge_localization_penalty", "expected_selected"),
    [(1.29, True), (1.3, False)],
)
def test_select_best_candidate_applies_charge_localization_margin(
    charge_localization_penalty: float,
    expected_selected: bool,
) -> None:
    minimum = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Zn", 30, 2, 0, 0.0, 0.0, 0.0),),
        -1,
        0,
        combination_index=0,
    )
    challenger = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Zn", 30, 4, 0, 0.0, 0.0, 0.0),),
        -3,
        0,
        combination_index=1,
    )
    for candidate, localization, score in (
        (minimum, 1.0, 10.0),
        (challenger, charge_localization_penalty, 8.0),
    ):
        candidate.score = score
        candidate.metadata.update(
            {
                "score": score,
                "metal_discordance_count": 0.0,
                "metal_discordance_structural_count": 0.0,
                "organic_aromatic_atom_count": 0,
                "organic_aromatic_ring_count": 0,
                "organic_aromatic_stability_score": 0.0,
                "organic_conjugated_atom_count": 0,
                "organic_conjugated_bond_count": 0,
                "organic_max_conjugated_component_size": 0,
                "organic_radical_localization_penalty": 0.0,
                "organic_charge_localization_penalty": localization,
            }
        )

    result = metal_scoring_module.select_best_candidate([minimum, challenger])

    assert (result is challenger) is expected_selected
    assert challenger.metadata["organic_charge_localization_margin_exceeded"] is (
        not expected_selected
    )


def test_select_best_candidate_uses_configured_charge_localization_margin() -> None:
    minimum = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Zn", 30, 2, 0, 0.0, 0.0, 0.0),),
        -1,
        0,
        combination_index=0,
    )
    challenger = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Zn", 30, 4, 0, 0.0, 0.0, 0.0),),
        -3,
        0,
        combination_index=1,
    )
    for candidate, localization, score in (
        (minimum, 1.0, 10.0),
        (challenger, 1.15, 8.0),
    ):
        candidate.score = score
        candidate.metadata.update(
            {
                "score": score,
                "metal_discordance_count": 0.0,
                "metal_discordance_structural_count": 0.0,
                "organic_aromatic_atom_count": 0,
                "organic_aromatic_ring_count": 0,
                "organic_aromatic_stability_score": 0.0,
                "organic_conjugated_atom_count": 0,
                "organic_conjugated_bond_count": 0,
                "organic_max_conjugated_component_size": 0,
                "organic_radical_localization_penalty": 0.0,
                "organic_charge_localization_penalty": localization,
            }
        )
    config = replace(
        MolGRConfig(),
        metal_scoring=replace(
            MolGRConfig().metal_scoring,
            charge_localization_selection_margin=0.1,
        ),
    )

    result = metal_scoring_module.select_best_candidate(
        [minimum, challenger],
        config=config,
    )

    assert result is minimum
    assert challenger.metadata["organic_charge_localization_selection_margin"] == pytest.approx(0.1)
    assert challenger.metadata["organic_charge_localization_margin_exceeded"] is True


def test_select_best_candidate_keeps_large_oxidation_state_jumps_charge_localized() -> None:
    minimum = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Pt", 78, 0, 0, 0.0, 0.0, 0.0),),
        0,
        0,
        combination_index=0,
    )
    challenger = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Pt", 78, 4, 0, 0.0, 0.0, 0.0),),
        -4,
        0,
        combination_index=1,
    )
    for candidate, localization, score in (
        (minimum, 1.0, 10.0),
        (challenger, 1.2, 1.0),
    ):
        candidate.score = score
        candidate.metadata.update(
            {
                "score": score,
                "metal_discordance_count": 0.0,
                "metal_discordance_structural_count": 0.0,
                "organic_aromatic_atom_count": 0,
                "organic_aromatic_ring_count": 0,
                "organic_aromatic_stability_score": 0.0,
                "organic_conjugated_atom_count": 0,
                "organic_conjugated_bond_count": 0,
                "organic_max_conjugated_component_size": 0,
                "organic_radical_localization_penalty": 0.0,
                "organic_charge_localization_penalty": localization,
            }
        )

    result = metal_scoring_module.select_best_candidate([challenger, minimum])

    assert result is minimum
    assert challenger.metadata["organic_charge_localization_margin_exceeded"] is True
    assert challenger.metadata["organic_charge_localization_reference_metal_valence_max_delta"] == 4
    assert challenger.metadata["organic_charge_localization_metal_valence_jump_exceeded"] is True


def test_select_best_candidate_uses_charge_localization_before_radical_localization() -> None:
    charge_localized = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Au", 79, 1, 0, 0.0, 0.0, 0.0),),
        -1,
        2,
        combination_index=4,
    )
    radical_localized = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Au", 79, 0, 0, 0.0, 0.0, 0.0),),
        0,
        2,
        combination_index=5,
    )
    for candidate, radical_penalty, charge_penalty in (
        (charge_localized, 0.0, 0.514),
        (radical_localized, 0.6, 0.0),
    ):
        candidate.score = 10.0
        candidate.metadata.update(
            {
                "score": 10.0,
                "metal_discordance_count": 0.0,
                "metal_discordance_structural_count": 0.0,
                "organic_aromatic_atom_count": 6,
                "organic_aromatic_ring_count": 1,
                "organic_aromatic_stability_score": 1.0,
                "organic_conjugated_atom_count": 6,
                "organic_conjugated_bond_count": 6,
                "organic_max_conjugated_component_size": 6,
                "organic_radical_localization_penalty": radical_penalty,
                "organic_charge_localization_penalty": charge_penalty,
            }
        )

    result = metal_scoring_module.select_best_candidate([radical_localized, charge_localized])

    assert result is radical_localized
    assert radical_localized.metadata["selection_key"][1] == 0
    assert charge_localized.metadata["selection_key"][1] == 1


@pytest.mark.parametrize("ligand_conjugated_atoms", [6, 8])
@pytest.mark.parametrize("reverse_candidates", [False, True])
def test_charge_reference_does_not_cross_ligand_radical_budgets(
    ligand_conjugated_atoms: int,
    reverse_candidates: bool,
) -> None:
    metal_spin = make_metal_candidate_state(
        (),
        tuple(MetalAtomPosition(i, "Mo", 42, 3, 1, 0.0, 0.0, 0.0) for i in (1, 2)),
        -6,
        0,
        combination_index=0,
    )
    ligand_spin = make_metal_candidate_state(
        (),
        tuple(MetalAtomPosition(i, "Mo", 42, 2, 0, 0.0, 0.0, 0.0) for i in (1, 2)),
        -4,
        2,
        combination_index=1,
    )
    for candidate, charge_penalty, radical_penalty, conjugated_atoms in (
        (metal_spin, 2.1, 0.0, 6),
        (ligand_spin, 1.4, 2.0, ligand_conjugated_atoms),
    ):
        candidate.score = 10.0
        candidate.metadata.update(
            {
                "metal_discordance_count": 0.0,
                "metal_discordance_structural_count": 0.0,
                "organic_conjugated_atom_count": conjugated_atoms,
                "organic_aromatic_atom_count": 6,
                "organic_aromatic_ring_count": 1,
                "organic_charge_localization_penalty": charge_penalty,
                "organic_radical_localization_penalty": radical_penalty,
            }
        )
    candidates = [metal_spin, ligand_spin]
    if reverse_candidates:
        candidates.reverse()
    selected = metal_scoring_module.select_best_candidate(candidates)

    # When topology ties, avoid creating ligand radicals just to reduce charge.
    # Ligand-centered spin can still win when it preserves more conjugation.
    assert selected is (metal_spin if ligand_conjugated_atoms == 6 else ligand_spin)
    assert metal_spin.metadata["organic_charge_localization_reference_penalty"] == 2.1
    assert ligand_spin.metadata["organic_charge_localization_reference_penalty"] == 1.4
    assert metal_spin.metadata["organic_charge_localization_margin_exceeded"] is False
    assert ligand_spin.metadata["organic_charge_localization_margin_exceeded"] is False


@pytest.mark.parametrize("visible_carbene", [False, True])
@pytest.mark.parametrize("singlet_aromatic_atoms", [5, 10, 15])
@pytest.mark.parametrize("reverse_candidates", [False, True])
def test_charge_reference_preserves_carbene_unless_aromaticity_changes(
    singlet_aromatic_atoms: int, reverse_candidates: bool, visible_carbene: bool
) -> None:
    anionic = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Pd", 46, 2, 0, 0.0, 0.0, 0.0),),
        -2,
        0,
        combination_index=0,
    )
    singlet = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Pd", 46, 0, 0, 0.0, 0.0, 0.0),),
        0,
        0,
        combination_index=1,
    )
    for candidate, charge_penalty, singlet_count, aromatic_atoms in (
        (anionic, 2.8, 0, 10),
        (singlet, 1.8, 1, singlet_aromatic_atoms),
    ):
        candidate.score = 10.0
        candidate.metadata.update(
            {
                "metal_discordance_count": 0.0,
                "metal_discordance_structural_count": 0.0,
                "organic_singlet_two_electron_center_count": singlet_count,
                "organic_visible_singlet_carbene_count": singlet_count if visible_carbene else 0,
                "organic_charge_localization_penalty": charge_penalty,
                "organic_aromatic_atom_count": aromatic_atoms,
                "organic_aromatic_ring_count": aromatic_atoms // 5,
            }
        )
    candidates = [anionic, singlet]
    if reverse_candidates:
        candidates.reverse()

    selected = metal_scoring_module.select_best_candidate(candidates)

    # At equal aromaticity, preserve the neutral singlet center. With different
    # aromaticity, occupation-specific references allow the aromatic anion to win.
    shares_reference = singlet_aromatic_atoms == 10 and visible_carbene
    expected_winner = singlet if singlet_aromatic_atoms == 15 or shares_reference else anionic
    assert selected is expected_winner
    expected_reference = 1.8 if shares_reference else 2.8
    assert anionic.metadata["organic_charge_localization_reference_penalty"] == expected_reference
    assert singlet.metadata["organic_charge_localization_reference_penalty"] == 1.8
    assert anionic.metadata["organic_charge_localization_margin_exceeded"] is shares_reference
    assert singlet.metadata["organic_charge_localization_margin_exceeded"] is False


def test_select_best_candidate_uses_aromatic_stability_before_force_field() -> None:
    benzene_like = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Li", 3, 1, 0, 0.0, 0.0, 0.0),),
        0,
        0,
        combination_index=0,
    )
    benzene_like.score = 2.0
    benzene_like.metadata.update(
        {
            "score": 2.0,
            "metal_discordance_count": 0,
            "metal_discordance_structural_count": 0,
            "organic_aromatic_atom_count": 6,
            "organic_aromatic_ring_count": 1,
            "organic_aromatic_stability_score": 1.0,
            "organic_conjugated_atom_count": 6,
            "organic_conjugated_bond_count": 6,
            "organic_max_conjugated_component_size": 6,
            "organic_radical_localization_penalty": 0.0,
            "organic_charge_localization_penalty": 0.0,
        }
    )
    hetero_like = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Li", 3, 1, 0, 0.0, 0.0, 0.0),),
        0,
        0,
        combination_index=1,
    )
    hetero_like.score = 1.0
    hetero_like.metadata.update(
        {
            "score": 1.0,
            "metal_discordance_count": 0,
            "metal_discordance_structural_count": 0,
            "organic_aromatic_atom_count": 6,
            "organic_aromatic_ring_count": 1,
            "organic_aromatic_stability_score": 0.82,
            "organic_conjugated_atom_count": 6,
            "organic_conjugated_bond_count": 6,
            "organic_max_conjugated_component_size": 6,
            "organic_radical_localization_penalty": 0.0,
            "organic_charge_localization_penalty": 0.0,
        }
    )

    result = metal_scoring_module.select_best_candidate([benzene_like, hetero_like])

    assert result is benzene_like
    assert benzene_like.metadata["metal_discordance_aromatic_ring_deficit_count"] == 0
    assert hetero_like.metadata["metal_discordance_aromatic_ring_deficit_count"] == 0
    assert hetero_like.metadata["metal_discordance_aromatic_stability_deficit"] == pytest.approx(
        0.18
    )


def test_select_best_candidate_preserves_fractional_discordance() -> None:
    fractional = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Mo", 42, -1, 0, 0.0, 0.0, 0.0),),
        0,
        0,
        combination_index=0,
    )
    fractional.score = 2.0
    fractional.metadata.update(
        {
            "score": 2.0,
            "metal_discordance_count": 0.5,
            "metal_discordance_structural_count": 0.5,
            "organic_aromatic_atom_count": 0,
            "organic_aromatic_ring_count": 0,
            "organic_conjugated_atom_count": 0,
            "organic_conjugated_bond_count": 0,
            "organic_max_conjugated_component_size": 0,
            "organic_radical_localization_penalty": 0.0,
            "organic_charge_localization_penalty": 0.0,
        }
    )
    full = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Li", 3, 1, 0, 0.0, 0.0, 0.0),),
        0,
        0,
        combination_index=1,
    )
    full.score = 1.0
    full.metadata.update(
        {
            "score": 1.0,
            "metal_discordance_count": 1.0,
            "metal_discordance_structural_count": 1.0,
            "organic_aromatic_atom_count": 0,
            "organic_aromatic_ring_count": 0,
            "organic_conjugated_atom_count": 0,
            "organic_conjugated_bond_count": 0,
            "organic_max_conjugated_component_size": 0,
            "organic_radical_localization_penalty": 0.0,
            "organic_charge_localization_penalty": 0.0,
        }
    )

    result = metal_scoring_module.select_best_candidate([fractional, full])

    assert result is fractional
    assert fractional.metadata["passes_metal_discordance_filter"] is True
    assert full.metadata["passes_metal_discordance_filter"] is False
    assert fractional.metadata["metal_discordance_count"] == pytest.approx(0.5)


def test_metal_discordance_counts_charge_asymmetry_only_for_repeated_components() -> None:
    asymmetric = pybel.readstring("smi", "[CH3-].[CH3]")
    symmetric = pybel.readstring("smi", "[CH3-].[CH3-]")
    unrelated = pybel.readstring("smi", "[CH3-].N")

    assert metal_scoring_module._repeated_component_charge_asymmetry_count(asymmetric.OBMol) == 1
    assert metal_scoring_module._repeated_component_charge_asymmetry_count(symmetric.OBMol) == 0
    assert metal_scoring_module._repeated_component_charge_asymmetry_count(unrelated.OBMol) == 0


def test_metal_discordance_haptic_ring_reduction_is_ring_and_metal_local() -> None:
    ring = pybel.readstring("smi", "C1CCCCC1")
    ring.OBMol.GetAtom(1).SetFormalCharge(-1)

    assert (
        metal_scoring_module._haptic_arene_reduction_count(
            ring.OBMol,
            ((ring.OBMol.GetAtom(1), ring.OBMol.GetAtom(2)),),
        )
        == 0
    )
    assert (
        metal_scoring_module._haptic_arene_reduction_count(
            ring.OBMol,
            (
                (ring.OBMol.GetAtom(1), ring.OBMol.GetAtom(2)),
                (ring.OBMol.GetAtom(3), ring.OBMol.GetAtom(4)),
            ),
        )
        == 0
    )
    assert (
        metal_scoring_module._haptic_arene_reduction_count(
            ring.OBMol,
            ((ring.OBMol.GetAtom(1), ring.OBMol.GetAtom(2), ring.OBMol.GetAtom(3)),),
        )
        == 1
    )


@pytest.mark.parametrize("smiles", ["C1=CC=CC=[C-]1", "C1=CC=C[C-]1"])
def test_metal_discordance_haptic_ring_reduction_preserves_complete_pi_rings(
    smiles: str,
) -> None:
    ring = pybel.readstring("smi", smiles)

    assert (
        metal_scoring_module._haptic_arene_reduction_count(
            ring.OBMol,
            ((ring.OBMol.GetAtom(1), ring.OBMol.GetAtom(2), ring.OBMol.GetAtom(3)),),
        )
        == 0
    )


def test_metal_discordance_coordination_geometry_is_metal_state_specific() -> None:
    square_planar = pybel.readstring(
        "xyz",
        """4
square planar donors
N 1.0 0.0 0.0
N -1.0 0.0 0.0
N 0.0 1.0 0.0
N 0.0 -1.0 0.0
""",
    )
    linear = pybel.readstring(
        "xyz",
        """2
linear donors
S 1.0 0.0 0.0
S -1.0 0.0 0.0
""",
    )
    square_planar_atoms = tuple(square_planar.OBMol.GetAtom(idx) for idx in range(1, 5))
    linear_atoms = tuple(linear.OBMol.GetAtom(idx) for idx in range(1, 3))

    assert (
        metal_scoring_module._coordination_geometry_discordance_count(
            (square_planar_atoms,),
            (MetalAtomPosition(1, "Pd", 46, 4, 0, 0.0, 0.0, 0.0),),
        )
        == 1
    )
    assert (
        metal_scoring_module._coordination_geometry_discordance_count(
            (square_planar_atoms,),
            (MetalAtomPosition(1, "Pd", 46, 2, 0, 0.0, 0.0, 0.0),),
        )
        == 0
    )
    assert (
        metal_scoring_module._coordination_geometry_discordance_count(
            (linear_atoms,),
            (MetalAtomPosition(1, "Au", 79, 3, 0, 0.0, 0.0, 0.0),),
        )
        == 1
    )
    assert (
        metal_scoring_module._coordination_geometry_discordance_count(
            (linear_atoms,),
            (MetalAtomPosition(1, "Au", 79, 1, 0, 0.0, 0.0, 0.0),),
        )
        == 0
    )


@pytest.mark.parametrize(
    ("organic_charge", "expected_same", "expected_opposite"),
    [(-1, 0, 1), (1, 1, 0)],
)
def test_metal_discordance_adjacent_double_charge_sign_is_relative_to_metal_center(
    organic_charge: int,
    expected_same: int,
    expected_opposite: int,
) -> None:
    no_metal = pybel.readstring(
        "xyz",
        """2
CC
C 4.0 0.0 0.0
C 5.2 0.0 0.0
""",
    )
    no_metal.OBMol.GetAtom(1).SetFormalCharge(organic_charge)
    no_metal.OBMol.GetAtom(2).SetFormalCharge(organic_charge)
    no_metal_state = ReconstructionState(
        omol=no_metal,
        given_charge=0,
        total_charge=2 * organic_charge,
        total_radical_electrons=0,
        metadata={
            "organic_core_score": organic_force_field_energy(no_metal),
            "force_field_score_key": build_force_field_score_key(no_metal),
        },
    )
    candidate = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Li", 3, 1, 0, 0.0, 0.0, 0.0),),
        2 * organic_charge,
        0,
        combination_index=0,
    )

    scored = metal_scoring_module._prepare_candidate_with_no_metal_state(
        candidate,
        no_metal_state,
    )

    assert scored.metadata["metal_discordance_outer_or_invisible_adjacent_double_charge_count"] == 1
    assert (
        scored.metadata[
            "metal_discordance_outer_or_invisible_adjacent_same_sign_double_charge_count"
        ]
        == expected_same
    )
    assert (
        scored.metadata[
            "metal_discordance_outer_or_invisible_adjacent_opposite_sign_double_charge_count"
        ]
        == expected_opposite
    )


@pytest.mark.parametrize(
    ("distance", "expected_count"),
    [(2.4, 1), (2.6, 1)],
)
def test_metal_discordance_inner_same_sign_charge_uses_coordination_radius(
    distance: float,
    expected_count: int,
) -> None:
    no_metal = pybel.readstring(
        "xyz",
        f"""1
N
N {distance} 0.0 0.0
""",
    )
    no_metal.OBMol.GetAtom(1).SetFormalCharge(1)
    no_metal_state = ReconstructionState(
        omol=no_metal,
        given_charge=0,
        total_charge=1,
        total_radical_electrons=0,
        metadata={
            "organic_core_score": organic_force_field_energy(no_metal),
            "force_field_score_key": build_force_field_score_key(no_metal),
        },
    )
    candidate = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Cd", 48, 2, 0, 0.0, 0.0, 0.0),),
        1,
        0,
        combination_index=0,
    )

    scored = metal_scoring_module._prepare_candidate_with_no_metal_state(
        candidate,
        no_metal_state,
    )

    assert (
        scored.metadata["metal_discordance_inner_visible_same_sign_charge_count"] == expected_count
    )


@pytest.mark.parametrize(
    (
        "metal_valence",
        "inner_atom_symbol",
        "inner_atom_charge",
        "adjacent_atom_symbol",
        "adjacent_atom_charge",
        "expected_count",
    ),
    [
        (2, "N", 1, "B", -1, 1),
        (-1, "O", -1, "N", 1, 1),
        (2, "N", 1, "O", -1, 0),
        (0, "N", 1, "O", -1, 0),
    ],
)
def test_metal_discordance_inner_same_sign_charge_exempts_local_anionic_polarization(
    metal_valence: int,
    inner_atom_symbol: str,
    inner_atom_charge: int,
    adjacent_atom_symbol: str,
    adjacent_atom_charge: int,
    expected_count: int,
) -> None:
    no_metal = pybel.readstring(
        "xyz",
        f"""2
{inner_atom_symbol}{adjacent_atom_symbol}
{inner_atom_symbol} 2.0 0.0 0.0
{adjacent_atom_symbol} 3.2 0.0 0.0
""",
    )
    if no_metal.OBMol.GetBond(1, 2) is None:
        no_metal.OBMol.AddBond(1, 2, 1)
    no_metal.OBMol.GetAtom(1).SetFormalCharge(inner_atom_charge)
    no_metal.OBMol.GetAtom(2).SetFormalCharge(adjacent_atom_charge)
    total_charge = inner_atom_charge + adjacent_atom_charge
    no_metal_state = ReconstructionState(
        omol=no_metal,
        given_charge=0,
        total_charge=total_charge,
        total_radical_electrons=0,
        metadata={
            "organic_core_score": organic_force_field_energy(no_metal),
            "force_field_score_key": build_force_field_score_key(no_metal),
        },
    )
    candidate = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Cd", 48, metal_valence, 0, 0.0, 0.0, 0.0),),
        total_charge,
        0,
        combination_index=0,
    )

    scored = metal_scoring_module._prepare_candidate_with_no_metal_state(
        candidate,
        no_metal_state,
    )

    assert (
        scored.metadata["metal_discordance_inner_visible_same_sign_charge_count"] == expected_count
    )


def test_local_anionic_polarization_cancellation_covers_perchlorate() -> None:
    perchlorate = pybel.readstring("smi", "[O-][Cl+3]([O-])([O-])[O-]")
    chlorine = perchlorate.OBMol.GetAtom(2)

    assert metal_scoring_module._has_adjacent_anionic_polarization_cancellation(chlorine)


@pytest.mark.parametrize(
    ("formal_charge", "expected_count"),
    [(1, 1), (-1, 0)],
)
def test_metal_discordance_zero_valence_counts_inner_positive_charge(
    formal_charge: int,
    expected_count: int,
) -> None:
    no_metal = pybel.readstring(
        "xyz",
        """1
N
N 2.4 0.0 0.0
""",
    )
    no_metal.OBMol.GetAtom(1).SetFormalCharge(formal_charge)
    no_metal_state = ReconstructionState(
        omol=no_metal,
        given_charge=0,
        total_charge=formal_charge,
        total_radical_electrons=0,
        metadata={
            "organic_core_score": organic_force_field_energy(no_metal),
            "force_field_score_key": build_force_field_score_key(no_metal),
        },
    )
    candidate = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Cd", 48, 0, 0, 0.0, 0.0, 0.0),),
        formal_charge,
        0,
        combination_index=0,
    )

    scored = metal_scoring_module._prepare_candidate_with_no_metal_state(
        candidate,
        no_metal_state,
    )

    assert (
        scored.metadata["metal_discordance_inner_visible_same_sign_charge_count"] == expected_count
    )
    expected_zero_valent_cation_count = 0
    assert (
        scored.metadata["metal_discordance_zero_valent_metals_with_organic_cation_count"]
        == expected_zero_valent_cation_count
    )
    expected_unsaturated_cation_count = 1 if formal_charge > 0 else 0
    assert (
        scored.metadata["metal_discordance_unsaturated_organic_cation_count"]
        == expected_unsaturated_cation_count
    )
    assert (
        scored.metadata["metal_discordance_structural_count"]
        == expected_count + expected_zero_valent_cation_count + expected_unsaturated_cation_count
    )


def test_metal_discordance_all_zero_valence_exempts_organic_cation_when_global_charge_positive() -> (
    None
):
    no_metal = pybel.readstring(
        "xyz",
        """1
N
N 5.0 0.0 0.0
""",
    )
    no_metal.OBMol.GetAtom(1).SetFormalCharge(1)
    no_metal_state = ReconstructionState(
        omol=no_metal,
        given_charge=0,
        total_charge=1,
        total_radical_electrons=0,
        metadata={
            "organic_core_score": organic_force_field_energy(no_metal),
            "force_field_score_key": build_force_field_score_key(no_metal),
        },
    )
    candidate = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Cd", 48, 0, 0, 0.0, 0.0, 0.0),),
        1,
        0,
        combination_index=0,
    )

    scored = metal_scoring_module._prepare_candidate_with_no_metal_state(
        candidate,
        no_metal_state,
    )

    assert scored.metadata["metal_discordance_inner_visible_same_sign_charge_count"] == 0
    assert scored.metadata["metal_discordance_zero_valent_metals_with_organic_cation_count"] == 0
    assert scored.metadata["metal_discordance_unsaturated_organic_cation_count"] == 1
    assert scored.metadata["metal_discordance_structural_count"] == 1


@pytest.mark.parametrize(
    (
        "xyz_block",
        "positive_atom_idx",
        "metal_valence",
        "expected_unsaturated_cation_count",
    ),
    [
        pytest.param(
            """4
NH3
N 5.0 0.0 0.0
H 5.0 0.0 1.0
H 5.0 1.0 0.0
H 6.0 0.0 0.0
""",
            1,
            1,
            1,
            id="under-valent-ammonium",
        ),
        pytest.param(
            """5
NH4
N 5.0 0.0 0.0
H 5.0 0.0 1.0
H 5.0 1.0 0.0
H 6.0 0.0 0.0
H 4.0 0.0 0.0
""",
            1,
            1,
            0,
            id="saturated-onium",
        ),
        pytest.param(
            """2
CO
C 5.0 0.0 0.0
O 6.2 0.0 0.0
""",
            1,
            1,
            1,
            id="bond-order-exceeds-degree",
        ),
        pytest.param(
            """4
NH3
N 5.0 0.0 0.0
H 5.0 0.0 1.0
H 5.0 1.0 0.0
H 6.0 0.0 0.0
""",
            1,
            -1,
            1,
            id="negative-metal-candidate",
        ),
    ],
)
def test_metal_discordance_counts_unsaturated_organic_cation_for_any_metal_valence(
    xyz_block: str,
    positive_atom_idx: int,
    metal_valence: int,
    expected_unsaturated_cation_count: int,
) -> None:
    no_metal = pybel.readstring("xyz", xyz_block)
    no_metal.OBMol.GetAtom(positive_atom_idx).SetFormalCharge(1)
    no_metal_state = ReconstructionState(
        omol=no_metal,
        given_charge=0,
        total_charge=1,
        total_radical_electrons=0,
        metadata={
            "organic_core_score": organic_force_field_energy(no_metal),
            "force_field_score_key": build_force_field_score_key(no_metal),
        },
    )
    candidate = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Cd", 48, metal_valence, 0, 0.0, 0.0, 0.0),),
        1,
        0,
        combination_index=0,
    )

    scored = metal_scoring_module._prepare_candidate_with_no_metal_state(
        candidate,
        no_metal_state,
    )

    assert (
        scored.metadata["metal_discordance_unsaturated_organic_cation_count"]
        == expected_unsaturated_cation_count
    )


def test_metal_discordance_counts_three_coordinate_carbocation() -> None:
    no_metal = pybel.readstring("smi", "CC[C+]C")
    no_metal_state = ReconstructionState(
        omol=no_metal,
        given_charge=0,
        total_charge=1,
        total_radical_electrons=0,
        metadata={
            "organic_core_score": organic_force_field_energy(no_metal),
            "force_field_score_key": build_force_field_score_key(no_metal),
        },
    )
    candidate = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Ir", 77, -1, 0, 0.0, 0.0, 0.0),),
        0,
        0,
        combination_index=0,
    )

    scored = metal_scoring_module._prepare_candidate_with_no_metal_state(
        candidate,
        no_metal_state,
    )

    assert scored.metadata["metal_discordance_unsaturated_organic_cation_count"] == 1


def test_metal_discordance_does_not_count_valence_three_oxygen_cation() -> None:
    molecule = pybel.readstring("smi", "[O+](C)=C")
    oxygen = molecule.OBMol.GetAtom(1)
    assert oxygen.GetTotalDegree() == 2
    assert oxygen.GetTotalValence() == 3
    assert not metal_scoring_module._is_unsaturated_organic_cation(oxygen)


def test_metal_discordance_exempts_aromatic_unsaturated_organic_cation() -> None:
    no_metal = pybel.readstring("smi", "[nH+]1ccccc1")
    positive_atom = next(
        atom for atom in no_metal.atoms if int(atom.OBAtom.GetFormalCharge()) > 0
    ).OBAtom
    assert positive_atom.IsAromatic()
    assert not metal_scoring_module._is_unsaturated_organic_cation(positive_atom)
    no_metal_state = ReconstructionState(
        omol=no_metal,
        given_charge=0,
        total_charge=1,
        total_radical_electrons=0,
        metadata={
            "organic_core_score": organic_force_field_energy(no_metal),
            "force_field_score_key": build_force_field_score_key(no_metal),
        },
    )
    candidate = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Cd", 48, 1, 0, 100.0, 0.0, 0.0),),
        1,
        0,
        combination_index=0,
    )

    scored = metal_scoring_module._prepare_candidate_with_no_metal_state(
        candidate,
        no_metal_state,
    )

    assert scored.metadata["metal_discordance_unsaturated_organic_cation_count"] == 0
    assert scored.metadata["metal_discordance_structural_count"] == 0


def test_metal_discordance_counts_unsaturated_cation_when_only_global_charge_is_zero() -> None:
    no_metal = pybel.readstring(
        "xyz",
        """4
NOH2
N 5.0 0.0 0.0
O 15.0 0.0 0.0
H 5.0 1.0 0.0
H 5.0 0.0 1.0
""",
    )
    no_metal.OBMol.GetAtom(1).SetFormalCharge(1)
    no_metal.OBMol.GetAtom(2).SetFormalCharge(-1)
    no_metal_state = ReconstructionState(
        omol=no_metal,
        given_charge=0,
        total_charge=0,
        total_radical_electrons=0,
        metadata={
            "organic_core_score": organic_force_field_energy(no_metal),
            "force_field_score_key": build_force_field_score_key(no_metal),
        },
    )
    candidate = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Cd", 48, 0, 0, 100.0, 0.0, 0.0),),
        0,
        0,
        combination_index=0,
    )

    scored = metal_scoring_module._prepare_candidate_with_no_metal_state(
        candidate,
        no_metal_state,
    )

    assert scored.metadata["metal_discordance_zero_valent_metals_with_organic_cation_count"] == 1
    assert scored.metadata["metal_discordance_unsaturated_organic_cation_count"] == 1
    assert scored.metadata["metal_discordance_structural_count"] == 2


def test_metal_discordance_exempts_locally_zwitterionic_unsaturated_cation() -> None:
    no_metal = pybel.readstring(
        "xyz",
        """5
NOH2_O
N 5.0 0.0 0.0
O 6.2 0.0 0.0
H 5.0 1.0 0.0
H 5.0 0.0 1.0
O 15.0 0.0 0.0
""",
    )
    for begin_idx, end_idx in ((1, 2), (1, 3), (1, 4)):
        if no_metal.OBMol.GetBond(begin_idx, end_idx) is None:
            no_metal.OBMol.AddBond(begin_idx, end_idx, 1)
    no_metal.OBMol.GetAtom(1).SetFormalCharge(1)
    no_metal.OBMol.GetAtom(2).SetFormalCharge(-1)
    no_metal.OBMol.GetAtom(5).SetFormalCharge(-1)
    no_metal_state = ReconstructionState(
        omol=no_metal,
        given_charge=0,
        total_charge=-1,
        total_radical_electrons=0,
        metadata={
            "organic_core_score": organic_force_field_energy(no_metal),
            "force_field_score_key": build_force_field_score_key(no_metal),
        },
    )
    candidate = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Cd", 48, 0, 0, 100.0, 0.0, 0.0),),
        -1,
        0,
        combination_index=0,
    )

    scored = metal_scoring_module._prepare_candidate_with_no_metal_state(
        candidate,
        no_metal_state,
    )

    assert scored.metadata["metal_discordance_zero_valent_metals_with_organic_cation_count"] == 0
    assert scored.metadata["metal_discordance_unsaturated_organic_cation_count"] == 0
    assert scored.metadata["metal_discordance_structural_count"] == 0


def test_metal_discordance_exempts_high_valent_nonmetal_cation_with_excess_adjacent_anions() -> (
    None
):
    no_metal = pybel.readstring(
        "smi",
        "[O-][N+](=O)[O-] nitrate",
    )
    no_metal_state = ReconstructionState(
        omol=no_metal,
        given_charge=0,
        total_charge=-1,
        total_radical_electrons=0,
        metadata={
            "organic_core_score": organic_force_field_energy(no_metal),
            "force_field_score_key": build_force_field_score_key(no_metal),
        },
    )
    candidate = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Zn", 30, 2, 0, 100.0, 0.0, 0.0),),
        -1,
        0,
        combination_index=0,
    )

    scored = metal_scoring_module._prepare_candidate_with_no_metal_state(
        candidate,
        no_metal_state,
    )

    assert scored.metadata["metal_discordance_zero_valent_metals_with_organic_cation_count"] == 0
    assert scored.metadata["metal_discordance_unsaturated_organic_cation_count"] == 0
    assert scored.metadata["metal_discordance_structural_count"] == 0


def test_metal_inner_sphere_radius_uses_coordination_tolerance_config() -> None:
    no_metal = pybel.readstring(
        "xyz",
        """1
N
N 2.10 0.0 0.0
""",
    )
    set_unpaired_electron_count(no_metal.OBMol.GetAtom(1), 2)
    no_metal_state = ReconstructionState(
        omol=no_metal,
        given_charge=0,
        total_charge=0,
        total_radical_electrons=1,
        metadata={
            "organic_core_score": organic_force_field_energy(no_metal),
            "force_field_score_key": build_force_field_score_key(no_metal),
        },
    )
    candidate = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Zn", 30, 0, 0, 0.0, 0.0, 0.0),),
        0,
        1,
        combination_index=0,
    )
    tight_config = replace(
        MolGRConfig(),
        metal_scoring=replace(
            MolGRConfig().metal_scoring,
            metal_coordination_extra_tolerance_angstrom=0.10,
        ),
    )
    loose_config = replace(
        MolGRConfig(),
        metal_scoring=replace(
            MolGRConfig().metal_scoring,
            metal_coordination_extra_tolerance_angstrom=0.35,
        ),
    )

    tight_scored = metal_scoring_module._prepare_candidate_with_no_metal_state(
        candidate,
        no_metal_state,
        config=tight_config,
    )
    loose_scored = metal_scoring_module._prepare_candidate_with_no_metal_state(
        candidate,
        no_metal_state,
        config=loose_config,
    )

    assert tight_scored.metadata["metal_discordance_inner_visible_diradical_count"] == 0
    assert loose_scored.metadata["metal_discordance_inner_visible_diradical_count"] == 1


def test_metal_inner_sphere_radius_uses_metal_access_radius_scale_for_scoring() -> None:
    no_metal = pybel.readstring(
        "xyz",
        """1
N
N 2.22 0.0 0.0
""",
    )
    set_unpaired_electron_count(no_metal.OBMol.GetAtom(1), 2)
    no_metal_state = ReconstructionState(
        omol=no_metal,
        given_charge=0,
        total_charge=0,
        total_radical_electrons=1,
        metadata={
            "organic_core_score": organic_force_field_energy(no_metal),
            "force_field_score_key": build_force_field_score_key(no_metal),
        },
    )
    candidate = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Zn", 30, 0, 0, 0.0, 0.0, 0.0),),
        0,
        1,
        combination_index=0,
    )
    unscaled_config = replace(
        MolGRConfig(),
        metal_scoring=replace(
            MolGRConfig().metal_scoring,
            metal_access_radius_scale=1.0,
            metal_coordination_extra_tolerance_angstrom=0.10,
        ),
    )
    scaled_config = replace(
        MolGRConfig(),
        metal_scoring=replace(
            MolGRConfig().metal_scoring,
            metal_access_radius_scale=1.10,
            metal_coordination_extra_tolerance_angstrom=0.10,
        ),
    )

    unscaled_scored = metal_scoring_module._prepare_candidate_with_no_metal_state(
        candidate,
        no_metal_state,
        config=unscaled_config,
    )
    scaled_scored = metal_scoring_module._prepare_candidate_with_no_metal_state(
        candidate,
        no_metal_state,
        config=scaled_config,
    )

    assert unscaled_scored.metadata["metal_discordance_inner_visible_diradical_count"] == 0
    assert scaled_scored.metadata["metal_discordance_inner_visible_diradical_count"] == 1


@pytest.mark.parametrize(
    "atomic_symbol",
    ["N", "O", "P", "S", "Cl", "Br", "I"],
)
def test_metal_discordance_counts_explicit_diradicals_for_every_element(
    atomic_symbol: str,
) -> None:
    no_metal = pybel.readstring(
        "xyz",
        f"""1
{atomic_symbol}
{atomic_symbol} 2.0 0.0 0.0
""",
    )
    set_unpaired_electron_count(no_metal.OBMol.GetAtom(1), 2)
    no_metal_state = ReconstructionState(
        omol=no_metal,
        given_charge=0,
        total_charge=0,
        total_radical_electrons=1,
        metadata={
            "organic_core_score": organic_force_field_energy(no_metal),
            "force_field_score_key": build_force_field_score_key(no_metal),
        },
    )
    candidate = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Zn", 30, 0, 0, 0.0, 0.0, 0.0),),
        0,
        1,
        combination_index=0,
    )

    scored = metal_scoring_module._prepare_candidate_with_no_metal_state(
        candidate,
        no_metal_state,
    )

    assert scored.metadata["metal_discordance_inner_visible_diradical_count"] == 1


@pytest.mark.parametrize("atomic_symbol", ["P", "S", "Cl", "Br", "I"])
def test_metal_discordance_does_not_count_explicit_lone_pairs(atomic_symbol: str) -> None:
    no_metal = pybel.readstring(
        "xyz",
        f"""1
{atomic_symbol}
{atomic_symbol} 2.0 0.0 0.0
""",
    )
    set_lone_pair_count(no_metal.OBMol.GetAtom(1), 1)
    no_metal_state = ReconstructionState(
        omol=no_metal,
        given_charge=0,
        total_charge=0,
        total_radical_electrons=0,
        metadata={
            "organic_core_score": organic_force_field_energy(no_metal),
            "force_field_score_key": build_force_field_score_key(no_metal),
        },
    )
    candidate = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Zn", 30, 0, 0, 0.0, 0.0, 0.0),),
        0,
        0,
        combination_index=0,
    )

    scored = metal_scoring_module._prepare_candidate_with_no_metal_state(
        candidate,
        no_metal_state,
    )

    assert scored.metadata["metal_discordance_inner_visible_diradical_count"] == 0


@pytest.mark.parametrize(
    ("center_count", "expected_excess_count"),
    [(1, 0), (2, 1)],
)
def test_metal_discordance_counts_only_excess_visible_singlet_two_electron_centers(
    center_count: int,
    expected_excess_count: int,
) -> None:
    no_metal = pybel.readstring("smi", ".".join("[CH2]" for _ in range(center_count)))
    for atom, coordinates in zip(
        no_metal.atoms,
        ((2.0, 0.0, 0.0), (0.0, 2.0, 0.0)),
    ):
        atom.OBAtom.SetVector(*coordinates)
        set_unpaired_electron_count(atom.OBAtom, 0)
        set_lone_pair_count(atom.OBAtom, 1)
    no_metal_state = ReconstructionState(
        omol=no_metal,
        given_charge=0,
        total_charge=0,
        total_radical_electrons=0,
    )
    candidate = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "W", 74, 2, 0, 0.0, 0.0, 0.0),),
        0,
        0,
        combination_index=0,
    )
    candidate.no_metal_state = no_metal_state

    metal_scoring_module._annotate_candidate_discordance_features(candidate)

    assert (
        candidate.metadata["metal_discordance_excess_visible_singlet_two_electron_center_count"]
        == expected_excess_count
    )


def test_metal_discordance_counts_bent_but_not_linear_ring_allene() -> None:
    no_metal = pybel.readstring("smi", "C1=C=CCC1")
    coordinates = (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (-0.5, 0.866, 0.0),
        (-1.0, 1.7, 0.0),
        (0.5, 1.0, 0.0),
    )
    for atom, position in zip(no_metal.atoms, coordinates):
        atom.OBAtom.SetVector(*position)

    assert metal_scoring_module._bent_cumulated_ring_allene_count(no_metal.OBMol) == 1

    no_metal.OBMol.GetAtom(3).SetVector(-1.0, 0.05, 0.0)

    assert metal_scoring_module._bent_cumulated_ring_allene_count(no_metal.OBMol) == 0


@pytest.mark.parametrize(
    ("metal_valence", "expected_negative_metal_count"),
    [(-1, 1), (-2, 2)],
)
def test_metal_discordance_counts_negative_metal_without_charge_balance_exception(
    metal_valence: int,
    expected_negative_metal_count: int,
) -> None:
    no_metal = pybel.readstring(
        "xyz",
        """1
C
C 5.0 0.0 0.0
""",
    )
    no_metal_state = ReconstructionState(
        omol=no_metal,
        given_charge=0,
        total_charge=0,
        total_radical_electrons=0,
        metadata={
            "organic_core_score": organic_force_field_energy(no_metal),
            "force_field_score_key": build_force_field_score_key(no_metal),
        },
    )
    candidate = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Mo", 42, metal_valence, 0, 0.0, 0.0, 0.0),),
        0,
        0,
        combination_index=0,
    )

    scored = metal_scoring_module._prepare_candidate_with_no_metal_state(
        candidate,
        no_metal_state,
    )

    assert (
        scored.metadata["metal_discordance_negative_metal_count"] == expected_negative_metal_count
    )
    assert (
        scored.metadata["metal_discordance_negative_metal_outer_sphere_cation_exception"] is False
    )
    assert (
        scored.metadata["metal_discordance_negative_metal_positive_metal_counterion_exception"]
        is False
    )
    assert scored.metadata["metal_discordance_negative_metal_penalty"] == pytest.approx(
        0.5 * expected_negative_metal_count
    )
    assert scored.metadata["metal_discordance_structural_count"] == pytest.approx(
        0.5 * expected_negative_metal_count
        + scored.metadata["metal_discordance_unsaturated_organic_cation_count"]
    )
    assert scored.metadata["metal_discordance_count"] == pytest.approx(
        0.5 * expected_negative_metal_count
    )


@pytest.mark.parametrize(
    (
        "atomic_symbol",
        "atomic_num",
        "cation_distance",
        "expected_negative_metal_count",
        "expected_outer_sphere_exception",
    ),
    [("H", 1, 1.8, 1, False), ("H", 1, 6.0, 0, True), ("N", 7, 6.0, 1, False)],
)
def test_metal_discordance_negative_metal_only_uses_outer_sphere_proton_exception(
    atomic_symbol: str,
    atomic_num: int,
    cation_distance: float,
    expected_negative_metal_count: int,
    expected_outer_sphere_exception: bool,
) -> None:
    no_metal = pybel.readstring(
        "xyz",
        f"""1
{atomic_symbol}
{atomic_symbol} {cation_distance} 0.0 0.0
""",
    )
    no_metal.OBMol.GetAtom(1).SetFormalCharge(1)
    assert no_metal.OBMol.GetAtom(1).GetAtomicNum() == atomic_num
    no_metal_state = ReconstructionState(
        omol=no_metal,
        given_charge=0,
        total_charge=1,
        total_radical_electrons=0,
        metadata={
            "organic_core_score": organic_force_field_energy(no_metal),
            "force_field_score_key": build_force_field_score_key(no_metal),
        },
    )
    candidate = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Mo", 42, -1, 0, 0.0, 0.0, 0.0),),
        1,
        0,
        combination_index=0,
    )

    scored = metal_scoring_module._prepare_candidate_with_no_metal_state(
        candidate,
        no_metal_state,
    )

    assert (
        scored.metadata["metal_discordance_negative_metal_count"] == expected_negative_metal_count
    )
    assert (
        scored.metadata["metal_discordance_negative_metal_outer_sphere_cation_exception"]
        is expected_outer_sphere_exception
    )
    assert (
        scored.metadata["metal_discordance_negative_metal_positive_metal_counterion_exception"]
        is False
    )
    assert scored.metadata["metal_discordance_negative_metal_penalty"] == pytest.approx(
        0.5 * expected_negative_metal_count
    )
    assert scored.metadata["metal_discordance_structural_count"] == pytest.approx(
        0.5 * expected_negative_metal_count
        + scored.metadata["metal_discordance_unsaturated_organic_cation_count"]
    )


def test_metal_discordance_allows_negative_metal_with_positive_metal_counterion() -> None:
    no_metal = pybel.readstring(
        "xyz",
        """1
C
C 5.0 0.0 0.0
""",
    )
    no_metal_state = ReconstructionState(
        omol=no_metal,
        given_charge=0,
        total_charge=0,
        total_radical_electrons=0,
        metadata={
            "organic_core_score": organic_force_field_energy(no_metal),
            "force_field_score_key": build_force_field_score_key(no_metal),
        },
    )
    candidate = make_metal_candidate_state(
        (),
        (
            MetalAtomPosition(1, "Mo", 42, -1, 0, 0.0, 0.0, 0.0),
            MetalAtomPosition(2, "Li", 3, 1, 0, 8.0, 0.0, 0.0),
        ),
        0,
        0,
        combination_index=0,
    )

    scored = metal_scoring_module._prepare_candidate_with_no_metal_state(
        candidate,
        no_metal_state,
    )

    assert scored.metadata["metal_discordance_negative_metal_count"] == 0
    assert (
        scored.metadata["metal_discordance_negative_metal_outer_sphere_cation_exception"] is False
    )
    assert (
        scored.metadata["metal_discordance_negative_metal_positive_metal_counterion_exception"]
        is True
    )
    assert scored.metadata["metal_discordance_structural_count"] == 0


def test_metal_discordance_adjacent_double_charge_counts_outer_pair_by_radius() -> None:
    no_metal = pybel.readstring(
        "xyz",
        """2
CN
C 2.0 0.0 0.0
N 3.2 0.0 0.0
""",
    )
    no_metal.OBMol.GetAtom(1).SetFormalCharge(1)
    no_metal.OBMol.GetAtom(2).SetFormalCharge(1)
    no_metal_state = ReconstructionState(
        omol=no_metal,
        given_charge=0,
        total_charge=2,
        total_radical_electrons=0,
        metadata={
            "organic_core_score": organic_force_field_energy(no_metal),
            "force_field_score_key": build_force_field_score_key(no_metal),
        },
    )
    candidate = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Cd", 48, 0, 0, 0.0, 0.0, 0.0),),
        2,
        0,
        combination_index=0,
    )

    scored = metal_scoring_module._prepare_candidate_with_no_metal_state(
        candidate,
        no_metal_state,
    )

    assert scored.metadata["metal_discordance_outer_or_invisible_adjacent_double_charge_count"] == 1
    assert (
        scored.metadata[
            "metal_discordance_outer_or_invisible_adjacent_unknown_metal_sign_double_charge_count"
        ]
        == 1
    )


def test_metal_discordance_counts_inner_visible_adjacent_cation_pair() -> None:
    no_metal = pybel.readstring(
        "xyz",
        """2
CC
C 0.0 0.0 0.0
C 1.2 0.0 0.0
""",
    )
    no_metal.OBMol.GetAtom(1).SetFormalCharge(1)
    no_metal.OBMol.GetAtom(2).SetFormalCharge(1)
    no_metal_state = ReconstructionState(
        omol=no_metal,
        given_charge=0,
        total_charge=2,
        total_radical_electrons=0,
        metadata={
            "organic_core_score": organic_force_field_energy(no_metal),
            "force_field_score_key": build_force_field_score_key(no_metal),
        },
    )
    candidate = make_metal_candidate_state(
        (),
        (
            MetalAtomPosition(1, "Cd", 48, -2, 0, -1.8, 0.0, 0.0),
            MetalAtomPosition(2, "Cd", 48, -2, 0, 3.0, 0.0, 0.0),
            MetalAtomPosition(3, "Cd", 48, 1, 0, 100.0, 0.0, 0.0),
        ),
        2,
        0,
        combination_index=0,
    )

    scored = metal_scoring_module._prepare_candidate_with_no_metal_state(
        candidate,
        no_metal_state,
    )

    assert scored.metadata["metal_discordance_outer_or_invisible_adjacent_double_charge_count"] == 0
    assert scored.metadata["metal_discordance_inner_visible_adjacent_carbanion_pair_count"] == 1
    assert scored.metadata["metal_discordance_unsaturated_organic_cation_count"] == 1
    assert scored.metadata["metal_discordance_structural_count"] == 2


def test_metal_discordance_counts_inner_visible_adjacent_carbanion_pair() -> None:
    no_metal = pybel.readstring(
        "xyz",
        """2
CC
C 0.0 0.0 0.0
C 1.2 0.0 0.0
""",
    )
    no_metal.OBMol.GetAtom(1).SetFormalCharge(-1)
    no_metal.OBMol.GetAtom(2).SetFormalCharge(-1)
    no_metal_state = ReconstructionState(
        omol=no_metal,
        given_charge=0,
        total_charge=-2,
        total_radical_electrons=0,
        metadata={
            "organic_core_score": organic_force_field_energy(no_metal),
            "force_field_score_key": build_force_field_score_key(no_metal),
        },
    )
    candidate = make_metal_candidate_state(
        (),
        (
            MetalAtomPosition(1, "Cd", 48, 0, 0, -1.8, 0.0, 0.0),
            MetalAtomPosition(2, "Cd", 48, 0, 0, 3.0, 0.0, 0.0),
        ),
        -2,
        0,
        combination_index=0,
    )

    scored = metal_scoring_module._prepare_candidate_with_no_metal_state(
        candidate,
        no_metal_state,
    )

    assert scored.metadata["metal_discordance_outer_or_invisible_adjacent_double_charge_count"] == 0
    assert scored.metadata["metal_discordance_inner_visible_adjacent_carbanion_pair_count"] == 1
    assert scored.metadata["metal_discordance_inner_visible_conjugated_carbanion_pair_count"] == 0
    assert scored.metadata["metal_discordance_structural_count"] == 1


def test_metal_discordance_counts_inner_visible_conjugated_carbanion_pair() -> None:
    no_metal = pybel.readstring(
        "xyz",
        """4
C4
C 0.0 0.0 0.0
C 1.4 0.0 0.0
C 2.7 0.0 0.0
C 4.1 0.0 0.0
""",
    )
    for begin_idx, end_idx, bond_order in ((1, 2, 1), (2, 3, 2), (3, 4, 1)):
        bond = no_metal.OBMol.GetBond(begin_idx, end_idx)
        if bond is None:
            no_metal.OBMol.AddBond(begin_idx, end_idx, bond_order)
            bond = no_metal.OBMol.GetBond(begin_idx, end_idx)
        bond.SetBondOrder(bond_order)
    no_metal.OBMol.GetAtom(1).SetFormalCharge(-1)
    no_metal.OBMol.GetAtom(4).SetFormalCharge(-1)
    no_metal_state = ReconstructionState(
        omol=no_metal,
        given_charge=0,
        total_charge=-2,
        total_radical_electrons=0,
        metadata={
            "organic_core_score": organic_force_field_energy(no_metal),
            "force_field_score_key": build_force_field_score_key(no_metal),
        },
    )
    candidate = make_metal_candidate_state(
        (),
        (
            MetalAtomPosition(1, "Cd", 48, 0, 0, -1.8, 0.0, 0.0),
            MetalAtomPosition(2, "Cd", 48, 0, 0, 5.9, 0.0, 0.0),
        ),
        -2,
        0,
        combination_index=0,
    )

    scored = metal_scoring_module._prepare_candidate_with_no_metal_state(
        candidate,
        no_metal_state,
    )

    assert scored.metadata["metal_discordance_outer_or_invisible_adjacent_double_charge_count"] == 0
    assert scored.metadata["metal_discordance_inner_visible_adjacent_carbanion_pair_count"] == 0
    assert scored.metadata["metal_discordance_inner_visible_conjugated_carbanion_pair_count"] == 1
    assert scored.metadata["metal_discordance_unsaturated_organic_cation_count"] == 0
    assert scored.metadata["metal_discordance_structural_count"] == 1


def test_metal_discordance_counts_inner_visible_conjugated_cation_pair() -> None:
    no_metal = pybel.readstring(
        "xyz",
        """4
C4
C 0.0 0.0 0.0
C 1.4 0.0 0.0
C 2.7 0.0 0.0
C 4.1 0.0 0.0
""",
    )
    for begin_idx, end_idx, bond_order in ((1, 2, 1), (2, 3, 2), (3, 4, 1)):
        bond = no_metal.OBMol.GetBond(begin_idx, end_idx)
        if bond is None:
            no_metal.OBMol.AddBond(begin_idx, end_idx, bond_order)
            bond = no_metal.OBMol.GetBond(begin_idx, end_idx)
        bond.SetBondOrder(bond_order)
    no_metal.OBMol.GetAtom(1).SetFormalCharge(1)
    no_metal.OBMol.GetAtom(4).SetFormalCharge(1)
    no_metal_state = ReconstructionState(
        omol=no_metal,
        given_charge=0,
        total_charge=2,
        total_radical_electrons=0,
        metadata={
            "organic_core_score": organic_force_field_energy(no_metal),
            "force_field_score_key": build_force_field_score_key(no_metal),
        },
    )
    candidate = make_metal_candidate_state(
        (),
        (
            MetalAtomPosition(1, "Cd", 48, -2, 0, -1.8, 0.0, 0.0),
            MetalAtomPosition(2, "Cd", 48, -2, 0, 5.9, 0.0, 0.0),
            MetalAtomPosition(3, "Cd", 48, 1, 0, 100.0, 0.0, 0.0),
        ),
        2,
        0,
        combination_index=0,
    )

    scored = metal_scoring_module._prepare_candidate_with_no_metal_state(
        candidate,
        no_metal_state,
    )

    assert scored.metadata["metal_discordance_outer_or_invisible_adjacent_double_charge_count"] == 0
    assert scored.metadata["metal_discordance_inner_visible_adjacent_carbanion_pair_count"] == 0
    assert scored.metadata["metal_discordance_inner_visible_conjugated_carbanion_pair_count"] == 1
    assert scored.metadata["metal_discordance_unsaturated_organic_cation_count"] == 1
    assert scored.metadata["metal_discordance_structural_count"] == 2


def test_organic_radical_localization_penalty_counts_real_diradical() -> None:
    sulfur = pybel.readstring(
        "xyz",
        """1
S
S 0.0 0.0 0.0
""",
    )
    sulfur_atom = sulfur.OBMol.GetAtom(1)
    set_unpaired_electron_count(sulfur_atom, 2)

    assert metal_scoring_module._radical_localization_penalty_for_atom(
        sulfur_atom,
        is_conjugated=False,
    ) == pytest.approx(6.0)


def test_xyz2omol_state_does_not_treat_nonprior_valence_as_discordance(monkeypatch) -> None:
    no_metal = pybel.readstring(
        "xyz",
        """2
CO
C 2.0 0.0 0.0
O 3.2 0.0 0.0
""",
    )
    no_metal_state = ReconstructionState(
        omol=no_metal,
        given_charge=0,
        total_charge=0,
        total_radical_electrons=0,
        metadata={
            "organic_core_score": organic_force_field_energy(no_metal),
            "force_field_score_key": build_force_field_score_key(no_metal),
        },
    )
    plausible = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Li", 3, 1, 0, 0.0, 0.0, 0.0),),
        -1,
        0,
        combination_index=0,
    )
    implausible = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Li", 3, 2, 0, 0.0, 0.0, 0.0),),
        -2,
        0,
        combination_index=1,
    )

    monkeypatch.setattr(
        metal_search_module,
        "_group_candidates_by_target_dp",
        lambda *args, **kwargs: {
            (plausible.no_metal_charge_target, plausible.no_metal_radical_target): [plausible],
            (implausible.no_metal_charge_target, implausible.no_metal_radical_target): [
                implausible
            ],
        },
    )
    _patch_no_metal_seed_reconstruction(monkeypatch, no_metal_state)

    def fake_combined_score(self):
        valence = self.metal_states[0].valence
        self.combined_omol = {"valence": valence}
        self.score = 1.19 if valence == 1 else 1.0
        self.metadata["score"] = self.score
        return self.score

    monkeypatch.setattr(MetalCandidateState, "combined_score", fake_combined_score)

    result = with_metals_module.xyz2omol_state(
        """3
LiCO
Li 0.0 0.0 0.0
C 2.0 0.0 0.0
O 3.2 0.0 0.0
""",
        0,
        0,
    )

    assert result is not None
    assert result.score == 1.0
    assert result.combined_omol == {"valence": 2}


def test_xyz2omol_state_uses_organic_score_before_prepare_metadata(
    monkeypatch,
) -> None:
    no_metal = pybel.readstring(
        "xyz",
        """2
CO
C 2.0 0.0 0.0
O 3.2 0.0 0.0
""",
    )
    no_metal_state = ReconstructionState(
        omol=no_metal,
        given_charge=0,
        total_charge=0,
        total_radical_electrons=0,
        metadata={
            "organic_core_score": organic_force_field_energy(no_metal),
            "force_field_score_key": build_force_field_score_key(no_metal),
        },
    )
    plausible = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Li", 3, 1, 0, 0.0, 0.0, 0.0),),
        -1,
        0,
        combination_index=0,
    )
    forced = make_metal_candidate_state(
        (),
        (MetalAtomPosition(1, "Li", 3, 1, 0, 0.0, 0.0, 0.0),),
        -1,
        0,
        combination_index=1,
    )

    monkeypatch.setattr(
        metal_search_module,
        "_group_candidates_by_target_dp",
        lambda *args, **kwargs: {
            (plausible.no_metal_charge_target, plausible.no_metal_radical_target): [
                plausible,
                forced,
            ],
        },
    )
    _patch_no_metal_seed_reconstruction(monkeypatch, no_metal_state)

    def fake_prepare_candidate(candidate, no_metal_state, *, config=None):
        del no_metal_state, config
        candidate.score = 1.19 if candidate.metadata["combination_index"] == 0 else 1.0
        candidate.metadata["score"] = candidate.score
        candidate.metadata["ignored_prepare_metadata"] = (
            0.0 if candidate.metadata["combination_index"] == 0 else 5.0
        )
        candidate.metadata["metal_assignment_rank"] = 0.0
        candidate.metadata["organic_aromatic_atom_count"] = 6
        candidate.metadata["organic_aromatic_ring_count"] = 1
        candidate.metadata["organic_conjugated_atom_count"] = 6
        candidate.metadata["organic_conjugated_bond_count"] = 6
        candidate.metadata["organic_max_conjugated_component_size"] = 6
        candidate.metadata["organic_radical_localization_penalty"] = 0.0
        candidate.metadata["organic_charge_localization_penalty"] = 0.0
        candidate.combined_omol = {"combination_index": candidate.metadata["combination_index"]}
        return candidate

    monkeypatch.setattr(
        metal_scoring_module,
        "_prepare_candidate_with_no_metal_state",
        fake_prepare_candidate,
    )

    result = with_metals_module.xyz2omol_state(
        """3
LiCO
Li 0.0 0.0 0.0
C 2.0 0.0 0.0
O 3.2 0.0 0.0
""",
        0,
        0,
    )

    assert result is not None
    assert result.score == 1.0
    assert result.metadata["ignored_prepare_metadata"] == pytest.approx(5.0)
    assert result.combined_omol == {"combination_index": 1}
