from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest
from rdkit import Chem


pytest.importorskip("openbabel")
pytest.importorskip("rdkit")

from openbabel import openbabel as ob

from molgr import _core
from molgr.config import MolGRConfig
from molgr.fallback.pipeline import reconstruct_with_metals, reconstruct_without_metals
from molgr.fallback.utils.electrons import (
    get_lone_pair_count,
    get_unpaired_electron_count,
    has_unresolved_two_electron_center,
)
from molgr.fallback.utils.force_field import organic_force_field_evaluation
from molgr.fallback.utils.metals import preparation, scoring, search
from molgr.fallback.utils.no_metals import preparation as no_metal_preparation
from molgr.fallback.utils.no_metals import selection as no_metal_selection
from molgr.fallback.utils.organic_topology import compute_organic_topology_metrics
from molgr.interface import xyz_to_rdmol
from scripts.tmqmg_fixture_data import fixture


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_XYZ_FIXTURES = _PROJECT_ROOT / "tests/data/xyz"


@pytest.mark.parametrize(
    "case_id,symbol,valence",
    [
        ("CENPAN", "Ta", 5),
        ("XEMDEZ", "Ti", 4),
        ("DERYUV", "Mo", 6),
        ("HUPHOR", "Ir", 1),
    ],
)
def test_terminal_hydrazido_seed_and_geometry_scoring_parity(case_id, symbol, valence):

    if case_id == "CENPAN":
        case = fixture(case_id)
        xyz, charge = case["xyz"], case["charge"]
    else:
        case = next(c for c in _backend_parity_cases() if c["case_id"] == case_id)
        xyz = (_PROJECT_ROOT / "tests/data/tmqmg/backend_parity" / case["xyz_file"]).read_text()
        charge = int(case["charge"])
    native = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        xyz, charge, 0, config=MolGRConfig()
    )
    _, candidates, best = _python_scored_candidates(xyz, charge, 0)
    assert best is not None
    assert [(s.symbol, s.valence) for s in best.metal_states] == [(symbol, valence)]
    assert len(native["candidates"]) == len(candidates)
    for cpp, python in zip(native["candidates"], candidates):
        assert tuple(cpp["no_metal_atom_signature"]) == _pybel_atom_signature(
            python.no_metal_state.omol
        )
        assert tuple(cpp["no_metal_bond_signature"]) == _pybel_bond_signature(
            python.no_metal_state.omol
        )
        assert cpp["selected"] == (python is best)
        key = "metal_terminal_hydrazido_geometry_conflict_count"
        assert cpp[key] == python.metadata[key]
        assert (
            cpp["metal_discordance_structural_count"]
            == python.metadata["metal_discordance_structural_count"]
        )
    assert scoring.select_best_candidate(list(reversed(candidates))) is best
    assert scoring.select_best_candidate(candidates) is best


def test_cation_resonance_preselection_preserves_reviewed_zinc_state():
    """A local carbocation contributor must not hide the accepted Zn(II) state."""

    case = fixture("CEJDUU")
    assert case is not None
    xyz, charge = case["xyz"], case["charge"]
    radicals = case["multiplicity"] - 1
    native = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        xyz, charge, radicals, config=MolGRConfig()
    )
    _, candidates, best = _python_scored_candidates(xyz, charge, radicals)
    assert best is not None
    assert [(s.symbol, s.valence) for s in best.metal_states] == [("Zn", 2)]
    assert best.metadata["metal_discordance_unsaturated_organic_cation_count"] == 0
    assert len(native["candidates"]) == len(candidates)
    for cpp, python in zip(native["candidates"], candidates):
        assert cpp["combination_index"] == python.metadata["combination_index"]
        assert cpp["selected"] == (python is best)
        assert (
            cpp["metal_discordance_unsaturated_organic_cation_count"]
            == python.metadata["metal_discordance_unsaturated_organic_cation_count"]
        )


@pytest.mark.parametrize(
    "case_id, metal",
    [("APOVOS", ("Ti", 4)), ("LUQXAZ", ("Fe", 2)), ("NUDRUZ", ("Ni", 2)), ("WEDVAF", ("Y", 3))],
)
def test_aryl_donor_nhc_candidate_preference_parity(case_id, metal):
    """Geometry compares complete partners equally in both candidate pools."""

    case = fixture(case_id)
    xyz, charge, radicals = case["xyz"], case["charge"], case["multiplicity"] - 1
    native = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        xyz, charge, radicals, config=MolGRConfig()
    )
    _, candidates, best = _python_scored_candidates(xyz, charge, radicals)
    assert best is not None
    assert [(s.symbol, s.valence) for s in best.metal_states] == [metal]
    assert len(native["candidates"]) == len(candidates)
    for cpp, python in zip(native["candidates"], candidates):
        assert cpp["combination_index"] == python.metadata["combination_index"]
        ligand = python.no_metal_state.omol
        assert tuple(cpp["no_metal_atom_signature"]) == _pybel_atom_signature(ligand)
        assert tuple(cpp["no_metal_bond_signature"]) == _pybel_bond_signature(ligand)
        assert cpp["selected"] == (python is best)
        assert (
            cpp["metal_discordance_structural_count"]
            == python.metadata["metal_discordance_structural_count"]
        )
        assert cpp["score"] == pytest.approx(python.score)


def test_338_s_mixed_metal_candidate_order_parity() -> None:
    """Equal scores must resolve to the same assignment in both backends."""
    xyz = (_XYZ_FIXTURES / "338-S.xyz").read_text()
    config = MolGRConfig()
    cpp = _core.dev.pipeline.reconstruct_with_metals.debug_metal_search_summaries(
        xyz, 0, 0, config=config
    )
    assert _cpp_metal_search_summary(cpp) == _python_metal_search_summary(xyz, 0, 0, config=config)

    cpp_scored = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        xyz, 0, 0, config=config
    )
    layer, candidates, best = _python_scored_candidates(xyz, 0, 0)
    assert cpp_scored["layer_index"] == layer
    assert len(cpp_scored["candidates"]) == len(candidates)
    for native, fallback in zip(cpp_scored["candidates"], candidates):
        assert native["combination_index"] == fallback.metadata["combination_index"]
        assert [(s["valence"], s["radical_num"]) for s in native["metal_states"]] == [
            (s.valence, s.radical_num) for s in fallback.metal_states
        ]
        assert (
            native["organic_singlet_two_electron_center_count"]
            == fallback.metadata["organic_singlet_two_electron_center_count"]
        )
        assert (
            native["organic_visible_singlet_carbene_count"]
            == fallback.metadata["organic_visible_singlet_carbene_count"]
        )
        assert native["score"] == pytest.approx(fallback.score)
        assert native["selected"] == (fallback is best)
    assert best is not None
    assert [(s.symbol, s.valence) for s in best.metal_states] == [("Fe", 2), ("Pd", 2), ("Fe", 2)]


@pytest.mark.parametrize(
    ("case_id", "charge"),
    [
        ("GUMBAS", 0),
        ("OPAGES", 0),
        ("ZUPYAN", 1),
        ("BODFIJ", 0),
        ("OXMTMI", 0),
        ("BADGUL", 1),
        ("KARMIZ", 0),
        ("WECYUB", 0),
    ],
)
def test_shared_seed_preserves_scored_candidate_parity(case_id: str, charge: int) -> None:
    """Seed perception must agree before independent electronic-state searches."""
    xyz = (_PROJECT_ROOT / "tests/data/tmqmg/reconstruction" / f"{case_id}.xyz").read_text()
    cpp = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        xyz, charge, 0, config=MolGRConfig()
    )
    layer, candidates, best = _python_scored_candidates(xyz, charge, 0)
    assert best is not None
    assert cpp["layer_index"] == layer
    assert len(cpp["candidates"]) == len(candidates)
    for native, fallback in zip(cpp["candidates"], candidates):
        assert native["combination_index"] == fallback.metadata["combination_index"]
        assert tuple(native["no_metal_atom_signature"]) == _pybel_atom_signature(
            fallback.no_metal_state.omol
        )
        assert tuple(native["no_metal_bond_signature"]) == _pybel_bond_signature(
            fallback.no_metal_state.omol
        )
        assert native["score"] == pytest.approx(fallback.score)
        if native["selection_key"] is not None:
            assert native["selection_key"] == pytest.approx(fallback.metadata["selection_key"])
        assert native["selected"] == (fallback is best)


def _backend_parity_cases() -> list[dict[str, str]]:
    path = _PROJECT_ROOT / "tests/data/tmqmg/backend_parity/manifest.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_on_seesaw_phosphoranide_avoids_reducing_aryl_rings() -> None:
    """Explore the phosphoranide without treating a pending standard as approved.

    The same ligand combination is reported as a Pd(II) phosphoranide complex
    in doi:10.1016/j.ica.2019.119239. Geometry opens the alternative; the normal
    charge/spin search selects it without changing scoring weights.
    """
    xyz = (_PROJECT_ROOT / "tests/data/tmqmg/backend_parity/BUBNUK.xyz").read_text()
    source = Chem.MolFromXYZBlock(xyz)
    results = []
    for backend in ("python", "cpp"):
        result = xyz_to_rdmol(xyz, 0, 1, backend=backend)
        assert [a.GetAtomicNum() for a in result.GetAtoms()] == [
            a.GetAtomicNum() for a in source.GetAtoms()
        ]
        assert sum(a.GetAtomicNum() == 1 for a in result.GetAtoms()) == 29
        assert all(a.GetNumImplicitHs() == 0 for a in result.GetAtoms())
        assert Chem.GetFormalCharge(result) == 0
        assert sum(a.GetNumRadicalElectrons() for a in result.GetAtoms()) == 0
        assert [a.GetFormalCharge() for a in result.GetAtoms() if a.GetAtomicNum() == 46] == [2]
        assert sorted(a.GetFormalCharge() for a in result.GetAtoms() if a.GetAtomicNum() == 15) == [
            -1,
            0,
        ]
        assert all(a.GetFormalCharge() == 0 for a in result.GetAtoms() if a.GetAtomicNum() == 6)
        for i in range(source.GetNumAtoms()):
            assert tuple(result.GetConformer().GetAtomPosition(i)) == pytest.approx(
                tuple(source.GetConformer().GetAtomPosition(i)), abs=1e-8
            )
        Chem.RemoveStereochemistry(result)
        results.append(Chem.MolToSmiles(result))
    assert results[0] == results[1]


@pytest.mark.parametrize("case", _backend_parity_cases(), ids=lambda case: case["case_id"])
def test_recovery_candidates_and_final_graph_match_between_backends(
    case: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check production recovery paths without treating either graph as a reference."""
    xyz = (_PROJECT_ROOT / "tests/data/tmqmg/backend_parity" / case["xyz_file"]).read_text()
    charge = int(case["charge"])
    config = MolGRConfig()
    config.cpp_backend.max_threads = 1
    cpp = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        xyz, charge, 0, config=config
    )
    captured = []
    select = scoring.select_best_candidate

    def capture(candidates, **kwargs):
        best = select(candidates, **kwargs)
        captured.extend(candidates)
        return best

    monkeypatch.setattr(scoring, "select_best_candidate", capture)
    best = reconstruct_with_metals.xyz2omol_state(xyz, charge, 0, config=config)
    monkeypatch.setattr(scoring, "select_best_candidate", select)
    assert best is not None
    assert len(cpp["candidates"]) == len(captured) > 0
    for native, fallback in zip(cpp["candidates"], captured):
        assert native["combination_index"] == fallback.metadata["combination_index"]
        assert tuple(native["no_metal_atom_signature"]) == _pybel_atom_signature(
            fallback.no_metal_state.omol
        )
        assert tuple(native["no_metal_bond_signature"]) == _pybel_bond_signature(
            fallback.no_metal_state.omol
        )
        assert native["score"] == pytest.approx(fallback.score)
        if native["selection_key"] is None:
            assert fallback.metadata.get("selection_key") is None
        else:
            assert native["selection_key"] == pytest.approx(fallback.metadata["selection_key"])
        for key in (
            "metal_discordance_long_carbonyl_multiple_bond_count",
            "metal_discordance_short_carbonyl_reduction_count",
            "metal_discordance_long_dioxygen_multiple_bond_count",
        ):
            assert native[key] == fallback.metadata[key]
        assert native["selected"] == (
            fallback.metadata["combination_index"] == best.metadata["combination_index"]
        )

    results = [
        _canonical_reconstructed_smiles(
            xyz,
            backend=backend,
            total_charge=charge,
            total_radical_electrons=0,
            config=config,
        )
        for backend in ("cpp", "python")
    ]
    assert results[0] == results[1]


def _load_tmqmg_xyz(case_id: str) -> str:
    return fixture(case_id)["xyz"]


@pytest.mark.parametrize(
    ("case_id", "pairs", "order", "charges", "metal_charge"),
    [
        ("BOWPEL", [(14, 52), (33, 54)], 2, (0, 0), 2),
        ("EGEHIJ", [(6, 73)], 2, (0, 0), 3),
        ("GOHCOY", [(3, 10)], 2, (0, 0), 2),
        ("JAYGEV", [(1, 24)], 2, (0, 0), 0),
        ("YEVWAX", [(1, 2), (15, 16)], 1, (-1, -1), 6),
        ("ALUSEI", [(42, 43)], 1, (-1, -1), 4),
        ("FOXZOH", [(1, 2), (3, 4)], 1, (-1, -1), 6),
        ("RITCIG", [(1, 16)], 1, (-1, -1), 4),
        ("FUDRIF", [(5, 7), (6, 10)], 1, (-1, -1), 6),
        ("TILXIX", [(2, 5), (3, 18)], 1, (-1, -1), 6),
        ("KUPMOX", [(5, 8)], 1, (-1, -1), 6),
        ("SIPDEB", [(5, 6)], 1, (-1, -1), 4),
        ("AVEPEW", [(6, 43), (5, 9)], 1, (-1, -1), 4),
        ("COCJOT", [(1, 2), (4, 5)], 1, (-1, -1), 4),
        ("XASJEK", [(2, 12)], 1, (-1, -1), 2),
    ],
)
def test_carbonyl_and_peroxide_local_reconstruction(case_id, pairs, order, charges, metal_charge):
    """Assert the affected local structures without approving the whole reference."""
    case = next(c for c in _backend_parity_cases() if c["case_id"] == case_id)
    xyz = (_PROJECT_ROOT / "tests/data/tmqmg/backend_parity" / case["xyz_file"]).read_text()
    config = MolGRConfig()
    config.cpp_backend.max_threads = 1
    config.cpp_backend.target_bucket_parallel_max_threads = 1
    for backend in ("cpp", "python"):
        mol = xyz_to_rdmol(xyz, int(case["charge"]), 1, backend=backend, config=config)
        assert Chem.GetFormalCharge(mol) == int(case["charge"])
        assert sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()) == 0
        for i, j in pairs:
            assert mol.GetBondBetweenAtoms(i, j).GetBondTypeAsDouble() == order
            assert tuple(mol.GetAtomWithIdx(k).GetFormalCharge() for k in (i, j)) == charges
        if metal_charge is not None:
            metals = [a for a in mol.GetAtoms() if a.GetSymbol() == case["metal_center"]]
            assert len(metals) == 1 and metals[0].GetFormalCharge() == metal_charge


@pytest.mark.parametrize("case_id", ["ITUCUW", "QEJSIK"])
def test_cumulene_exception_preserves_aromatic_macrocycle(case_id):
    case = next(c for c in _backend_parity_cases() if c["case_id"] == case_id)
    xyz = (_PROJECT_ROOT / "tests/data/tmqmg/backend_parity" / case["xyz_file"]).read_text()
    for backend in ("cpp", "python"):
        mol = xyz_to_rdmol(xyz, int(case["charge"]), 1, backend=backend)
        metal = next(a for a in mol.GetAtoms() if a.GetSymbol() == case["metal_center"])
        assert metal.GetFormalCharge() == 2
        assert not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]=[#6]=[#6]"))


@pytest.mark.parametrize("case_id", ["JAYGEV", "QEHQUP"])
def test_accepted_nhc_charge_separated_representation_is_retained(case_id):
    case = next(c for c in _backend_parity_cases() if c["case_id"] == case_id)
    xyz = (_PROJECT_ROOT / "tests/data/tmqmg/backend_parity" / case["xyz_file"]).read_text()
    pattern = Chem.MolFromSmarts("[#6-]=[#7+]")
    for backend in ("cpp", "python"):
        mol = xyz_to_rdmol(xyz, int(case["charge"]), 1, backend=backend)
        assert mol.HasSubstructMatch(pattern)


@pytest.mark.parametrize("charge,multiplicity,order", [(0, 1, 2), (-1, 2, 1), (-2, 1, 1)])
def test_elongated_o2_normalization_respects_charge_and_spin(charge, multiplicity, order):
    xyz = "2\nO2 charge and spin control\nO 0 0 0\nO 1.45 0 0\n"
    for backend in ("cpp", "python"):
        mol = xyz_to_rdmol(xyz, charge, multiplicity, backend=backend)
        assert Chem.GetFormalCharge(mol) == charge
        assert sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()) == multiplicity - 1
        assert mol.GetBondBetweenAtoms(0, 1).GetBondTypeAsDouble() == order


@pytest.mark.parametrize(
    ("case_id", "expected_charge"),
    [("FAKDOL", 1), ("BIBZEV", 1), ("BOBYAS", 1), ("PEDNIY", -1), ("RASBOC", 1)],
)
def test_terminal_nitrosyl_selection_and_backend_metrics(case_id, expected_charge):
    """Assert reviewed local NO behavior without freezing unrelated ligand states."""
    case = next(c for c in _backend_parity_cases() if c["case_id"] == case_id)
    xyz = (_PROJECT_ROOT / "tests/data/tmqmg/backend_parity" / case["xyz_file"]).read_text()
    charge = int(case["charge"])
    config = MolGRConfig()
    config.cpp_backend.max_threads = 1
    config.cpp_backend.target_bucket_parallel_max_threads = 1
    cpp = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        xyz, charge, 0, config=config
    )
    _, candidates, best = _python_scored_candidates(xyz, charge, 0)
    assert best is not None
    assert len(cpp["candidates"]) == len(candidates)
    for native, fallback in zip(cpp["candidates"], candidates):
        assert (
            native["nitrosyl_geometry_site_count"]
            == fallback.metadata["nitrosyl_geometry_site_count"]
        )
        assert native["nitrosyl_geometry_penalty"] == pytest.approx(
            fallback.metadata["nitrosyl_geometry_penalty"]
        )
        assert native["selected"] == (fallback is best)
        assert (
            native["nitrosonium_redistribution_exception"]
            == fallback.metadata["nitrosonium_redistribution_exception"]
        )
        if fallback.metadata.get("selection_key") is not None:
            assert native["selection_key"] == pytest.approx(fallback.metadata["selection_key"])
    assert best.metadata["nitrosyl_geometry_site_count"] == (expected_charge == 1)
    outputs = []
    for backend in ("cpp", "python"):
        mol = xyz_to_rdmol(xyz, charge, 1, backend=backend, config=config)
        assert Chem.GetFormalCharge(mol) == charge
        assert sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()) == 0
        organic = Chem.RWMol(mol)
        for atom in reversed(list(mol.GetAtoms())):
            if atom.GetSymbol() in {"Os", "Fe", "Re", "Co", "Mo"}:
                organic.RemoveAtom(atom.GetIdx())
        no_fragments = [
            indices
            for indices in Chem.GetMolFrags(organic)
            if sorted(organic.GetAtomWithIdx(i).GetAtomicNum() for i in indices) == [7, 8]
        ]
        assert len(no_fragments) == 1
        assert (
            sum(organic.GetAtomWithIdx(i).GetFormalCharge() for i in no_fragments[0])
            == expected_charge
        )
        outputs.append(Chem.MolToSmiles(mol))
    assert outputs[0] == outputs[1]
    if case_id == "RASBOC":
        # Oximate geometry makes the approved Mo(II) state reachable directly;
        # the former Mo(0)/nitroso winner needed a redistribution exception.
        assert best.metal_states[0].valence == 2
        assert best.metadata["nitrosonium_redistribution_exception"] == 0
        assert scoring.select_best_candidate(candidates) is best
        assert scoring.select_best_candidate(candidates) is best
        scoring._annotate_candidate_discordance_features(best)
        assert best.metadata["nitrosonium_redistribution_exception"] == 0
        assert scoring.select_best_candidate(candidates) is best
        assert best.metadata["metal_discordance_structural_count"] == 0


def _canonical_reconstructed_smiles(
    xyz_block: str,
    *,
    backend: Literal["cpp", "python"],
    total_charge: int,
    total_radical_electrons: int,
    config,
) -> str:
    rdmol = xyz_to_rdmol(
        xyz_block,
        total_charge,
        total_radical_electrons + 1,
        backend=backend,
        make_dative_bonds=True,
        make_stereochemistry=True,
        config=config,
    )
    return Chem.MolToSmiles(
        Chem.RemoveHs(rdmol),
        canonical=True,
        isomericSmiles=True,
    )


def _python_scored_candidates(xyz_block: str, total_charge: int, total_radical_electrons: int):
    # Capture the production pool after geometry-supported seeds have been scored.
    from unittest.mock import patch

    candidates = []
    original = scoring.select_best_candidate
    selected = None

    def capture(pool, **kwargs):
        nonlocal selected
        candidates[:] = pool
        selected = original(pool, **kwargs)
        return selected

    config = MolGRConfig()
    with patch.object(scoring, "select_best_candidate", capture):
        best = reconstruct_with_metals.xyz2omol_state(
            xyz_block,
            total_charge,
            total_radical_electrons,
            config=config,
        )
    if best is None:
        return None, [], None
    # xyz2omol_state freezes its winner for materialization. Return the selected
    # pool object so identity-based assertions keep checking the actual choice.
    return best.metadata["search_layer_index"], candidates, selected


def _python_no_metal_resonance_candidate_summaries(
    no_metal_xyz_block: str,
    total_charge: int,
    total_radical_electrons: int,
    *,
    config,
) -> tuple[dict[str, object], ...]:
    seed_state = no_metal_preparation._seed_state(
        no_metal_xyz_block,
        total_charge,
        total_radical_electrons,
    )
    candidates = []
    reconstruct_without_metals._run_no_metal_pipeline_from_state(
        seed_state,
        config=config,
        candidate_pool=candidates,
    )

    summaries = []
    for candidate in candidates:
        no_metal_selection._score_reconstruction_candidate(candidate, config=config)
        selection_key = no_metal_selection._no_metal_candidate_selection_key(
            candidate,
            config=config,
        )
        smiles = candidate.omol.write("smi")
        assert smiles is not None
        summaries.append(
            {
                "smiles": smiles.split()[0],
                "resonance_index": candidate.metadata.get("resonance_index"),
                "score": float(candidate.metadata.get("score", 0.0)),
                "selection_key": selection_key,
            }
        )
    return tuple(summaries)


def _cpp_no_metal_resonance_candidate_summaries(
    no_metal_xyz_block: str,
    total_charge: int,
    total_radical_electrons: int,
    *,
    config,
) -> tuple[dict[str, object], ...]:
    summaries = []
    for item in _core.dev.pipeline.reconstruct_without_metals.debug_resonance_candidate_summaries(
        no_metal_xyz_block,
        total_charge,
        total_radical_electrons,
        config=config,
    ):
        selection_key = (
            int(item["azinium_geometry_conflict_count"]),
            int(item["formal_charge_absolute_sum"]),
            int(item["long_carbonyl_multiple_bond_count"]),
            -int(item["aromatic_atom_count"]),
            -int(item["aromatic_ring_count"]),
            -float(item["aromatic_stability_score"]),
            -float(item["adjusted_max_conjugated_component_size"]),
            -float(item["adjusted_conjugated_atom_count"]),
            -float(item["adjusted_conjugated_bond_count"]),
            int(item["excess_radical_labels"]),
            -int(item["hyperconjugation_score"]),
            float(item["score"]),
        )
        summaries.append(
            {
                "smiles": item["smiles"],
                "resonance_index": item["resonance_index"],
                "score": float(item["score"]),
                "selection_key": selection_key,
            }
        )
    return tuple(summaries)


def _metal_state_signature(metal_state) -> tuple[object, ...]:
    return (
        int(metal_state.idx),
        str(metal_state.symbol),
        int(metal_state.element_idx),
        int(metal_state.valence),
        int(metal_state.radical_num),
        round(float(metal_state.position_x) * 1_000_000),
        round(float(metal_state.position_y) * 1_000_000),
        round(float(metal_state.position_z) * 1_000_000),
    )


def _metal_choice_signature(choice) -> tuple[tuple[object, ...], ...]:
    return tuple(_metal_state_signature(metal_state) for metal_state in choice)


def _metal_search_group_signature(group) -> tuple[tuple[tuple[object, ...], ...], ...]:
    return tuple(_metal_choice_signature(choice) for choice in group)


def _metal_search_layer_signature(layer) -> tuple[tuple[tuple[tuple[object, ...], ...], ...], ...]:
    return tuple(_metal_search_group_signature(group) for group in layer)


def _cpp_metal_search_group_signature(group) -> tuple[tuple[tuple[object, ...], ...], ...]:
    return tuple(tuple(tuple(metal_state) for metal_state in choice) for choice in group)


def _cpp_metal_search_layer_signature(
    layer,
) -> tuple[tuple[tuple[tuple[object, ...], ...], ...], ...]:
    return tuple(_cpp_metal_search_group_signature(group) for group in layer)


def _candidate_search_signature(candidate) -> dict[str, object]:
    return {
        "combination_index": int(candidate.metadata.get("combination_index", -1)),
        "no_metal_charge_target": int(candidate.no_metal_charge_target),
        "no_metal_radical_target": int(candidate.no_metal_radical_target),
        "metal_assignment_rank": float(candidate.metadata.get("metal_assignment_rank", 0.0)),
        "metal_states": _metal_choice_signature(candidate.metal_states),
        "phase_history": tuple(candidate.phase_history),
    }


def _python_metal_search_summary(
    xyz_block: str,
    total_charge: int,
    total_radical_electrons: int,
    *,
    config,
) -> dict[str, object]:
    base_state = preparation.prepare_metal_state(
        xyz_block,
        total_charge,
        total_radical_electrons,
        config=config,
    )
    state_search_groups = search._build_metal_state_search_groups(
        base_state.available_valence_radical_states,
        config=config,
    )
    layered_groups = search._build_layered_metal_state_search_groups(
        state_search_groups,
        total_radical_electrons,
        config=config,
    )
    target_buckets_by_layer = []
    for layer in layered_groups:
        grouped_candidates = search._group_candidates_by_target_dp(
            base_state.phase_history,
            layer,
            total_charge,
            total_radical_electrons,
            config=config,
        )
        layer_buckets = []
        for target, candidates in sorted(grouped_candidates.items()):
            layer_buckets.append(
                {
                    "target": tuple(target),
                    "candidates": tuple(
                        _candidate_search_signature(candidate) for candidate in candidates
                    ),
                }
            )
        target_buckets_by_layer.append(tuple(layer_buckets))

    return {
        "available_valence_radical_states": tuple(
            tuple(_metal_state_signature(metal_state) for metal_state in state_options)
            for state_options in base_state.available_valence_radical_states
        ),
        "base_phase_history": tuple(base_state.phase_history),
        "state_search_groups": tuple(
            _metal_search_group_signature(group) for group in state_search_groups
        ),
        "layered_state_search_groups": tuple(
            _metal_search_layer_signature(layer) for layer in layered_groups
        ),
        "target_buckets_by_layer": tuple(target_buckets_by_layer),
    }


def _cpp_candidate_search_signature(candidate: dict) -> dict[str, object]:
    return {
        "combination_index": int(candidate["combination_index"]),
        "no_metal_charge_target": int(candidate["no_metal_charge_target"]),
        "no_metal_radical_target": int(candidate["no_metal_radical_target"]),
        "metal_assignment_rank": float(candidate["metal_assignment_rank"]),
        "metal_states": tuple(tuple(item) for item in candidate["metal_states"]),
        "phase_history": tuple(candidate["phase_history"]),
    }


def _cpp_metal_search_summary(raw: dict) -> dict[str, object]:
    target_buckets_by_layer = []
    for layer_buckets in raw["target_buckets_by_layer"]:
        normalized_buckets = []
        for bucket in layer_buckets:
            normalized_buckets.append(
                {
                    "target": tuple(bucket["target"]),
                    "candidates": tuple(
                        _cpp_candidate_search_signature(candidate)
                        for candidate in bucket["candidates"]
                    ),
                }
            )
        target_buckets_by_layer.append(tuple(normalized_buckets))

    return {
        "available_valence_radical_states": tuple(
            tuple(tuple(metal_state) for metal_state in state_options)
            for state_options in raw["available_valence_radical_states"]
        ),
        "base_phase_history": tuple(raw["base_phase_history"]),
        "state_search_groups": tuple(
            _cpp_metal_search_group_signature(group) for group in raw["state_search_groups"]
        ),
        "layered_state_search_groups": tuple(
            _cpp_metal_search_layer_signature(layer) for layer in raw["layered_state_search_groups"]
        ),
        "target_buckets_by_layer": tuple(target_buckets_by_layer),
    }


def _pybel_atom_signature(
    omol,
) -> tuple[tuple[int, int, int, int, bool, int, int, int, int, bool], ...]:
    atom_signature = []
    for atom in ob.OBMolAtomIter(omol.OBMol):
        atom_signature.append(
            (
                int(atom.GetAtomicNum()),
                int(atom.GetFormalCharge()),
                int(get_unpaired_electron_count(atom)),
                int(get_lone_pair_count(atom)),
                has_unresolved_two_electron_center(atom),
                int(atom.GetHyb()),
                round(float(atom.GetX()) * 1_000_000),
                round(float(atom.GetY()) * 1_000_000),
                round(float(atom.GetZ()) * 1_000_000),
                bool(atom.IsAromatic()),
            )
        )
    return tuple(atom_signature)


def _pybel_bond_signature(omol) -> tuple[tuple[int, int, int, bool], ...]:
    bond_signature = []
    for bond in ob.OBMolBondIter(omol.OBMol):
        begin_idx = int(bond.GetBeginAtom().GetIdx())
        end_idx = int(bond.GetEndAtom().GetIdx())
        if begin_idx > end_idx:
            begin_idx, end_idx = end_idx, begin_idx
        bond_signature.append(
            (begin_idx, end_idx, int(bond.GetBondOrder()), bool(bond.IsAromatic()))
        )
    return tuple(sorted(bond_signature))


def _get_ptr(obmol: ob.OBMol) -> int:
    this = getattr(obmol, "this", None)
    if this is not None:
        return int(this)  # type: ignore[arg-type]
    return int(obmol)  # type: ignore[arg-type]


def _metrics_signature(metrics) -> dict[str, object]:
    return {
        "aromatic_atom_count": metrics.aromatic_atom_count,
        "aromatic_ring_count": metrics.aromatic_ring_count,
        "aromatic_stability_score": metrics.aromatic_stability_score,
        "conjugated_atom_count": metrics.conjugated_atom_count,
        "conjugated_bond_count": metrics.conjugated_bond_count,
        "max_conjugated_component_size": metrics.max_conjugated_component_size,
        "conjugated_atom_indices": tuple(metrics.conjugated_atom_indices),
        "hyperconjugative_donor_count": metrics.hyperconjugative_donor_count,
        "hyperconjugation_score": metrics.hyperconjugation_score,
    }


def _assert_cpp_python_topology_metrics_match(omol, *, config) -> None:
    py_metrics = _metrics_signature(
        compute_organic_topology_metrics(omol, config=config.organic_topology)
    )
    cpp_metrics = dict(
        _core.dev.utils.compute_organic_topology_metrics_ptr(
            _get_ptr(omol.OBMol),
            config=config,
        )
    )
    cpp_metrics["conjugated_atom_indices"] = tuple(cpp_metrics["conjugated_atom_indices"])

    for key, py_value in py_metrics.items():
        cpp_value = cpp_metrics[key]
        if isinstance(py_value, float):
            assert cpp_value == pytest.approx(py_value), key
        else:
            assert cpp_value == py_value, key


@pytest.mark.parametrize(
    "smiles",
    [
        "c1ccccc1",
        "C=CC=C",
        "C=C=C",
        "S=C1C=CC=C2C1=C=CC=C2",
        "C=C=CC(=O)OCC",
        "C=[C-]/C=C(\\[O-])/OCC",
        "[c-]1[c-][c-][cH][cH][cH]1",
        "c1cc2ccccc2cc1",
        "[Br-].Clc1c2ccc3ccc4[C-](Cl)[CH-]C=Nc4c3c2ncc1.[O+]#[C-].[O+]#[C-].[O+]#[C-]",
        "[Br-].Clc1c2ccc3ccc4c(Cl)ccnc4c3c2ncc1.[O+]#[C-].[O+]#[C-].[O+]#[C-]",
        "[n-]1cccc1",
        "C/C(=C/C(=N/c1c(C)cccc1C)/C)/[N-]c1c(C)cccc1C.Cc1ccccc1",
    ],
)
def test_cpp_python_organic_topology_metrics_match_smiles(smiles: str) -> None:
    from openbabel import pybel

    config = MolGRConfig()
    omol = pybel.readstring("smi", smiles)

    _assert_cpp_python_topology_metrics_match(omol, config=config)


def test_cpp_python_organic_topology_metrics_match_case_522_no_metal_candidate() -> None:
    row = fixture("AFAKAW")
    xyz_block = _load_tmqmg_xyz(row["id"])
    total_charge = int(row["charge"])
    total_radical_electrons = 0
    config = MolGRConfig()
    _layer_index, py_candidates, _py_best = _python_scored_candidates(
        xyz_block,
        total_charge,
        total_radical_electrons,
    )

    assert py_candidates
    assert py_candidates[0].no_metal_state is not None
    _assert_cpp_python_topology_metrics_match(py_candidates[0].no_metal_state.omol, config=config)


def test_cpp_python_organic_force_field_energy_matches_case_522_no_metal_candidates() -> None:
    row = fixture("AFAKAW")
    xyz_block = _load_tmqmg_xyz(row["id"])
    total_charge = int(row["charge"])
    total_radical_electrons = 0
    config = MolGRConfig()
    _layer_index, py_candidates, _py_best = _python_scored_candidates(
        xyz_block,
        total_charge,
        total_radical_electrons,
    )

    assert py_candidates
    for candidate in py_candidates:
        assert candidate.no_metal_state is not None
        py_energy = organic_force_field_evaluation(candidate.no_metal_state).energy_kj_mol
        cpp_energy = _core.dev.utils.organic_force_field_energy_ptr(
            _get_ptr(candidate.no_metal_state.omol.OBMol),
            config=config,
        )
        assert cpp_energy == pytest.approx(py_energy)


@pytest.mark.parametrize("case_id", ["ADIXUI", "ADIYAP"])
def test_cpp_python_scored_candidate_parity_for_dmso_state_regressions(case_id: str) -> None:
    row = fixture(case_id)
    xyz_block = _load_tmqmg_xyz(row["id"])
    total_charge = int(row["charge"])
    total_radical_electrons = 0
    config = MolGRConfig()
    cpp = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        xyz_block,
        total_charge,
        total_radical_electrons,
        config=config,
    )
    py_layer_index, py_candidates, py_best = _python_scored_candidates(
        xyz_block,
        total_charge,
        total_radical_electrons,
    )

    assert cpp["layer_index"] == py_layer_index
    assert len(cpp["candidates"]) == len(py_candidates)
    for cpp_item, py_item in zip(cpp["candidates"], py_candidates):
        assert cpp_item["combination_index"] == py_item.metadata.get("combination_index")
        assert cpp_item["no_metal_charge_target"] == py_item.no_metal_charge_target
        assert cpp_item["score"] == pytest.approx(py_item.score or 0.0)
        assert cpp_item["selection_key"] == py_item.metadata.get("selection_key")
        assert cpp_item["selected"] == (py_best is py_item)


def test_cpp_python_scored_candidate_parity_for_acifea_nitrate_exemption() -> None:
    row = fixture("ACIFEA")
    assert row["id"] == "ACIFEA"
    xyz_block = _load_tmqmg_xyz(row["id"])
    total_charge = int(row["charge"])
    config = MolGRConfig()
    cpp = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        xyz_block,
        total_charge,
        0,
        config=config,
    )
    py_layer_index, py_candidates, py_best = _python_scored_candidates(
        xyz_block,
        total_charge,
        0,
    )

    assert cpp["layer_index"] == py_layer_index == 0
    # Exceptional Zn(+4) remains searchable with an explicit preference cost.
    assert len(cpp["candidates"]) == len(py_candidates) == 3
    assert [item["metal_states"][0]["valence"] for item in cpp["candidates"]] == [4, 2, 0]
    assert [item["metal_discordance_high_zinc_valence_count"] for item in cpp["candidates"]] == [
        1,
        0,
        0,
    ]
    for cpp_item, py_item in zip(cpp["candidates"], py_candidates):
        assert cpp_item["metal_states"] == [
            {
                "idx": metal_state.idx,
                "symbol": metal_state.symbol,
                "valence": metal_state.valence,
                "radical_num": metal_state.radical_num,
            }
            for metal_state in py_item.metal_states
        ]
        assert cpp_item["metal_discordance_structural_count"] == pytest.approx(
            float(py_item.metadata.get("metal_discordance_structural_count", 0.0))
        )
        assert cpp_item["organic_charge_localization_penalty"] == pytest.approx(
            float(py_item.metadata.get("organic_charge_localization_penalty", 0.0))
        )
        assert cpp_item["organic_charge_localization_component_cancellation"] == pytest.approx(
            float(
                py_item.metadata.get(
                    "organic_charge_localization_component_cancellation",
                    0.0,
                )
            )
        )
        assert cpp_item["organic_charge_localization_polarity_inversion_penalty"] == pytest.approx(
            float(
                py_item.metadata.get(
                    "organic_charge_localization_polarity_inversion_penalty",
                    0.0,
                )
            )
        )
        assert cpp_item["organic_charge_localization_reference_penalty"] == pytest.approx(
            float(py_item.metadata.get("organic_charge_localization_reference_penalty", 0.0))
        )
        assert cpp_item["organic_charge_localization_selection_margin"] == pytest.approx(
            float(py_item.metadata.get("organic_charge_localization_selection_margin", 0.0))
        )
        assert cpp_item["organic_charge_localization_margin_difference"] == pytest.approx(
            float(py_item.metadata.get("organic_charge_localization_margin_difference", 0.0))
        )
        assert cpp_item["organic_charge_localization_margin_exceeded"] is bool(
            py_item.metadata.get("organic_charge_localization_margin_exceeded", False)
        )
        assert cpp_item["organic_charge_localization_reference_metal_valence_max_delta"] == int(
            py_item.metadata.get(
                "organic_charge_localization_reference_metal_valence_max_delta",
                0,
            )
        )
        assert cpp_item["organic_charge_localization_metal_valence_jump_exceeded"] is bool(
            py_item.metadata.get(
                "organic_charge_localization_metal_valence_jump_exceeded",
                False,
            )
        )
        assert cpp_item["selection_key"] == py_item.metadata.get("selection_key")
        assert cpp_item["selected"] == (py_best is py_item)

    zinc_ii = next(item for item in cpp["candidates"] if item["metal_states"][0]["valence"] == 2)
    assert zinc_ii["metal_discordance_structural_count"] == pytest.approx(0.0)
    assert zinc_ii["passes_metal_discordance_filter"] is True
    assert zinc_ii["selected"] is True


def test_cpp_python_arojew_titanium_two_counts_visible_same_sign_nitrogen() -> None:
    xyz_block = _load_tmqmg_xyz("AROJEW")
    config = MolGRConfig()
    cpp = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        xyz_block,
        0,
        0,
        config=config,
    )
    _, py_candidates, _ = _python_scored_candidates(xyz_block, 0, 0)

    cpp_titanium_two = next(
        item
        for item in cpp["candidates"]
        if item["metal_states"] == [{"idx": 1, "symbol": "Ti", "valence": 2, "radical_num": 0}]
    )
    py_titanium_two = next(
        candidate
        for candidate in py_candidates
        if len(candidate.metal_states) == 1
        and candidate.metal_states[0].symbol == "Ti"
        and candidate.metal_states[0].valence == 2
        and candidate.metal_states[0].radical_num == 0
    )

    assert cpp_titanium_two["metal_discordance_inner_visible_same_sign_charge_count"] == 1
    assert py_titanium_two.metadata["metal_discordance_inner_visible_same_sign_charge_count"] == 1


def test_cpp_python_cehhab_preserves_iron_two_after_nitroso_geometry_screen() -> None:
    xyz_block = _load_tmqmg_xyz("CEHHAB")
    config = MolGRConfig()
    cpp = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        xyz_block,
        0,
        0,
        config=config,
    )
    _, py_candidates, py_best = _python_scored_candidates(xyz_block, 0, 0)

    assert py_best is not None
    assert [(state.valence, state.radical_num) for state in py_best.metal_states] == [(2, 0)]
    selected_cpp = next(item for item in cpp["candidates"] if item["selected"])
    assert [(state["valence"], state["radical_num"]) for state in selected_cpp["metal_states"]] == [
        (2, 0)
    ]

    iron_zero_cpp_candidates = [
        item for item in cpp["candidates"] if item["metal_states"][0]["valence"] == 0
    ]
    iron_zero_py_candidates = [
        candidate for candidate in py_candidates if candidate.metal_states[0].valence == 0
    ]
    # Missing candidates are a backend-search regression, not a fixture lookup
    # error or an unexplained StopIteration. Keep the original presence control.
    assert len(iron_zero_cpp_candidates) == len(iron_zero_py_candidates) == 1, (
        "CEHHAB Fe(0) candidate coverage differs",
        len(iron_zero_cpp_candidates),
        len(iron_zero_py_candidates),
    )
    iron_zero_cpp = iron_zero_cpp_candidates[0]
    iron_zero_py = iron_zero_py_candidates[0]
    # The old inverted C+/N- alternative is no longer this target's winner.
    # Its replacement still fails the zero-metal/organic-cation safeguard.
    assert iron_zero_cpp["metal_discordance_zero_valent_metals_with_organic_cation_count"] == 1
    assert (
        iron_zero_py.metadata["metal_discordance_zero_valent_metals_with_organic_cation_count"] == 1
    )
    assert iron_zero_cpp["organic_charge_localization_polarity_inversion_penalty"] == pytest.approx(
        iron_zero_py.metadata["organic_charge_localization_polarity_inversion_penalty"]
    )


def test_cpp_python_akonew_preserves_cobalt_three_local_charge_state() -> None:
    row = fixture("AKONEW")
    assert row["id"] == "AKONEW"
    xyz_block = _load_tmqmg_xyz(row["id"])
    config = MolGRConfig()
    cpp = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        xyz_block,
        int(row["charge"]),
        0,
        config=config,
    )
    py_layer_index, py_candidates, py_best = _python_scored_candidates(
        xyz_block,
        int(row["charge"]),
        0,
    )

    assert cpp["layer_index"] == py_layer_index == 0
    assert py_best is not None
    assert py_best.no_metal_charge_target == -3
    assert py_best.metal_states[0].valence == 3
    selected_cpp = next(item for item in cpp["candidates"] if item["selected"])
    assert selected_cpp["no_metal_charge_target"] == -3
    assert selected_cpp["metal_states"][0]["valence"] == 3
    assert selected_cpp["organic_charge_localization_component_cancellation"] == pytest.approx(0.7)
    assert len(cpp["candidates"]) == len(py_candidates)


def test_cpp_python_metal_search_stage_parity_case_522() -> None:
    row = fixture("AFAKAW")
    xyz_block = _load_tmqmg_xyz(row["id"])
    total_charge = int(row["charge"])
    total_radical_electrons = 0
    config = MolGRConfig()

    py_summary = _python_metal_search_summary(
        xyz_block,
        total_charge,
        total_radical_electrons,
        config=config,
    )
    cpp_summary = _cpp_metal_search_summary(
        _core.dev.pipeline.reconstruct_with_metals.debug_metal_search_summaries(
            xyz_block,
            total_charge,
            total_radical_electrons,
            config=config,
        )
    )

    assert cpp_summary == py_summary


def test_cpp_python_scored_candidate_parity_case_522() -> None:
    row = fixture("AFAKAW")
    xyz_block = _load_tmqmg_xyz(row["id"])
    total_charge = int(row["charge"])
    total_radical_electrons = 0
    config = MolGRConfig()

    _core.dev.pipeline.clear_force_field_evaluation_cache()
    _core.dev.pipeline.clear_uff_atom_typing_cache()
    cpp = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        xyz_block,
        total_charge,
        total_radical_electrons,
        config=config,
    )
    py_layer_index, py_candidates, py_best = _python_scored_candidates(
        xyz_block,
        total_charge,
        total_radical_electrons,
    )

    assert cpp["layer_index"] == py_layer_index
    assert len(cpp["candidates"]) == len(py_candidates)

    for cpp_item, py_item in zip(cpp["candidates"], py_candidates):
        assert cpp_item["combination_index"] == py_item.metadata.get("combination_index")
        assert cpp_item["no_metal_charge_target"] == py_item.no_metal_charge_target
        assert cpp_item["no_metal_radical_target"] == py_item.no_metal_radical_target
        assert py_item.no_metal_state is not None
        assert cpp_item["no_metal_total_charge"] == py_item.no_metal_state.total_charge
        assert (
            cpp_item["no_metal_total_radical_electrons"]
            == py_item.no_metal_state.total_radical_electrons
        )
        assert tuple(cpp_item["no_metal_atom_signature"]) == _pybel_atom_signature(
            py_item.no_metal_state.omol
        )
        assert tuple(cpp_item["no_metal_bond_signature"]) == _pybel_bond_signature(
            py_item.no_metal_state.omol
        )
        assert cpp_item["score"] == pytest.approx(py_item.score or 0.0)
        assert cpp_item["metal_assignment_rank"] == pytest.approx(
            float(py_item.metadata.get("metal_assignment_rank", 0.0))
        )
        assert cpp_item["organic_aromatic_ring_count"] == py_item.metadata.get(
            "organic_aromatic_ring_count"
        )
        assert cpp_item["organic_aromatic_stability_score"] == pytest.approx(
            float(py_item.metadata.get("organic_aromatic_stability_score", 0.0))
        )
        for metadata_key in (
            "organic_conjugated_atom_count",
            "organic_conjugated_bond_count",
            "organic_max_conjugated_component_size",
            "organic_hyperconjugative_donor_count",
            "organic_hyperconjugation_score",
            "organic_hyperconjugation_max_score",
            "organic_hyperconjugation_deficit",
            "metal_discordance_conjugated_atom_deficit_count",
            "metal_discordance_conjugated_bond_deficit_count",
            "metal_discordance_aromatic_atom_deficit_count",
            "metal_discordance_aromatic_ring_deficit_count",
        ):
            assert cpp_item[metadata_key] == py_item.metadata.get(metadata_key)
        assert cpp_item["organic_charge_localization_penalty"] == pytest.approx(
            float(py_item.metadata.get("organic_charge_localization_penalty", 0.0))
        )
        assert cpp_item["organic_radical_localization_penalty"] == pytest.approx(
            float(py_item.metadata.get("organic_radical_localization_penalty", 0.0))
        )
        assert cpp_item["metal_discordance_structural_count"] == pytest.approx(
            float(py_item.metadata.get("metal_discordance_structural_count", 0.0))
        )
        assert cpp_item["metal_discordance_count"] == pytest.approx(
            float(py_item.metadata.get("metal_discordance_count", 0.0))
        )
        assert cpp_item["metal_discordance_aromatic_stability_deficit"] == pytest.approx(
            float(py_item.metadata.get("metal_discordance_aromatic_stability_deficit", 0.0))
        )
        for metadata_key in (
            "metal_discordance_inner_visible_diradical_count",
            "metal_discordance_excess_visible_singlet_two_electron_center_count",
            "metal_discordance_bent_cumulated_ring_allene_count",
            "metal_discordance_outer_or_invisible_adjacent_double_charge_count",
            "metal_discordance_outer_or_invisible_adjacent_same_sign_double_charge_count",
            "metal_discordance_outer_or_invisible_adjacent_opposite_sign_double_charge_count",
            "metal_discordance_outer_or_invisible_adjacent_unknown_metal_sign_double_charge_count",
            "metal_discordance_inner_visible_adjacent_carbanion_pair_count",
            "metal_discordance_inner_visible_conjugated_carbanion_pair_count",
            "metal_discordance_inner_visible_same_sign_charge_count",
            "metal_discordance_negative_metal_count",
            "metal_discordance_zero_valent_metals_with_organic_cation_count",
            "metal_discordance_unsaturated_organic_cation_count",
            "metal_discordance_repeated_component_charge_asymmetry_count",
            "metal_discordance_haptic_arene_reduction_count",
            "metal_discordance_coordination_geometry_count",
        ):
            assert cpp_item[metadata_key] == py_item.metadata.get(metadata_key)
        assert cpp_item["metal_discordance_negative_metal_penalty"] == pytest.approx(
            float(py_item.metadata.get("metal_discordance_negative_metal_penalty", 0.0))
        )
        for metadata_key in (
            "metal_discordance_negative_metal_outer_sphere_cation_exception",
            "metal_discordance_negative_metal_positive_metal_counterion_exception",
        ):
            assert cpp_item[metadata_key] == py_item.metadata.get(metadata_key)
        assert cpp_item["passes_metal_discordance_filter"] == py_item.metadata.get(
            "passes_metal_discordance_filter"
        )
        assert cpp_item["selection_key"] == py_item.metadata.get("selection_key")

        assert cpp_item["selected"] == (py_best is py_item)


def test_cpp_python_scored_candidate_state_matches_abagam_fixture() -> None:
    xyz_block = Path("tests/data/tmqmg/reconstruction/ABAGAM.xyz").read_text(encoding="utf-8")
    config = MolGRConfig()

    cpp = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        xyz_block,
        0,
        0,
        config=config,
    )
    py_layer_index, py_candidates, py_best = _python_scored_candidates(xyz_block, 0, 0)

    assert cpp["layer_index"] == py_layer_index
    assert len(cpp["candidates"]) == len(py_candidates)
    for cpp_item, py_item in zip(cpp["candidates"], py_candidates):
        assert py_item.no_metal_state is not None
        assert tuple(cpp_item["no_metal_atom_signature"]) == _pybel_atom_signature(
            py_item.no_metal_state.omol
        )
        assert tuple(cpp_item["no_metal_bond_signature"]) == _pybel_bond_signature(
            py_item.no_metal_state.omol
        )
        assert cpp_item["selection_key"] == py_item.metadata.get("selection_key")
        assert cpp_item["selected"] == (py_best is py_item)


@pytest.mark.parametrize("case_id", ["ABISEL", "AFUGEQ", "AGAZOA", "AGICOJ"])
def test_cpp_python_no_metal_resonance_candidate_parity_cases(case_id: str) -> None:
    row = fixture(case_id)
    xyz_block = _load_tmqmg_xyz(row["id"])
    total_charge = int(row["charge"])
    total_radical_electrons = 0
    config = MolGRConfig()
    base_state = preparation.prepare_metal_state(
        xyz_block,
        total_charge,
        total_radical_electrons,
        config=config,
    )
    _layer_index, py_candidates, py_best = _python_scored_candidates(
        xyz_block,
        total_charge,
        total_radical_electrons,
    )

    assert py_best is not None
    assert py_candidates
    target_charge = py_best.no_metal_charge_target
    target_radicals = py_best.no_metal_radical_target

    py_summaries = _python_no_metal_resonance_candidate_summaries(
        base_state.no_metal_xyz_block,
        target_charge,
        target_radicals,
        config=config,
    )
    cpp_summaries = _cpp_no_metal_resonance_candidate_summaries(
        base_state.no_metal_xyz_block,
        target_charge,
        target_radicals,
        config=config,
    )

    assert len(cpp_summaries) == len(py_summaries)
    for cpp_item, py_item in zip(cpp_summaries, py_summaries):
        assert cpp_item["smiles"] == py_item["smiles"]
        assert cpp_item["resonance_index"] == py_item["resonance_index"]
        assert cpp_item["score"] == pytest.approx(py_item["score"])
        assert cpp_item["selection_key"] == pytest.approx(py_item["selection_key"])


@pytest.mark.parametrize("case_id", ["ABISEL", "AFUGEQ", "AGAZOA", "AGICOJ"])
def test_cpp_python_final_smiles_match_resonance_representative_cases(case_id: str) -> None:
    row = fixture(case_id)
    xyz_block = _load_tmqmg_xyz(row["id"])
    total_charge = int(row["charge"])
    total_radical_electrons = 0
    config = MolGRConfig()

    cpp_smiles = _canonical_reconstructed_smiles(
        xyz_block,
        backend="cpp",
        total_charge=total_charge,
        total_radical_electrons=total_radical_electrons,
        config=config,
    )
    py_smiles = _canonical_reconstructed_smiles(
        xyz_block,
        backend="python",
        total_charge=total_charge,
        total_radical_electrons=total_radical_electrons,
        config=config,
    )

    assert cpp_smiles == py_smiles


@pytest.mark.parametrize("case_id", ["ABISEL", "AFAKAW", "AFUGEQ"])
def test_cpp_all_accelerations_preserve_python_final_smiles(case_id: str) -> None:
    row = fixture(case_id)
    xyz_block = _load_tmqmg_xyz(row["id"])
    total_charge = int(row["charge"])
    total_radical_electrons = 0
    baseline_config = MolGRConfig()
    cpp_config = replace(
        baseline_config,
        cpp_backend=replace(
            baseline_config.cpp_backend,
            max_threads=4,
            enable_target_bucket_parallelism=True,
            enable_candidate_scoring_parallelism=False,
            enable_uff_atom_typing_cache=True,
            target_bucket_parallel_threshold=1,
            target_bucket_parallel_max_threads=1,
            candidate_score_parallel_threshold=32,
        ),
    )

    cpp_smiles = _canonical_reconstructed_smiles(
        xyz_block,
        backend="cpp",
        total_charge=total_charge,
        total_radical_electrons=total_radical_electrons,
        config=cpp_config,
    )
    py_smiles = _canonical_reconstructed_smiles(
        xyz_block,
        backend="python",
        total_charge=total_charge,
        total_radical_electrons=total_radical_electrons,
        config=baseline_config,
    )

    assert cpp_smiles == py_smiles


@pytest.mark.parametrize(
    "case_id,metal_charge,rings,ring_charges,contacts",
    [
        (
            "GAYZEO",
            3,
            [[6, 7, 8, 10, 2, 4], [25, 26, 27, 29, 22, 23]],
            [-1, -1],
            [(55, 49), (6, 49), (25, 49)],
        ),
        ("OHIMEA", 4, [[38, 39, 41, 34, 35, 36], [55, 56, 58, 59, 61, 54]], [0, 0], []),
        ("ACADCO", 3, [[8, 14, 7, 17, 16, 15]], [1], []),
        ("SOZCIW", 2, [[45, 47, 49, 51, 53, 11]], [1], []),
    ],
)
def test_reviewed_arene_and_coordination_repairs(
    case_id, metal_charge, rings, ring_charges, contacts
):
    """Assert reviewed local states, without freezing the whole dataset reference."""
    case = next(c for c in _backend_parity_cases() if c["case_id"] == case_id)
    xyz = (_PROJECT_ROOT / "tests/data/tmqmg/backend_parity" / case["xyz_file"]).read_text()
    outputs = []
    for backend in ("cpp", "python"):
        mol = xyz_to_rdmol(xyz, int(case["charge"]), 1, backend=backend, kekulize=False)
        assert Chem.GetFormalCharge(mol) == int(case["charge"])
        assert sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()) == 0
        metals = [a for a in mol.GetAtoms() if a.GetSymbol() == case["metal_center"]]
        assert len(metals) == 1 and metals[0].GetFormalCharge() == metal_charge
        for ring, charge in zip(rings, ring_charges):
            assert all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
            assert sum(mol.GetAtomWithIdx(i).GetFormalCharge() for i in ring) == charge
        for donor, metal in contacts:
            bond = mol.GetBondBetweenAtoms(donor, metal)
            assert bond is not None and bond.GetBondType() == Chem.BondType.DATIVE
        if case_id == "OHIMEA":
            assert mol.HasSubstructMatch(Chem.MolFromSmarts("[#6-]=[#7+]"))
        outputs.append(Chem.MolToSmiles(mol))
    assert outputs[0] == outputs[1]


def test_geometry_triggered_recovery_reports_the_production_gold_candidate():
    xyz = (_PROJECT_ROOT / "tests/data/tmqmg/gold_electronic_assignment/MIKLOK.xyz").read_text()
    config = MolGRConfig()
    native = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        xyz, 0, 0, config=config
    )
    _, candidates, best = _python_scored_candidates(xyz, 0, 0)
    assert best is not None
    assert [s.valence for s in best.metal_states] == [1]
    assert len(native["candidates"]) == len(candidates)
    for cpp, python in zip(native["candidates"], candidates):
        assert cpp["combination_index"] == python.metadata["combination_index"]
        assert cpp["selected"] == (python is best)
        assert cpp["score"] == pytest.approx(python.score)


def test_equalized_three_carbon_candidate_selection_matches_backends():

    case = fixture("CEGGAW")
    xyz, charge = case["xyz"], case["charge"]
    native = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        xyz, charge, 0, config=MolGRConfig()
    )
    _, candidates, best = _python_scored_candidates(xyz, charge, 0)
    assert best is not None
    assert [(state.symbol, state.valence) for state in best.metal_states] == [("W", 4)]
    assert best.metadata["metal_equalized_carbon_reduction_exemption_count"] == 1
    assert len(native["candidates"]) == len(candidates)
    for cpp, python in zip(native["candidates"], candidates):
        assert cpp["combination_index"] == python.metadata["combination_index"]
        assert cpp["selected"] == (python is best)
        for key in (
            "metal_equalized_carbon_reduction_exemption_count",
            "metal_equalized_carbon_localization_count",
            "metal_discordance_structural_count",
        ):
            assert cpp[key] == python.metadata[key]


@pytest.mark.parametrize("case_id", ["GILPUQ", "GIMGOC"])
def test_alkoxy_allylic_candidate_counts_and_selection_match_backends(case_id):

    case = fixture(case_id)
    xyz, charge = case["xyz"], case["charge"]
    native = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        xyz, charge, 0, config=MolGRConfig()
    )
    _, candidates, best = _python_scored_candidates(xyz, charge, 0)
    key = "metal_alkoxy_allylic_geometry_conflict_count"
    assert best is not None
    assert [(s.symbol, s.valence) for s in best.metal_states] == [("W", 2)]
    assert any(c.metadata[key] for c in candidates)
    assert best.metadata[key] == 0
    assert len(native["candidates"]) == len(candidates)
    for cpp, python in zip(native["candidates"], candidates):
        assert cpp["combination_index"] == python.metadata["combination_index"]
        assert cpp["selected"] == (python is best)
        assert cpp[key] == python.metadata[key]
        assert (
            cpp["metal_discordance_structural_count"]
            == python.metadata["metal_discordance_structural_count"]
        )
    before = [c.metadata[key] for c in candidates]
    assert scoring.select_best_candidate(list(reversed(candidates))) is best
    assert scoring.select_best_candidate(candidates) is best
    assert [c.metadata[key] for c in candidates] == before


@pytest.mark.parametrize("case_id,symbol,valence", [("XEYQIB", "Mn", 1)])
def test_bent_nitronium_recovery_and_selection_match_backends(case_id, symbol, valence):

    case = fixture(case_id)
    xyz, charge = case["xyz"], case["charge"]
    native = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        xyz, charge, 0, config=MolGRConfig()
    )
    _, candidates, best = _python_scored_candidates(xyz, charge, 0)
    key = "metal_discordance_zero_valent_metals_with_organic_cation_count"
    assert best is not None
    assert [(s.symbol, s.valence) for s in best.metal_states] == [(symbol, valence)]
    assert best.metadata[key] == 0
    assert len(native["candidates"]) == len(candidates)
    for cpp, python in zip(native["candidates"], candidates):
        assert cpp["combination_index"] == python.metadata["combination_index"]
        assert cpp["selected"] == (python is best)
        assert cpp[key] == python.metadata[key]
        assert cpp["score"] == pytest.approx(python.score)


@pytest.mark.parametrize("case_id,symbol", [("BASDAC", "Fe"), ("BIBZEV", "Fe"), ("BEBQIK", "Ni")])
def test_macrocycle_recovery_counts_and_selection_match_backends(case_id, symbol):

    case = fixture(case_id)
    xyz, charge = case["xyz"], case["charge"]
    native = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        xyz, charge, 0, config=MolGRConfig()
    )
    _, candidates, best = _python_scored_candidates(xyz, charge, 0)
    key = "metal_macrocycle_electron_conflict_count"
    assert best is not None
    assert [(s.symbol, s.valence) for s in best.metal_states] == [(symbol, 2)]
    assert any(c.metadata[key] for c in candidates)
    assert best.metadata[key] == 0
    assert len(native["candidates"]) == len(candidates)
    for cpp, python in zip(native["candidates"], candidates):
        assert cpp["combination_index"] == python.metadata["combination_index"]
        assert cpp["selected"] == (python is best)
        assert cpp[key] == python.metadata[key]
        assert (
            cpp["metal_discordance_structural_count"]
            == python.metadata["metal_discordance_structural_count"]
        )
    before = [c.metadata[key] for c in candidates]
    assert scoring.select_best_candidate(list(reversed(candidates))) is best
    assert scoring.select_best_candidate(candidates) is best
    assert [c.metadata[key] for c in candidates] == before


@pytest.mark.parametrize("case_id,symbol,valence", [("XUMTAD", "Ru", 2)])
def test_local_multiple_bond_geometry_recovery_matches_backends(case_id, symbol, valence):

    case = fixture(case_id)
    xyz, charge = case["xyz"], case["charge"]
    native = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        xyz, charge, 0, config=MolGRConfig()
    )
    _, candidates, best = _python_scored_candidates(xyz, charge, 0)
    assert best is not None
    assert [(s.symbol, s.valence) for s in best.metal_states] == [(symbol, valence)]
    assert len(native["candidates"]) == len(candidates)
    for cpp, python in zip(native["candidates"], candidates):
        assert cpp["combination_index"] == python.metadata["combination_index"]
        assert cpp["selected"] == (python is best)
        assert cpp["score"] == pytest.approx(python.score)


@pytest.mark.parametrize(
    "case_id,symbol,valence", [("BUSWOD", "Fe", 4), ("EHILEM", "Nb", 5), ("KETGOG", "Ni", 4)]
)
def test_folded_ring_and_contributor_ties_match_backends(case_id, symbol, valence):

    case = fixture(case_id)
    xyz, charge, radicals = case["xyz"], case["charge"], case["multiplicity"] - 1
    native = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        xyz, charge, radicals, config=MolGRConfig()
    )
    _, candidates, best = _python_scored_candidates(xyz, charge, radicals)
    key = "metal_folded_ring_geometry_conflict_count"
    assert best is not None
    assert [(s.symbol, s.valence) for s in best.metal_states] == [(symbol, valence)]
    assert best.metadata[key] == 0
    if case_id == "BUSWOD":
        assert max(c.metadata[key] for c in candidates) == 2
    assert len(native["candidates"]) == len(candidates)
    for cpp, python in zip(native["candidates"], candidates):
        assert cpp["combination_index"] == python.metadata["combination_index"]
        assert cpp["selected"] == (python is best)
        assert cpp[key] == python.metadata[key]
        assert (
            cpp["metal_discordance_structural_count"]
            == python.metadata["metal_discordance_structural_count"]
        )
        assert cpp["score"] == pytest.approx(python.score)
    before = [c.metadata[key] for c in candidates]
    assert scoring.select_best_candidate(list(reversed(candidates))) is best
    assert scoring.select_best_candidate(candidates) is best
    assert [c.metadata[key] for c in candidates] == before


@pytest.mark.parametrize("case_id,symbol,valence", [("BIXBUG", "Co", 3)])
def test_formazan_geometry_alternatives_match_backends(case_id, symbol, valence):

    case = fixture(case_id)
    xyz, charge, radicals = case["xyz"], case["charge"], case["multiplicity"] - 1
    native = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        xyz, charge, radicals, config=MolGRConfig()
    )
    _, candidates, best = _python_scored_candidates(xyz, charge, radicals)
    key = "metal_formazan_geometry_conflict_count"
    assert best is not None
    assert [(s.symbol, s.valence) for s in best.metal_states] == [(symbol, valence)]
    assert best.metadata[key] == 0
    assert max(c.metadata[key] for c in candidates) == 1
    assert len(native["candidates"]) == len(candidates)
    for cpp, python in zip(native["candidates"], candidates):
        assert cpp["combination_index"] == python.metadata["combination_index"]
        assert cpp["selected"] == (python is best)
        assert cpp[key] == python.metadata[key]
        assert (
            cpp["metal_discordance_structural_count"]
            == python.metadata["metal_discordance_structural_count"]
        )
        assert cpp["score"] == pytest.approx(python.score)
    before = [c.metadata[key] for c in candidates]
    assert scoring.select_best_candidate(list(reversed(candidates))) is best
    assert scoring.select_best_candidate(candidates) is best
    assert [c.metadata[key] for c in candidates] == before


@pytest.mark.parametrize("case_id", ["BIXTUY", "AYEVEF"])
def test_cationic_donor_penalties_and_selection_match_backends(case_id):

    case = fixture(case_id)
    xyz, charge, radicals = case["xyz"], case["charge"], case["multiplicity"] - 1
    native = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        xyz, charge, radicals, config=MolGRConfig()
    )
    _, candidates, best = _python_scored_candidates(xyz, charge, radicals)
    assert best is not None
    assert len(native["candidates"]) == len(candidates)
    assert any(
        c.metadata["metal_discordance_inner_visible_cationic_donor_count"] for c in candidates
    )
    for cpp, python in zip(native["candidates"], candidates):
        assert cpp["combination_index"] == python.metadata["combination_index"]
        assert cpp["selected"] == (python is best)
        for key in (
            "metal_discordance_inner_visible_same_sign_charge_count",
            "metal_discordance_inner_visible_cationic_donor_count",
            "metal_discordance_inner_visible_same_sign_charge_penalty",
            "metal_discordance_structural_count",
        ):
            assert cpp[key] == pytest.approx(python.metadata[key])


@pytest.mark.parametrize(
    "case_id,symbol,valence",
    [
        ("LEBNIR", "Zn", 4),
        ("YURTOW", "Zn", 2),
        ("UNOQAR", "Zn", 2),
        ("AROJIA", "Ti", 4),
        ("WUQJAT", "Ti", 4),
        ("AYEVEF", "W", 4),
    ],
)
def test_reviewed_cationic_allocation_repairs_match_native_scores(case_id, symbol, valence):

    case = fixture(case_id)
    xyz, charge, radicals = case["xyz"], case["charge"], case["multiplicity"] - 1
    native = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        xyz, charge, radicals, config=MolGRConfig()
    )
    _, candidates, best = _python_scored_candidates(xyz, charge, radicals)
    assert best is not None
    assert [(s.symbol, s.valence) for s in best.metal_states] == [(symbol, valence)]
    assert len(native["candidates"]) == len(candidates)
    for cpp, python in zip(native["candidates"], candidates):
        assert cpp["combination_index"] == python.metadata["combination_index"]
        assert cpp["selected"] == (python is best)
        for key in (
            "metal_discordance_inner_visible_same_sign_charge_penalty",
            "metal_discordance_high_zinc_valence_count",
            "metal_discordance_unsaturated_organic_cation_count",
            "metal_aryl_carbenoid_geometry_conflict_count",
            "metal_discordance_structural_count",
        ):
            assert cpp[key] == pytest.approx(python.metadata[key])
    assert scoring.select_best_candidate(list(reversed(candidates))) is best
    assert scoring.select_best_candidate(candidates) is best


@pytest.mark.parametrize(
    "case_id,symbol,valence",
    [
        ("ARUTIR", "Fe", -2),
        ("DEYCUI", "Fe", -2),
        ("BISJOC", "Co", -1),
        ("NSTPRU", "Ru", -2),
        ("BALDIE", "Fe", -2),
        ("HIKPUP", "Fe", -2),
        ("CEBWUB", "Mn", -1),
        ("OHIHEV", "Re", -1),
        ("WEDNEY", "Re", -1),
    ],
)
def test_nitrosonium_formal_charge_scoring_matches_backends(case_id, symbol, valence):
    xyz = (_PROJECT_ROOT / "tests/data/tmqmg/backend_parity" / f"{case_id}.xyz").read_text()
    native = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        xyz, 0, 0, config=MolGRConfig()
    )
    _, candidates, best = _python_scored_candidates(xyz, 0, 0)
    assert best is not None
    assert [(s.symbol, s.valence) for s in best.metal_states] == [(symbol, valence)]
    assert best.metadata["metal_nitrosonium_formal_charge_exemption"] > 0
    assert len(native["candidates"]) == len(candidates)
    for cpp, python in zip(native["candidates"], candidates):
        assert cpp["combination_index"] == python.metadata["combination_index"]
        assert cpp["selected"] == (python is best)
        for key in (
            "metal_nitrosonium_formal_charge_exemption",
            "metal_discordance_negative_metal_penalty",
            "metal_discordance_inner_visible_same_sign_charge_penalty",
            "metal_discordance_structural_count",
            "nitrosyl_geometry_penalty",
        ):
            assert cpp[key] == pytest.approx(python.metadata[key])
        assert cpp["score"] == pytest.approx(python.score)
    assert scoring.select_best_candidate(list(reversed(candidates))) is best
    assert scoring.select_best_candidate(candidates) is best
