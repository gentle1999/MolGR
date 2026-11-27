"""Shared SMILES/molfile scoring regressions (independent of dataset import)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from rdkit import Chem

from benchmarks._timeout import CaseTimeoutError
from benchmarks.molfile_xyz_benchmark.run import _run_case_method as _run_molfile_case_method
from benchmarks.smiles_xyz_benchmark.io import summarize_results
from benchmarks.smiles_xyz_benchmark.methods import get_method_registry
from benchmarks.smiles_xyz_benchmark.methods.base import MethodRunOutput
from benchmarks.smiles_xyz_benchmark.methods.postprocess import remove_hs_without_sanitize
from benchmarks.smiles_xyz_benchmark.run import _run_case_method as _run_smiles_case_method
from benchmarks.smiles_xyz_benchmark.schema import BenchmarkResult


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCORING_RUNNERS = [
    pytest.param(_run_smiles_case_method, id="smiles"),
    pytest.param(_run_molfile_case_method, id="molfile"),
]


def test_method_registry_only_imports_selected_method_modules(monkeypatch) -> None:
    imported_modules: list[str] = []

    def fake_import_module(module_name: str):
        imported_modules.append(module_name)

        class _Method:
            method_id = "molgr_cpp"

        return type("_Module", (), {"MolGRCppMethod": _Method})

    monkeypatch.setattr(
        "benchmarks.smiles_xyz_benchmark.methods.import_module",
        fake_import_module,
    )

    methods = get_method_registry(("molgr_cpp",))

    assert [method.method_id for method in methods] == ["molgr_cpp"]
    assert imported_modules == ["benchmarks.smiles_xyz_benchmark.methods.molgr_cpp"]


def test_summary_separates_raw_decisions_reference_quality_and_runtime_failures() -> None:
    summary = summarize_results(
        [
            BenchmarkResult(
                case_idx=idx,
                method_id="fake",
                input_smiles="C",
                ground_truth_smiles="C",
                status=status,
                error="method failed" if status == "error" else None,
                predicted_smiles="C" if status == "ok" else None,
                equivalent=equivalent,
                equivalence_method=None,
                evaluator_decision=decision,
                reference_status=reference_status,
                reference_reason="reference evidence" if reference_status != "valid" else None,
                comparison_skipped=comparison_skipped,
                timing_ms_total=0.0 if status == "skipped" else 3.0,
                timing_ms_breakdown={},
            )
            for idx, (
                status,
                decision,
                equivalent,
                reference_status,
                comparison_skipped,
            ) in enumerate(
                [
                    ("ok", "equivalent", True, "excluded", False),
                    ("ok", "equivalent", True, "valid", False),
                    ("error", "not_equivalent", False, "valid", False),
                    ("ok", "not_equivalent", False, "valid", False),
                    ("ok", "inconclusive", None, "valid", False),
                    ("ok", "not_equivalent", False, "unresolved", False),
                    ("skipped", None, None, "valid", True),
                    ("ok", None, True, "valid", False),
                ],
                start=1,
            )
        ]
    )[0]

    assert summary["count"] == 6
    assert summary["success_count"] == 1
    assert summary["fail_count"] == 2
    assert summary["inconclusive_count"] == 1
    assert summary["runtime_failure_count"] == 2
    assert summary["api_error_count"] == 1
    assert summary["reference_excluded_count"] == 1
    assert summary["reference_unresolved_count"] == 1
    assert summary["skip_count"] == 1
    assert summary["comparison_skip_count"] == 1
    assert summary["avg_ms_total"] == 3.0


def test_benchmark_hydrogen_removal_does_not_require_kekulization() -> None:
    mol = Chem.MolFromSmiles("c1cccc1", sanitize=False)
    assert mol is not None

    mol_no_h = remove_hs_without_sanitize(mol)

    assert Chem.MolToSmiles(mol_no_h) == "c1cccc1"


@pytest.mark.parametrize("scoring_runner", _SCORING_RUNNERS)
def test_benchmark_abelok_uses_published_native_target(scoring_runner) -> None:
    from molgr import MolGRConfig
    from molgr.interface import xyz_to_rdmol
    from scripts.tmqmg_fixture_data import reference_graph

    fixtures = _PROJECT_ROOT / "tests" / "data" / "tmqmg" / "reconstruction"
    with (fixtures / "manifest.csv").open() as handle:
        fixture = next(row for row in csv.DictReader(handle) if row["case_id"] == "ABELOK")
    config = MolGRConfig()
    config.cpp_backend.max_threads = 1
    candidate = Chem.RemoveHs(
        xyz_to_rdmol(
            (fixtures / fixture["xyz_file"]).read_text(),
            int(fixture["charge"]),
            1,
            backend="cpp",
            config=config,
        )
    )
    reference = reference_graph("ABELOK")
    result = scoring_runner(
        {
            "case_idx": int(fixture["row_index"]),
            "input_smiles": fixture["reference_smiles"],
            "ground_truth_smiles": fixture["reference_smiles"],
            "ground_truth_rdmol": reference,
            "corrected_standard": True,
            "xyz_block": (fixtures / fixture["xyz_file"]).read_text(),
            "total_charge": int(fixture["charge"]),
            "total_radical_electrons": int(fixture["multiplicity"]) - 1,
        },
        "scoring",
        lambda case: MethodRunOutput(
            status="ok",
            predicted_smiles=Chem.MolToSmiles(candidate),
            rdkit_mol=candidate,
        ),
        case_timeout_seconds=None,
    )

    assert result.status == "ok"
    assert result.evaluator_decision == "equivalent"


@pytest.mark.parametrize("scoring_runner", _SCORING_RUNNERS)
@pytest.mark.parametrize(
    ("reference_smiles", "candidate_smiles", "decision", "equivalent", "count_key"),
    [
        ("C", "C", "equivalent", True, "success_count"),
        ("C", "N", "not_equivalent", False, "fail_count"),
        (
            "[O]N=O",
            "[O-][N+]=O",  # No supported joint charge/radical migration witness.
            "inconclusive",
            None,
            "inconclusive_count",
        ),
    ],
)
def test_benchmark_scoring_preserves_real_evaluator_decisions(
    scoring_runner,
    reference_smiles,
    candidate_smiles,
    decision,
    equivalent,
    count_key,
) -> None:
    result = scoring_runner(
        {
            "case_idx": 1,
            "input_smiles": reference_smiles,
            "ground_truth_smiles": reference_smiles,
            "ground_truth_rdmol": Chem.MolFromSmiles(reference_smiles),
        },
        "scoring",
        lambda case: MethodRunOutput(
            status="ok",
            predicted_smiles=reference_smiles,
            rdkit_mol=Chem.MolFromSmiles(candidate_smiles),
            equivalent=True,
        ),
        case_timeout_seconds=None,
    )

    assert result.status == "ok"
    assert result.evaluator_decision == decision
    assert result.equivalent is equivalent
    summary = summarize_results([result])[0]
    assert summary["count"] == 1
    assert summary[count_key] == 1
    assert summary["runtime_failure_count"] == 0


@pytest.mark.parametrize("scoring_runner", _SCORING_RUNNERS)
@pytest.mark.parametrize("reference_status", ["excluded", "unresolved"])
def test_reference_quality_filters_statistics_not_raw_decision(
    scoring_runner, reference_status
) -> None:
    result = scoring_runner(
        {
            "case_idx": 1,
            "input_smiles": "C",
            "ground_truth_smiles": "C",
            "ground_truth_rdmol": Chem.MolFromSmiles("C"),
            "reference_status": reference_status,
            "reference_reason": "Independent reference review",
        },
        "scoring",
        lambda case: MethodRunOutput(status="ok", rdkit_mol=Chem.MolFromSmiles("N")),
        case_timeout_seconds=None,
    )

    assert result.status == "ok"
    assert result.evaluator_decision == "not_equivalent"
    assert result.equivalent is False
    assert result.reference_status == reference_status
    assert result.reference_reason == "Independent reference review"
    assert result.comparison_skipped is False
    summary = summarize_results([result])[0]
    assert summary["count"] == 0
    assert summary["fail_count"] == 0
    assert summary[f"reference_{reference_status}_count"] == 1


@pytest.mark.parametrize("scoring_runner", _SCORING_RUNNERS)
@pytest.mark.parametrize("failure_type", [CaseTimeoutError, RuntimeError])
def test_evaluator_failure_keeps_valid_reference_in_denominator(
    scoring_runner, failure_type, monkeypatch
) -> None:
    def fail_evaluation(*args, **kwargs):
        raise failure_type("scoring failed")

    monkeypatch.setattr("molgr.utils.equivalence.evaluate_equivalence", fail_evaluation)
    result = scoring_runner(
        {
            "case_idx": 1,
            "input_smiles": "C",
            "ground_truth_smiles": "C",
            "ground_truth_rdmol": Chem.MolFromSmiles("C"),
        },
        "scoring",
        lambda case: MethodRunOutput(status="ok", rdkit_mol=Chem.MolFromSmiles("C")),
        case_timeout_seconds=None,
    )

    assert result.status == "error"
    assert "scoring failed" in result.error
    assert result.reference_status == "valid"
    assert result.evaluator_decision is None
    assert result.equivalent is None
    assert result.comparison_skipped is False
    assert result.comparison_skip_reason is None
    summary = summarize_results([result])[0]
    assert summary["count"] == 1
    assert summary["fail_count"] == 0
    assert summary["runtime_failure_count"] == 1


@pytest.mark.parametrize("scoring_runner", _SCORING_RUNNERS)
@pytest.mark.parametrize("reference_status", ["valid", "excluded", "unresolved"])
@pytest.mark.parametrize("provider_failure", [False, True])
def test_early_failures_preserve_reference_classification(
    scoring_runner, reference_status, provider_failure
) -> None:
    def method_runner(case):
        raise CaseTimeoutError("method timeout")

    result = scoring_runner(
        {
            "case_idx": 1,
            "input_smiles": "C",
            "ground_truth_smiles": "C",
            "ground_truth_rdmol": Chem.MolFromSmiles("C"),
            "reference_status": reference_status,
            "reference_reason": "Frozen reference review",
            "provider_error": "missing input" if provider_failure else None,
        },
        "scoring",
        method_runner,
        case_timeout_seconds=None,
    )

    assert result.status == ("skipped" if provider_failure else "error")
    assert result.reference_status == reference_status
    assert result.reference_reason == "Frozen reference review"
    assert result.evaluator_decision is None
    summary = summarize_results([result])[0]
    assert summary["count"] == int(reference_status == "valid")
    assert summary["runtime_failure_count"] == int(reference_status == "valid")


@pytest.mark.parametrize("scoring_runner", _SCORING_RUNNERS)
def test_missing_reference_is_unresolved_without_trusting_method_equivalence(
    scoring_runner,
) -> None:
    result = scoring_runner(
        {"case_idx": 1, "input_smiles": "C"},
        "scoring",
        lambda case: MethodRunOutput(
            status="ok", rdkit_mol=Chem.MolFromSmiles("C"), equivalent=True
        ),
        case_timeout_seconds=None,
    )

    assert result.reference_status == "unresolved"
    assert result.evaluator_decision is None
    assert result.equivalent is None
    summary = summarize_results([result])[0]
    assert summary["count"] == 0
    assert summary["reference_unresolved_count"] == 1


def test_xyzgraph_explicit_hydrogen_geometry_keeps_methyl_radical() -> None:
    from benchmarks.smiles_xyz_benchmark.methods.xyzgraph_cheminf_full import _graph_to_rdkit_mol

    nx = pytest.importorskip("networkx")
    graph = nx.Graph()
    graph.add_node(0, symbol="C")
    for i in (1, 2, 3):
        graph.add_node(i, symbol="H")
        graph.add_edge(0, i, bond_order=1)
    mol = _graph_to_rdkit_mol(graph)
    Chem.SanitizeMol(mol)
    carbon = mol.GetAtomWithIdx(0)
    assert carbon.GetNumImplicitHs() == 0
    assert carbon.GetNumRadicalElectrons() == 1


@pytest.mark.parametrize("method_id", ["rdkit_determine_bonds", "xyz2mol_tm_local"])
def test_neutral_methyl_radical_without_charge_assignment_error(method_id: str) -> None:
    if method_id == "xyz2mol_tm_local":
        pytest.importorskip("xyz2mol_tm.huckel_to_smiles.xyz2mol_local")
    (method,) = get_method_registry((method_id,))
    result = method.run(
        {
            "xyz_block": "4\nmethyl\nC 0 0 0\nH 1.08 0 0\nH -0.54 .935 0\nH -0.54 -.935 0\n",
            "total_charge": 0,
            "total_radical_electrons": 1,
        }
    )
    assert result.status == "ok", result.error
    assert Chem.MolToSmiles(Chem.RemoveHs(result.rdkit_mol)) == "[CH3]"


@pytest.mark.parametrize(
    ("charge", "radicals", "charged_fragments"),
    [(0, 0, True), (0, 1, False), (0, 2, False), (-1, 1, True), (1, 0, True)],
)
def test_fragment_assignment_uses_charge_and_spin(charge, radicals, charged_fragments):
    from benchmarks.smiles_xyz_benchmark.methods.base import allow_charged_fragments

    assert allow_charged_fragments(charge, radicals) is charged_fragments


def test_xyz2mol_radical_valences_do_not_leak_into_the_next_call(monkeypatch):
    upstream = pytest.importorskip("xyz2mol_tm.huckel_to_smiles.xyz2mol_local")
    baseline = upstream.atomic_valence
    before = {z: tuple(values) for z, values in baseline.items()}
    seen = []

    def record_and_exit(*args, allow_charged_fragments, **kwargs):
        seen.append(
            (
                allow_charged_fragments,
                tuple(upstream.atomic_valence[7]),
                tuple(upstream.atomic_valence[8]),
            )
        )
        upstream.atomic_valence[7] = [99]
        raise SystemExit(1)

    monkeypatch.setattr(upstream, "xyz2mol", record_and_exit)
    (method,) = get_method_registry(("xyz2mol_tm_local",))
    case = {"xyz_block": "1\ncarbon\nC 0 0 0\n", "total_charge": 0}
    for radicals in (1, 0):
        result = method.run(dict(case, total_radical_electrons=radicals))
        assert result.status == "error"
        assert upstream.atomic_valence is baseline
        assert {z: tuple(values) for z, values in baseline.items()} == before
    assert seen == [(False, (3, 2), (2, 1)), (True, before[7], before[8])]
