from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from rdkit import Chem

from benchmarks._timeout import CaseTimeoutError
from benchmarks.molfile_xyz_benchmark.run import _run_case_method as _run_molfile_case_method
from benchmarks.smiles_xyz_benchmark.methods.base import MethodRunOutput
from benchmarks.smiles_xyz_benchmark.methods.postprocess import remove_hs_without_sanitize
from benchmarks.smiles_xyz_benchmark.run import _run_case_method as _run_smiles_case_method
from benchmarks.tmqmg_xyz_benchmark.comparison_annotations import (
    find_comparison_annotation,
    load_comparison_annotations,
)
from benchmarks.tmqmg_xyz_benchmark.io import summarize_results
from benchmarks.tmqmg_xyz_benchmark.run import (
    TmqmgBenchmarkInput,
    _build_case,
    _build_cpp_backend_config_payload,
    _fallback_results_for_method,
    _resolve_total_radical_electrons,
    _run_case_method,
    _run_method_cases_worker,
    _run_method_subprocesses,
    _select_methods,
    _selected_rows,
    get_method_registry,
    run,
)
from benchmarks.tmqmg_xyz_benchmark.schema import BenchmarkResult
from molgr.utils.converter import LONE_PAIR_COUNT_PROP


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TMQMG_DATA_DIR = (
    _PROJECT_ROOT / ".local" / "datasets" / "tmqmg" / "e1dc9887b8f20a217a1db6ca972d726bcbaab45b"
)
_TMQMG_CSV = _TMQMG_DATA_DIR / "tmQMg_properties_and_targets.csv"
_TMQMG_XYZ_DIR = _TMQMG_DATA_DIR / "tmQMg_xyz" / "xyz"
_SCORING_RUNNERS = [
    pytest.param(_run_case_method, id="tmqmg"),
    pytest.param(_run_smiles_case_method, id="smiles"),
    pytest.param(_run_molfile_case_method, id="molfile"),
]


@pytest.mark.parametrize(
    "smiles",
    [
        "CC",
        "[CH3]",
        "[C]=S.[C]=S",
    ],
)
def test_resolve_total_radical_electrons_is_fixed_closed_shell_target(smiles: str) -> None:
    assert _resolve_total_radical_electrons({"smiles": smiles}) == 0


def test_build_case_does_not_use_reference_smiles_for_reconstruction_state(tmp_path: Path) -> None:
    xyz_dir = tmp_path / "xyz"
    xyz_dir.mkdir()
    (xyz_dir / "CASE.xyz").write_text("1\nCASE\nC 0 0 0\n", encoding="utf-8")

    case = _build_case(
        1,
        {"id": "CASE", "charge": "-1", "smiles": "[CH3]"},
        xyz_dir=xyz_dir,
    )

    assert case["xyz_block"] == "1\nCASE\nC 0 0 0\n"
    assert case["total_charge"] == -1
    assert case["total_radical_electrons"] == 0
    assert case["ground_truth_smiles"] == "[CH3]"


def test_tmqmg_subset_selection_respects_ids_and_row_bounds() -> None:
    rows = [
        {"id": "A", "smiles": "C", "charge": "0"},
        {"id": "B", "smiles": "CC", "charge": "0"},
        {"id": "C", "smiles": "CCC", "charge": "0"},
        {"id": "D", "smiles": "CCCC", "charge": "0"},
    ]

    selected = _selected_rows(
        rows,
        ids={"B", "D"},
        start_row=2,
        end_row=4,
        limit=1,
    )

    assert [item.row["id"] for item in selected] == ["B"]
    assert [item.row_index for item in selected] == [2]


def test_tmqmg_run_writes_results_for_selected_subset(tmp_path: Path, monkeypatch) -> None:
    csv_path = tmp_path / "tmqmg.csv"
    xyz_dir = tmp_path / "xyz"
    out_dir = tmp_path / "out"
    xyz_dir.mkdir()

    csv_path.write_text(
        "id,smiles,charge\nA,C,0\nB,CC,0\nC,CCC,0\n",
        encoding="utf-8",
    )
    (xyz_dir / "B.xyz").write_text("1\nB\nB 0 0 0\n", encoding="utf-8")

    def _fake_methods(method_ids=None):
        class _Method:
            method_id = "fake"

            def run(self, case):
                return _Output(
                    status="ok",
                    error=None,
                    predicted_smiles=case["input_smiles"],
                    rdkit_mol=None,
                    equivalent=True,
                    equivalence_method="ideal",
                    timing_ms_breakdown={"method_ms": 1.0},
                )

        return [_Method()]

    class _Output:
        def __init__(
            self,
            *,
            status: str,
            error: str | None,
            predicted_smiles: str | None,
            rdkit_mol,
            equivalent: bool | None,
            equivalence_method: str | None,
            timing_ms_breakdown: dict[str, float],
        ) -> None:
            self.status = status
            self.error = error
            self.predicted_smiles = predicted_smiles
            self.rdkit_mol = rdkit_mol
            self.equivalent = equivalent
            self.equivalence_method = equivalence_method
            self.timing_ms_breakdown = timing_ms_breakdown

    monkeypatch.setattr("benchmarks.tmqmg_xyz_benchmark.run.get_method_registry", _fake_methods)
    monkeypatch.setattr(
        "benchmarks.tmqmg_xyz_benchmark.run._run_method_subprocess",
        lambda **kwargs: [
            type(
                "_Result",
                (),
                {
                    "case_idx": 2,
                    "case_id": "B",
                    "method_id": "fake",
                    "input_smiles": "CC",
                    "ground_truth_smiles": "CC",
                    "status": "ok",
                    "error": None,
                    "predicted_smiles": "CC",
                    "equivalent": True,
                    "evaluator_decision": "equivalent",
                    "reference_status": "valid",
                    "reference_reason": None,
                    "equivalence_method": "ideal",
                    "comparison_skipped": False,
                    "comparison_skip_reason": None,
                    "timing_ms_total": 1.0,
                    "timing_ms_breakdown": {"method_ms": 1.0},
                },
            )
        ],
    )

    results = run(
        csv_path=csv_path,
        xyz_dir=xyz_dir,
        out_dir=out_dir,
        limit=1,
        start_row=2,
        end_row=3,
        ids=["B"],
    )

    assert len(results) == 1
    assert (out_dir / "results.csv").exists()
    assert (out_dir / "summary.csv").exists()

    with (out_dir / "results.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [row["id"] for row in rows] == ["B"]
    assert [row["input_smiles"] for row in rows] == ["CC"]


def test_tmqmg_run_filters_selected_methods(tmp_path: Path, monkeypatch) -> None:
    csv_path = tmp_path / "tmqmg.csv"
    xyz_dir = tmp_path / "xyz"
    out_dir = tmp_path / "out"
    xyz_dir.mkdir()

    csv_path.write_text("id,smiles,charge\nA,C,0\n", encoding="utf-8")
    (xyz_dir / "A.xyz").write_text("1\nA\nC 0 0 0\n", encoding="utf-8")

    class _Method:
        def __init__(self, method_id: str) -> None:
            self.method_id = method_id

    monkeypatch.setattr(
        "benchmarks.tmqmg_xyz_benchmark.run.get_method_registry",
        lambda method_ids=None: [
            _Method(method_id)
            for method_id in (method_ids or ("skip_me", "molgr_fallback", "molgr_cpp"))
        ],
    )

    called_method_ids: list[str] = []

    def _fake_run_method_subprocess(**kwargs):
        method_id = kwargs["method_id"]
        called_method_ids.append(method_id)
        return [
            BenchmarkResult(
                case_idx=1,
                method_id=method_id,
                input_smiles="C",
                ground_truth_smiles="C",
                status="ok",
                error=None,
                predicted_smiles="C",
                equivalent=True,
                equivalence_method="ideal",
                evaluator_decision="equivalent",
                reference_status="valid",
                timing_ms_total=1.0,
                timing_ms_breakdown={"method_ms": 1.0},
                case_id="A",
            )
        ]

    monkeypatch.setattr(
        "benchmarks.tmqmg_xyz_benchmark.run._run_method_subprocess",
        _fake_run_method_subprocess,
    )

    results = run(
        csv_path=csv_path,
        xyz_dir=xyz_dir,
        out_dir=out_dir,
        method_ids=["molgr_cpp", "molgr_fallback"],
    )

    assert called_method_ids == ["molgr_cpp", "molgr_fallback"]
    assert [result.method_id for result in results] == ["molgr_cpp", "molgr_fallback"]


def test_tmqmg_method_filter_rejects_unknown_method_id() -> None:
    class _Method:
        method_id = "known"

    try:
        _select_methods([_Method()], ["missing"])
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_tmqmg_method_registry_only_imports_selected_method_modules(monkeypatch) -> None:
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


def test_cpp_acceleration_preset_enables_available_cpp_backend_switches() -> None:
    cpp_backend_config = _build_cpp_backend_config_payload(use_all_accelerations=True)

    assert cpp_backend_config["enable_target_bucket_parallelism"] is True
    assert cpp_backend_config["enable_candidate_scoring_parallelism"] is False
    assert cpp_backend_config["enable_uff_atom_typing_cache"] is True
    assert cpp_backend_config["enable_target_bucket_score_bundle_preheat"] is True
    assert cpp_backend_config["target_bucket_parallel_threshold"] == 1
    assert cpp_backend_config["target_bucket_parallel_max_threads"] is None
    assert cpp_backend_config["candidate_score_parallel_threshold"] == 32


def test_cpp_uff_atom_typing_cache_can_be_enabled_with_default_preset() -> None:
    cpp_backend_config = _build_cpp_backend_config_payload(
        use_all_accelerations=False,
        enable_uff_atom_typing_cache=True,
    )

    assert cpp_backend_config["enable_target_bucket_parallelism"] is True
    assert cpp_backend_config["enable_candidate_scoring_parallelism"] is False
    assert cpp_backend_config["enable_uff_atom_typing_cache"] is True
    assert cpp_backend_config["target_bucket_parallel_threshold"] == 1
    assert cpp_backend_config["target_bucket_parallel_max_threads"] is None


def test_tmqmg_worker_applies_cpp_backend_config_before_running_methods(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seen_configs = []
    xyz_dir = tmp_path / "xyz"
    xyz_dir.mkdir()
    (xyz_dir / "A.xyz").write_text("2\nA\nC 0 0 0\nC 0 0 1\n", encoding="utf-8")

    class _Output:
        def __init__(self, config):
            self.status = "ok"
            self.error = None
            self.predicted_smiles = "CC"
            self.rdkit_mol = None
            self.equivalent = True
            self.equivalence_method = "ideal"
            self.timing_ms_breakdown = {"method_ms": 1.0}
            self._config = config

    class _Method:
        method_id = "fake"

        def run(self, case):
            from molgr.config import CONFIG

            config = CONFIG
            seen_configs.append(config.cpp_backend)
            return _Output(config)

    monkeypatch.setattr(
        "benchmarks.tmqmg_xyz_benchmark.run.get_method_registry",
        lambda method_ids=None: [_Method()],
    )

    payload = {
        "method_id": "fake",
        "cases": [{"row_index": 1, "row": {"id": "A", "smiles": "CC", "charge": "0"}}],
        "xyz_dir": str(xyz_dir),
        "case_timeout_seconds": 1.0,
        "cpp_backend_config": {
            "max_threads": None,
            "enable_target_bucket_parallelism": True,
            "enable_candidate_scoring_parallelism": True,
            "enable_uff_atom_typing_cache": False,
            "enable_target_bucket_score_bundle_preheat": True,
            "target_bucket_parallel_threshold": 4,
            "target_bucket_parallel_max_threads": 1,
            "candidate_score_parallel_threshold": 1,
        },
    }

    results = _run_method_cases_worker(payload)

    assert len(results) == 1
    assert seen_configs
    assert seen_configs[0].enable_target_bucket_parallelism is True
    assert seen_configs[0].enable_candidate_scoring_parallelism is True
    assert seen_configs[0].enable_uff_atom_typing_cache is False
    assert seen_configs[0].enable_target_bucket_score_bundle_preheat is True
    assert seen_configs[0].target_bucket_parallel_threshold == 4
    assert seen_configs[0].target_bucket_parallel_max_threads == 1


def test_tmqmg_cpp_method_uses_one_subprocess_with_native_workers(
    monkeypatch,
) -> None:
    run_calls = []
    payloads = []

    def fake_run(*args, **kwargs):
        run_calls.append((args, kwargs))
        payload = json.loads(
            Path(kwargs["env"]["MOLGR_TMQMG_SUBPROCESS_PAYLOAD_PATH"]).read_text(encoding="utf-8")
        )
        payloads.append(payload)
        output_path = Path(kwargs["env"]["MOLGR_TMQMG_SUBPROCESS_OUT"])
        with output_path.open("w", encoding="utf-8") as fh:
            for item in payload["cases"]:
                row_index = int(item["row_index"])
                fh.write(
                    json.dumps(
                        BenchmarkResult(
                            case_idx=row_index,
                            method_id=payload["method_id"],
                            input_smiles="C",
                            ground_truth_smiles="C",
                            status="ok",
                            error=None,
                            predicted_smiles="C",
                            equivalent=True,
                            equivalence_method="ideal",
                            timing_ms_total=float(row_index),
                            timing_ms_breakdown={"method_ms": float(row_index)},
                            case_id=item["row"]["id"],
                        ).to_dict(),
                        ensure_ascii=True,
                    )
                    + "\n"
                )
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    cases = [TmqmgBenchmarkInput(row_index=idx, row={"id": f"C{idx}"}) for idx in range(1, 6)]
    results = _run_method_subprocesses(
        method_id="molgr_cpp",
        cases=cases,
        xyz_dir=Path("xyz"),
        case_timeout_seconds=1.0,
        cpp_backend_config=_build_cpp_backend_config_payload(use_all_accelerations=True),
        process_workers=3,
        roundtrip_diagnostics=False,
    )

    assert len(run_calls) == 1
    payload = payloads[0]
    assert payload["native_workers"] == 3
    assert payload["native_batch"] is True
    assert payload["roundtrip_diagnostics"] is False
    assert len(payload["cases"]) == 5
    assert [result.case_idx for result in results] == [1, 2, 3, 4, 5]
    assert [result.case_id for result in results] == ["C1", "C2", "C3", "C4", "C5"]


def test_build_case_marks_provider_error_for_missing_xyz(tmp_path: Path) -> None:
    case = _build_case(
        2,
        {"id": "B", "smiles": "CC", "charge": "0"},
        xyz_dir=tmp_path / "xyz",
    )

    assert case["provider_error"] is not None
    assert case["xyz_block"] is None
    assert case["reference_status"] == "valid"


def test_build_case_marks_reference_error_for_reference_xyz_element_mismatch(
    tmp_path: Path,
) -> None:
    xyz_dir = tmp_path / "xyz"
    xyz_dir.mkdir()
    (xyz_dir / "A.xyz").write_text(
        "2\nA\nC 0 0 0\nH 0 0 1\n",
        encoding="utf-8",
    )

    case = _build_case(
        1,
        {"id": "A", "smiles": "C", "charge": "0"},
        xyz_dir=xyz_dir,
    )

    assert case["provider_error"] is None
    assert case["xyz_block"] is not None
    assert case["reference_error"] is not None
    assert "Reference SMILES element counts differ from XYZ" in case["reference_error"]
    assert case["reference_status"] == "excluded"
    assert case["reference_reason"] == case["reference_error"]


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
    assert summary["fail_count"] == 1
    assert summary["inconclusive_count"] == 1
    assert summary["runtime_failure_count"] == 3
    assert summary["reference_excluded_count"] == 1
    assert summary["reference_unresolved_count"] == 1
    assert summary["skip_count"] == 1
    assert summary["comparison_skip_count"] == 1
    assert summary["avg_ms_total"] == 3.0


def test_reference_element_mismatch_preserves_raw_comparison_and_method(
    tmp_path: Path,
) -> None:
    xyz_dir = tmp_path / "xyz"
    xyz_dir.mkdir()
    (xyz_dir / "A.xyz").write_text(
        "2\nA\nC 0 0 0\nH 0 0 1\n",
        encoding="utf-8",
    )
    case = _build_case(
        1,
        {"id": "A", "smiles": "C", "charge": "0"},
        xyz_dir=xyz_dir,
    )
    calls = {"count": 0}

    class _Output:
        status = "ok"
        error = None
        predicted_smiles = "C"
        rdkit_mol = Chem.MolFromSmiles("C")
        equivalent = True
        equivalence_method = "method-native"
        timing_ms_breakdown = {"method_ms": 1.0}

    def _runner(_case):
        calls["count"] += 1
        return _Output()

    result = _run_case_method(
        case,
        "fake",
        _runner,
        case_timeout_seconds=1.0,
    )

    assert calls["count"] == 1
    assert result.status == "ok"
    assert result.error is None
    assert result.predicted_smiles == "C"
    assert result.case_id == "A"
    assert result.equivalent is True
    assert result.evaluator_decision == "equivalent"
    assert result.reference_status == "excluded"
    assert "Reference SMILES element counts differ from XYZ" in str(result.reference_reason)
    assert result.comparison_skipped is False
    assert result.comparison_skip_reason is None


def test_boron_cluster_annotation_records_1176_cases_and_yulboy_exception() -> None:
    annotations = load_comparison_annotations()

    assert len(annotations) == 1
    assert annotations[0].status == "no_clear_evidence_boron_cluster"
    assert annotations[0].expected_case_count == 1176
    assert find_comparison_annotation("ADOCOL", {"B": 4}) == annotations[0]
    assert find_comparison_annotation("YULBOY", {"B": 4}) is None
    assert find_comparison_annotation("ADOCOL", {"B": 3}) is None


def test_boron_cluster_runs_reconstruction_but_skips_answer_comparison() -> None:
    annotation = find_comparison_annotation("ADOCOL", {"B": 4})
    assert annotation is not None
    calls = {"count": 0}

    class _Output:
        status = "ok"
        error = None
        predicted_smiles = "B.B.B.B"
        rdkit_mol = Chem.MolFromSmiles("B.B.B.B")
        equivalent = True
        equivalence_method = "method-native"
        timing_ms_breakdown = {"method_ms": 1.0}

    def _runner(_case):
        calls["count"] += 1
        return _Output()

    result = _run_case_method(
        {
            "case_idx": 1,
            "id": "ADOCOL",
            "input_smiles": "B.B.B.B",
            "ground_truth_smiles": "B.B.B.B",
            "ground_truth_rdmol": Chem.MolFromSmiles("B.B.B.B"),
            "reference_error": None,
            "provider_error": None,
            "comparison_annotation": annotation,
        },
        "fake",
        _runner,
        case_timeout_seconds=1.0,
    )

    assert calls["count"] == 1
    assert result.status == "ok"
    assert result.predicted_smiles == "B.B.B.B"
    assert result.equivalent is None
    assert result.equivalence_method is None
    assert result.evaluator_decision is None
    assert result.reference_status == "unresolved"
    assert result.comparison_skipped is True
    assert "neither answer is treated as assessable" in str(result.comparison_skip_reason)


def test_benchmark_hydrogen_removal_does_not_require_kekulization() -> None:
    mol = Chem.MolFromSmiles("c1cccc1", sanitize=False)
    assert mol is not None

    mol_no_h = remove_hs_without_sanitize(mol)

    assert Chem.MolToSmiles(mol_no_h) == "c1cccc1"


def test_equivalence_uses_internal_candidate_not_recorded_smiles() -> None:
    ground_truth = Chem.MolFromSmiles("C")
    backend_object = Chem.MolFromSmiles("N")
    assert ground_truth is not None
    assert backend_object is not None

    class _Output:
        status = "ok"
        error = None
        predicted_smiles = "C"
        rdkit_mol = backend_object
        equivalent = None
        equivalence_method = None
        timing_ms_breakdown = {"method_ms": 1.0}

    result = _run_case_method(
        {
            "case_idx": 1,
            "id": "A",
            "input_smiles": "C",
            "ground_truth_smiles": "C",
            "ground_truth_rdmol": ground_truth,
            "reference_error": None,
            "provider_error": None,
        },
        "fake",
        lambda case: _Output(),
        case_timeout_seconds=1.0,
    )

    assert result.status == "ok"
    assert result.equivalent is False
    assert result.evaluator_decision == "not_equivalent"
    assert result.comparison_skipped is False
    assert result.smiles_roundtrip_status == "roundtrip_changed"
    assert result.smiles_roundtrip_parse_success is True
    assert result.smiles_roundtrip_equivalent is False


def test_skipping_roundtrip_keeps_reference_equivalence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ground_truth = Chem.MolFromSmiles("C")
    backend_object = Chem.MolFromSmiles("C")
    assert ground_truth is not None
    assert backend_object is not None

    class _Output:
        status = "ok"
        error = None
        predicted_smiles = "C"
        rdkit_mol = backend_object
        timing_ms_breakdown = {"method_ms": 1.0}

    def unexpected_roundtrip(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("roundtrip diagnostics should be disabled")

    monkeypatch.setattr(
        "benchmarks.tmqmg_xyz_benchmark.run._smiles_roundtrip_diagnostic",
        unexpected_roundtrip,
    )
    result = _run_case_method(
        {
            "case_idx": 1,
            "id": "A",
            "input_smiles": "C",
            "ground_truth_smiles": "C",
            "ground_truth_rdmol": ground_truth,
            "reference_error": None,
            "provider_error": None,
        },
        "fake",
        lambda case: _Output(),
        case_timeout_seconds=1.0,
        roundtrip_diagnostics=False,
    )

    assert result.evaluator_decision == "equivalent"
    assert result.smiles_roundtrip_status is None
    assert "roundtrip_ms" not in (result.timing_ms_breakdown or {})


def test_smiles_reparse_failure_does_not_override_correct_internal_equivalence() -> None:
    ground_truth = Chem.MolFromSmiles("C")
    assert ground_truth is not None

    class _Output:
        status = "ok"
        error = None
        predicted_smiles = "not valid smiles"
        rdkit_mol = ground_truth
        equivalent = None
        equivalence_method = None
        timing_ms_breakdown = {"method_ms": 1.0}

    result = _run_case_method(
        {
            "case_idx": 1,
            "id": "A",
            "input_smiles": "C",
            "ground_truth_smiles": "C",
            "ground_truth_rdmol": ground_truth,
            "reference_error": None,
            "provider_error": None,
        },
        "fake",
        lambda case: _Output(),
        case_timeout_seconds=1.0,
    )

    assert result.status == "ok"
    assert result.error is None
    assert result.predicted_smiles == "not valid smiles"
    assert result.equivalent is True
    assert result.evaluator_decision == "equivalent"
    assert result.equivalence_method == "ideal"
    assert result.comparison_skipped is False
    assert result.comparison_skip_reason is None
    assert result.smiles_roundtrip_status == "reparse_failed"
    assert result.smiles_roundtrip_parse_success is False
    assert result.smiles_roundtrip_equivalent is None


def test_low_valence_internal_candidate_can_match_reference_despite_lossy_smiles() -> None:
    reference = Chem.MolFromSmiles("C[C]C")
    assert reference is not None
    internal = Chem.Mol(reference)
    center = internal.GetAtomWithIdx(1)
    center.SetNumRadicalElectrons(0)
    center.SetIntProp(LONE_PAIR_COUNT_PROP, 1)
    internal.UpdatePropertyCache(strict=False)

    class _Output:
        status = "ok"
        error = None
        predicted_smiles = Chem.MolToSmiles(internal)
        rdkit_mol = internal
        equivalent = None
        equivalence_method = None
        timing_ms_breakdown = {"method_ms": 1.0}

    result = _run_case_method(
        {
            "case_idx": 1,
            "id": "LOW_VALENCE",
            "input_smiles": "C[C]C",
            "ground_truth_smiles": "C[C]C",
            "ground_truth_rdmol": reference,
            "reference_error": None,
            "provider_error": None,
        },
        "fake",
        lambda case: _Output(),
        case_timeout_seconds=1.0,
    )

    assert result.equivalent is True
    assert result.equivalence_method == "ideal"
    assert result.smiles_roundtrip_status == "roundtrip_changed"
    assert result.smiles_roundtrip_parse_success is True
    assert result.smiles_roundtrip_equivalent is False
    assert result.smiles_roundtrip_diagnostics is not None
    assert result.smiles_roundtrip_diagnostics["delta"] == {
        "total_h": 2,
        "formal_charge": 0,
        "radical_electrons": 0,
        "active_lone_pairs": -1,
        "formula_changed": True,
    }


def test_internal_mismatch_remains_failure_when_smiles_reparse_also_fails() -> None:
    reference = Chem.MolFromSmiles("C")
    internal = Chem.MolFromSmiles("N")
    assert reference is not None
    assert internal is not None

    class _Output:
        status = "ok"
        error = None
        predicted_smiles = "not valid smiles"
        rdkit_mol = internal
        equivalent = None
        equivalence_method = None
        timing_ms_breakdown = {"method_ms": 1.0}

    result = _run_case_method(
        {
            "case_idx": 1,
            "id": "INTERNAL_MISMATCH",
            "input_smiles": "C",
            "ground_truth_smiles": "C",
            "ground_truth_rdmol": reference,
            "reference_error": None,
            "provider_error": None,
        },
        "fake",
        lambda case: _Output(),
        case_timeout_seconds=1.0,
    )

    assert result.equivalent is False
    assert result.comparison_skipped is False
    assert result.smiles_roundtrip_status == "reparse_failed"


@pytest.mark.parametrize(
    ("case_id", "candidate_smiles", "reference_smiles", "roundtrip_status", "expected_equivalent"),
    [
        (
            "WEXMIW_PATTERN",
            "CCOC(=O)[C@H]1CSC(=[Cr@OH27](<-[C-]#[O+])(<-[C-]#[O+])"
            "(<-[C-]#[O+])(<-[C-]#[O+])<-[C-]#[O+])C=C(c2ccccc2)N1",
            "CCOC(=O)C1CS#C(->[Cr](<-[C-]#[O+])(<-[C-]#[O+])"
            "(<-[C-]#[O+])(<-[C-]#[O+])<-[C-]#[O+])C=C(c2ccccc2)N1",
            "preserved",
            None,  # Matching InChIKey without a bounded structural witness is inconclusive.
        ),
        (
            "NON_KEKULIZABLE_PATTERN",
            "O=C=C=Cc12->[Pt](<-P(c3ccccc3)(c3ccccc3)c3ccccc3)"
            "(<-P(c3ccccc3)(c3ccccc3)c3ccccc3)<-c1c(=O)co2",
            "[O-]C#C[CH-]c12->[Pt+4](<-c1c([O-])[c-]o2)"
            "(<-P(c1ccccc1)(c1ccccc1)c1ccccc1)<-P(c1ccccc1)(c1ccccc1)c1ccccc1",
            "reparse_failed",
            None,  # Native input rejects a graph that cannot be Kekulized.
        ),
    ],
)
def test_internal_graph_decision_is_independent_of_roundtrip_status(
    case_id: str,
    candidate_smiles: str,
    reference_smiles: str,
    roundtrip_status: str,
    expected_equivalent: bool | None,
) -> None:
    internal = Chem.MolFromSmiles(candidate_smiles, sanitize=False)
    reference = Chem.MolFromSmiles(reference_smiles)
    assert internal is not None
    assert reference is not None
    internal.UpdatePropertyCache(strict=False)

    class _Output:
        status = "ok"
        error = None
        predicted_smiles = Chem.MolToSmiles(internal)
        rdkit_mol = internal
        equivalent = None
        equivalence_method = None
        timing_ms_breakdown = {"method_ms": 1.0}

    result = _run_case_method(
        {
            "case_idx": 1,
            "id": case_id,
            "input_smiles": reference_smiles,
            "ground_truth_smiles": reference_smiles,
            "ground_truth_rdmol": reference,
            "reference_error": None,
            "provider_error": None,
        },
        "fake",
        lambda case: _Output(),
        case_timeout_seconds=1.0,
    )

    assert result.equivalent is expected_equivalent
    assert (
        result.evaluator_decision
        == {
            True: "equivalent",
            False: "not_equivalent",
            None: "inconclusive",
        }[expected_equivalent]
    )
    assert result.comparison_skipped is False
    assert result.smiles_roundtrip_status == roundtrip_status


def test_tmqmg_cpp_all_accelerations_worker_survives_target_bucket_parallelism(
    tmp_path: Path,
) -> None:
    pytest.importorskip("molgr._core.pipeline")
    if not _TMQMG_CSV.exists() or not _TMQMG_XYZ_DIR.exists():
        pytest.skip("tmQMg benchmark dataset is not available")

    with _TMQMG_CSV.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    case_row = next((row for row in rows if row.get("id") == "ABAJAP"), None)
    if case_row is None or not (_TMQMG_XYZ_DIR / "ABAJAP.xyz").exists():
        pytest.skip("tmQMg ABAJAP crash-regression case is not available")

    payload_path = tmp_path / "payload.json"
    out_jsonl = tmp_path / "results.jsonl"
    repeats = 8
    payload_path.write_text(
        json.dumps(
            {
                "method_id": "molgr_cpp",
                "cases": [{"row_index": 5, "row": case_row} for _ in range(repeats)],
                "xyz_dir": str(_TMQMG_XYZ_DIR),
                "case_timeout_seconds": 2.0,
                "cpp_backend_config": _build_cpp_backend_config_payload(use_all_accelerations=True),
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    out_jsonl.write_text("", encoding="utf-8")

    env = os.environ.copy()
    env["MOLGR_TMQMG_SUBPROCESS_PAYLOAD_PATH"] = str(payload_path)
    env["MOLGR_TMQMG_SUBPROCESS_OUT"] = str(out_jsonl)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.tmqmg_xyz_benchmark.run",
            "--subprocess-worker",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    result_rows = [
        json.loads(line)
        for line in out_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(result_rows) == repeats
    assert {row["method_id"] for row in result_rows} == {"molgr_cpp"}
    assert {row["case_id"] for row in result_rows} == {"ABAJAP"}


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


@pytest.mark.parametrize(
    ("manual_entry", "expected_status"),
    [
        (
            {
                "status": "accepted_reference_issue",
                "reference_answer_wrong": True,
                "reason": "Independent reference error",
            },
            "excluded",
        ),
        ({"status": "accepted_reference_issue", "reason": "Needs confirmation"}, "unresolved"),
        ({"status": "accepted_ambiguous", "reason": "Needs confirmation"}, "unresolved"),
    ],
)
def test_build_case_uses_manual_reference_evidence_independently(
    tmp_path, monkeypatch, manual_entry, expected_status
) -> None:
    (tmp_path / "REVIEW_CASE.xyz").write_text("1\nreview\nHe 0 0 0\n", encoding="utf-8")
    monkeypatch.setattr(
        "benchmarks.tmqmg_xyz_benchmark.run.load_manual_reference_records",
        lambda: {"REVIEW_CASE": manual_entry},
    )
    case = _build_case(
        1,
        {"id": "REVIEW_CASE", "smiles": "[He]", "charge": "0", "review_note": "retained"},
        xyz_dir=tmp_path,
    )

    assert case["review_note"] == "retained"
    assert case["reference_status"] == expected_status
    assert case["reference_reason"] == manual_entry["reason"]
    result = _run_case_method(
        case,
        "scoring",
        lambda case: MethodRunOutput(status="ok", rdkit_mol=Chem.MolFromSmiles("[He]")),
        case_timeout_seconds=None,
    )
    assert result.evaluator_decision == "equivalent"
    assert result.equivalent is True
    assert result.reference_status == expected_status


def test_fallback_classifies_references_using_actual_xyz_directory(tmp_path) -> None:
    (tmp_path / "VALID_CASE.xyz").write_text("1\nvalid\nHe 0 0 0\n", encoding="utf-8")
    (tmp_path / "WRONG_CASE.xyz").write_text("1\nwrong\nNe 0 0 0\n", encoding="utf-8")
    results = _fallback_results_for_method(
        "scoring",
        [
            TmqmgBenchmarkInput(idx, {"id": case_id, "smiles": "[He]", "charge": "0"})
            for idx, case_id in enumerate(["VALID_CASE", "WRONG_CASE"], start=1)
        ],
        "subprocess failed",
        xyz_dir=tmp_path,
    )

    assert [result.reference_status for result in results] == ["valid", "excluded"]
    assert results[1].reference_reason.startswith(
        "ValueError: Reference SMILES element counts differ from XYZ:"
    )
    summary = summarize_results(results)[0]
    assert summary["count"] == 1
    assert summary["runtime_failure_count"] == 1
    assert summary["reference_excluded_count"] == 1


@pytest.mark.parametrize(
    ("flags", "expected_status"),
    [
        (
            {
                "reference_answer_wrong": "true",
                "reference_answer_reason": "Independent reference error",
            },
            "excluded",
        ),
        ({"reference_status": "excluded"}, "unresolved"),
        (
            {
                "reference_status": "unresolved",
                "reference_reason": "Review still pending",
            },
            "unresolved",
        ),
    ],
)
def test_build_case_preserves_csv_reference_review_flags(tmp_path, flags, expected_status) -> None:
    (tmp_path / "CSV_REVIEW.xyz").write_text("1\nreview\nHe 0 0 0\n", encoding="utf-8")
    case = _build_case(
        1,
        {"id": "CSV_REVIEW", "smiles": "[He]", "charge": "0", **flags},
        xyz_dir=tmp_path,
    )

    assert case["reference_status"] == expected_status
    for key in ("reference_answer_wrong", "reference_answer_reason"):
        if key in flags:
            assert case[key] == flags[key]
