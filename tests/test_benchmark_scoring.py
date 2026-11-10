from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from benchmarks.smiles_xyz_benchmark.io import (
    summarize_results,
    write_results_csv,
    write_summary_csv,
)
from benchmarks.smiles_xyz_benchmark.schema import BenchmarkResult
from benchmarks.tmqmg_xyz_benchmark.comparison_annotations import (
    load_manual_reference_records,
    reference_validity,
)


@pytest.mark.parametrize("case_id", ["ABEGOD", "ABATEC"])
@pytest.mark.parametrize("decision", ["equivalent", "not_equivalent", "inconclusive"])
def test_historical_reference_records_do_not_certify_correctness(
    case_id: str, decision: str
) -> None:
    record = load_manual_reference_records()[case_id]

    status, _ = reference_validity({"equivalence_decision": decision}, record)

    assert status == "unresolved"


@pytest.mark.parametrize(
    ("confirmed", "reason", "expected"),
    [
        (True, "Confirmed reference error", "excluded"),
        (True, "", "unresolved"),
        (False, "Candidate preferred", "unresolved"),
    ],
)
def test_reference_issue_requires_explicit_confirmation(
    confirmed: bool, reason: str, expected: str
) -> None:
    status, _ = reference_validity(
        {},
        {
            "status": "accepted_reference_issue",
            "reference_answer_wrong": confirmed,
            "reason": reason,
        },
    )

    assert status == expected


def test_reference_parse_failure_is_not_a_confirmed_reference_error() -> None:
    status, _ = reference_validity(
        {"reference_error": "ValueError: reference SMILES could not be parsed"}
    )

    assert status == "unresolved"


@pytest.mark.parametrize(
    ("formula_match", "expected_status"),
    [("False", "excluded"), (" false ", "excluded"), ("0", "excluded"), ("True", "valid")],
)
def test_reference_formula_evidence_survives_csv_boolean_strings(
    formula_match: str, expected_status: str
) -> None:
    rows = csv.DictReader(
        io.StringIO(
            "reference_formula_match,reference_formula_mismatch_detail,reference_status\n"
            f"{formula_match},confirmed XYZ/reference formula mismatch,valid\n"
        )
    )

    status, reason = reference_validity(next(rows))

    assert status == expected_status
    if expected_status == "excluded":
        assert reason == "confirmed XYZ/reference formula mismatch"


def test_summary_and_csv_keep_reference_partition_and_inconclusive(tmp_path: Path) -> None:
    rows = []
    specifications = [
        ("valid", "ok", "equivalent"),
        ("valid", "ok", "not_equivalent"),
        ("valid", "ok", "inconclusive"),
        ("valid", "error", None),
        ("valid", "skipped", None),
        ("excluded", "ok", "equivalent"),
        ("excluded", "ok", "not_equivalent"),
        ("unresolved", "ok", "equivalent"),
        ("unresolved", "ok", "inconclusive"),
    ]
    for index, (reference_status, status, decision) in enumerate(specifications):
        rows.append(
            BenchmarkResult(
                case_idx=index,
                method_id="frozen_graphs",
                input_smiles="",
                ground_truth_smiles=None,
                status=status,
                error="method failed" if status == "error" else None,
                predicted_smiles=None,
                equivalent=True
                if decision == "equivalent"
                else False
                if decision == "not_equivalent"
                else None,
                equivalence_method=None,
                timing_ms_total=1.0,
                timing_ms_breakdown={},
                evaluator_decision=decision,
                reference_status=reference_status,
                reference_reason=None
                if reference_status == "valid"
                else "Reference-quality record",
            )
        )
    summary = summarize_results(rows)
    counts = summary[0]
    assert counts["total_count"] == 9
    assert counts["count"] == counts["reference_valid_count"] == 5
    assert counts["reference_excluded_count"] == 2
    assert counts["reference_unresolved_count"] == 2
    assert counts["success_count"] == 1
    assert counts["fail_count"] == 1
    assert counts["inconclusive_count"] == 1
    assert counts["runtime_failure_count"] == 2

    results_path = tmp_path / "results.csv"
    summary_path = tmp_path / "summary.csv"
    write_results_csv(results_path, rows)
    write_summary_csv(summary_path, summary)
    with results_path.open() as handle:
        saved = list(csv.DictReader(handle))
    assert saved[2]["evaluator_decision"] == "inconclusive"
    assert saved[2]["equivalent"] == ""
    assert saved[5]["reference_status"] == "excluded"
    assert saved[5]["evaluator_decision"] == "equivalent"
    with summary_path.open() as handle:
        saved_summary = next(csv.DictReader(handle))
    assert saved_summary["reference_valid_count"] == "5"
    assert saved_summary["inconclusive_count"] == "1"

    from benchmarks.tmqmg_xyz_benchmark.report import _accuracy_by_method, _load_results

    report = _accuracy_by_method(_load_results(results_path)).iloc[0]
    assert report["count"] == 5
    assert report["accuracy"] == pytest.approx(1 / 5)


def test_frozen_valid_reference_is_not_reopened_by_legacy_whitelist() -> None:
    record = load_manual_reference_records()["ABEGOD"]

    status, _ = reference_validity(
        {"reference_status": "valid", "reference_reason": "Frozen approved reference set"},
        record,
    )

    assert status == "valid"
