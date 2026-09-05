# ruff: noqa: I001
from __future__ import annotations

import argparse
import sys
import time
from importlib import import_module
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks._timeout import CaseTimeoutError, case_timeout
from benchmarks.molfile_xyz_benchmark.io import (
    summarize_results,
    write_results_csv,
    write_summary_csv,
)
from benchmarks.molfile_xyz_benchmark.methods import get_method_registry
from benchmarks.molfile_xyz_benchmark.schema import BenchmarkResult
from scripts.molgr_cases_molfile import load_molfile_cases


try:
    from tqdm import tqdm as _tqdm_impl
except ModuleNotFoundError:

    def _tqdm_impl(iterable, **_kwargs):
        return iterable


def _tqdm(*args, **kwargs):
    return _tqdm_impl(*args, **kwargs)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run molfile/SDF XYZ benchmark using the shared method registry."
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Input molfile/SDF path or directory."
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional case limit.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument(
        "--case-timeout-seconds",
        type=float,
        default=1.0,
        help="Per-method per-case wall-time limit. Use 0 to disable.",
    )
    return parser.parse_args()


def _run_case_method(
    case: dict,
    method_id: str,
    method_runner,
    *,
    case_timeout_seconds: float | None,
) -> BenchmarkResult:
    evaluate_equivalence = import_module("molgr.utils.equivalence").evaluate_equivalence

    if case.get("provider_error"):
        breakdown = {"method_ms": 0.0, "equivalence_ms": 0.0}
        return BenchmarkResult(
            case_idx=int(case["case_idx"]),
            method_id=method_id,
            input_smiles=str(case["input_smiles"]),
            ground_truth_smiles=case.get("ground_truth_smiles"),
            status="skipped",
            error=str(case["provider_error"]),
            predicted_smiles=None,
            equivalent=None,
            equivalence_method=None,
            timing_ms_total=0.0,
            timing_ms_breakdown=breakdown,
        )

    started = time.perf_counter()
    try:
        with case_timeout(case_timeout_seconds, f"{method_id} case {case['case_idx']}"):
            output = method_runner(case)
    except CaseTimeoutError as exc:
        method_elapsed_ms = (time.perf_counter() - started) * 1000.0
        breakdown = {"method_ms": method_elapsed_ms, "equivalence_ms": 0.0}
        return BenchmarkResult(
            case_idx=int(case["case_idx"]),
            method_id=method_id,
            input_smiles=str(case["input_smiles"]),
            ground_truth_smiles=case.get("ground_truth_smiles"),
            status="error",
            error=str(exc),
            predicted_smiles=None,
            equivalent=None,
            equivalence_method=None,
            timing_ms_total=method_elapsed_ms,
            timing_ms_breakdown=breakdown,
        )
    method_elapsed_ms = (time.perf_counter() - started) * 1000.0

    breakdown = dict(output.timing_ms_breakdown or {})
    breakdown["method_ms"] = method_elapsed_ms
    breakdown.setdefault("equivalence_ms", 0.0)

    status = output.status
    error = output.error
    equivalent = output.equivalent
    equivalence_method = output.equivalence_method
    evaluator_decision = None

    ground_truth_rdmol = case.get("ground_truth_rdmol")
    if ground_truth_rdmol is not None and output.rdkit_mol is not None:
        equivalence_started = time.perf_counter()
        try:
            with case_timeout(case_timeout_seconds, f"{method_id} equivalence {case['case_idx']}"):
                info = evaluate_equivalence(
                    ground_truth_rdmol,
                    output.rdkit_mol,
                    use_chirality=True,
                    max_resonance=100,
                )
            evaluator_decision = info.decision.value
            equivalent = (
                True
                if evaluator_decision == "equivalent"
                else False
                if evaluator_decision == "not_equivalent"
                else None
            )
            equivalence_method = info.method.value if info.method is not None else None
        except CaseTimeoutError as exc:
            status = "error"
            error = str(exc) if error is None else f"{error}; {exc}"
            equivalent = None
            equivalence_method = None
            evaluator_decision = None
        except Exception as exc:  # noqa: BLE001
            status = "error"
            equivalence_error = f"equivalence check failed: {exc}"
            error = f"{error}; {equivalence_error}" if error else equivalence_error
            equivalent = None
            equivalence_method = None
            evaluator_decision = None
        finally:
            breakdown["equivalence_ms"] = (time.perf_counter() - equivalence_started) * 1000.0

    return BenchmarkResult(
        case_idx=int(case["case_idx"]),
        method_id=method_id,
        input_smiles=str(case["input_smiles"]),
        ground_truth_smiles=case.get("ground_truth_smiles"),
        status=status,
        error=error,
        predicted_smiles=output.predicted_smiles,
        equivalent=equivalent,
        equivalence_method=equivalence_method,
        evaluator_decision=evaluator_decision,
        timing_ms_total=method_elapsed_ms,
        timing_ms_breakdown=breakdown,
    )


def run(
    input_path: Path,
    out_dir: Path,
    limit: int | None = None,
    case_timeout_seconds: float | None = 1.0,
) -> list[BenchmarkResult]:
    cases = load_molfile_cases(input_path=input_path, limit=limit)
    methods = get_method_registry()

    results: list[BenchmarkResult] = []
    for method in methods:
        for case in _tqdm(cases, desc=f"Running {method.method_id}", total=len(cases)):
            results.append(
                _run_case_method(
                    case,
                    method.method_id,
                    method.run,
                    case_timeout_seconds=case_timeout_seconds,
                )
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_results_csv(out_dir / "results.csv", results)
    write_summary_csv(out_dir / "summary.csv", summarize_results(results))
    return results


def main() -> int:
    args = _parse_args()
    run(
        input_path=args.input,
        out_dir=args.out,
        limit=args.limit,
        case_timeout_seconds=args.case_timeout_seconds or None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
