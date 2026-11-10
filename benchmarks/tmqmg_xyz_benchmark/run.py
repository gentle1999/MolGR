# ruff: noqa: I001
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks._timeout import CaseTimeoutError, case_timeout
from benchmarks.tmqmg_xyz_benchmark.io import (
    summarize_results,
    write_results_csv,
    write_summary_csv,
)
from benchmarks.tmqmg_xyz_benchmark.comparison_annotations import (
    find_comparison_annotation,
    load_manual_reference_records,
    reference_validity,
)
from benchmarks.tmqmg_xyz_benchmark.methods import METHOD_IDS, get_method_registry
from benchmarks.tmqmg_xyz_benchmark.schema import BenchmarkResult
from benchmarks.smiles_xyz_benchmark.methods.base import MethodRunOutput

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


try:
    from tqdm import tqdm as _tqdm_impl
except ModuleNotFoundError:

    def _tqdm_impl(iterable, **_kwargs):
        return iterable


def _tqdm(*args, **kwargs):
    return _tqdm_impl(*args, **kwargs)


@dataclass(frozen=True)
class TmqmgBenchmarkInput:
    row_index: int
    row: dict[str, str]


def _roundtrip_molecule_snapshot(mol: Chem.Mol) -> dict[str, Any]:
    """Return cheap electronic/formula diagnostics without modifying ``mol``."""

    from molgr.utils.converter import get_atom_lone_pair_count

    def safe_value(callback) -> Any:
        try:
            return callback()
        except Exception as exc:  # noqa: BLE001
            return f"unavailable: {type(exc).__name__}: {exc}"

    return {
        "formula": safe_value(lambda: rdMolDescriptors.CalcMolFormula(mol)),
        "total_h": safe_value(
            lambda: sum(
                int(atom.GetTotalNumHs(includeNeighbors=True))
                for atom in mol.GetAtoms()
                if atom.GetAtomicNum() != 1
            )
        ),
        "formal_charge": safe_value(
            lambda: sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms())
        ),
        "radical_electrons": safe_value(
            lambda: sum(int(atom.GetNumRadicalElectrons()) for atom in mol.GetAtoms())
        ),
        "active_lone_pairs": safe_value(
            lambda: sum(get_atom_lone_pair_count(atom) for atom in mol.GetAtoms())
        ),
    }


def _roundtrip_deltas(
    internal: dict[str, Any],
    reparsed: dict[str, Any],
) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    for key in ("total_h", "formal_charge", "radical_electrons", "active_lone_pairs"):
        left = internal.get(key)
        right = reparsed.get(key)
        if isinstance(left, int) and isinstance(right, int):
            deltas[key] = right - left
    deltas["formula_changed"] = internal.get("formula") != reparsed.get("formula")
    return deltas


def _smiles_roundtrip_diagnostic(
    internal_mol: Chem.Mol,
    predicted_smiles: str,
    *,
    timeout_seconds: float | None,
    method_id: str,
    case_idx: int,
) -> dict[str, Any]:
    """Diagnose I -> SMILES -> S independently of reference accuracy."""

    from molgr.utils.equivalence import EquivalenceDecision, evaluate_equivalence

    internal_snapshot = _roundtrip_molecule_snapshot(internal_mol)
    try:
        reparsed = Chem.MolFromSmiles(predicted_smiles)
    except Exception as exc:  # noqa: BLE001
        reparsed = None
        parse_reason = f"{type(exc).__name__}: {exc}"
    else:
        parse_reason = ""
    if reparsed is None:
        return {
            "status": "reparse_failed",
            "parse_success": False,
            "equivalent": None,
            "equivalence_method": None,
            "reason": parse_reason or "predicted_smiles could not be reparsed",
            "diagnostics": {"internal": internal_snapshot, "reparsed": None},
        }

    reparsed_snapshot = _roundtrip_molecule_snapshot(reparsed)
    diagnostics = {
        "internal": internal_snapshot,
        "reparsed": reparsed_snapshot,
        "delta": _roundtrip_deltas(internal_snapshot, reparsed_snapshot),
    }
    try:
        with case_timeout(
            timeout_seconds,
            f"{method_id} SMILES roundtrip {case_idx}",
        ):
            info = evaluate_equivalence(
                internal_mol,
                reparsed,
                use_chirality=False,
                max_resonance=100,
            )
            equivalent = (
                None
                if info.decision == EquivalenceDecision.INCONCLUSIVE
                else info.decision == EquivalenceDecision.EQUIVALENT
            )
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "check_failed",
            "parse_success": True,
            "equivalent": None,
            "equivalence_method": None,
            "reason": f"roundtrip equivalence check failed: {type(exc).__name__}: {exc}",
            "diagnostics": diagnostics,
        }
    return {
        "status": (
            "inconclusive"
            if equivalent is None
            else "preserved"
            if equivalent
            else "roundtrip_changed"
        ),
        "parse_success": True,
        "equivalent": equivalent,
        "equivalence_method": info.method.value if info.method is not None else None,
        "reason": info.reason,
        "diagnostics": diagnostics,
    }


def _build_cpp_backend_config_payload(
    *,
    use_all_accelerations: bool,
    enable_uff_atom_typing_cache: bool = False,
) -> dict[str, Any]:
    use_uff_atom_typing_cache = use_all_accelerations or enable_uff_atom_typing_cache
    if not use_all_accelerations:
        return {
            "max_threads": None,
            "enable_target_bucket_parallelism": True,
            "enable_candidate_scoring_parallelism": False,
            "enable_uff_atom_typing_cache": use_uff_atom_typing_cache,
            "enable_target_bucket_score_bundle_preheat": True,
            "target_bucket_parallel_threshold": 1,
            "target_bucket_parallel_max_threads": None,
            "candidate_score_parallel_threshold": 32,
        }
    return {
        "max_threads": None,
        "enable_target_bucket_parallelism": True,
        "enable_candidate_scoring_parallelism": False,
        "enable_uff_atom_typing_cache": use_uff_atom_typing_cache,
        "enable_target_bucket_score_bundle_preheat": True,
        "target_bucket_parallel_threshold": 1,
        "target_bucket_parallel_max_threads": None,
        "candidate_score_parallel_threshold": 32,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tmQMg XYZ benchmark using the shared method registry."
    )
    parser.add_argument("--csv", type=Path, required=True, help="tmQMg metadata CSV path.")
    parser.add_argument(
        "--xyz-dir",
        type=Path,
        required=True,
        help="Directory containing tmQMg XYZ files named <id>.xyz.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit.")
    parser.add_argument("--start-row", type=int, default=1, help="1-based CSV row index to start.")
    parser.add_argument("--end-row", type=int, default=None, help="1-based CSV row index to end.")
    parser.add_argument(
        "--ids",
        action="append",
        default=None,
        help="Optional tmQMg id filter. Repeat or pass comma-separated values.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument("--progress-every", type=int, default=10, help="Progress print cadence.")
    parser.add_argument(
        "--process-workers",
        type=int,
        default=1,
        help=(
            "External worker budget for non-C++ methods. For molgr_cpp this is "
            "the native batch worker count; 1 keeps the benchmark serial while "
            "retaining MolGR's internal metal-bucket parallelism."
        ),
    )
    parser.add_argument(
        "--case-timeout-seconds",
        type=float,
        default=1.0,
        help="Per-method per-case wall-time limit. Use 0 to disable.",
    )
    parser.add_argument(
        "--skip-smiles-roundtrip",
        action="store_true",
        help="Skip the independent internal-molecule to SMILES roundtrip diagnostic.",
    )
    parser.add_argument(
        "--cpp-accelerations",
        choices=("default", "all"),
        default="default",
        help="C++ backend acceleration preset to use in benchmark workers.",
    )
    parser.add_argument(
        "--enable-uff-atom-typing-cache",
        dest="enable_uff_atom_typing_cache",
        action="store_true",
        help=(
            "Enable MolGR's vendor UFF atom-typing cache when using the "
            "default C++ acceleration preset. The C++ backend always uses "
            "the thread-safe vendor UFF implementation."
        ),
    )
    parser.add_argument(
        "--methods",
        action="append",
        default=None,
        help=(
            "Optional method id filter. Repeat or pass comma-separated values. "
            f"Available: {', '.join(METHOD_IDS)}."
        ),
    )
    parser.add_argument("--subprocess-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.start_row < 1:
        parser.error("--start-row must be >= 1")
    if args.end_row is not None and args.end_row < args.start_row:
        parser.error("--end-row must be >= --start-row")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.progress_every < 0:
        parser.error("--progress-every must be >= 0")
    if args.process_workers < 1:
        parser.error("--process-workers must be >= 1")
    if args.case_timeout_seconds < 0:
        parser.error("--case-timeout-seconds must be >= 0")
    return args


def _split_repeated_values(values: Sequence[str] | None) -> list[str]:
    if not values:
        return []
    split_values: list[str] = []
    for raw in values:
        split_values.extend(part.strip() for part in raw.split(",") if part.strip())
    return split_values


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"tmQMg CSV has no header row: {csv_path}")
        return [dict(row) for row in reader]


def _selected_rows(
    rows: list[dict[str, str]],
    *,
    ids: set[str] | None,
    start_row: int,
    end_row: int | None,
    limit: int | None,
) -> list[TmqmgBenchmarkInput]:
    selected: list[TmqmgBenchmarkInput] = []
    for row_index, row in enumerate(rows, start=1):
        if row_index < start_row:
            continue
        if end_row is not None and row_index > end_row:
            break
        if ids is not None and row.get("id", "").strip() not in ids:
            continue
        selected.append(TmqmgBenchmarkInput(row_index=row_index, row=row))
        if limit is not None and len(selected) >= limit:
            break
    return selected


def _load_case_xyz(xyz_dir: Path, row: dict[str, str]) -> tuple[str, Path]:
    case_id = row.get("id", "").strip()
    if not case_id:
        raise ValueError("Missing id column")
    xyz_path = xyz_dir / f"{case_id}.xyz"
    return xyz_path.read_text(encoding="utf-8"), xyz_path


def _format_element_counts(counts: Counter[str]) -> str:
    return ",".join(f"{symbol}:{counts[symbol]}" for symbol in sorted(counts))


def _xyz_element_counts(xyz_block: str) -> Counter[str]:
    lines = xyz_block.splitlines()
    if not lines:
        raise ValueError("XYZ block is empty")
    try:
        atom_count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError("XYZ atom count line is invalid") from exc
    atom_lines = lines[2 : 2 + atom_count]
    if len(atom_lines) != atom_count:
        raise ValueError("XYZ block has fewer atom lines than declared")
    counts: Counter[str] = Counter()
    for line in atom_lines:
        parts = line.split()
        if not parts:
            raise ValueError("XYZ atom line is empty")
        counts[parts[0]] += 1
    return counts


def _reference_element_counts(reference_smiles: str) -> Counter[str]:
    mol = Chem.MolFromSmiles(reference_smiles)
    if mol is None:
        raise ValueError("reference SMILES could not be parsed")
    mol = Chem.AddHs(mol)
    return Counter(atom.GetSymbol() for atom in cast(Any, mol).GetAtoms())


def _validate_reference_matches_xyz(reference_smiles: str, xyz_block: str) -> None:
    if not reference_smiles:
        return
    reference_counts = _reference_element_counts(reference_smiles)
    xyz_counts = _xyz_element_counts(xyz_block)
    if reference_counts != xyz_counts:
        raise ValueError(
            "Reference SMILES element counts differ from XYZ: "
            f"reference={_format_element_counts(reference_counts)}; "
            f"xyz={_format_element_counts(xyz_counts)}"
        )


def _resolve_total_radical_electrons(row: dict[str, str]) -> int:
    """Return the tmQMg benchmark's fixed closed-shell target without reading SMILES."""

    del row
    return 0


def _build_case(
    row_index: int,
    row: dict[str, str],
    *,
    xyz_dir: Path,
) -> dict[str, Any]:
    case: dict[str, Any] = {
        **row,
        "case_idx": row_index,
        "input_smiles": row.get("smiles", "").strip(),
        "ground_truth_rdmol": None,
        "ground_truth_smiles": None,
        "xyz_block": None,
        "total_charge": None,
        "total_radical_electrons": None,
        "provider_error": None,
        "reference_error": row.get("reference_error") or None,
        "comparison_annotation": None,
        "xyz_path": None,
        "id": row.get("id", "").strip(),
    }

    reference_smiles = case["input_smiles"]
    case["ground_truth_smiles"] = reference_smiles or None
    try:
        case["ground_truth_rdmol"] = (
            Chem.MolFromSmiles(reference_smiles) if reference_smiles else None
        )
    except Exception as exc:  # noqa: BLE001
        case["reference_error"] = f"{type(exc).__name__}: {exc}"
    if case["ground_truth_rdmol"] is None and not case["reference_error"]:
        case["reference_error"] = "Missing or unparseable reference SMILES"

    try:
        xyz_block, xyz_path = _load_case_xyz(xyz_dir, row)
        case["comparison_annotation"] = find_comparison_annotation(
            case["id"],
            _xyz_element_counts(xyz_block),
        )
        if case["ground_truth_rdmol"] is not None:
            try:
                _validate_reference_matches_xyz(reference_smiles, xyz_block)
            except ValueError as exc:
                case["reference_error"] = f"{type(exc).__name__}: {exc}"
        total_charge = int(row.get("charge", "0").strip() or 0)
        total_radical_electrons = _resolve_total_radical_electrons(row)

        case.update(
            {
                "xyz_block": xyz_block,
                "total_charge": total_charge,
                "total_radical_electrons": total_radical_electrons,
                "xyz_path": xyz_path,
            }
        )
    except Exception as exc:  # noqa: BLE001
        case["provider_error"] = f"{type(exc).__name__}: {exc}"

    case["reference_status"], case["reference_reason"] = reference_validity(
        case, load_manual_reference_records().get(case["id"])
    )
    annotation = case["comparison_annotation"]
    if (
        annotation is not None
        and annotation.skip_comparison
        and case["reference_status"] == "valid"
    ):
        case["reference_status"] = "unresolved"
        case["reference_reason"] = annotation.comparison_skip_reason

    return case


def _run_case_method(
    case: dict,
    method_id: str,
    method_runner,
    *,
    case_timeout_seconds: float | None,
    precomputed_output: MethodRunOutput | None = None,
    roundtrip_diagnostics: bool = True,
) -> BenchmarkResult:
    from molgr.utils.equivalence import EquivalenceDecision, evaluate_equivalence

    comparison_annotation = case.get("comparison_annotation")
    comparison_skipped = bool(
        comparison_annotation is not None and comparison_annotation.skip_comparison
    )
    comparison_skip_reason = (
        comparison_annotation.comparison_skip_reason
        if comparison_annotation is not None and comparison_skipped
        else None
    )
    reference_status = case.get("reference_status")
    reference_reason = case.get("reference_reason")
    if reference_status is None:
        reference_status, reference_reason = reference_validity(
            case, load_manual_reference_records().get(case.get("id", ""))
        )
    if case.get("ground_truth_rdmol") is None and reference_status == "valid":
        reference_status, reference_reason = "unresolved", "Missing or unparseable reference SMILES"
    if comparison_skipped and reference_status == "valid":
        reference_status, reference_reason = "unresolved", comparison_skip_reason
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
            reference_status=reference_status,
            reference_reason=reference_reason,
            case_id=case.get("id"),
            comparison_skipped=comparison_skipped,
            comparison_skip_reason=comparison_skip_reason,
        )

    started = time.perf_counter()
    if precomputed_output is None:
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
                reference_status=reference_status,
                reference_reason=reference_reason,
                case_id=case.get("id"),
                comparison_skipped=comparison_skipped,
                comparison_skip_reason=comparison_skip_reason,
            )
        method_elapsed_ms = (time.perf_counter() - started) * 1000.0
    else:
        output = precomputed_output
        output_breakdown = output.timing_ms_breakdown or {}
        method_elapsed_ms = float(output_breakdown.get("method_ms", sum(output_breakdown.values())))
    breakdown = dict(output.timing_ms_breakdown or {})
    breakdown["method_ms"] = method_elapsed_ms
    breakdown.setdefault("equivalence_ms", 0.0)

    status = output.status
    error = output.error
    equivalent = None
    equivalence_method = None
    evaluator_decision = None
    roundtrip: dict[str, Any] | None = None
    if roundtrip_diagnostics and output.rdkit_mol is not None and output.predicted_smiles:
        roundtrip_started = time.perf_counter()
        roundtrip = _smiles_roundtrip_diagnostic(
            output.rdkit_mol,
            output.predicted_smiles,
            timeout_seconds=case_timeout_seconds,
            method_id=method_id,
            case_idx=int(case["case_idx"]),
        )
        breakdown["roundtrip_ms"] = (time.perf_counter() - roundtrip_started) * 1000.0
    ground_truth_rdmol = case.get("ground_truth_rdmol")
    if comparison_skipped:
        equivalent = None
        equivalence_method = None
    elif ground_truth_rdmol is not None and output.rdkit_mol is not None:
        eq_started = time.perf_counter()
        try:
            with case_timeout(case_timeout_seconds, f"{method_id} equivalence {case['case_idx']}"):
                info = evaluate_equivalence(
                    ground_truth_rdmol,
                    output.rdkit_mol,
                    # tmQMg reference SMILES do not consistently encode stereochemistry.
                    use_chirality=False,
                    max_resonance=100,
                )
            evaluator_decision = info.decision.value
            equivalent = (
                None
                if info.decision == EquivalenceDecision.INCONCLUSIVE
                else info.decision == EquivalenceDecision.EQUIVALENT
            )
            equivalence_method = info.method.value if info.method is not None else None
        except CaseTimeoutError as exc:
            status = "error"
            error = f"{error}; {exc}" if error else str(exc)
        except Exception as exc:  # noqa: BLE001
            status = "error"
            equivalence_error = f"equivalence check failed: {exc}"
            error = f"{error}; {equivalence_error}" if error else equivalence_error
        finally:
            breakdown["equivalence_ms"] = (time.perf_counter() - eq_started) * 1000.0

    return BenchmarkResult(
        case_idx=int(case["case_idx"]),
        method_id=method_id,
        input_smiles=str(case["input_smiles"]),
        ground_truth_smiles=case.get("ground_truth_smiles"),
        status=cast(Any, status),
        error=error,
        predicted_smiles=output.predicted_smiles,
        equivalent=equivalent,
        equivalence_method=equivalence_method,
        evaluator_decision=evaluator_decision,
        reference_status=reference_status,
        reference_reason=reference_reason,
        timing_ms_total=method_elapsed_ms,
        timing_ms_breakdown=breakdown,
        case_id=case.get("id"),
        comparison_skipped=comparison_skipped,
        comparison_skip_reason=comparison_skip_reason,
        smiles_roundtrip_status=roundtrip["status"] if roundtrip else None,
        smiles_roundtrip_parse_success=roundtrip["parse_success"] if roundtrip else None,
        smiles_roundtrip_equivalent=roundtrip["equivalent"] if roundtrip else None,
        smiles_roundtrip_equivalence_method=(
            roundtrip["equivalence_method"] if roundtrip else None
        ),
        smiles_roundtrip_reason=roundtrip["reason"] if roundtrip else None,
        smiles_roundtrip_diagnostics=roundtrip["diagnostics"] if roundtrip else None,
    )


def _run_method_cases_worker(payload: dict[str, Any]) -> list[BenchmarkResult]:
    method_id = payload["method_id"]
    cases = payload["cases"]
    xyz_dir = Path(payload["xyz_dir"])
    case_timeout_seconds = payload["case_timeout_seconds"]
    roundtrip_diagnostics = payload.get("roundtrip_diagnostics", True)
    cpp_backend_config_payload = payload.get("cpp_backend_config") or {}
    from molgr.config import CONFIG, CppBackendConfig, MolGRConfig

    runtime_config = MolGRConfig()
    CONFIG.resonance = runtime_config.resonance
    CONFIG.cpp_backend = CppBackendConfig(**cpp_backend_config_payload)
    CONFIG.organic_topology = runtime_config.organic_topology
    CONFIG.metal_scoring = runtime_config.metal_scoring
    CONFIG.metal_radical_inference = runtime_config.metal_radical_inference
    CONFIG.interface = runtime_config.interface

    methods = {method.method_id: method for method in get_method_registry((method_id,))}
    method = methods[method_id]
    built_cases = [
        _build_case(int(item["row_index"]), item["row"], xyz_dir=xyz_dir) for item in cases
    ]
    results: list[BenchmarkResult] = []
    batch_runner = getattr(method, "run_batch", None)
    if method_id == "molgr_cpp" and payload.get("native_batch", False) and callable(batch_runner):
        batch_outputs = cast(Any, batch_runner)(
            built_cases,
            max_workers=payload.get("native_workers"),
        )
        for case in built_cases:
            output = batch_outputs.get(
                int(case["case_idx"]),
                MethodRunOutput(
                    status="error",
                    error="native batch returned no result for case",
                    timing_ms_breakdown={"method_ms": 0.0},
                ),
            )
            results.append(
                _run_case_method(
                    case,
                    method_id,
                    method.run,
                    case_timeout_seconds=case_timeout_seconds,
                    precomputed_output=output,
                    roundtrip_diagnostics=roundtrip_diagnostics,
                )
            )
        return results
    for case in built_cases:
        results.append(
            _run_case_method(
                case,
                method_id,
                method.run,
                case_timeout_seconds=case_timeout_seconds,
                roundtrip_diagnostics=roundtrip_diagnostics,
            )
        )
    return results


def _select_methods(methods, method_ids: Sequence[str] | None):
    if not method_ids:
        return list(methods)
    selected_ids = set(method_ids)
    selected = [method for method in methods if method.method_id in selected_ids]
    found_ids = {method.method_id for method in selected}
    unknown_ids = sorted(selected_ids - found_ids)
    if unknown_ids:
        raise ValueError(f"Unknown benchmark method id(s): {', '.join(unknown_ids)}")
    return selected


def _write_worker_results(out_jsonl: Path, results: list[BenchmarkResult]) -> None:
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for result in results:
            fh.write(json.dumps(result.to_dict(), ensure_ascii=True) + "\n")


def _run_method_subprocess(
    *,
    method_id: str,
    cases: list[TmqmgBenchmarkInput],
    xyz_dir: Path,
    case_timeout_seconds: float | None,
    cpp_backend_config: dict[str, Any],
    native_workers: int | None = None,
    native_batch: bool = False,
    roundtrip_diagnostics: bool = True,
) -> list[BenchmarkResult]:
    payload = {
        "method_id": method_id,
        "cases": [{"row_index": item.row_index, "row": item.row} for item in cases],
        "xyz_dir": str(xyz_dir),
        "case_timeout_seconds": case_timeout_seconds,
        "cpp_backend_config": cpp_backend_config,
        "native_workers": native_workers,
        "native_batch": native_batch,
        "roundtrip_diagnostics": roundtrip_diagnostics,
    }
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp_payload:
        payload_path = Path(tmp_payload.name)
        json.dump(payload, tmp_payload, ensure_ascii=True)
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as tmp_results:
        out_jsonl = Path(tmp_results.name)
    try:
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
            env=env,
            capture_output=True,
            text=True,
        )
        results: list[BenchmarkResult] = []
        with out_jsonl.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                results.append(BenchmarkResult(**json.loads(line)))
        if completed.returncode not in (0,):
            return results
        return results
    finally:
        payload_path.unlink(missing_ok=True)
        out_jsonl.unlink(missing_ok=True)


def _run_method_subprocesses(
    *,
    method_id: str,
    cases: list[TmqmgBenchmarkInput],
    xyz_dir: Path,
    case_timeout_seconds: float | None,
    cpp_backend_config: dict[str, Any],
    process_workers: int,
    roundtrip_diagnostics: bool = True,
) -> list[BenchmarkResult]:
    if method_id == "molgr_cpp" and process_workers > 1:
        # One process owns one native batch executor. Splitting this method
        # into subprocesses would recreate the oversubscription hazard that
        # the native batch backend is designed to remove.
        return _run_method_subprocess(
            method_id=method_id,
            cases=cases,
            xyz_dir=xyz_dir,
            case_timeout_seconds=case_timeout_seconds,
            cpp_backend_config=cpp_backend_config,
            native_workers=process_workers,
            native_batch=True,
            roundtrip_diagnostics=roundtrip_diagnostics,
        )
    if process_workers <= 1 or len(cases) <= 1:
        return _run_method_subprocess(
            method_id=method_id,
            cases=cases,
            xyz_dir=xyz_dir,
            case_timeout_seconds=case_timeout_seconds,
            cpp_backend_config=cpp_backend_config,
            roundtrip_diagnostics=roundtrip_diagnostics,
        )

    worker_count = min(process_workers, len(cases))
    chunks = [cases[worker_idx::worker_count] for worker_idx in range(worker_count)]
    payload_paths: list[Path] = []
    out_jsonl_paths: list[Path] = []
    processes: list[subprocess.Popen[str]] = []
    try:
        for chunk in chunks:
            payload = {
                "method_id": method_id,
                "cases": [{"row_index": item.row_index, "row": item.row} for item in chunk],
                "xyz_dir": str(xyz_dir),
                "case_timeout_seconds": case_timeout_seconds,
                "cpp_backend_config": cpp_backend_config,
                "roundtrip_diagnostics": roundtrip_diagnostics,
            }
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp_payload:
                payload_path = Path(tmp_payload.name)
                json.dump(payload, tmp_payload, ensure_ascii=True)
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as tmp_results:
                out_jsonl = Path(tmp_results.name)
            payload_paths.append(payload_path)
            out_jsonl_paths.append(out_jsonl)

            env = os.environ.copy()
            env["MOLGR_TMQMG_SUBPROCESS_PAYLOAD_PATH"] = str(payload_path)
            env["MOLGR_TMQMG_SUBPROCESS_OUT"] = str(out_jsonl)
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "benchmarks.tmqmg_xyz_benchmark.run",
                        "--subprocess-worker",
                    ],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )

        completed = [process.communicate() for process in processes]
        results: list[BenchmarkResult] = []
        for out_jsonl in out_jsonl_paths:
            with out_jsonl.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    results.append(BenchmarkResult(**json.loads(line)))
        for process, (_stdout, _stderr) in zip(processes, completed):
            if process.returncode not in (0,):
                return results
        return sorted(results, key=lambda result: result.case_idx)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
        for path in payload_paths:
            path.unlink(missing_ok=True)
        for path in out_jsonl_paths:
            path.unlink(missing_ok=True)


def _subprocess_worker_main() -> int:
    payload_path_raw = os.environ.get("MOLGR_TMQMG_SUBPROCESS_PAYLOAD_PATH")
    out_jsonl_raw = os.environ.get("MOLGR_TMQMG_SUBPROCESS_OUT")
    if not payload_path_raw:
        raise SystemExit("missing MOLGR_TMQMG_SUBPROCESS_PAYLOAD_PATH")
    if not out_jsonl_raw:
        raise SystemExit("missing MOLGR_TMQMG_SUBPROCESS_OUT")
    payload = json.loads(Path(payload_path_raw).read_text(encoding="utf-8"))
    out_results = _run_method_cases_worker(payload)
    _write_worker_results(Path(out_jsonl_raw), out_results)
    return 0


def _fallback_results_for_method(
    method_id: str,
    cases: list[TmqmgBenchmarkInput],
    error_message: str,
    *,
    xyz_dir: Path,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    for item in cases:
        case = _build_case(item.row_index, item.row, xyz_dir=xyz_dir)
        results.append(
            BenchmarkResult(
                case_idx=item.row_index,
                method_id=method_id,
                input_smiles=case["input_smiles"],
                ground_truth_smiles=case.get("ground_truth_smiles"),
                status="error",
                error=error_message,
                predicted_smiles=None,
                equivalent=None,
                equivalence_method=None,
                timing_ms_total=0.0,
                timing_ms_breakdown={"method_ms": 0.0, "equivalence_ms": 0.0},
                case_id=item.row.get("id", "").strip() or None,
                reference_status=case["reference_status"],
                reference_reason=case["reference_reason"],
            )
        )
    return results


def run(
    *,
    csv_path: Path,
    xyz_dir: Path,
    out_dir: Path,
    limit: int | None = None,
    start_row: int = 1,
    end_row: int | None = None,
    ids: Sequence[str] | None = None,
    progress_every: int = 10,
    process_workers: int = 1,
    case_timeout_seconds: float | None = 1.0,
    cpp_backend_config: dict[str, Any] | None = None,
    method_ids: Sequence[str] | None = None,
    roundtrip_diagnostics: bool = True,
) -> list[BenchmarkResult]:
    rows = _load_rows(csv_path)
    selected = _selected_rows(
        rows,
        ids=set(ids) if ids else None,
        start_row=start_row,
        end_row=end_row,
        limit=limit,
    )
    runtime_cpp_backend_config = cpp_backend_config or _build_cpp_backend_config_payload(
        use_all_accelerations=False
    )
    methods = _select_methods(
        get_method_registry(tuple(method_ids) if method_ids else None), method_ids
    )
    results: list[BenchmarkResult] = []
    for method in _tqdm(methods, desc="Running methods", total=len(methods)):
        method_results = _run_method_subprocesses(
            method_id=method.method_id,
            cases=selected,
            xyz_dir=xyz_dir,
            case_timeout_seconds=case_timeout_seconds,
            cpp_backend_config=runtime_cpp_backend_config,
            process_workers=process_workers,
            roundtrip_diagnostics=roundtrip_diagnostics,
        )
        if len(method_results) != len(selected):
            results.extend(
                _fallback_results_for_method(
                    method.method_id,
                    selected,
                    "subprocess terminated before producing complete results",
                    xyz_dir=xyz_dir,
                )
            )
            continue
        results.extend(method_results)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_results_csv(out_dir / "results.csv", results)
    write_summary_csv(out_dir / "summary.csv", summarize_results(results))
    return results


def main() -> int:
    if "--subprocess-worker" in sys.argv[1:]:
        return _subprocess_worker_main()
    args = _parse_args()
    ids = _split_repeated_values(args.ids)
    method_ids = _split_repeated_values(args.methods)
    cpp_backend_config = _build_cpp_backend_config_payload(
        use_all_accelerations=args.cpp_accelerations == "all",
        enable_uff_atom_typing_cache=args.enable_uff_atom_typing_cache,
    )
    run(
        csv_path=args.csv,
        xyz_dir=args.xyz_dir,
        out_dir=args.out,
        limit=args.limit,
        start_row=args.start_row,
        end_row=args.end_row,
        ids=ids,
        progress_every=args.progress_every,
        process_workers=args.process_workers,
        case_timeout_seconds=args.case_timeout_seconds or None,
        cpp_backend_config=cpp_backend_config,
        method_ids=method_ids,
        roundtrip_diagnostics=not args.skip_smiles_roundtrip,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
