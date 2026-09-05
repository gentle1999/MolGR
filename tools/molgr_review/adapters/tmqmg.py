#!/usr/bin/env python3
"""Generate a tmQMg review queue from py38/py310 MolGR benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmarks.tmqmg_xyz_benchmark.comparison_annotations import (  # noqa: E402
    ComparisonAnnotation,
    find_comparison_annotation,
)


STATE_DIR = ROOT_DIR / ".local" / "molgr_review" / "tmqmg"
DEFAULT_CASES_CSV = STATE_DIR / "tmqmg_cases.csv"
DEFAULT_REVIEW_DB = ROOT_DIR / ".local" / "molgr_review" / "review.sqlite"
TMQMG_REVISION = "e1dc9887b8f20a217a1db6ca972d726bcbaab45b"
DEFAULT_DATA_DIR = ROOT_DIR / ".local" / "datasets" / "tmqmg" / TMQMG_REVISION
DEFAULT_CSV = Path(
    os.environ.get("TMQMG_CSV", DEFAULT_DATA_DIR / "tmQMg_properties_and_targets.csv")
)
DEFAULT_XYZ_DIR = Path(os.environ.get("TMQMG_XYZ_DIR", DEFAULT_DATA_DIR / "tmQMg_xyz" / "xyz"))
KNOWN_METHOD_IDS = ("molgr_fallback", "molgr_cpp")
DEFAULT_METHOD_IDS = ("molgr_cpp",)
DEFAULT_PROCESS_WORKERS = 12
BENCHMARK_RDKIT_REQUIREMENT = "rdkit==2024.3.5"
BENCHMARK_RDKIT_RUNTIME_VERSION = "2024.03.5"
REFERENCE_FORMULA_MISMATCH_PREFIX = "ValueError: Reference SMILES element counts differ from XYZ:"

REVIEW_COLUMNS = (
    "case_id",
    "source",
    "category",
    "total_charge",
    "total_radical_electrons",
    "spin_multiplicity",
    "reference_smiles",
    "candidate_smiles",
    "candidate_status",
    "candidate_organic_smiles",
    "review_category",
    "row_index",
    "id",
    "metal_center",
    "charge",
    "xyz_path",
    "reference_smiles_input",
    "reference_smiles_canonical",
    "molgr_smiles_canonical",
    "equivalent",
    "strict_equivalent",
    "equivalence_method",
    "equivalence_reason",
    "spin_source",
    "total_radical_electrons_used",
    "spin_multiplicity_used",
    "reference_parse_status",
    "reference_formula_check_status",
    "reference_formula_match",
    "xyz_atom_count",
    "reference_atom_count_with_h",
    "xyz_formula",
    "reference_formula_with_h",
    "reference_formula_mismatch_detail",
    "molgr_status",
    "reference_answer_wrong",
    "reference_answer_status",
    "reference_answer_reason",
    "accuracy_assessment_status",
    "accuracy_assessment_reason",
    "tmqmg_answer_assessment",
    "molgr_answer_assessment",
    "manual_whitelist_status",
    "manual_whitelist_reason",
    "effective_equivalent",
    "molgr_organic_smiles",
    "reference_organic_smiles",
    "molgr_organic_atom_count",
    "reference_organic_atom_count",
    "molgr_organic_heavy_atom_count",
    "reference_organic_heavy_atom_count",
    "molgr_organic_uff_status",
    "molgr_organic_uff_kj_mol",
    "reference_organic_mapping_status",
    "reference_organic_uff_status",
    "reference_organic_uff_kj_mol",
    "organic_uff_delta_kj_mol",
    "elapsed_seconds",
    "error",
    "py38_molgr_fallback_status",
    "py38_molgr_cpp_status",
    "py310_molgr_fallback_status",
    "py310_molgr_cpp_status",
    "py38_molgr_fallback_smiles",
    "py38_molgr_cpp_smiles",
    "py310_molgr_fallback_smiles",
    "py310_molgr_cpp_smiles",
)


@dataclass(frozen=True)
class PythonRun:
    label: str
    requested_python: str
    venv: Path


@dataclass
class BenchmarkRow:
    label: str
    method_id: str
    case_idx: int
    case_id: str
    input_smiles: str
    ground_truth_smiles: str
    status: str
    error: str
    predicted_smiles: str
    equivalent: str
    equivalence_method: str
    comparison_skipped: str
    comparison_skip_reason: str
    timing_ms_total: float
    evaluator_decision: str = ""

    @classmethod
    def from_csv(cls, label: str, row: dict[str, str]) -> BenchmarkRow:
        raw_timing = row.get("timing_ms_total", "")
        try:
            timing_ms_total = float(raw_timing) if raw_timing else 0.0
        except ValueError:
            timing_ms_total = 0.0
        raw_case_idx = row.get("case_idx", "") or "0"
        return cls(
            label=label,
            method_id=row.get("method_id", ""),
            case_idx=int(raw_case_idx),
            case_id=row.get("id", ""),
            input_smiles=row.get("input_smiles", ""),
            ground_truth_smiles=row.get("ground_truth_smiles", ""),
            status=row.get("status", ""),
            error=row.get("error", ""),
            predicted_smiles=row.get("predicted_smiles", ""),
            equivalent=row.get("equivalent", ""),
            equivalence_method=row.get("equivalence_method", ""),
            comparison_skipped=row.get("comparison_skipped", ""),
            comparison_skip_reason=row.get("comparison_skip_reason", ""),
            timing_ms_total=timing_ms_total,
            evaluator_decision=row.get("evaluator_decision", ""),
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="tmQMg metadata CSV.")
    parser.add_argument(
        "--xyz-dir", type=Path, default=DEFAULT_XYZ_DIR, help="tmQMg XYZ directory."
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional benchmark row limit.")
    parser.add_argument("--start-row", type=int, default=1, help="1-based CSV row index to start.")
    parser.add_argument("--end-row", type=int, default=None, help="1-based CSV row index to end.")
    parser.add_argument("--ids", action="append", default=None, help="Optional id filter.")
    parser.add_argument(
        "--progress-every", type=int, default=50, help="Benchmark progress cadence."
    )
    parser.add_argument(
        "--process-workers",
        type=int,
        default=DEFAULT_PROCESS_WORKERS,
        help=f"Benchmark process workers. Defaults to {DEFAULT_PROCESS_WORKERS} for review refreshes.",
    )
    parser.add_argument("--case-timeout-seconds", type=float, default=1.0, help="Per-case timeout.")
    parser.add_argument(
        "--cpp-accelerations",
        choices=("default", "all"),
        default="all",
        help="C++ acceleration preset used by benchmark workers.",
    )
    parser.add_argument(
        "--enable-uff-atom-typing-cache",
        action="store_true",
        help="Enable the optional vendor UFF atom-typing cache.",
    )
    parser.add_argument(
        "--methods",
        action="append",
        default=None,
        help=(
            "Benchmark method ids to run. Repeat or pass comma-separated values. "
            f"Defaults to {','.join(DEFAULT_METHOD_IDS)}. Available: {', '.join(KNOWN_METHOD_IDS)}."
        ),
    )
    parser.add_argument(
        "--python-version-comparison",
        choices=("graph",),
        default="graph",
        help=(
            "Compare py38/py310 candidate molecular graphs directly. Reference benchmark "
            "verdicts are not used to determine Python-version agreement."
        ),
    )
    parser.add_argument("--run-dir", type=Path, default=None, help="Output run directory.")
    parser.add_argument(
        "--py38-results",
        type=Path,
        default=None,
        help="Reuse an existing py38 benchmark results.csv instead of rerunning the benchmark.",
    )
    parser.add_argument(
        "--py310-results",
        type=Path,
        default=None,
        help="Reuse an existing py310 benchmark results.csv instead of rerunning the benchmark.",
    )
    parser.add_argument(
        "--cases-csv",
        type=Path,
        default=DEFAULT_CASES_CSV,
        help="Review-compatible CSV to update.",
    )
    parser.add_argument(
        "--review-db",
        type=Path,
        default=DEFAULT_REVIEW_DB,
        help="Review SQLite database synchronized after the queue CSV is updated.",
    )
    parser.add_argument(
        "--no-sync-review-db",
        action="store_true",
        help="Update queue artifacts without synchronizing the review SQLite database.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Summary JSON path. Defaults to <cases-csv>.summary.json.",
    )
    parser.add_argument(
        "--skip-env-create",
        action="store_true",
        help="Use existing venvs/interpreters instead of creating benchmark venvs.",
    )
    parser.add_argument("--py38-python", default="python3.8", help="Python 3.8 interpreter.")
    parser.add_argument("--py310-python", default="python3.10", help="Python 3.10 interpreter.")
    parser.add_argument(
        "--py38-venv",
        type=Path,
        default=STATE_DIR / ".venv-benchmark-py38",
        help="Python 3.8 benchmark venv.",
    )
    parser.add_argument(
        "--py310-venv",
        type=Path,
        default=STATE_DIR / ".venv-benchmark-py310",
        help="Python 3.10 benchmark venv.",
    )
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _split_repeated(values: Sequence[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for value in values:
        out.extend(part.strip() for part in value.split(",") if part.strip())
    return out


def _resolve_method_ids(values: Sequence[str] | None) -> tuple[str, ...]:
    method_ids = tuple(_split_repeated(values) or DEFAULT_METHOD_IDS)
    unknown = sorted(set(method_ids) - set(KNOWN_METHOD_IDS))
    if unknown:
        raise SystemExit(f"Unknown benchmark method id(s): {', '.join(unknown)}")
    return method_ids


def _run(cmd: Sequence[str], *, cwd: Path = ROOT_DIR, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _benchmark_build_dependencies() -> list[str]:
    """Keep RDKit fixed while intentionally exercising both Open Babel releases."""

    return [
        "scikit-build-core>=0.11.6",
        "setuptools-scm>=9.2.2",
        "pybind11>=3.0.1",
        BENCHMARK_RDKIT_REQUIREMENT,
        "openbabel-wheel==3.1.1.22; python_version < '3.9' and sys_platform == 'win32'",
        "openbabel-wheel; python_version < '3.10' and (sys_platform != 'win32' or python_version >= '3.9')",
        "openbabel>=3.2.0; python_version >= '3.10'",
        "numpy<2",
        "pandas>=2.0.3",
        "tqdm",
    ]


def _benchmark_environment_versions(python_exe: Path) -> dict[str, str]:
    probe = (
        "import json, platform; "
        "from openbabel import openbabel as ob; "
        "from rdkit import rdBase; "
        "print(json.dumps({"
        "'python': platform.python_version(), "
        "'openbabel': ob.OBReleaseVersion(), "
        "'rdkit': rdBase.rdkitVersion"
        "}, sort_keys=True))"
    )
    completed = subprocess.run(
        [str(python_exe), "-c", probe],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        payload = json.loads(completed.stdout.strip())
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"Could not read benchmark dependency versions from {python_exe}: {completed.stdout!r}"
        ) from exc
    return {str(key): str(value) for key, value in payload.items()}


def _validate_benchmark_environment_versions(
    run: PythonRun,
    versions: dict[str, str],
) -> None:
    rdkit_version = versions.get("rdkit", "")
    if rdkit_version != BENCHMARK_RDKIT_RUNTIME_VERSION:
        raise SystemExit(
            f"{run.label} benchmark environment uses RDKit {rdkit_version or 'unknown'}; "
            f"expected {BENCHMARK_RDKIT_RUNTIME_VERSION}. Cross-Open-Babel comparison requires "
            "the same RDKit sanitizer, SMILES writer, and equivalence implementation."
        )


def _ensure_benchmark_env(run: PythonRun, *, skip_env_create: bool) -> Path:
    venv_python = _venv_python(run.venv)
    if skip_env_create:
        if venv_python.exists():
            return venv_python
        return Path(run.requested_python)

    if shutil.which("uv") is None:
        raise SystemExit("uv is required to create benchmark environments.")

    run.venv.parent.mkdir(parents=True, exist_ok=True)
    _run(["uv", "venv", "--allow-existing", "--python", run.requested_python, str(run.venv)])
    build_deps = _benchmark_build_dependencies()
    _run(["uv", "pip", "install", "--python", str(venv_python), *build_deps])
    env = os.environ.copy()
    env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(venv_python),
            "--no-build-isolation",
            "--reinstall-package",
            "molgr",
            "-e",
            str(ROOT_DIR),
        ],
        env=env,
    )
    return venv_python


def _benchmark_cmd(
    python_exe: Path,
    *,
    args: argparse.Namespace,
    out_dir: Path,
    ids: Sequence[str],
    method_ids: Sequence[str],
) -> list[str]:
    cmd = [
        str(python_exe),
        "-m",
        "benchmarks.tmqmg_xyz_benchmark.run",
        "--csv",
        str(args.csv),
        "--xyz-dir",
        str(args.xyz_dir),
        "--out",
        str(out_dir),
        "--progress-every",
        str(args.progress_every),
        "--process-workers",
        str(args.process_workers),
        "--case-timeout-seconds",
        str(args.case_timeout_seconds),
        "--cpp-accelerations",
        args.cpp_accelerations,
        "--methods",
        ",".join(method_ids),
        "--start-row",
        str(args.start_row),
    ]
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.end_row is not None:
        cmd.extend(["--end-row", str(args.end_row)])
    for case_id in ids:
        cmd.extend(["--ids", case_id])
    if args.enable_uff_atom_typing_cache:
        cmd.append("--enable-uff-atom-typing-cache")
    return cmd


def _run_benchmark(
    run: PythonRun,
    *,
    python_exe: Path,
    args: argparse.Namespace,
    run_dir: Path,
    ids: Sequence[str],
    method_ids: Sequence[str],
) -> Path:
    out_dir = run_dir / run.label
    cmd = _benchmark_cmd(python_exe, args=args, out_dir=out_dir, ids=ids, method_ids=method_ids)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR)
    _run(cmd, env=env)
    results_csv = out_dir / "results.csv"
    if not results_csv.exists():
        raise SystemExit(f"Benchmark did not write expected results CSV: {results_csv}")
    return results_csv


def _load_tmqmg_metadata(path: Path) -> dict[tuple[int, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {
        (index, row.get("id", "").strip()): {**row, "_row_index": str(index)}
        for index, row in enumerate(rows, start=1)
    }


def _load_results(label: str, path: Path) -> list[BenchmarkRow]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [BenchmarkRow.from_csv(label, row) for row in csv.DictReader(fh)]


def _truthy(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def _signature(row: BenchmarkRow | None) -> str:
    if row is None:
        return "missing"
    if row.status != "ok":
        return f"{row.status}:{row.error}"
    return f"ok:{row.predicted_smiles}"


def _results_equivalent(
    left: BenchmarkRow | None,
    right: BenchmarkRow | None,
) -> tuple[bool | None, str]:
    if left is None or right is None:
        return left is right, "missing result"
    if left.status != "ok" or right.status != "ok":
        return _signature(left) == _signature(right), "non-ok status"
    if left.predicted_smiles == right.predicted_smiles:
        return True, "identical_smiles"
    if not left.predicted_smiles or not right.predicted_smiles:
        return False, "missing predicted_smiles"

    try:
        from rdkit import Chem

        from molgr.utils.equivalence import evaluate_equivalence

        left_mol = Chem.MolFromSmiles(left.predicted_smiles)
        right_mol = Chem.MolFromSmiles(right.predicted_smiles)
        if left_mol is None or right_mol is None:
            return False, "predicted_smiles_parse_failed"
        info = evaluate_equivalence(left_mol, right_mol, use_chirality=False, max_resonance=100)
        reason = info.reason if info is not None else ""
        equivalent = (
            True
            if info.decision.value == "equivalent"
            else False
            if info.decision.value == "not_equivalent"
            else None
        )
        return equivalent, reason or "molgr_equivalence"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _python_results_equivalent(
    left: BenchmarkRow | None,
    right: BenchmarkRow | None,
) -> tuple[bool | None, str]:
    return _results_equivalent(left, right)


def _is_reference_missing(rows: Iterable[BenchmarkRow]) -> bool:
    return all(not row.ground_truth_smiles.strip() for row in rows)


def _is_reference_formula_mismatch(row: BenchmarkRow) -> bool:
    return _truthy(row.comparison_skipped) and row.comparison_skip_reason.startswith(
        REFERENCE_FORMULA_MISMATCH_PREFIX
    )


def _formula_string(counts: Counter[str]) -> str:
    return ",".join(f"{symbol}:{counts[symbol]}" for symbol in sorted(counts))


def _formula_mismatch_detail(
    xyz_counts: Counter[str],
    reference_counts: Counter[str],
) -> str:
    return "; ".join(
        f"{symbol}:xyz={xyz_counts.get(symbol, 0)},ref={reference_counts.get(symbol, 0)}"
        for symbol in sorted(set(xyz_counts) | set(reference_counts))
        if xyz_counts.get(symbol, 0) != reference_counts.get(symbol, 0)
    )


def _xyz_element_counts(xyz_path: Path) -> Counter[str]:
    lines = xyz_path.read_text(encoding="utf-8").splitlines()
    atom_count = int(lines[0].strip())
    atom_lines = lines[2 : 2 + atom_count]
    if len(atom_lines) != atom_count:
        raise ValueError(f"XYZ atom count does not match coordinate lines: {xyz_path}")
    return Counter(line.split()[0] for line in atom_lines)


def _may_match_boron_annotation(rows: Iterable[BenchmarkRow]) -> bool:
    for row in rows:
        if _truthy(row.comparison_skipped):
            return True
        if not row.input_smiles or not row.ground_truth_smiles or not row.predicted_smiles:
            return True
        if "B" in f"{row.input_smiles}{row.ground_truth_smiles}{row.predicted_smiles}":
            return True
    return False


def _reference_formula_mismatch_fields(
    reference_smiles: str,
    xyz_path: Path,
) -> dict[str, str]:
    from rdkit import Chem

    xyz_counts = _xyz_element_counts(xyz_path)

    reference_mol = Chem.MolFromSmiles(reference_smiles)
    if reference_mol is None:
        return {
            "reference_formula_check_status": "reference_parse_error",
            "reference_formula_match": "False",
            "xyz_atom_count": str(sum(xyz_counts.values())),
            "reference_atom_count_with_h": "",
            "xyz_formula": _formula_string(xyz_counts),
            "reference_formula_with_h": "",
            "reference_formula_mismatch_detail": "reference SMILES could not be parsed",
            "reference_answer_wrong": "True",
            "reference_answer_status": "reference_parse_error",
            "reference_answer_reason": "Reference SMILES could not be parsed for formula validation.",
        }
    reference_with_h = Chem.AddHs(reference_mol)
    reference_counts: Counter[str] = Counter(
        atom.GetSymbol()
        for atom in reference_with_h.GetAtoms()  # pyright: ignore[reportCallIssue]
    )
    mismatch_detail = _formula_mismatch_detail(xyz_counts, reference_counts)
    return {
        "reference_formula_check_status": "formula_mismatch",
        "reference_formula_match": "False",
        "xyz_atom_count": str(sum(xyz_counts.values())),
        "reference_atom_count_with_h": str(sum(reference_counts.values())),
        "xyz_formula": _formula_string(xyz_counts),
        "reference_formula_with_h": _formula_string(reference_counts),
        "reference_formula_mismatch_detail": mismatch_detail,
        "reference_answer_wrong": "True",
        "reference_answer_status": "formula_mismatch",
        "reference_answer_reason": (
            "Reference formula does not conserve XYZ atom counts: " + mismatch_detail
        ),
    }


def _select_display_row(rows_by_key: dict[tuple[str, str], BenchmarkRow]) -> BenchmarkRow | None:
    for key in (
        ("py310", "molgr_cpp"),
        ("py38", "molgr_cpp"),
        ("py310", "molgr_fallback"),
        ("py38", "molgr_fallback"),
    ):
        row = rows_by_key.get(key)
        if row is not None and row.predicted_smiles:
            return row
    for key in (
        ("py310", "molgr_cpp"),
        ("py38", "molgr_cpp"),
        ("py310", "molgr_fallback"),
        ("py38", "molgr_fallback"),
    ):
        row = rows_by_key.get(key)
        if row is not None:
            return row
    return None


def _case_issue(
    rows_by_key: dict[tuple[str, str], BenchmarkRow],
    *,
    labels: Sequence[str],
    method_ids: Sequence[str],
    comparison_annotation: ComparisonAnnotation | None = None,
) -> tuple[str | None, dict[str, Any]]:
    rows = list(rows_by_key.values())
    if comparison_annotation is not None:
        return comparison_annotation.status, {
            "comparison_annotation": [comparison_annotation.comparison_skip_reason],
            "rows": {
                f"{row.label}/{row.method_id}": {
                    "status": row.status,
                    "error": row.error,
                    "predicted_smiles": row.predicted_smiles,
                    "equivalent": row.equivalent,
                    "comparison_skipped": row.comparison_skipped,
                    "comparison_skip_reason": row.comparison_skip_reason,
                    "timing_ms_total": row.timing_ms_total,
                }
                for row in rows
            },
        }
    missing_keys = [
        f"{label}/{method_id}"
        for label in labels
        for method_id in method_ids
        if (label, method_id) not in rows_by_key
    ]
    status_failures = [
        f"{row.label}/{row.method_id}:{row.status}:{row.error}"
        for row in rows
        if row.status != "ok"
    ]
    reference_not_comparable = [
        f"{row.label}/{row.method_id}:{row.comparison_skip_reason}"
        for row in rows
        if _truthy(row.comparison_skipped)
    ]
    reference_formula_mismatches = [
        f"{row.label}/{row.method_id}:{row.comparison_skip_reason}"
        for row in rows
        if _is_reference_formula_mismatch(row)
    ]
    reference_failures = [
        f"{row.label}/{row.method_id}:equivalent={row.equivalent}"
        for row in rows
        if not _truthy(row.comparison_skipped) and row.equivalent and not _truthy(row.equivalent)
    ]

    backend_mismatches = []
    backend_mismatch_reasons = []
    if {"molgr_fallback", "molgr_cpp"}.issubset(set(method_ids)):
        for label in labels:
            fallback = rows_by_key.get((label, "molgr_fallback"))
            cpp = rows_by_key.get((label, "molgr_cpp"))
            equivalent, reason = _results_equivalent(fallback, cpp)
            if fallback is not None and cpp is not None and equivalent is False:
                backend_mismatches.append(label)
                backend_mismatch_reasons.append(f"{label}:{reason}")

    python_mismatches = []
    python_mismatch_reasons = []
    for method_id in method_ids:
        baseline_label = labels[0] if labels else ""
        baseline = rows_by_key.get((baseline_label, method_id))
        mismatched_reasons = []
        for label in labels[1:]:
            candidate = rows_by_key.get((label, method_id))
            equivalent, reason = _python_results_equivalent(
                baseline,
                candidate,
            )
            if equivalent is False:
                mismatched_reasons.append(f"{baseline_label}_vs_{label}:{reason}")
        if mismatched_reasons:
            python_mismatches.append(method_id)
            python_mismatch_reasons.extend(f"{method_id}:{reason}" for reason in mismatched_reasons)

    if not (
        missing_keys
        or status_failures
        or reference_not_comparable
        or reference_failures
        or backend_mismatches
        or python_mismatches
        or _is_reference_missing(rows)
    ):
        return None, {}

    if reference_formula_mismatches:
        category = "reference_formula_mismatch"
    elif missing_keys or status_failures:
        category = "molgr_failed"
    elif backend_mismatches:
        category = "backend_mismatch"
    elif python_mismatches:
        category = "python_version_mismatch"
    elif _is_reference_missing(rows):
        category = "missing_reference_smiles"
    elif reference_not_comparable:
        category = "reference_not_comparable"
    else:
        category = "graph_not_equivalent"

    details = {
        "missing_results": missing_keys,
        "status_failures": status_failures,
        "backend_mismatches": backend_mismatches,
        "backend_mismatch_reasons": backend_mismatch_reasons,
        "python_mismatches": python_mismatches,
        "python_mismatch_reasons": python_mismatch_reasons,
        "reference_not_comparable": reference_not_comparable,
        "reference_formula_mismatches": reference_formula_mismatches,
        "reference_failures": reference_failures,
        "rows": {
            f"{row.label}/{row.method_id}": {
                "status": row.status,
                "error": row.error,
                "predicted_smiles": row.predicted_smiles,
                "equivalent": row.equivalent,
                "comparison_skipped": row.comparison_skipped,
                "comparison_skip_reason": row.comparison_skip_reason,
                "timing_ms_total": row.timing_ms_total,
            }
            for row in rows
        },
    }
    return category, details


def _compact_reason(details: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "missing_results",
        "status_failures",
        "backend_mismatches",
        "backend_mismatch_reasons",
        "python_mismatches",
        "python_mismatch_reasons",
        "reference_not_comparable",
        "reference_formula_mismatches",
        "reference_failures",
        "comparison_annotation",
    ):
        values = details.get(key) or []
        if values:
            parts.append(f"{key}={values}")
    if not parts:
        return "tmQMg review case generated from py38/py310 benchmark."
    return "; ".join(parts)


def _status_for(
    rows_by_key: dict[tuple[str, str], BenchmarkRow], label: str, method_id: str
) -> str:
    row = rows_by_key.get((label, method_id))
    if row is None:
        return "missing"
    return row.status


def _display_status_for(
    rows_by_key: dict[tuple[str, str], BenchmarkRow],
    label: str,
    method_id: str,
    method_ids: Sequence[str],
) -> str:
    if method_id not in method_ids:
        return "not_run"
    return _status_for(rows_by_key, label, method_id)


def _smiles_for(
    rows_by_key: dict[tuple[str, str], BenchmarkRow], label: str, method_id: str
) -> str:
    row = rows_by_key.get((label, method_id))
    if row is None:
        return ""
    return row.predicted_smiles


def _build_review_rows(
    *,
    metadata: dict[tuple[int, str], dict[str, str]],
    result_paths: dict[str, Path],
    xyz_dir: Path,
    method_ids: Sequence[str],
    python_version_comparison: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if python_version_comparison != "graph":
        raise ValueError("Python-version comparison must use direct candidate graph comparison")
    grouped: dict[tuple[int, str], dict[tuple[str, str], BenchmarkRow]] = defaultdict(dict)
    labels = tuple(result_paths)
    for label, path in result_paths.items():
        for row in _load_results(label, path):
            grouped[(row.case_idx, row.case_id)][(label, row.method_id)] = row

    review_rows: list[dict[str, str]] = []
    category_counts: Counter[str] = Counter()
    for case_key in sorted(grouped):
        rows_by_key = grouped[case_key]
        row_index, case_id = case_key
        element_counts = (
            _xyz_element_counts(xyz_dir / f"{case_id}.xyz")
            if _may_match_boron_annotation(rows_by_key.values())
            else Counter()
        )
        comparison_annotation = find_comparison_annotation(
            case_id,
            element_counts,
        )
        category, details = _case_issue(
            rows_by_key,
            labels=labels,
            method_ids=method_ids,
            comparison_annotation=comparison_annotation,
        )
        if category is None:
            continue

        meta = metadata.get(case_key, {})
        display_row = _select_display_row(rows_by_key)
        reference_smiles = (
            meta.get("smiles", "")
            or (display_row.ground_truth_smiles if display_row is not None else "")
            or (display_row.input_smiles if display_row is not None else "")
        )
        representative = rows_by_key.get(("py310", "molgr_cpp")) or display_row
        molgr_smiles = representative.predicted_smiles if representative is not None else ""
        representative_equivalent = representative.equivalent if representative is not None else ""
        representative_timing = (
            representative.timing_ms_total if representative is not None else 0.0
        )
        any_comparison_skip = any(_truthy(row.comparison_skipped) for row in rows_by_key.values())
        first_skip_reason = next(
            (
                row.comparison_skip_reason
                for row in rows_by_key.values()
                if row.comparison_skip_reason
            ),
            "",
        )
        any_ok = any(row.status == "ok" for row in rows_by_key.values())
        formula_fields = {
            "reference_formula_check_status": (
                "comparison_skipped" if any_comparison_skip else "ok"
            ),
            "reference_formula_match": "False" if any_comparison_skip else "True",
            "xyz_atom_count": meta.get("n_atoms", ""),
            "reference_atom_count_with_h": meta.get("n_atoms", ""),
            "xyz_formula": "",
            "reference_formula_with_h": "",
            "reference_formula_mismatch_detail": first_skip_reason,
            "reference_answer_wrong": "False",
            "reference_answer_status": "not_flagged",
            "reference_answer_reason": "",
        }
        if category == "reference_formula_mismatch":
            formula_fields = _reference_formula_mismatch_fields(
                reference_smiles,
                xyz_dir / f"{case_id}.xyz",
            )
        if comparison_annotation is not None:
            formula_fields = {
                "reference_formula_check_status": "not_applicable",
                "reference_formula_match": "",
                "xyz_atom_count": meta.get("n_atoms", ""),
                "reference_atom_count_with_h": "",
                "xyz_formula": "",
                "reference_formula_with_h": "",
                "reference_formula_mismatch_detail": "",
                "reference_answer_wrong": "False",
                "reference_answer_status": "not_assessable",
                "reference_answer_reason": comparison_annotation.reason,
            }
            representative_equivalent = ""
        row = {
            "case_id": case_id,
            "source": "tmqmg",
            "category": "candidate_failed" if category == "molgr_failed" else category,
            "total_charge": meta.get("charge", ""),
            "total_radical_electrons": "",
            "spin_multiplicity": "",
            "reference_smiles": reference_smiles,
            "candidate_smiles": molgr_smiles,
            "candidate_status": "ok" if any_ok else "error",
            "candidate_organic_smiles": molgr_smiles,
            "review_category": category,
            "row_index": str(row_index),
            "id": case_id,
            "metal_center": meta.get("metal_center", ""),
            "charge": meta.get("charge", ""),
            "xyz_path": str(xyz_dir / f"{case_id}.xyz") if case_id else "",
            "reference_smiles_input": reference_smiles,
            "reference_smiles_canonical": reference_smiles,
            "molgr_smiles_canonical": molgr_smiles,
            "equivalent": representative_equivalent,
            "strict_equivalent": representative_equivalent,
            "equivalence_method": (
                ""
                if comparison_annotation is not None
                else "tmqmg_py38_py310_" + "_".join(method_ids) + "_benchmark"
            ),
            "equivalence_reason": (
                comparison_annotation.reason
                if comparison_annotation is not None
                else _compact_reason(details)
            ),
            "spin_source": "reference_smiles",
            "total_radical_electrons_used": "",
            "spin_multiplicity_used": "",
            "reference_parse_status": "missing_reference_smiles" if not reference_smiles else "ok",
            "reference_formula_check_status": formula_fields["reference_formula_check_status"],
            "reference_formula_match": formula_fields["reference_formula_match"],
            "xyz_atom_count": formula_fields["xyz_atom_count"],
            "reference_atom_count_with_h": formula_fields["reference_atom_count_with_h"],
            "xyz_formula": formula_fields["xyz_formula"],
            "reference_formula_with_h": formula_fields["reference_formula_with_h"],
            "reference_formula_mismatch_detail": formula_fields[
                "reference_formula_mismatch_detail"
            ],
            "molgr_status": "ok" if any_ok else "error",
            "reference_answer_wrong": formula_fields["reference_answer_wrong"],
            "reference_answer_status": formula_fields["reference_answer_status"],
            "reference_answer_reason": formula_fields["reference_answer_reason"],
            "accuracy_assessment_status": (
                comparison_annotation.status if comparison_annotation is not None else "assessable"
            ),
            "accuracy_assessment_reason": (
                comparison_annotation.reason if comparison_annotation is not None else ""
            ),
            "tmqmg_answer_assessment": (
                "not_assessable" if comparison_annotation is not None else "assessable"
            ),
            "molgr_answer_assessment": (
                "not_assessable" if comparison_annotation is not None else "assessable"
            ),
            "manual_whitelist_status": "",
            "manual_whitelist_reason": "",
            "effective_equivalent": "" if comparison_annotation is not None else "False",
            "molgr_organic_smiles": molgr_smiles,
            "reference_organic_smiles": reference_smiles,
            "molgr_organic_atom_count": "",
            "reference_organic_atom_count": "",
            "molgr_organic_heavy_atom_count": "",
            "reference_organic_heavy_atom_count": "",
            "molgr_organic_uff_status": "",
            "molgr_organic_uff_kj_mol": "",
            "reference_organic_mapping_status": "",
            "reference_organic_uff_status": "",
            "reference_organic_uff_kj_mol": "",
            "organic_uff_delta_kj_mol": "",
            "elapsed_seconds": f"{representative_timing / 1000.0:.6f}",
            "error": json.dumps(details, ensure_ascii=False, sort_keys=True),
            "py38_molgr_fallback_status": _display_status_for(
                rows_by_key, "py38", "molgr_fallback", method_ids
            ),
            "py38_molgr_cpp_status": _display_status_for(
                rows_by_key, "py38", "molgr_cpp", method_ids
            ),
            "py310_molgr_fallback_status": _display_status_for(
                rows_by_key, "py310", "molgr_fallback", method_ids
            ),
            "py310_molgr_cpp_status": _display_status_for(
                rows_by_key, "py310", "molgr_cpp", method_ids
            ),
            "py38_molgr_fallback_smiles": _smiles_for(rows_by_key, "py38", "molgr_fallback"),
            "py38_molgr_cpp_smiles": _smiles_for(rows_by_key, "py38", "molgr_cpp"),
            "py310_molgr_fallback_smiles": _smiles_for(rows_by_key, "py310", "molgr_fallback"),
            "py310_molgr_cpp_smiles": _smiles_for(rows_by_key, "py310", "molgr_cpp"),
        }
        review_rows.append(row)
        category_counts[category] += 1

    summary = {
        "record_count": len(review_rows),
        "category_counts": dict(sorted(category_counts.items())),
        "result_paths": {label: str(path) for label, path in result_paths.items()},
        "python_version_comparison": python_version_comparison,
    }
    return review_rows, summary


def _write_review_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            temp_path = Path(fh.name)
            writer = csv.DictWriter(fh, fieldnames=REVIEW_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)  # pyright: ignore[reportArgumentType]
        temp_path.replace(path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _load_review_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _review_row_sort_key(row: dict[str, str]) -> tuple[int, str]:
    try:
        row_index = int(row.get("row_index", ""))
    except ValueError:
        row_index = sys.maxsize
    return row_index, row.get("case_id", "")


def _merge_partial_review_rows(
    existing_rows: Sequence[dict[str, str]],
    refreshed_rows: Sequence[dict[str, str]],
    *,
    refreshed_case_ids: set[str],
) -> list[dict[str, str]]:
    merged = {
        row.get("case_id", ""): dict(row)
        for row in existing_rows
        if row.get("case_id", "") and row.get("case_id", "") not in refreshed_case_ids
    }
    for row in refreshed_rows:
        case_id = row.get("case_id", "")
        if case_id:
            merged[case_id] = dict(row)
    return sorted(merged.values(), key=_review_row_sort_key)


def _result_case_ids(result_paths: dict[str, Path]) -> set[str]:
    return {
        row.case_id
        for path in result_paths.values()
        for row in _load_results("refresh", path)
        if row.case_id
    }


def _is_partial_refresh(args: argparse.Namespace, ids: Sequence[str]) -> bool:
    return bool(ids or args.limit is not None or args.start_row != 1 or args.end_row is not None)


def _sync_review_db(*, cases_csv: Path, review_db: Path) -> None:
    _run(
        [
            sys.executable,
            str(ROOT_DIR / "tools" / "molgr_review" / "import_cases.py"),
            "--input",
            str(cases_csv),
            "--db",
            str(review_db),
        ]
    )


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = _parse_args()
    stamp = _timestamp()
    ids = _split_repeated(args.ids)
    method_ids = _resolve_method_ids(args.methods)
    run_dir = args.run_dir or (STATE_DIR / "runs" / stamp)
    run_dir.mkdir(parents=True, exist_ok=True)

    python_runs = (
        PythonRun("py38", args.py38_python, args.py38_venv),
        PythonRun("py310", args.py310_python, args.py310_venv),
    )
    provided_results = {"py38": args.py38_results, "py310": args.py310_results}
    result_paths: dict[str, Path] = {}
    environment_versions: dict[str, dict[str, str]] = {}
    for python_run in python_runs:
        provided_result = provided_results[python_run.label]
        if provided_result is not None:
            if not provided_result.exists():
                raise SystemExit(
                    f"Provided {python_run.label} results CSV does not exist: {provided_result}"
                )
            result_paths[python_run.label] = provided_result
        else:
            python_exe = _ensure_benchmark_env(python_run, skip_env_create=args.skip_env_create)
            versions = _benchmark_environment_versions(python_exe)
            _validate_benchmark_environment_versions(python_run, versions)
            environment_versions[python_run.label] = versions
            result_paths[python_run.label] = _run_benchmark(
                python_run,
                python_exe=python_exe,
                args=args,
                run_dir=run_dir,
                ids=ids,
                method_ids=method_ids,
            )

    metadata = _load_tmqmg_metadata(args.csv)
    review_rows, summary = _build_review_rows(
        metadata=metadata,
        result_paths=result_paths,
        xyz_dir=args.xyz_dir,
        method_ids=method_ids,
        python_version_comparison=args.python_version_comparison,
    )
    refreshed_case_ids = _result_case_ids(result_paths)
    partial_refresh = _is_partial_refresh(args, ids)
    existing_review_rows = _load_review_csv(args.cases_csv)
    if (
        partial_refresh
        and not existing_review_rows
        and not args.no_sync_review_db
        and args.review_db.exists()
    ):
        raise SystemExit(
            "Partial refresh cannot synchronize an existing review database without the "
            f"current queue CSV: {args.cases_csv}"
        )
    if partial_refresh and existing_review_rows:
        queue_rows = _merge_partial_review_rows(
            existing_review_rows,
            review_rows,
            refreshed_case_ids=refreshed_case_ids,
        )
    else:
        queue_rows = review_rows
    run_cases_csv = run_dir / "review_cases.csv"
    _write_review_csv(run_cases_csv, review_rows)
    _write_review_csv(args.cases_csv, queue_rows)

    summary_payload = {
        **summary,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tmqmg_csv": str(args.csv),
        "xyz_dir": str(args.xyz_dir),
        "run_dir": str(run_dir),
        "run_cases_csv": str(run_cases_csv),
        "default_cases_csv": str(args.cases_csv),
        "review_db": str(args.review_db),
        "review_db_synchronized": not args.no_sync_review_db,
        "partial_refresh": partial_refresh,
        "refreshed_case_count": len(refreshed_case_ids),
        "run_review_case_count": len(review_rows),
        "queue_record_count": len(queue_rows),
        "methods": list(method_ids),
        "process_workers": args.process_workers,
        "python_version_comparison": args.python_version_comparison,
        "reused_results": {
            label: str(path) for label, path in provided_results.items() if path is not None
        },
        "python_runs": {
            python_run.label: {
                "requested_python": python_run.requested_python,
                "venv": str(python_run.venv),
                "environment_versions": environment_versions.get(python_run.label),
            }
            for python_run in python_runs
        },
    }
    summary_path = args.summary_json or args.cases_csv.with_name(
        f"{args.cases_csv.name}.summary.json"
    )
    _write_summary(run_dir / "review_cases.summary.json", summary_payload)
    _write_summary(summary_path, summary_payload)

    if not args.no_sync_review_db:
        _sync_review_db(cases_csv=args.cases_csv, review_db=args.review_db)

    print(
        f"Prepared {len(review_rows)} refreshed review cases; "
        f"queue now has {len(queue_rows)} cases at {args.cases_csv}"
    )
    if args.no_sync_review_db:
        print(f"Review database synchronization disabled: {args.review_db}")
    else:
        print(f"Synchronized review database: {args.review_db}")
    print(f"Run artifacts: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
