#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
import platform
import sqlite3
import sys
import time
import traceback
import uuid
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase
from rdkit.Chem import rdchem, rdinchi

import molgr
from molgr import _core


VERSIONS = ("original", "v1", "v2")
REFERENCE_STATUSES = ("valid", "confirmed_excluded", "unresolved")
OUTCOMES = ("EQ", "NEQ", "INC", "error", "timeout", "input_error", "candidate_unavailable")
DECISIONS = {"equivalent": "EQ", "not_equivalent": "NEQ", "inconclusive": "INC"}
CSV_FIELDS = (
    "case_id",
    "version",
    "status",
    "decision",
    "raw_decision",
    "relation",
    "method",
    "reason",
    "elapsed_seconds",
    "reference_smiles_sha256",
    "candidate_sha256",
    "candidate_artifact_path",
    "source_sha256",
    "bounded_search",
    "invariants",
    "raw_result",
    "error",
    "input_diagnostic",
)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _serializable(value: Any) -> Any:
    if isinstance(value, Enum):
        return _serializable(value.value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _serializable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_serializable(item) for item in value), key=_json)
    if hasattr(value, "__dict__"):
        return _serializable(vars(value))
    raise TypeError(f"Cannot serialize evaluator diagnostic of type {type(value).__name__}")


def _diagnostic(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}


def _case_map(document: Any, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(document, dict) or not isinstance(document.get("schema"), str):
        raise ValueError(f"{label} must be an object with a schema string")
    if not isinstance(document.get("cases"), list):
        raise ValueError(f"{label}.cases must be a list")
    cases = {}
    for case in document["cases"]:
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("case_id"), str)
            or not case["case_id"]
        ):
            raise ValueError(f"{label} requires nonempty string case_id values")
        case_id = case["case_id"]
        if case_id in cases:
            raise ValueError(f"Duplicate case_id in {label}: {case_id}")
        cases[case_id] = case
    return cases


def _artifact_path(root: Path, entry: dict[str, Any]) -> Path:
    relative = entry.get("artifact_path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"Invalid relative artifact_path for {entry['case_id']}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError(f"artifact_path escapes candidate root: {relative}") from None
    if path == root:
        raise ValueError(f"artifact_path escapes candidate root: {relative}")
    return path


def _read_inputs(args: argparse.Namespace) -> tuple[dict, dict, dict, dict, list[str]]:
    manifest_bytes = args.manifest.read_bytes()
    index_bytes = args.candidate_index.read_bytes()
    manifest = json.loads(manifest_bytes)
    inventory = json.loads(index_bytes)
    references = _case_map(manifest, "manifest")
    candidates = _case_map(inventory, "candidate inventory")
    if references.keys() != candidates.keys():
        raise ValueError(
            "Manifest and candidate inventory must contain exactly the same case_id universe"
        )
    for case_id, reference in references.items():
        smiles = reference.get("reference_smiles")
        if not isinstance(smiles, str):
            raise ValueError(f"reference_smiles must be a string: {case_id}")
        if _sha256(smiles.encode("utf-8")) != reference.get("reference_smiles_sha256"):
            raise ValueError(f"Reference SMILES SHA256 mismatch: {case_id}")
        if reference.get("reference_status") not in REFERENCE_STATUSES:
            raise ValueError(f"Invalid reference_status: {case_id}")
        if not isinstance(reference.get("reference_reason"), str):
            raise ValueError(f"reference_reason must be a string: {case_id}")
    root = args.candidate_root.resolve()
    verified = hashlib.sha256()
    available_count = 0
    for case_id, entry in candidates.items():
        if not isinstance(entry.get("available"), bool):
            raise ValueError(f"available must be a boolean: {case_id}")
        path = _artifact_path(root, entry)
        actual = _file_sha256(path) if path.exists() else None
        if actual != entry.get("sha256") or (entry["available"] and actual is None):
            raise ValueError(f"Candidate SHA256 mismatch or missing artifact: {case_id}")
        available_count += int(entry["available"])
        verified.update((_json([case_id, actual]) + "\n").encode("utf-8"))
    source_bytes = {
        version: getattr(args, f"source_{version}").read_bytes() for version in VERSIONS
    }
    selected = list(references)
    if args.limit is not None:
        selected = selected[: args.limit]
    identity = {
        "schema": "evaluator-replay-checkpoint/v1",
        "runner_sha256": _file_sha256(Path(__file__)),
        "manifest": {"path": str(args.manifest.resolve()), "sha256": _sha256(manifest_bytes)},
        "candidate_index": {
            "path": str(args.candidate_index.resolve()),
            "sha256": _sha256(index_bytes),
        },
        "candidate_root": str(root),
        "verified_candidate_inputs_sha256": verified.hexdigest(),
        "sources": {
            version: {
                "path": str(getattr(args, f"source_{version}").resolve()),
                "sha256": _sha256(data),
            }
            for version, data in source_bytes.items()
        },
        "rdkit_version": rdBase.rdkitVersion,
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "runtime_binaries_sha256": {
            "python": _file_sha256(Path(sys.executable)),
            "rdBase": _file_sha256(Path(rdBase.__file__)),
            "rdchem": _file_sha256(Path(rdchem.__file__)),
            "rdinchi": _file_sha256(Path(rdinchi.__file__)),
            "molgr_core": _file_sha256(Path(_core.__file__)),
        },
        # Native evaluator snapshots import adapters/types and the compiled
        # kernel. A wrapper-only hash must not authorize a mixed-code resume.
        "molgr_python_sha256": {
            str(path.relative_to(Path(molgr.__file__).parent)): _file_sha256(path)
            for path in sorted(Path(molgr.__file__).parent.rglob("*.py"))
        },
        "parameters": {
            "max_resonance": args.max_resonance,
            "use_chirality": args.use_chirality,
            "timeout_seconds": args.timeout,
            "expected_rdkit_version": args.rdkit_version,
            "limit": args.limit,
            "argument_order": ["reference", "candidate"],
            "candidate_parse": {
                "sanitize": False,
                "removeHs": False,
                "strictParsing": True,
                "update_property_cache_strict": False,
            },
            "reference_parse": {"sanitize": True},
        },
        "manifest_population": len(references),
        "inventory_available": available_count,
        "selected_population": len(selected),
    }
    return references, candidates, source_bytes, identity, selected


def _load_source(version: str, path: Path, content: bytes) -> Any:
    name = f"_evaluator_replay_{version}_{_sha256(content)}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise ValueError(f"Cannot load evaluator source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(content, str(path), "exec"), module.__dict__)
        function = "check_equivalence" if version == "original" else "evaluate_equivalence"
        if not callable(getattr(module, function, None)):
            raise ValueError(f"{version} source has no callable {function}")
        return module
    except BaseException:
        sys.modules.pop(name, None)
        raise


def _result_fields(result: Any, version: str) -> dict[str, Any]:
    raw = _serializable(result)
    if version == "original":
        verdict = result[0] if isinstance(result, tuple) else result
        if not isinstance(verdict, bool):
            raise TypeError("Original check_equivalence must return bool or (bool, diagnostics)")
        detail = raw[1] if isinstance(raw, list) and len(raw) > 1 else {}
        detail = detail if isinstance(detail, dict) else {}
        decision = "EQ" if verdict else "NEQ"
        raw_decision = verdict
    else:
        if not isinstance(raw, dict):
            raise TypeError("evaluate_equivalence must return a structured result")
        detail = raw
        raw_decision = detail.get("decision")
        if raw_decision not in DECISIONS:
            raise ValueError(f"Unknown evaluator decision: {raw_decision!r}")
        decision = DECISIONS[raw_decision]
    return {
        "status": "scored",
        "decision": decision,
        "raw_decision": raw_decision,
        "relation": detail.get("relation"),
        "method": detail.get("method"),
        "reason": detail.get("reason"),
        "bounded_search": detail.get("bounded_search"),
        "invariants": detail.get("invariants"),
        "raw_result": raw,
        "error": None,
    }


def _call_worker(
    connection: Any,
    module: Any,
    version: str,
    candidate: Any,
    reference: Any,
    max_resonance: int,
    use_chirality: bool,
) -> None:
    started = time.perf_counter()
    try:
        function = (
            module.check_equivalence if version == "original" else module.evaluate_equivalence
        )
        result = function(
            Chem.Mol(reference),
            Chem.Mol(candidate),
            max_resonance=max_resonance,
            use_chirality=use_chirality,
        )
        fields = _result_fields(result, version)
    except TimeoutError as exc:
        fields = {"status": "timeout", "error": _diagnostic(exc)}
    except Exception as exc:
        fields = {"status": "error", "error": _diagnostic(exc)}
    fields["elapsed_seconds"] = time.perf_counter() - started
    try:
        connection.send(fields)
    finally:
        connection.close()


def _run_call(
    module: Any, version: str, candidate: Any, reference: Any, args: argparse.Namespace
) -> dict[str, Any]:
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_call_worker,
        args=(
            sender,
            module,
            version,
            candidate,
            reference,
            args.max_resonance,
            args.use_chirality,
        ),
    )
    started = time.perf_counter()
    try:
        process.start()
        sender.close()
        if not receiver.poll(args.timeout):
            return {
                "status": "timeout",
                "elapsed_seconds": time.perf_counter() - started,
                "error": {
                    "type": "EvaluatorTimeout",
                    "message": f"Exceeded {args.timeout} seconds",
                },
            }
        try:
            return receiver.recv()
        except (EOFError, OSError) as exc:
            process.join(timeout=0.1)
            return {
                "status": "error",
                "elapsed_seconds": time.perf_counter() - started,
                "error": {**_diagnostic(exc), "worker_exitcode": process.exitcode},
            }
    finally:
        sender.close()
        receiver.close()
        if process.pid is not None:
            process.join(timeout=0.1)
            if process.is_alive():
                process.kill()
                process.join()
        process.close()


def _parse_pair(reference: dict, entry: dict, root: Path) -> tuple[Any, Any, dict | None, dict]:
    diagnostic = {"inventory_status": entry.get("status"), "inventory_error": entry.get("error")}
    if not entry["available"]:
        return (
            None,
            None,
            {"status": "candidate_unavailable", "reason": "Inventory marks candidate unavailable"},
            diagnostic,
        )
    data = _artifact_path(root, entry).read_bytes()
    if _sha256(data) != entry["sha256"]:
        raise ValueError(f"Candidate SHA256 changed during replay: {entry['case_id']}")
    try:
        artifact = json.loads(data)
        if not isinstance(artifact, dict):
            raise ValueError("Candidate artifact must be a JSON object")
        if artifact.get("case_id") != entry["case_id"]:
            raise ValueError("Candidate artifact case_id mismatch")
        diagnostic.update(
            {
                "artifact_status": artifact.get("status"),
                "artifact_outcome": artifact.get("outcome"),
                "artifact_error": artifact.get("error"),
            }
        )
        if artifact.get("status") != "ok" or artifact.get("outcome") != "success":
            return (
                None,
                None,
                {
                    "status": "candidate_unavailable",
                    "reason": "Candidate artifact is not ok/success",
                },
                diagnostic,
            )
        graph = artifact["predicted_graph"]
        field = "molblock_lossless" if graph.get("molblock_lossless") else "molblock_v2000"
        diagnostic["molblock_field"] = field
        molblock = graph[field]
        candidate = Chem.MolFromMolBlock(
            molblock, sanitize=False, removeHs=False, strictParsing=True
        )
        if candidate is None:
            raise ValueError("Candidate MolBlock did not parse")
        candidate.UpdatePropertyCache(False)
        if not reference["reference_smiles"].strip():
            raise ValueError("Reference SMILES is missing")
        parsed_reference = Chem.MolFromSmiles(reference["reference_smiles"], sanitize=True)
        if parsed_reference is None:
            raise ValueError("Reference SMILES did not parse")
        return candidate, parsed_reference, None, diagnostic
    except Exception as exc:
        return None, None, {"status": "input_error", "error": _diagnostic(exc)}, diagnostic


def _checkpoint(path: Path, identity: dict) -> tuple[sqlite3.Connection, str]:
    canonical = _json(identity)
    fingerprint = _sha256(canonical.encode("utf-8"))
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA synchronous=FULL")
        with connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY CHECK (id=1), fingerprint TEXT NOT NULL, identity_json TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS results (case_id TEXT NOT NULL, version TEXT NOT NULL, record_json TEXT NOT NULL, PRIMARY KEY (case_id, version))"
            )
            saved = connection.execute(
                "SELECT fingerprint, identity_json FROM metadata WHERE id=1"
            ).fetchone()
            if saved is None:
                if connection.execute("SELECT COUNT(*) FROM results").fetchone()[0]:
                    raise ValueError("Checkpoint contains results without identity metadata")
                connection.execute(
                    "INSERT INTO metadata VALUES (1, ?, ?)", (fingerprint, canonical)
                )
            elif saved != (fingerprint, canonical):
                raise ValueError(
                    "Checkpoint fingerprint mismatch: source, input, runtime, or parameters changed; use a new output directory"
                )
            connection.execute(
                "CREATE TRIGGER IF NOT EXISTS immutable_metadata_update BEFORE UPDATE ON metadata BEGIN SELECT RAISE(ABORT, 'immutable checkpoint metadata'); END"
            )
            connection.execute(
                "CREATE TRIGGER IF NOT EXISTS immutable_metadata_delete BEFORE DELETE ON metadata BEGIN SELECT RAISE(ABORT, 'immutable checkpoint metadata'); END"
            )
        return connection, fingerprint
    except BaseException:
        connection.close()
        raise


def _atomic_write(path: Path, writer: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _counts(total: int) -> dict[str, int]:
    return {"total": total, "completed": 0, **dict.fromkeys(OUTCOMES, 0), "not_scored": total}


def _export(
    connection: sqlite3.Connection,
    out_dir: Path,
    references: dict,
    selected: list[str],
    identity: dict,
    fingerprint: str,
) -> dict:
    selected_statuses = Counter(references[case_id]["reference_status"] for case_id in selected)
    summary = {
        "schema": "evaluator-replay-summary/v1",
        "fingerprint": fingerprint,
        "mode": "smoke" if identity["parameters"]["limit"] is not None else "full",
        "manifest_population": len(references),
        "selected_population": len(selected),
        "manifest_reference_status": dict(
            Counter(case["reference_status"] for case in references.values())
        ),
        "selected_reference_status": dict(selected_statuses),
        "expected_result_slots": len(selected) * len(VERSIONS),
        "completed_result_slots": 0,
        "versions": {
            version: {
                "all_population": _counts(len(selected)),
                "by_reference_status": {
                    status: _counts(selected_statuses[status]) for status in REFERENCE_STATUSES
                },
            }
            for version in VERSIONS
        },
    }
    query = "SELECT record_json FROM results ORDER BY case_id, CASE version WHEN 'original' THEN 0 WHEN 'v1' THEN 1 ELSE 2 END"
    for (record_json,) in connection.execute(query):
        row = json.loads(record_json)
        outcome = row["decision"] if row["status"] == "scored" else row["status"]
        version_summary = summary["versions"][row["version"]]
        for bucket in (
            version_summary["all_population"],
            version_summary["by_reference_status"][references[row["case_id"]]["reference_status"]],
        ):
            bucket[outcome] += 1
            bucket["completed"] += 1
            bucket["not_scored"] -= 1
        summary["completed_result_slots"] += 1
    summary["not_scored_result_slots"] = (
        summary["expected_result_slots"] - summary["completed_result_slots"]
    )
    summary["complete"] = summary["not_scored_result_slots"] == 0

    def write_jsonl(stream: Any) -> None:
        for (record_json,) in connection.execute(query):
            stream.write(record_json + "\n")

    def write_csv(stream: Any) -> None:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for (record_json,) in connection.execute(query):
            row = json.loads(record_json)
            writer.writerow(
                {
                    key: _json(value) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )

    _atomic_write(out_dir / "raw.jsonl", write_jsonl)
    _atomic_write(out_dir / "raw.csv", write_csv)
    _atomic_write(out_dir / "summary.json", lambda stream: stream.write(_json(summary) + "\n"))
    metadata = {"fingerprint": fingerprint, "identity": identity}
    _atomic_write(
        out_dir / "checkpoint-metadata.json", lambda stream: stream.write(_json(metadata) + "\n")
    )
    return summary


def run(args: argparse.Namespace) -> dict:
    if rdBase.rdkitVersion != args.rdkit_version:
        raise ValueError(
            f"RDKit version mismatch: expected {args.rdkit_version}, actual {rdBase.rdkitVersion}; an explicit --rdkit-version alternate is required for local development"
        )
    if "fork" not in multiprocessing.get_all_start_methods():
        raise ValueError(
            "This sequential replay runner requires a platform supporting multiprocessing fork"
        )
    if args.max_resonance < 1 or args.timeout <= 0 or not math.isfinite(args.timeout):
        raise ValueError("max_resonance and timeout must be positive and finite")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive and is for smoke runs only")
    references, candidates, source_bytes, identity, selected = _read_inputs(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    connection, fingerprint = _checkpoint(args.out_dir / "checkpoint.sqlite", identity)
    try:
        modules = {
            version: _load_source(version, getattr(args, f"source_{version}"), data)
            for version, data in source_bytes.items()
        }
        completed = set(connection.execute("SELECT case_id, version FROM results"))
        for case_id in selected:
            missing = [version for version in VERSIONS if (case_id, version) not in completed]
            if not missing:
                continue
            reference, entry = references[case_id], candidates[case_id]
            candidate, parsed_reference, failure, diagnostic = _parse_pair(
                reference, entry, args.candidate_root.resolve()
            )
            for version in missing:
                record: dict[str, Any] = dict.fromkeys(CSV_FIELDS)
                record.update(
                    {
                        "case_id": case_id,
                        "version": version,
                        "reference_smiles_sha256": reference["reference_smiles_sha256"],
                        "candidate_sha256": entry.get("sha256"),
                        "candidate_artifact_path": entry["artifact_path"],
                        "source_sha256": identity["sources"][version]["sha256"],
                        "input_diagnostic": diagnostic,
                    }
                )
                record.update(
                    failure
                    if failure is not None
                    else _run_call(modules[version], version, candidate, parsed_reference, args)
                )
                with connection:
                    connection.execute(
                        "INSERT INTO results VALUES (?, ?, ?)", (case_id, version, _json(record))
                    )
        return _export(connection, args.out_dir, references, selected, identity, fingerprint)
    except BaseException:
        _export(connection, args.out_dir, references, selected, identity, fingerprint)
        raise
    finally:
        connection.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay three frozen standalone evaluators without reconstruction; validity membership is applied only to saved-row aggregates."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--candidate-index", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    for version in VERSIONS:
        parser.add_argument(f"--source-{version}", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--limit", type=int, help="SMOKE ONLY: first N manifest cases, without validity filtering"
    )
    parser.add_argument("--max-resonance", type=int, default=5000)
    parser.add_argument("--use-chirality", action="store_true", default=False)
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Hard evaluator-call timeout in seconds (default: 5)",
    )
    parser.add_argument(
        "--rdkit-version",
        default="2026.03.6",
        help="Required exact runtime version; explicitly override only for local development",
    )
    return parser.parse_args(argv)


def main() -> int:
    try:
        args = parse_args()
        summary = run(args)
    except KeyboardInterrupt:
        print(
            "Interrupted; completed SQLite rows are durable. Repeat the identical command to resume.",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        _json(
            {
                "summary": str(args.out_dir / "summary.json"),
                "completed_result_slots": summary["completed_result_slots"],
                "not_scored_result_slots": summary["not_scored_result_slots"],
                "complete": summary["complete"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
