"""Resume source-geometry benchmarks with persistent workers and phase watchdogs.

Run one method at a time for latency measurements. Raw results remain outside Git.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
import platform
import sqlite3
import time
import zlib
from contextlib import suppress
from multiprocessing.connection import wait
from pathlib import Path

from benchmarks.dataset_xyz_benchmark.comparison import (
    compare_output,
    comparison_policy,
    native_reference,
)
from benchmarks.dataset_xyz_benchmark.prepare import SCHEMA_VERSION, file_hash

# Must precede MolGR/Open Babel native imports; see this registry's import guard.
from benchmarks.smiles_xyz_benchmark.methods import METHOD_IDS, get_method_registry


DATASET_METHOD_IDS = (*METHOD_IDS, "xyz2mol_tm_local")
ROOT = Path(__file__).resolve().parents[2]
# Mirrors xyz2mol_tmc.TRANSITION_METALS. Keep routing available without the optional package.
# fmt: off
_XYZ2MOL_TMC_ELEMENTS = frozenset({
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "La", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
})
# fmt: on


def xyz2mol_entry(xyz: str) -> str:
    """Choose the upstream entry from the XYZ atom list before timing reconstruction."""
    for row in xyz.splitlines()[2:]:
        fields = row.split(maxsplit=1)
        if fields and fields[0] in _XYZ2MOL_TMC_ELEMENTS:
            return "xyz2mol_tm"
    return "xyz2mol_tm_local"


def reconstruction_input(xyz, charge, multiplicity):
    """Adapters receive geometry and source electronic state, never reference data."""
    return {
        "xyz_block": xyz,
        "total_charge": charge,
        "total_radical_electrons": multiplicity - 1,
    }


def worker(conn, cpu, source, method_id, log, policy):
    os.sched_setaffinity(0, {cpu})
    with open(log, "a", buffering=1) as stream, sqlite3.connect(
        f"{Path(source).resolve().as_uri()}?mode=ro", uri=True
    ) as db:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
        from rdkit import RDLogger

        from molgr.config import CONFIG

        RDLogger.DisableLog("rdApp.*")
        CONFIG.cpp_backend.max_threads = 1
        CONFIG.cpp_backend.target_bucket_parallel_max_threads = 1
        entries = (method_id, "xyz2mol_tm_local") if method_id == "xyz2mol_tm" else (method_id,)
        methods = {method.method_id: method for method in get_method_registry(entries)}
        warmed = set()
        conn.send(("ready", None))
        while True:
            case_id = conn.recv()
            if case_id is None:
                return
            row = db.execute(
                "SELECT xyz,charge,multiplicity,reference_native,reference_sha256,reference_status FROM cases WHERE id=?",
                (case_id,),
            ).fetchone()
            xyz, charge, multiplicity, native, digest, reference_status = row
            case = reconstruction_input(zlib.decompress(xyz).decode(), charge, multiplicity)
            entry = xyz2mol_entry(case["xyz_block"]) if method_id == "xyz2mol_tm" else method_id
            method = methods[entry]
            if entry not in warmed:
                conn.send(("warmup", None))
                with suppress(Exception):
                    method.run(dict(case))
                warmed.add(entry)
            conn.send(("method", None))
            started = time.perf_counter()
            output = None
            try:
                output = method.run(dict(case))
                record = {
                    "status": output.status,
                    "error": output.error,
                    "predicted_smiles": output.predicted_smiles,
                }
            except Exception as exc:
                record = {"status": "error", "error": repr(exc)}
            record.update(method_ms=(time.perf_counter() - started) * 1000, decision=None)
            if method_id == "xyz2mol_tm":
                record["upstream_entry"] = entry
            conn.send(("comparison", record))
            # The reference graph is never passed to the reconstruction adapter.
            reference = {"reference_status": reference_status, "ground_truth_rdmol": None}
            if reference_status == "valid":
                reference["ground_truth_rdmol"] = native_reference(native, digest)
            record.update(
                compare_output(
                    reference,
                    output,
                    **{k: v for k, v in policy.items() if k != "comparison_timeout_seconds"},
                )
            )
            conn.send(("result", record))


def run(source, out, method, workers, timeout=30.0, comparison_timeout=None):
    if not hasattr(os, "sched_setaffinity"):
        raise ValueError("Latency harness requires Linux CPU affinity")
    if workers < 1 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Worker count and timeouts must be positive")
    if method not in DATASET_METHOD_IDS:
        raise ValueError(f"Unknown method: {method}")
    out.mkdir(parents=True, exist_ok=True)
    source = source.resolve()
    cpus = sorted(os.sched_getaffinity(0))[::2][:workers]
    with sqlite3.connect(f"{Path(source).resolve().as_uri()}?mode=ro", uri=True) as db:
        source_provenance = json.loads(
            db.execute("SELECT value FROM metadata WHERE key='provenance'").fetchone()[0]
        )
    if source_provenance.get("schema") != SCHEMA_VERSION:
        raise ValueError(
            "Case store schema is obsolete; prepare it again to retain native references"
        )
    policy = comparison_policy(source_provenance, comparison_timeout)
    comparison_timeout = policy["comparison_timeout_seconds"]
    if not source_provenance["complete"]:
        raise ValueError("Case preparation was interrupted; refusing incomplete input")
    packages = {}
    for name in (
        "molgr",
        "rdkit",
        "openbabel",
        "cell2mol",
        "xyzgraph",
        "numpy",
        "scipy",
        "xyz2mol-tm",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    code = sorted((ROOT / "benchmarks").rglob("*.py")) + sorted((ROOT / "src").rglob("*.py"))
    import molgr._core.pipeline as native

    provenance = {
        "schema": SCHEMA_VERSION,
        "source": source_provenance,
        "case_store_sha256": file_hash(source),
        "method": method,
        "packages": packages,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpus": cpus,
        "max_threads": 1,
        "timeout_seconds": timeout,
        "comparison_timeout_seconds": comparison_timeout,
        "comparison_policy": policy,
        "code_sha256": hashlib.sha256("".join(file_hash(p) for p in code).encode()).hexdigest(),
        "native_binary_sha256": file_hash(Path(native.__file__)),
        "timing": "wall time of adapter.run; excludes reference parsing/matching, input reading, startup and warmup",
    }
    provenance_path = out / f"{method}-provenance.json"
    result_path = out / f"{method}.sqlite"
    if result_path.exists() and not provenance_path.exists():
        raise ValueError("Cache provenance is missing; choose a new output directory")
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
        raise ValueError("Cache provenance differs; choose a new output directory")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    results = sqlite3.connect(result_path, uri=True)
    results.execute(
        "CREATE TABLE IF NOT EXISTS results (case_id INTEGER PRIMARY KEY, method_ms REAL, status TEXT, decision TEXT, detail BLOB)"
    )
    results.execute("ATTACH DATABASE ? AS inputs", (f"{Path(source).resolve().as_uri()}?mode=ro",))
    # Keyset pagination avoids retaining millions of pending cases in memory.
    cursor_id = 0
    pending = iter(())

    def next_case():
        nonlocal cursor_id, pending
        try:
            return next(pending)
        except StopIteration:
            rows = results.execute(
                "SELECT c.id FROM inputs.cases c LEFT JOIN results r ON r.case_id=c.id WHERE c.id>? AND r.case_id IS NULL AND c.multiplicity IS NOT NULL AND c.reference_status = 'valid' ORDER BY c.id LIMIT 1000",
                (cursor_id,),
            ).fetchall()
            if not rows:
                return None
            cursor_id = rows[-1][0]
            pending = iter(r[0] for r in rows)
            return next(pending)

    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["MPLCONFIGDIR"] = str(out.resolve() / "matplotlib-cache")
    ctx = mp.get_context("spawn")
    active = {}
    completed = 0
    started = last_report = time.monotonic()

    def launch(cpu):
        parent, child = ctx.Pipe()
        process = ctx.Process(
            target=worker,
            args=(child, cpu, str(source), method, str(out / f"{method}-worker-{cpu}.log"), policy),
        )
        process.start()
        child.close()
        active[parent] = {
            "process": process,
            "cpu": cpu,
            "phase": "startup",
            "since": time.monotonic(),
            "case_id": None,
            "record": None,
        }

    def dispatch(conn, state):
        case_id = next_case()
        if case_id is None:
            conn.send(None)
            state["process"].join(5)
            if state["process"].is_alive():
                state["process"].kill()
                state["process"].join()
            conn.close()
            del active[conn]
        else:
            state.update(case_id=case_id, phase="input", since=time.monotonic(), record=None)
            conn.send(case_id)

    def save(case_id, record):
        nonlocal completed
        # Keep exact timings/statuses; compress diagnostic strings and output graphs.
        detail = zlib.compress(json.dumps(record).encode())
        results.execute(
            "INSERT INTO results VALUES (?,?,?,?,?)",
            (case_id, record.get("method_ms"), record["status"], record.get("decision"), detail),
        )
        completed += 1
        if completed % 100 == 0:
            results.commit()

    for cpu in cpus:
        launch(cpu)
    try:
        while active:
            for conn in wait(list(active), timeout=0.05):
                state = active[conn]
                try:
                    phase, record = conn.recv()
                except EOFError:
                    continue
                if phase == "ready":
                    dispatch(conn, state)
                elif phase == "result":
                    save(state["case_id"], record)
                    dispatch(conn, state)
                else:
                    state.update(phase=phase, record=record, since=time.monotonic())
            now = time.monotonic()
            for conn, state in list(active.items()):
                phase = state["phase"]
                limit = (
                    120
                    if phase == "startup"
                    else comparison_timeout
                    if phase == "comparison"
                    else timeout
                )
                dead = not state["process"].is_alive()
                if dead or now - state["since"] > limit:
                    state["process"].kill()
                    state["process"].join()
                    conn.close()
                    del active[conn]
                    if state["case_id"] is None:
                        raise RuntimeError(f"Worker startup failed: CPU {state['cpu']}")
                    record = state["record"]
                    if record is None:
                        record = {
                            "status": "crash" if dead else "timeout",
                            "decision": None,
                            "method_ms": timeout * 1000 if phase == "method" and not dead else None,
                        }
                    else:
                        record["decision"] = "inconclusive"
                    record.update(
                        error=f"Worker {'exited' if dead else 'timed out'} in {phase}",
                        interrupted_phase=phase,
                    )
                    save(state["case_id"], record)
                    launch(state["cpu"])
            if now - last_report > 30:
                print(f"{method}: {completed} new attempts, {now - started:.0f}s", flush=True)
                last_report = now
    finally:
        for conn, state in active.items():
            state["process"].kill()
            state["process"].join()
            conn.close()
        results.commit()
        results.close()
    print(
        f"{method}: completed {completed} new attempts in {time.monotonic() - started:.1f}s",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--method", choices=DATASET_METHOD_IDS, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument(
        "--comparison-timeout",
        type=float,
        help="Override the source policy only when it is not publisher-pinned",
    )
    args = parser.parse_args()
    run(args.cases, args.out, args.method, args.workers, args.timeout, args.comparison_timeout)


if __name__ == "__main__":
    main()
