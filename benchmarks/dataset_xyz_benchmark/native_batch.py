"""Development full-set scoring using the unordered native C++ batch scheduler."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import sqlite3
import time
import zlib
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

from rdkit import Chem, RDLogger, rdBase

from benchmarks._timeout import case_timeout
from benchmarks.dataset_xyz_benchmark.comparison import (
    compare_output,
    comparison_policy,
    native_reference,
)
from benchmarks.dataset_xyz_benchmark.prepare import ROOT, SCHEMA_VERSION, file_hash
from benchmarks.smiles_xyz_benchmark.methods.base import MethodRunOutput
from benchmarks.smiles_xyz_benchmark.methods.postprocess import remove_hs_without_sanitize
from molgr import core
from molgr.config import MolGRConfig
from molgr.interface import _finalize_rdmol
from molgr.utils.converter import mol_data_to_rdkit


def check_group(group, config, policy):
    RDLogger.DisableLog("rdApp.*")
    settings = dict(policy)
    timeout = settings.pop("comparison_timeout_seconds")
    records = []
    for case_id, xyz, packed, error, reference, digest in group:
        mol = smiles = None
        if packed is not None:
            try:
                mol = _finalize_rdmol(
                    Chem.Mol(packed),
                    xyz,
                    backend="cpp",
                    make_dative_bonds=True,
                    make_stereochemistry=True,
                    config=config,
                )
            except Exception as exc:
                error = str(exc)
            if mol is not None:
                try:
                    mol = remove_hs_without_sanitize(mol)
                    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                except Exception as exc:
                    error = f"MolGR result postprocess failed: {exc}"
        output = MethodRunOutput("error" if error is not None else "ok", error, smiles, mol)
        record = {
            "status": output.status,
            "error": error,
            "predicted_smiles": smiles,
            "decision": None,
        }
        if mol is not None:
            ref = native_reference(reference, digest)
            started = time.monotonic()
            try:
                with case_timeout(timeout, "comparison"):
                    record.update(
                        compare_output(
                            {"reference_status": "valid", "ground_truth_rdmol": ref},
                            output,
                            **settings,
                        )
                    )
            except TimeoutError:
                record["interrupted_phase"] = "comparison"
            if record.get("interrupted_phase") or time.monotonic() - started >= timeout:
                record.update(
                    decision="inconclusive",
                    comparison_reason="Comparison timed out",
                    interrupted_phase="comparison",
                )
        records.append(
            (case_id, output.status, record["decision"], zlib.compress(json.dumps(record).encode()))
        )
    return records


def reconstruction_groups(rows, config, workers):
    """Only geometry and electronic targets enter C++. Reattach references by result index."""
    requests = [
        (zlib.decompress(xyz).decode(), charge, mult - 1) for _, xyz, charge, mult, _, _ in rows
    ]
    # The checker needs the original electron annotations and full-precision geometry.
    flags = int(Chem.PropertyPickleOptions.AllProps) | int(
        Chem.PropertyPickleOptions.CoordsAsDouble
    )
    batch = core.pipeline.reconstruct_with_metals.batch_xyz2omol(
        requests,
        config=config,
        max_workers=workers,
        queue_size=16,
        ordered=False,
    )
    seen = set()
    group = []
    with closing(batch):
        for item in batch:
            i = item["index"]
            if not 0 <= i < len(rows) or i in seen:
                raise RuntimeError(f"Unexpected native result index {i}")
            seen.add(i)
            case_id, _, _, _, reference, digest = rows[i]
            data = item["molecule_data"]
            packed = error = None
            if data is None:
                error = json.dumps(
                    item.get("diagnostics") or "Native reconstruction returned no molecule"
                )
            else:
                try:
                    packed = mol_data_to_rdkit(data).ToBinary(flags)
                except Exception as exc:
                    error = f"RDKit conversion failed: {exc}"
            group.append((case_id, requests[i][0], packed, error, reference, digest))
            if len(group) == 16:
                yield group
                group = []
    if len(seen) != len(rows):
        raise RuntimeError(f"Missing {len(rows) - len(seen)} native results")
    if group:
        yield group


def save_scores(db, futures):
    for future in futures:
        db.executemany(
            "INSERT INTO results(case_id,status,decision,detail) VALUES (?,?,?,?)", future.result()
        )


def run(cases, out, workers=8, checkers=4):
    if workers < 1 or checkers < 1:
        raise ValueError("Worker counts must be positive")
    if os.name != "posix":
        raise ValueError("This scoring runner needs POSIX comparison timers")
    started = time.monotonic()
    config = MolGRConfig()
    config.cpp_backend.max_threads = config.cpp_backend.target_bucket_parallel_max_threads = 1
    RDLogger.DisableLog("rdApp.*")
    with closing(sqlite3.connect(f"{cases.resolve().as_uri()}?mode=ro", uri=True)) as source:
        meta = json.loads(
            source.execute("SELECT value FROM metadata WHERE key='provenance'").fetchone()[0]
        )
        if meta["schema"] != SCHEMA_VERSION or not meta["complete"]:
            raise ValueError("Prepare a complete, current case store first")
        policy = comparison_policy(meta)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "molgr_cpp.sqlite"
        provenance_path = out / "molgr_cpp-provenance.json"
        files = sorted((ROOT / "benchmarks").rglob("*.py")) + sorted(
            (ROOT / "src/molgr").rglob("*.py")
        )
        provenance = {
            "method": "molgr_cpp",
            "scheduler": "batch_xyz2omol",
            "source": meta,
            "case_store_sha256": file_hash(cases),
            "native_binary_sha256": file_hash(Path(core.__file__)),
            "code_sha256": hashlib.sha256("".join(map(file_hash, files)).encode()).hexdigest(),
            "config": asdict(config),
            "rdkit": rdBase.rdkitVersion,
            "comparison_policy": policy,
            "workers": workers,
            "checkers": checkers,
            "timing": "end-to-end wall time; per-case latency unavailable",
        }
        if provenance_path.exists():
            if json.loads(provenance_path.read_text()) != provenance:
                raise ValueError("Result provenance differs; choose a new output directory")
        elif path.exists():
            raise ValueError("Result provenance is missing; choose a new output directory")
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")

        with closing(sqlite3.connect(path)) as db, ProcessPoolExecutor(
            max_workers=checkers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            db.execute(
                "CREATE TABLE IF NOT EXISTS results (case_id INTEGER PRIMARY KEY, method_ms REAL, status TEXT, decision TEXT, detail BLOB)"
            )
            source.execute("ATTACH DATABASE ? AS scored", (f"{path.resolve().as_uri()}?mode=ro",))
            last_id = completed = 0
            while True:
                rows = source.execute(
                    """
                    SELECT c.id,xyz,charge,multiplicity,reference_native,reference_sha256 FROM cases c
                    WHERE c.id>? AND reference_status='valid' AND multiplicity IS NOT NULL
                    AND NOT EXISTS (SELECT 1 FROM scored.results WHERE case_id=c.id)
                    ORDER BY id LIMIT 4096
                """,
                    (last_id,),
                ).fetchall()
                if not rows:
                    break
                last_id = rows[-1][0]
                pending = set()

                with db:
                    for group in reconstruction_groups(rows, config, workers):
                        pending.add(pool.submit(check_group, group, config, policy))
                        if len(pending) >= 2 * checkers:
                            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                            save_scores(db, finished)
                    save_scores(db, pending)
                completed += len(rows)
                print(f"{completed} new scores in {time.monotonic() - started:.1f}s", flush=True)
    elapsed = time.monotonic() - started
    print(
        f"{completed} new scores, {elapsed:.2f}s end to end, {completed / elapsed:.2f} cases/s",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", default=8, type=int, help="Native reconstruction workers")
    parser.add_argument(
        "--checkers", default=4, type=int, help="RDKit postprocessing and comparison processes"
    )
    args = parser.parse_args()
    run(args.cases, args.out, args.workers, args.checkers)


if __name__ == "__main__":
    main()
