#!/usr/bin/env python3
"""Sequential comparison of native and archived Python evaluators on saved CSV pairs.

This measures evaluation including input preparation, excluding XYZ reconstruction
and SMILES parsing. Verdict transitions are reported, never treated as an oracle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path

from rdkit import Chem, rdBase

from molgr import _core
from molgr.utils import _legacy_equivalence as legacy
from molgr.utils import equivalence as native


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    with args.csv.open(newline="") as stream:
        rows = [
            r
            for r in csv.DictReader(stream)
            if r.get("status", "ok") == "ok"
            and r.get("ground_truth_smiles")
            and r.get("predicted_smiles")
        ]
    if 0 < args.limit < len(rows):
        rows = [rows[i * (len(rows) - 1) // max(1, args.limit - 1)] for i in range(args.limit)]
    pairs = []
    with rdBase.BlockLogs():
        for row in rows:
            a, b = (
                Chem.MolFromSmiles(row["ground_truth_smiles"]),
                Chem.MolFromSmiles(row["predicted_smiles"]),
            )
            if a is not None and b is not None:
                pairs.append((row, a, b))
    if not pairs:
        raise ValueError("No parseable molecule pairs")
    package = Path(native.__file__).parents[1]
    report = {
        "count": len(pairs),
        "parse_failures": len(rows) - len(pairs),
        "rdkit": rdBase.rdkitVersion,
        "csv_sha256": digest(args.csv),
        "binary_sha256": digest(_core.__file__),
        "python_sha256": {
            str(p.relative_to(package)): digest(p) for p in sorted(package.rglob("*.py"))
        },
        "parameters": {
            "max_resonance": args.max_resonance,
            "use_chirality": False,
            "ignore_metal_coordination_bonds": True,
        },
        "scope": "sequential evaluation including preparation; excludes parsing and XYZ reconstruction",
        "runs": [],
        "cases": {},
    }
    with rdBase.BlockLogs():
        for repeat in range(args.repeat):
            engines = [("native", native), ("legacy", legacy)]
            if repeat % 2:
                engines.reverse()
            for name, module in engines:
                cases = []
                for row, a, b in pairs:
                    start = time.perf_counter()
                    result = module.evaluate_equivalence(
                        a, b, use_chirality=False, max_resonance=args.max_resonance
                    )
                    elapsed = 1000 * (time.perf_counter() - start)
                    cases.append(
                        {
                            "id": row.get("id", row.get("case_idx")),
                            "left": row["ground_truth_smiles"],
                            "right": row["predicted_smiles"],
                            "decision": result.decision.value,
                            "method": result.method.value if result.method else None,
                            "reason": result.reason,
                            "ms": elapsed,
                            "bounded": vars(result.bounded_search)
                            if result.bounded_search
                            else None,
                            "native_stats": getattr(result, "native_stats", {}),
                        }
                    )
                times = sorted(c["ms"] for c in cases)
                summary = {
                    "engine": name,
                    "repeat": repeat,
                    "total_ms": sum(times),
                    "p50_ms": statistics.median(times),
                    "p95_ms": times[int(0.95 * (len(times) - 1))],
                    "p99_ms": times[int(0.99 * (len(times) - 1))],
                    "max_ms": max(times),
                    "decisions": dict(Counter(c["decision"] for c in cases)),
                }
                if name == "native":
                    summary["stages_ms"] = {
                        stage: sum(c["native_stats"].get(stage, 0) for c in cases)
                        for stage in ("preparation_ms", "mapping_ms", "resonance_ms")
                    }
                    summary["python_and_binding_ms"] = sum(times) - sum(
                        summary["stages_ms"].values()
                    )
                report["runs"].append(summary)
                report["cases"][name] = cases
                print(json.dumps(summary), flush=True)
    paired = list(zip(report["cases"]["native"], report["cases"]["legacy"]))
    report["transitions"] = [
        {"native": a, "legacy": b} for a, b in paired if a["decision"] != b["decision"]
    ]
    report["transition_counts"] = dict(
        Counter(b["decision"] + " -> " + a["decision"] for a, b in paired)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["transition_counts"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=500, help="Evenly sampled rows; 0 selects all")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--max-resonance", type=int, default=100)
    args = parser.parse_args()
    if args.repeat < 1 or args.limit < 0:
        parser.error("repeat must be positive and limit must be nonnegative")
    run(args)
