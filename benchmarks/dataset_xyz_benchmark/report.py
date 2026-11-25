"""Aggregate every conformer with equal weight per parent molecule."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from benchmarks.dataset_xyz_benchmark.prepare import file_hash


def summarize(cases, results, *, allow_partial=False):
    provenance_path = results.with_name(results.stem + "-provenance.json")
    if not provenance_path.exists():
        raise ValueError("Result provenance is missing")
    provenance = json.loads(provenance_path.read_text())
    if provenance["case_store_sha256"] != file_hash(cases):
        raise ValueError("Results belong to a different case store")
    with sqlite3.connect(f"{results.resolve().as_uri()}?mode=ro", uri=True) as db:
        db.execute("ATTACH DATABASE ? AS inputs", (f"{cases.resolve().as_uri()}?mode=ro",))
        source = json.loads(
            db.execute("SELECT value FROM inputs.metadata WHERE key='provenance'").fetchone()[0]
        )
        if not source["complete"]:
            raise ValueError("Case store preparation is incomplete")
        eligible, attempted, matched = db.execute("""
            SELECT count(*), count(r.case_id), coalesce(sum(r.decision='equivalent'),0)
            FROM inputs.cases c LEFT JOIN results r ON c.id=r.case_id
            WHERE c.reference_status='valid'
        """).fetchone()
        if attempted != eligible and not allow_partial:
            raise ValueError(
                f"Incomplete run: {attempted}/{eligible} eligible conformers attempted"
            )
        groups = db.execute("""
            SELECT c.molecule, count(*), count(r.case_id), coalesce(sum(r.decision='equivalent'),0),
                   coalesce(sum(r.decision='not_equivalent'),0), coalesce(sum(r.decision='inconclusive'),0)
            FROM inputs.cases c LEFT JOIN results r ON c.id=r.case_id
            WHERE c.reference_status='valid' GROUP BY c.molecule
        """).fetchall()
        macro = (
            sum(matches / total for _, total, _, matches, _, _ in groups) / len(groups)
            if groups
            else None
        )
        return {
            "schema": 2,
            "provenance": provenance,
            "dataset": source["dataset"],
            "pilot": source.get("pilot", False),
            "complete": attempted == eligible,
            "source_molecules": source["molecules"],
            "source_conformers": source["cases"],
            "reference_statuses": source["reference_statuses"],
            "eligible_molecules": len(groups),
            "eligible_conformers": eligible,
            "attempted_eligible_conformers": attempted,
            "matched_conformers": matched,
            "molecule_macro_match_rate": macro,
            "all_eligible_conformers_match_molecule_rate": sum(
                matches == total for _, total, _, matches, _, _ in groups
            )
            / len(groups)
            if groups
            else None,
            "conformer_micro_match_rate": matched / eligible if eligible else None,
            "decision_counts": dict(
                db.execute(
                    "SELECT coalesce(r.decision,'unassessed'),count(*) FROM inputs.cases c LEFT JOIN results r ON c.id=r.case_id WHERE c.reference_status='valid' GROUP BY r.decision"
                )
            ),
            "aggregation": "mean over parents of matched/eligible conformers; every released conformer is imported; no energy, degeneracy or Boltzmann weighting; failures count as unmatched; unresolved/excluded source references reported separately",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = summarize(args.cases, args.results, allow_partial=args.allow_partial)
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
