#!/usr/bin/env python3
"""Regenerate offline tmQMg fixtures from the pinned published release, never CSV labels.

Run as ``python -m scripts.freeze_tmqmg_fixtures --dataset <checkout>``.
Selection and regression classifications do not determine published eligibility.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import zlib
from pathlib import Path

from rdkit import Chem

from benchmarks.dataset_xyz_benchmark import tmqmg as corrected
from molgr.fallback.utils.consts import NON_METAL_DICT


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/data"
# Preserve the historical candidate-search selection by original input ordinal,
# NOT target index. Current tests address these same inputs by stable case ID.
SEARCH_INPUT_INDICES = {113, 232, 430, 431, 522, 627, 684, 727, 1132}
# Direct consumers without a separate XYZ/selection manifest: concurrency and
# metal candidate-search controls and localized SMILES round trips.
DIRECT_CASE_IDS = {"ABAJAP", "AROJEW", "CEHHAB", "LIBZOM"}


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path, rows):
    if rows:
        fields = list(rows[0])
    else:
        with path.open(newline="", encoding="utf-8") as previous:
            fields = next(csv.reader(previous))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_release(directory, ids, input_indices=()):
    """Extract exact inputs/native graphs and explicit exclusions from verified parts."""
    manifest = corrected.load_manifest(directory)
    ids, input_indices = set(ids), set(input_indices)
    records, sdf_records = {}, {}
    input_offset = 0
    for part in manifest["parts"]:
        required = {x for x in ids if part["first"] <= x <= part["last"]}
        indices = {x for x in input_indices if input_offset < x <= input_offset + part["inputs"]}
        if not required and not indices:
            input_offset += part["inputs"]
            continue
        for key in ("xyz", "sdf", "metadata"):
            path = directory / part[key]
            if _digest(path.read_bytes()) != manifest["sha256"][part[key]]:
                raise ValueError(f"Release checksum mismatch: {path}")
        with (directory / part["metadata"]).open(encoding="utf-8") as stream:
            metadata = [json.loads(line) for line in stream]
        meta_ids = [m["id"] for m in metadata]
        if len(metadata) != part["inputs"] or len(set(meta_ids)) != len(meta_ids):
            raise ValueError(f"Invalid metadata count or duplicate ID: {part['metadata']}")
        for i, meta in enumerate(metadata, input_offset + 1):
            case_id = meta["id"]
            if case_id not in required and i not in indices:
                continue
            if case_id in records:
                raise ValueError(f"Duplicate fixture ID: {case_id}")
            if not isinstance(meta["benchmark"], bool) or meta["benchmark"] == bool(
                meta["exclusion"]
            ):
                raise ValueError(f"Invalid eligibility: {case_id}")
            required.add(case_id)
            records[case_id] = {
                "id": case_id,
                "input_index": i,
                "charge": meta["charge"],
                "multiplicity": meta["multiplicity"],
                "benchmark": meta["benchmark"],
                "exclusion": meta["exclusion"],
                "xyz_sha256": meta["xyz_sha256"],
                "graph_sha256": meta.get("graph_sha256"),
                "tier": meta.get("tier"),
                "source_part": part["metadata"],
            }
        xyz_ids = []
        with (directory / part["xyz"]).open("rb") as stream:
            for count in stream:
                comment = dict(v.split("=", 1) for v in stream.readline().decode().split())
                xyz_ids.append(comment["id"])
                xyz = count + b"\n" + b"".join(stream.readline() for _ in range(int(count)))
                if comment["id"] in required:
                    record = records[comment["id"]]
                    if _digest(xyz) != record["xyz_sha256"]:
                        raise ValueError(f"XYZ identity mismatch: {record['id']}")
                    record["xyz"] = xyz.decode()
                    atoms = Chem.MolFromXYZBlock(record["xyz"])
                    if atoms is None:
                        raise ValueError(f"Invalid XYZ: {record['id']}")
                    metals = [
                        a.GetSymbol()
                        for a in atoms.GetAtoms()
                        if a.GetAtomicNum() not in NON_METAL_DICT
                    ]
                    if len(metals) != 1:
                        raise ValueError(f"Expected one metal in tmQMg input: {record['id']}")
                    record["metal_center"] = metals[0]
                    if (
                        int(comment["charge"]) != record["charge"]
                        or int(comment["multiplicity"]) != record["multiplicity"]
                    ):
                        raise ValueError(f"XYZ electronic state mismatch: {record['id']}")
        if xyz_ids != meta_ids:
            raise ValueError(f"XYZ/metadata order or count mismatch: {part['xyz']}")
        sdf_ids = set()
        with (directory / part["sdf"]).open("rb") as stream:
            for raw in corrected._sdf_records(stream):
                case_id = raw.partition(b"\n")[0].decode()
                if case_id in sdf_ids:
                    raise ValueError(f"Duplicate SDF ID: {case_id}")
                sdf_ids.add(case_id)
                if case_id not in required:
                    continue
                props = {}
                for field in raw.split(b"\n>  <")[1:]:
                    key, value = field.split(b">\n", 1)
                    props[key.decode()] = value.partition(b"\n")[0].decode()
                graph = zlib.decompress(base64.b64decode(props["NATIVE_GRAPH_ZLIB_BASE64"]))
                record = records[case_id]
                if not record["benchmark"] or _digest(graph) != record["graph_sha256"]:
                    raise ValueError(f"Graph identity mismatch: {case_id}")
                if props["CASE_ID"] != case_id or props["GRAPH_SHA256"] != record["graph_sha256"]:
                    raise ValueError(f"SDF identity mismatch: {case_id}")
                if props["XYZ_SHA256"] != record["xyz_sha256"]:
                    raise ValueError(f"SDF/XYZ identity mismatch: {case_id}")
                if (
                    int(props["INPUT_CHARGE"]) != record["charge"]
                    or int(props["INPUT_MULTIPLICITY"]) != record["multiplicity"]
                ):
                    raise ValueError(f"Graph electronic state mismatch: {case_id}")
                record["native_graph"] = props["NATIVE_GRAPH_ZLIB_BASE64"]
                record["native_rdkit_version"] = props["NATIVE_RDKIT_VERSION"]
                mol = Chem.Mol(graph)
                xyz_mol = Chem.MolFromXYZBlock(record["xyz"])
                if (
                    mol is None
                    or [a.GetAtomicNum() for a in mol.GetAtoms()]
                    != [a.GetAtomicNum() for a in xyz_mol.GetAtoms()]
                    or Chem.GetFormalCharge(mol) != record["charge"]
                    or mol.GetNumConformers() != 1
                    or not (
                        mol.GetConformer().GetPositions() == xyz_mol.GetConformer().GetPositions()
                    ).all()
                ):
                    raise ValueError(f"Native graph/input conservation mismatch: {case_id}")
                record["reference_smiles"] = Chem.MolToSmiles(mol)
                record["reconstruction_scope"] = (
                    mol.GetProp("STANDARD_RECONSTRUCTION_SCOPE")
                    if mol.HasProp("STANDARD_RECONSTRUCTION_SCOPE")
                    else ""
                )
                record["sdf_part"] = part["sdf"]
                record["sdf_sha256"] = manifest["sha256"][part["sdf"]]
                sdf_records[case_id] = raw + b"$$$$\n"
        if len(sdf_ids) != part["targets"] or sdf_ids != {
            m["id"] for m in metadata if m["benchmark"]
        }:
            raise ValueError(f"SDF/metadata target count mismatch: {part['sdf']}")
        input_offset += part["inputs"]
    if input_indices - {c["input_index"] for c in records.values()}:
        raise ValueError("Fixture input indices absent from release")
    if ids - records.keys():
        raise ValueError(f"Fixture IDs absent from release: {sorted(ids - records.keys())}")
    for case in records.values():
        if bool(case.get("native_graph")) != case["benchmark"] or "xyz" not in case:
            raise ValueError(f"Incomplete published fixture: {case['id']}")
    return records, sdf_records


def freeze_fixtures(*, dataset: Path, data_dir: Path = DATA):
    root = data_dir / "tmqmg"
    spec_path = root / "fixture_sources.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    ids = {row["id"] for rows in spec.values() for row in rows}
    for group in spec:
        with (root / group / "manifest.csv").open(encoding="utf-8", newline="") as stream:
            ids.update(row["case_id"] for row in csv.DictReader(stream))
    mixed_path = data_dir / "reviewed/reconstruction/manifest.json"
    mixed = json.loads(mixed_path.read_text(encoding="utf-8"))
    ids.update(r["case_id"] for r in mixed["fixtures"] if r.get("source") == "tmQMg-Corrected")
    local_manifests = {
        path: json.loads(path.read_text(encoding="utf-8")) for path in root.rglob("manifest.json")
    }
    for payload in local_manifests.values():
        ids.update(row["case_id"] for row in payload.get("fixtures", []))
    xyz_paths = [p for p in data_dir.rglob("*.xyz") if p.stem.isalpha() and p.stem.isupper()]
    ids.update(p.stem for p in xyz_paths)
    review_path = data_dir / "reviewed/tmqmg/manifest.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    ids.update(r["case_id"] for r in review["fixtures"])
    ids.update(DIRECT_CASE_IDS)
    selection_path = root / "published/regression_case_ids.json"
    regression_ids = json.loads(selection_path.read_text(encoding="utf-8"))
    ids.update(regression_ids)
    source = corrected.provenance(dataset)
    records, sdfs = read_release(dataset, ids, SEARCH_INPUT_INDICES)
    notices = {
        name: (dataset / "sources" / name).read_bytes()
        for name in (
            "DATA-LICENSE.md",
            "CC-BY-4.0.txt",
            "tmQMg-LICENSE.txt",
            "xyz2mol_tm-LICENSE.txt",
        )
    }
    excluded = {key for key, case in records.items() if not case["benchmark"]}
    records = {key: case for key, case in records.items() if case["benchmark"]}
    _write_json(
        root / "published/cases.json",
        {
            "schema_version": 1,
            "source_dataset": source,
            "cases": {key: records[key] for key in sorted(records)},
        },
    )
    (root / "published/release-manifest.json").write_bytes((dataset / "manifest.json").read_bytes())
    for name, content in notices.items():
        path = root / "published/sources" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    # Excluded release inputs are not retained, even as local/boundary controls.
    for path in xyz_paths:
        if path.stem in excluded:
            path.unlink()
        else:
            path.write_bytes(records[path.stem]["xyz"].encode())
    for path, payload in local_manifests.items():
        payload["fixtures"] = [r for r in payload["fixtures"] if r["case_id"] not in excluded]
        _write_json(path, payload)
    for path in data_dir.rglob("*.sdf"):
        if path.stem in excluded:
            path.unlink()
    for group, selections in spec.items():
        path = root / group / "manifest.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            old = list(csv.DictReader(stream))
        existing_ids = {r["id"] for r in selections}
        for row in old:
            if row["case_id"] not in existing_ids:
                selections.append(
                    {
                        "id": row["case_id"],
                        "selection_classification": row.get(
                            "selection_classification", row.get("classification")
                        ),
                        "selection_reason": row.get("selection_reason", row.get("reason")),
                    }
                )
        selections = [row for row in selections if row["id"] not in excluded]
        spec[group] = selections
        rows = []
        for selection in selections:
            for name in ("classification", "reason"):
                if name in selection:
                    selection["selection_" + name] = selection.pop(name)
            case = records[selection["id"]]
            if "approved_graph" in selection or "approved_answer_database" in selection:
                selection["assert_published_graph"] = True
            for field in (
                "approved_graph",
                "approved_answer_database",
                "accepted_as_alternative",
                "rdkit_2024_coordination_compatibility",
                "comparison",
                "comparison_note",
                "review_source",
            ):
                selection.pop(field, None)
            rows.append(
                {
                    "case_id": case["id"],
                    "row_index": case["input_index"],
                    "charge": case["charge"],
                    "multiplicity": case["multiplicity"],
                    "metal_center": case["metal_center"],
                    "reference_smiles": case.get("reference_smiles", ""),
                    "benchmark": str(case["benchmark"]).lower(),
                    "exclusion": case["exclusion"] or "",
                    "selection_classification": selection["selection_classification"],
                    "selection_reason": selection["selection_reason"],
                    "xyz_file": case["id"] + ".xyz",
                    "xyz_sha256": case["xyz_sha256"],
                    "graph_sha256": case["graph_sha256"] or "",
                }
            )
        _write_csv(path, rows)
    _write_json(spec_path, spec)
    review["source_dataset"] = source
    review["schema_version"] = 2
    review["fixtures"] = [
        {
            "case_id": c["case_id"],
            "kind": "published_reference",
            "source": "tmQMg-Corrected",
            "structure_file": f"reference_graph/{c['case_id']}.sdf",
            "xyz_file": f"../../tmqmg/published/cases.json#{c['case_id']}",
            "total_charge": records[c["case_id"]]["charge"],
            "spin_multiplicity": records[c["case_id"]]["multiplicity"],
            "total_radical_electrons": records[c["case_id"]]["multiplicity"] - 1,
            "graph_sha256": records[c["case_id"]]["graph_sha256"],
            "reference_smiles": records[c["case_id"]].get("reference_smiles", ""),
        }
        for c in review["fixtures"]
        if c["case_id"] not in excluded
    ]
    for record in review["fixtures"]:
        target = review_path.parent / record["structure_file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(sdfs[record["case_id"]])
    _write_json(review_path, review)
    # Mixed reviewed fixtures retain non-tmQMg examples unchanged.
    mixed["fixtures"] = [r for r in mixed["fixtures"] if r["case_id"] not in excluded]
    for record in mixed["fixtures"]:
        if record["case_id"] not in records:
            continue
        case = records[record["case_id"]]
        record.update(
            kind="published_reference",
            source="tmQMg-Corrected",
            source_dataset=source,
            xyz_sha256=case["xyz_sha256"],
            total_charge=case["charge"],
            spin_multiplicity=case["multiplicity"],
            total_radical_electrons=case["multiplicity"] - 1,
            graph_sha256=case["graph_sha256"],
        )
        record.pop("review_status", None)
        record.pop("smiles_stereo_roundtrip", None)
        previous = mixed_path.parent / record["structure_file"]
        record["structure_file"] = f"reference_graph/{case['id']}.sdf"
        target = mixed_path.parent / record["structure_file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(sdfs[case["id"]])
        if previous != target and previous.suffix == ".sdf":
            previous.unlink(missing_ok=True)
    _write_json(mixed_path, mixed)
    _write_json(selection_path, sorted(case_id for case_id in regression_ids if case_id in records))
    # The stress runner only needs inputs; its reference links must not point to retired graphs.
    stress_path = data_dir / "xyz_stress_manifest.csv"
    with stress_path.open(encoding="utf-8", newline="") as stream:
        stress = list(csv.DictReader(stream))
    present = {str(p.relative_to(data_dir).as_posix()): p for p in data_dir.rglob("*.xyz")}
    stress = [row for row in stress if row["xyz_file"] in present]
    listed = {row["xyz_file"] for row in stress}
    for relative in sorted(present.keys() - listed):
        case_id = present[relative].stem
        if case_id not in records:
            raise ValueError(f"Unclassified non-tmQMg stress input: {relative}")
        stress.append(
            {
                "case_id": case_id,
                "charge": "",
                "spin_multiplicity": "",
                "xyz_file": relative,
                "reference_file": "",
                "reference_source": "",
                "reference_status": "",
            }
        )
    for row in stress:
        case_id = Path(row["xyz_file"]).stem
        if case_id in records:
            case = records[case_id]
            row.update(
                charge=case["charge"],
                spin_multiplicity=case["multiplicity"],
                reference_file="",
                reference_source=f"tests/data/tmqmg/published/cases.json#{case_id}",
                reference_status="dataset" if case["benchmark"] else "source_issue",
            )
    _write_csv(stress_path, stress)
    for directory in (
        root / "sulfur_electronic_assignment/approved_graph",
        root / "electronic_validity/approved_graph",
        review_path.parent / "approved_graph",
    ):
        if directory.is_dir():
            for path in directory.glob("*.sdf"):
                if path.stem in records or path.stem in excluded:
                    path.unlink()
            if not any(directory.iterdir()):
                directory.rmdir()
    print(
        f"Frozen {len(records)} inputs; {sum(c['benchmark'] for c in records.values())} released targets."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=corrected.DATASET)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    args = parser.parse_args()
    freeze_fixtures(dataset=args.dataset, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
