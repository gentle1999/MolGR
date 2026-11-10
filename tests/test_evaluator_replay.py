from __future__ import annotations

import csv
import importlib.util
import json
import multiprocessing
import os
import signal
import sqlite3
import sys
from pathlib import Path

import pytest
from rdkit import Chem, rdBase


ROOT = Path(__file__).resolve().parents[1]
# These tests exercise the historical original/v1/v2 replay contract.
SOURCE = ROOT / "src" / "molgr" / "utils" / "_legacy_equivalence.py"
SPEC = importlib.util.spec_from_file_location(
    "_replay_under_test", ROOT / "scripts" / "evaluator_replay.py"
)
assert SPEC is not None and SPEC.loader is not None
replay = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = replay
SPEC.loader.exec_module(replay)


pytestmark = pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="The replay integration runner requires multiprocessing fork (Linux/macOS).",
)


def _inputs(
    tmp_path: Path, pairs: list[tuple[str, str, str, str | None]], limit: int | None = None
):
    root = tmp_path / "inputs"
    (root / "candidates").mkdir(parents=True)
    references, candidates = [], []
    for case_id, status, reference, candidate in pairs:
        references.append(
            {
                "case_id": case_id,
                "reference_smiles": reference,
                "reference_smiles_sha256": replay._sha256(reference.encode("utf-8")),
                "reference_status": status,
                "reference_reason": "frozen test membership",
            }
        )
        path = root / "candidates" / f"{case_id}.json"
        if candidate is not None:
            mol = Chem.MolFromSmiles(candidate)
            assert mol is not None
            artifact = {
                "case_id": case_id,
                "status": "ok",
                "outcome": "success",
                "predicted_graph": {"molblock_v2000": Chem.MolToMolBlock(mol)},
            }
            path.write_text(json.dumps(artifact), encoding="utf-8")
        candidates.append(
            {
                "case_id": case_id,
                "artifact_path": f"candidates/{case_id}.json",
                "sha256": replay._file_sha256(path) if path.exists() else None,
                "available": candidate is not None,
                "status": "ok" if candidate else "error",
                "error": None if candidate else "native failure",
            }
        )
    manifest = root / "reference-validity-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "reference-validity/v1",
                "sources": {"fixture": str(SOURCE)},
                "cases": references,
            }
        ),
        encoding="utf-8",
    )
    index = root / "candidate-inventory.json"
    index.write_text(
        json.dumps({"schema": "candidate-inventory/v1", "cases": candidates}), encoding="utf-8"
    )
    source = root / "equivalence.py"
    source.write_bytes(SOURCE.read_bytes())
    argv = [
        "--manifest",
        str(manifest),
        "--candidate-index",
        str(index),
        "--candidate-root",
        str(root),
        "--out-dir",
        str(tmp_path / "output"),
        "--rdkit-version",
        rdBase.rdkitVersion,
    ]
    for version in replay.VERSIONS:
        argv.extend([f"--source-{version}", str(source)])
    if limit is not None:
        argv.extend(["--limit", str(limit)])
    return replay.parse_args(argv)


def _rows(args):
    return [json.loads(line) for line in (args.out_dir / "raw.jsonl").read_text().splitlines()]


def test_resume_identity_includes_native_binary_and_imported_python_sources(tmp_path):
    args = _inputs(tmp_path, [("a", "valid", "CCO", "CCO")])
    replay.run(args)
    # The identity stored alongside results must cover more than the wrapper.
    identity = json.loads((args.out_dir / "checkpoint-metadata.json").read_text())["identity"]
    assert identity["runtime_binaries_sha256"]["molgr_core"] == replay._file_sha256(
        Path(replay._core.__file__)
    )
    assert "utils/converter.py" in identity["molgr_python_sha256"]


def test_excluded_and_unresolved_are_scored_before_frozen_validity_aggregation(tmp_path):
    args = _inputs(
        tmp_path,
        [
            ("valid_eq", "valid", "CCO", "CCO"),
            ("valid_neq", "valid", "CCO", "COC"),
            ("excluded_neq", "confirmed_excluded", "CCO", "COC"),
            ("unresolved_eq", "unresolved", "CCO", "CCO"),
            ("bad_reference", "valid", "(", "CCO"),
            ("missing", "valid", "CCO", None),
            ("open_shell", "valid", "[NH3+]Cc1[n-]nnc1[CH]O", "[NH3+]CC1=N[N-][N]C1=CO"),
        ],
    )
    summary = replay.run(args)
    rows = {(row["case_id"], row["version"]): row for row in _rows(args)}
    assert summary["complete"] is True
    assert summary["expected_result_slots"] == summary["completed_result_slots"] == 21
    for version in replay.VERSIONS:
        assert rows["excluded_neq", version]["decision"] == "NEQ"
        assert rows["unresolved_eq", version]["decision"] == "EQ"
        invalid = rows["bad_reference", version]
        assert invalid["status"] == "input_error"
        assert invalid["decision"] is None and invalid["error"]["type"] == "ValueError"
        unavailable = rows["missing", version]
        assert unavailable["status"] == "candidate_unavailable"
        assert unavailable["decision"] is None
        assert unavailable["input_diagnostic"]["inventory_error"] == "native failure"
        valid = summary["versions"][version]["by_reference_status"]["valid"]
        all_population = summary["versions"][version]["all_population"]
        assert (
            valid["total"],
            valid["EQ"],
            valid["input_error"],
            valid["candidate_unavailable"],
        ) == (5, 1, 1, 1)
        assert (all_population["total"], all_population["EQ"], all_population["not_scored"]) == (
            7,
            2,
            0,
        )
        if version == "original":
            assert (valid["NEQ"], valid["INC"]) == (2, 0)
            assert rows["open_shell", version]["raw_decision"] is False
        else:
            assert (valid["NEQ"], valid["INC"]) == (1, 1)
            evidence = rows["open_shell", version]
            assert evidence["decision"] == "INC"
            assert evidence["raw_result"]["open_shell_heuristic_triggered"] is True


def test_interrupted_resume_keeps_completed_rows_and_exports_exactly_once(tmp_path, monkeypatch):
    args = _inputs(
        tmp_path, [("a", "valid", "CCO", "CCO"), ("b", "confirmed_excluded", "CCO", "COC")]
    )
    real_call = replay._run_call
    calls = []

    def interrupt(module, version, candidate, reference, parameters):
        if calls:
            raise KeyboardInterrupt
        calls.append(version)
        return real_call(module, version, candidate, reference, parameters)

    monkeypatch.setattr(replay, "_run_call", interrupt)
    with pytest.raises(KeyboardInterrupt):
        replay.run(args)
    saved = _rows(args)
    assert [(row["case_id"], row["version"]) for row in saved] == [("a", "original")]
    partial = json.loads((args.out_dir / "summary.json").read_text())
    assert partial["not_scored_result_slots"] == 5
    assert partial["versions"]["v1"]["all_population"]["not_scored"] == 2
    resumed_calls = []

    def resume_call(module, version, candidate, reference, parameters):
        resumed_calls.append(version)
        return real_call(module, version, candidate, reference, parameters)

    monkeypatch.setattr(replay, "_run_call", resume_call)
    assert replay.run(args)["completed_result_slots"] == 6
    assert resumed_calls == ["v1", "v2", "original", "v1", "v2"]
    assert _rows(args)[0] == saved[0]
    with sqlite3.connect(args.out_dir / "checkpoint.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 6
        with pytest.raises(sqlite3.IntegrityError, match="immutable checkpoint metadata"):
            connection.execute("DELETE FROM metadata")
    exports = {
        name: (args.out_dir / name).read_bytes()
        for name in ("raw.jsonl", "raw.csv", "summary.json")
    }
    resumed_calls.clear()
    replay.run(args)
    assert resumed_calls == []
    assert all((args.out_dir / name).read_bytes() == data for name, data in exports.items())
    with (args.out_dir / "raw.csv").open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == len({(row["case_id"], row["version"]) for row in csv_rows}) == 6


@pytest.mark.parametrize(
    "changed",
    [
        "manifest",
        "index",
        "source",
        "unselected_candidate",
        "limit",
        "max_resonance",
        "use_chirality",
        "timeout",
    ],
)
def test_resume_refuses_changed_identity_or_any_candidate_bytes(tmp_path, changed):
    args = _inputs(tmp_path, [("a", "valid", "CCO", "CCO"), ("b", "valid", "CCO", "COC")], limit=1)
    replay.run(args)
    if changed == "manifest":
        document = json.loads(args.manifest.read_text())
        document["cases"][0]["reference_status"] = "confirmed_excluded"
        args.manifest.write_text(json.dumps(document))
    elif changed in ("index", "source", "unselected_candidate"):
        path = {
            "index": args.candidate_index,
            "source": args.source_v2,
            "unselected_candidate": args.candidate_root / "candidates" / "b.json",
        }[changed]
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        setattr(
            args,
            changed,
            {"limit": 2, "max_resonance": 101, "use_chirality": True, "timeout": 6.0}[changed],
        )
    with pytest.raises(ValueError, match="mismatch"):
        replay.run(args)
    assert len(_rows(args)) == 3


def test_worker_failure_and_hard_timeout_are_not_molecular_decisions(tmp_path, monkeypatch):
    args = _inputs(tmp_path, [("a", "valid", "CCO", "CCO")])
    module = replay._load_source("v2", args.source_v2, args.source_v2.read_bytes())
    reference = Chem.MolFromSmiles("CCO")
    failed = replay._run_call(module, "v2", None, reference, args)
    assert failed["status"] == "error"
    assert failed.get("decision") is None
    real_worker = replay._call_worker

    def stopped_worker(*parameters):
        os.kill(os.getpid(), signal.SIGSTOP)
        real_worker(*parameters)

    monkeypatch.setattr(replay, "_call_worker", stopped_worker)
    args.timeout = 0.05
    timed_out = replay._run_call(module, "v2", reference, reference, args)
    assert timed_out["status"] == "timeout"
    assert timed_out.get("decision") is None
    assert timed_out["error"]["type"] == "EvaluatorTimeout"


def test_rdkit_version_mismatch_refuses_before_creating_checkpoint(tmp_path):
    args = _inputs(tmp_path, [("a", "valid", "CCO", "CCO")])
    args.rdkit_version = "not-the-installed-version"
    with pytest.raises(ValueError, match="RDKit version mismatch"):
        replay.run(args)
    assert not args.out_dir.exists()


@pytest.mark.parametrize("v2000", [None, "different_graph"])
def test_lossless_v3000_is_preferred_without_smiles_fallback(tmp_path, v2000):
    args = _inputs(tmp_path, [("a", "valid", "CCO", "CCO")])
    path = args.candidate_root / "candidates" / "a.json"
    artifact = json.loads(path.read_text())
    artifact["predicted_graph"] = {
        "molblock_lossless": Chem.MolToMolBlock(Chem.MolFromSmiles("CCO"), forceV3000=True),
        "molblock_v2000": None if v2000 is None else Chem.MolToMolBlock(Chem.MolFromSmiles("COC")),
        "predicted_smiles": "COC",
    }
    path.write_text(json.dumps(artifact))
    inventory = json.loads(args.candidate_index.read_text())
    inventory["cases"][0]["sha256"] = replay._file_sha256(path)
    inventory["cases"][0]["molblock_field"] = "molblock_lossless"
    args.candidate_index.write_text(json.dumps(inventory))
    replay.run(args)
    for row in _rows(args):
        assert row["decision"] == "EQ"
        assert row["input_diagnostic"]["molblock_field"] == "molblock_lossless"


def test_missing_reference_is_input_error_not_non_equivalence(tmp_path):
    args = _inputs(tmp_path, [("missing_reference", "unresolved", "", "CCO")])
    summary = replay.run(args)
    for row in _rows(args):
        assert row["status"] == "input_error"
        assert row["decision"] is None
    for version in replay.VERSIONS:
        unresolved = summary["versions"][version]["by_reference_status"]["unresolved"]
        assert (unresolved["input_error"], unresolved["NEQ"]) == (1, 0)
