"""Offline coverage for the published standard, without downloading the full release."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import zlib

import pytest
from rdkit import Chem

from benchmarks.dataset_xyz_benchmark import tmqmg as corrected


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _pin_fixture(root, manifest, monkeypatch):
    data = json.dumps(manifest).encode()
    (root / "manifest.json").write_bytes(data)
    monkeypatch.setattr(corrected, "MANIFEST_SHA256", _hash(data))


@pytest.fixture
def release(tmp_path, monkeypatch):
    """Mirror layout v4: nested, uncompressed parts and relative-path hash keys."""
    original = b"1\n\nH 0 0 0\n"
    mol = Chem.MolFromSmiles("[H]", sanitize=False)
    mol.GetAtomWithIdx(0).SetNumRadicalElectrons(1)
    graph = mol.ToBinary()
    props = {
        "CASE_ID": "B",
        "INPUT_CHARGE": "0",
        "INPUT_MULTIPLICITY": "2",
        "XYZ_SHA256": _hash(original),
        "NATIVE_GRAPH_ZLIB_BASE64": base64.b64encode(zlib.compress(graph)).decode(),
        "GRAPH_SHA256": _hash(graph),
    }
    sdf = b"B\n" + "".join(f"\n>  <{k}>\n{v}\n\n" for k, v in props.items()).encode() + b"$$$$\n"
    xyz = b"".join(f"1\nid={case} charge=0 multiplicity=2\nH 0 0 0\n".encode() for case in "ABC")
    part = {"sdf": "data/part.sdf", "xyz": "evidence/inputs/part.xyz", "inputs": 3, "targets": 1}
    hashes = {}
    for key, data in (("sdf", sdf), ("xyz", xyz)):
        path = tmp_path / part[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        hashes[part[key]] = _hash(data)
    manifest = {
        "dataset": "tmqmg-corrected",
        "version": corrected.VERSION,
        "layout_version": 4,
        "parts": [part],
        "sha256": hashes,
        "inputs": 3,
        "targets": 1,
        "exclusions": {"Synthetic exclusion": 2},
        "benchmark": {
            "max_resonance": 100000,
            "use_chirality": False,
            "ignore_metal_coordination_bonds": True,
            "comparison_timeout_seconds": 60,
        },
    }
    _pin_fixture(tmp_path, manifest, monkeypatch)
    return tmp_path, manifest


def test_published_layout_exclusions_and_input_state(release):
    root, _ = release
    rows = list(corrected.rows(root))
    assert [row["id"] for row in rows] == ["B"]
    from benchmarks.dataset_xyz_benchmark.run import reconstruction_input

    row = rows[0]
    case = reconstruction_input(row["xyz"], int(row["charge"]), int(row["multiplicity"]))
    assert case == {"xyz_block": "1\n\nH 0 0 0\n", "total_charge": 0, "total_radical_electrons": 1}
    graph = Chem.Mol(zlib.decompress(base64.b64decode(row["native_graph"])))
    assert graph.GetNumAtoms() == 1
    assert graph.GetAtomWithIdx(0).GetNumRadicalElectrons() == 1
    assert corrected.provenance(root)["revision"] == corrected.REVISION


def test_missing_release_is_actionable(tmp_path):
    from benchmarks.dataset_xyz_benchmark.prepare import prepare

    with pytest.raises(FileNotFoundError, match=r"benchmarks/dataset_xyz_benchmark/README\.md"):
        prepare(tmp_path / "missing", tmp_path / "cases.sqlite", "tmqmg-corrected")
    assert not (tmp_path / "cases.sqlite").exists()
    with pytest.raises(FileNotFoundError, match=r"benchmarks/dataset_xyz_benchmark/README\.md"):
        list(corrected.rows(tmp_path))


def test_old_bundled_manifest_is_rejected(tmp_path):
    (tmp_path / "manifest.json").write_text('{"version": "2026-11-19", "targets": 72578}')
    with pytest.raises(ValueError, match="Not the pinned"):
        list(corrected.rows(tmp_path))


@pytest.mark.parametrize("key", ["sdf", "xyz"])
def test_part_checksum_is_verified_before_yielding(release, key):
    root, manifest = release
    with (root / manifest["parts"][0][key]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Incorrect corrected dataset file"):
        next(corrected.rows(root))


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (b"INPUT_CHARGE>\n0", b"INPUT_CHARGE>\n1", "Incorrect XYZ"),
        (b"$$$$\n", b"", "Truncated corrected SDF"),
        (b"GRAPH_SHA256>\n", b"GRAPH_SHA256>\nwrong", "Incorrect native graph"),
    ],
)
def test_record_validation(release, monkeypatch, old, new, message):
    root, manifest = release
    name = manifest["parts"][0]["sdf"]
    path = root / name
    path.write_bytes(path.read_bytes().replace(old, new))
    manifest["sha256"][name] = _hash(path.read_bytes())
    _pin_fixture(root, manifest, monkeypatch)
    with pytest.raises(ValueError, match=message):
        list(corrected.rows(root))


def test_target_counts_checked(release, monkeypatch):
    root, manifest = release
    manifest["parts"][0]["targets"] = 2
    _pin_fixture(root, manifest, monkeypatch)
    with pytest.raises(ValueError, match="Incomplete corrected dataset part"):
        list(corrected.rows(root))


def test_corrected_release_input_and_single_target():
    from benchmarks.dataset_xyz_benchmark.run import reconstruction_input
    from scripts.tmqmg_fixture_data import fixture

    released = fixture("ABEVAG")
    assert released["benchmark"]
    # ABEVAG's original label fails conservation. Its corrected graph is eligible.
    graph = Chem.Mol(zlib.decompress(base64.b64decode(released["native_graph"])))
    assert graph.GetNumAtoms() == int(released["xyz"].splitlines()[0])
    assert set(
        reconstruction_input(released["xyz"], released["charge"], released["multiplicity"])
    ) == {
        "xyz_block",
        "total_charge",
        "total_radical_electrons",
    }


def test_published_structure_records_include_standalone_metadata():
    from pathlib import Path

    from scripts.tmqmg_fixture_data import fixture

    record = fixture("ABEVAG")
    path = Path(__file__).parent / "data/reviewed/tmqmg/reference_graph/ABEVAG.sdf"
    mol = next(iter(Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)))
    assert mol.GetNumAtoms() == int(record["xyz"].splitlines()[0])
    assert mol.GetProp("_Name") == mol.GetProp("CASE_ID") == record["id"]
    assert mol.GetProp("EVIDENCE_TIER") == str(record["tier"])
    assert mol.GetProp("SELECTION_REASON")
    assert json.loads(mol.GetProp("SOURCE_REFERENCES"))
    assert mol.GetProp("WBO_SUMMARY")
    energy = json.loads(mol.GetProp("GXTB_SUMMARY"))
    assert isinstance(energy["total_energy_Eh"], float)
    assert energy["charge"] == record["charge"]
    assert energy["uhf"] == record["multiplicity"] - 1
    assert mol.GetProp("XYZ_SHA256") == record["xyz_sha256"]
    assert mol.GetProp("GRAPH_SHA256") == record["graph_sha256"]


def test_unified_prepare_preserves_exact_native_graph_and_exclusion_coverage(release):
    import sqlite3

    from benchmarks.dataset_xyz_benchmark.prepare import prepare

    root, manifest = release
    store = root / "cases.sqlite"
    source = prepare(root, store, "tmqmg-corrected")
    assert source["cases"] == source["molecules"] == 3
    assert source["reference_statuses"] == {"valid": 1, "excluded": 2}
    assert source["release"]["inputs"] == 3
    assert source["release"]["targets"] == 1
    assert source["release"]["exclusions"] == manifest["exclusions"]
    assert source["comparison_policy"] == manifest["benchmark"]
    assert source["comparison_policy_locked"] is True
    assert source["complete"] and not source["pilot"]
    published = next(corrected.rows(root))
    original = zlib.decompress(base64.b64decode(published["native_graph"]))
    with sqlite3.connect(store) as db:
        native, digest, xyz, charge, mult, smiles = db.execute(
            "SELECT reference_native,reference_sha256,xyz,charge,multiplicity,reference_smiles "
            "FROM cases WHERE molecule='B'"
        ).fetchone()
        assert native == original and digest == _hash(original)
        assert zlib.decompress(xyz).decode() == published["xyz"]
        assert (charge, mult, smiles) == (0, 2, None)
        assert (
            db.execute("SELECT count(*) FROM cases WHERE reference_native IS NULL").fetchone()[0]
            == 2
        )


def test_release_spin_state_is_not_inferred_from_formal_radicals(release, monkeypatch):
    from benchmarks.dataset_xyz_benchmark.prepare import tmqmg_cases

    root, _ = release
    row = next(corrected.rows(root))
    # Authoritative coupled-spin inputs need not equal formal radical count + 1.
    row["multiplicity"] = "1"
    monkeypatch.setattr(corrected, "rows", lambda *a, **kw: iter([row]))
    case = next(tmqmg_cases(root))
    assert case[6:8] == (1, "valid")
    assert Chem.Mol(case[10]).GetAtomWithIdx(0).GetNumRadicalElectrons() == 1


def test_unified_prepare_pilot_and_invalid_release(release, monkeypatch):
    import sqlite3

    from benchmarks.dataset_xyz_benchmark.prepare import prepare

    root, manifest = release
    pilot = prepare(root, root / "pilot.sqlite", "tmqmg-corrected", limit_molecules=2)
    assert pilot["pilot"] and pilot["complete"]
    assert pilot["cases"] == 2
    assert pilot["reference_statuses"] == {"valid": 1, "excluded": 1}
    assert pilot["release"]["inputs"] == 3  # Release total is not the pilot denominator.
    manifest["parts"][0]["targets"] = 2
    _pin_fixture(root, manifest, monkeypatch)
    store = root / "bad.sqlite"
    with pytest.raises(ValueError, match="Incomplete corrected dataset part"):
        prepare(root, store, "tmqmg-corrected")
    with sqlite3.connect(store) as db:
        source = json.loads(db.execute("SELECT value FROM metadata").fetchone()[0])
        assert not source["complete"]


def test_unified_cli_prepares_registered_release_and_rejects_old_csv(release, monkeypatch, capsys):
    import sys

    from benchmarks.dataset_xyz_benchmark.prepare import main

    root, _ = release
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            "--dataset",
            "tmqmg-corrected",
            "--source",
            str(root),
            "--out",
            str(root / "cli.sqlite"),
        ],
    )
    main()
    assert json.loads(capsys.readouterr().out)["dataset"] == "tmqmg-corrected"
    monkeypatch.setattr(sys, "argv", ["prepare", "--csv", "old.csv"])
    with pytest.raises(SystemExit):
        main()


def test_unified_worker_uses_only_input_state_and_compares_native_error_graph(release, monkeypatch):
    from types import SimpleNamespace

    from benchmarks.dataset_xyz_benchmark import run as runner
    from benchmarks.dataset_xyz_benchmark.prepare import prepare
    from benchmarks.smiles_xyz_benchmark.methods.base import MethodRunOutput
    from molgr import CONFIG
    from molgr.utils import equivalence

    root, manifest = release
    store = root / "worker.sqlite"
    prepare(root, store, "tmqmg-corrected")
    row = next(corrected.rows(root))
    target = Chem.Mol(zlib.decompress(base64.b64decode(row["native_graph"])))
    inputs, comparisons, messages = [], [], []

    def reconstruct(case):
        inputs.append(dict(case))
        case["adapter_private_state"] = "must not leak out of warmup"
        return MethodRunOutput(
            status="error", error="display failed", predicted_smiles="wrong", rdkit_mol=target
        )

    def compare(left, right, **policy):
        assert left.ToBinary() == target.ToBinary()
        assert right is target
        comparisons.append(policy)
        return SimpleNamespace(decision=SimpleNamespace(value="equivalent"), reason="native graph")

    monkeypatch.setattr(
        runner,
        "get_method_registry",
        lambda ids: [SimpleNamespace(method_id="molgr_cpp", run=reconstruct)],
    )
    monkeypatch.setattr(equivalence, "evaluate_equivalence", compare)
    monkeypatch.setattr(runner.os, "sched_setaffinity", lambda *args: None, raising=False)
    monkeypatch.setattr(runner.os, "dup2", lambda *args: None)
    monkeypatch.setattr(CONFIG.cpp_backend, "max_threads", CONFIG.cpp_backend.max_threads)
    monkeypatch.setattr(
        CONFIG.cpp_backend,
        "target_bucket_parallel_max_threads",
        CONFIG.cpp_backend.target_bucket_parallel_max_threads,
    )
    work = iter([2, None])
    conn = SimpleNamespace(recv=lambda: next(work), send=messages.append)
    runner.worker(conn, 0, str(store), "molgr_cpp", root / "worker.log", manifest["benchmark"])
    assert len(inputs) == 2  # Warmup and timed call, both reference-blind.
    assert (
        inputs[0]
        == inputs[1]
        == {"xyz_block": "1\n\nH 0 0 0\n", "total_charge": 0, "total_radical_electrons": 1}
    )
    assert comparisons == [
        {k: v for k, v in manifest["benchmark"].items() if k != "comparison_timeout_seconds"}
    ]
    record = next(m[1] for m in messages if m[0] == "result")
    assert record["status"] == "error" and record["decision"] == "equivalent"
    assert record["error"] == "display failed"


@pytest.mark.skipif(
    not hasattr(os, "sched_setaffinity"), reason="Latency measurements require Linux CPU affinity"
)
def test_unified_run_report_resume_and_pinned_policy(release):
    import sqlite3

    from benchmarks.dataset_xyz_benchmark.prepare import prepare
    from benchmarks.dataset_xyz_benchmark.report import summarize
    from benchmarks.dataset_xyz_benchmark.run import run

    root, manifest = release
    store = root / "run.sqlite"
    prepare(root, store, "tmqmg-corrected")
    out = root / "results"
    method = "rdkit_determine_bonds"
    with pytest.raises(ValueError, match="Source pins"):
        run(store, out, method, 1, comparison_timeout=5)
    run(store, out, method, 1)
    result = out / f"{method}.sqlite"
    report = summarize(store, result)
    assert report["complete"] and not report["pilot"]
    assert report["eligible_conformers"] == report["attempted_eligible_conformers"] == 1
    assert report["reference_statuses"] == {"excluded": 2, "valid": 1}
    assert report["provenance"]["source"]["release"]["exclusions"] == manifest["exclusions"]
    assert report["provenance"]["comparison_policy"] == manifest["benchmark"]
    with sqlite3.connect(result) as db:
        case_id, status, decision = db.execute(
            "SELECT case_id,status,decision FROM results"
        ).fetchone()
        assert case_id == 2 and status == "ok" and decision == "not_equivalent"
    before = result.read_bytes()
    run(store, out, method, 1)  # Nothing is reconstructed again.
    assert result.read_bytes() == before
    with pytest.raises(ValueError, match="provenance differs"):
        run(store, out, method, 1, timeout=31)
    with sqlite3.connect(store) as db:
        db.execute("UPDATE cases SET multiplicity=1 WHERE molecule='B'")
    with pytest.raises(ValueError, match="different case store"):
        summarize(store, result)
    with pytest.raises(ValueError, match="provenance differs"):
        run(store, out, method, 1)


def _interrupted_worker(conn, cpu, source, method_id, log, policy):
    """Real spawned test worker; exercise parent watchdogs without chemical work."""
    import os
    import time

    conn.send(("ready", None))
    case_id = conn.recv()
    if case_id is None:
        return
    # Source policy is the only comparison timeout input to the worker.
    assert policy["comparison_timeout_seconds"] == 0.05
    if method_id == "molgr_cpp":
        conn.send(("comparison", {"status": "error", "method_ms": 12.5, "decision": None}))
        time.sleep(10)
    elif method_id == "molgr_fallback":
        conn.send(("method", None))
        time.sleep(10)
    else:
        conn.send(("method", None))
        os._exit(1)


@pytest.mark.parametrize(
    ("method", "status", "decision", "phase"),
    [
        ("molgr_cpp", "error", "inconclusive", "comparison"),
        ("molgr_fallback", "timeout", None, "method"),
        ("rdkit_determine_bonds", "crash", None, "method"),
    ],
)
def test_unified_watchdogs_preserve_failure_denominator(
    release, monkeypatch, method, status, decision, phase, mock_affinity
):
    import sqlite3

    from benchmarks.dataset_xyz_benchmark import run as runner
    from benchmarks.dataset_xyz_benchmark.prepare import prepare
    from benchmarks.dataset_xyz_benchmark.report import summarize

    root, manifest = release
    manifest["benchmark"]["comparison_timeout_seconds"] = 0.05
    _pin_fixture(root, manifest, monkeypatch)
    store = root / "watchdog.sqlite"
    prepare(root, store, "tmqmg-corrected")
    monkeypatch.setattr(runner, "worker", _interrupted_worker)
    out = root / "results"
    runner.run(store, out, method, 1, timeout=2)
    with sqlite3.connect(out / f"{method}.sqlite") as db:
        record = json.loads(zlib.decompress(db.execute("SELECT detail FROM results").fetchone()[0]))
    assert record["status"] == status
    assert record["decision"] == decision
    assert record["interrupted_phase"] == phase
    if phase == "comparison":
        assert record["method_ms"] == 12.5
    report = summarize(store, out / f"{method}.sqlite")
    assert report["complete"] and report["eligible_conformers"] == 1
    assert report["matched_conformers"] == 0
    assert report["molecule_macro_match_rate"] == 0


@pytest.fixture
def mock_affinity(monkeypatch):
    """Exercise worker supervision and cache validation without measuring pinned latency."""
    monkeypatch.setattr(os, "sched_setaffinity", lambda *args: None, raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0}, raising=False)


def test_unified_run_rejects_incomplete_obsolete_and_unattributed_stores(release, mock_affinity):
    import sqlite3

    from benchmarks.dataset_xyz_benchmark.prepare import prepare
    from benchmarks.dataset_xyz_benchmark.run import run

    root, _ = release
    store = root / "invalid.sqlite"
    source = prepare(root, store, "tmqmg-corrected")
    out = root / "results"
    for change, message in [
        ({"schema": 1}, "schema is obsolete"),
        ({"complete": False}, "interrupted"),
    ]:
        with sqlite3.connect(store) as db:
            db.execute("UPDATE metadata SET value=?", (json.dumps({**source, **change}),))
        with pytest.raises(ValueError, match=message):
            run(store, out, "molgr_cpp", 1)
    with sqlite3.connect(store) as db:
        db.execute("UPDATE metadata SET value=?", (json.dumps(source),))
    (out / "molgr_cpp.sqlite").touch()
    with pytest.raises(ValueError, match="Cache provenance is missing"):
        run(store, out, "molgr_cpp", 1)
