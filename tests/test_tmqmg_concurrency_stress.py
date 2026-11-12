from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import benchmarks.tmqmg_concurrency_stress as stress_benchmark
from benchmarks.tmqmg_concurrency_stress import available_cpu_count, stress_worker_counts


pytest.importorskip("openbabel")
pytest.importorskip("rdkit")


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_BENCHMARK = _PROJECT_ROOT / "benchmarks" / "tmqmg_concurrency_stress.py"


def _find_tmqmg_dataset() -> tuple[Path, Path, bool]:
    environment_csv = os.environ.get("MOLGR_TMQMG_CSV")
    environment_xyz_dir = os.environ.get("MOLGR_TMQMG_XYZ_DIR")
    if environment_csv and environment_xyz_dir:
        csv_path = Path(environment_csv)
        xyz_dir = Path(environment_xyz_dir)
        if csv_path.is_file() and xyz_dir.is_dir():
            return csv_path, xyz_dir, False

    candidates = [
        (
            _PROJECT_ROOT
            / ".local"
            / "datasets"
            / "tmqmg"
            / "e1dc9887b8f20a217a1db6ca972d726bcbaab45b"
            / "tmQMg_properties_and_targets.csv",
            _PROJECT_ROOT
            / ".local"
            / "datasets"
            / "tmqmg"
            / "e1dc9887b8f20a217a1db6ca972d726bcbaab45b"
            / "tmQMg_xyz"
            / "xyz",
        ),
        (
            Path("/mnt/e/download/tmQMg_properties_and_targets.csv"),
            Path("/mnt/e/download/tmQMg_xyz/xyz"),
        ),
    ]
    for csv_path, xyz_dir in candidates:
        if csv_path.is_file() and xyz_dir.is_dir():
            return csv_path, xyz_dir, False
    fixture_root = _PROJECT_ROOT / "tests" / "data"
    return fixture_root / "xyz_stress_manifest.csv", fixture_root, True


_TMQMG_CSV, _TMQMG_XYZ_DIR, _REPEAT_INPUTS = _find_tmqmg_dataset()
_STRESS_WORKER_COUNTS = stress_worker_counts()


def test_stress_worker_counts_follow_available_cpu_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stress_benchmark, "available_cpu_count", lambda: 3)
    assert stress_benchmark.stress_worker_counts() == (1, 2, 3)
    assert stress_benchmark.stress_worker_counts(max_workers=2) == (1, 2)


def test_repository_xyz_stress_manifest_covers_every_xyz_fixture() -> None:
    fixture_root = _PROJECT_ROOT / "tests" / "data"
    manifest_path = fixture_root / "xyz_stress_manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        listed = {row["xyz_file"] for row in csv.DictReader(stream)}
    discovered = {
        str(path.relative_to(fixture_root).as_posix()) for path in fixture_root.rglob("*.xyz")
    }

    assert listed == discovered


def test_repository_xyz_manifest_identifies_reference_sources() -> None:
    fixture_root = _PROJECT_ROOT / "tests" / "data"
    with (fixture_root / "xyz_stress_manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len({row["case_id"] for row in rows}) == len(rows)
    for row in rows:
        assert row["reference_status"] in {
            "reviewed",
            "dataset",
            "test_reference",
            "representation_issue",
            "source_issue",
            "missing",
            "rejected",
        }
        if row["reference_status"] == "missing":
            assert not row["reference_file"] and not row["reference_source"]
        else:
            source, anchor = row["reference_source"].split("#", 1)
            assert (_PROJECT_ROOT / source).is_file(), row["case_id"]
            assert anchor in (_PROJECT_ROOT / source).read_text(encoding="utf-8"), row["case_id"]
        if row["reference_file"]:
            assert (fixture_root / row["reference_file"]).is_file(), row["case_id"]


@pytest.mark.stress
@pytest.mark.parametrize("mode", ["native_batch", "spawn_xyz_to_rdmol"])
@pytest.mark.parametrize("worker_count", _STRESS_WORKER_COUNTS)
def test_native_batch_and_spawn_process_stress_on_1000_tmqmg_inputs(
    mode: str,
    worker_count: int,
) -> None:
    command = [
        sys.executable,
        str(_BENCHMARK),
        "--mode",
        mode,
        "--csv",
        str(_TMQMG_CSV),
        "--xyz-dir",
        str(_TMQMG_XYZ_DIR),
        "--items",
        "1000",
        "--workers",
        str(worker_count),
        "--timeout-seconds",
        "300",
    ]
    if _REPEAT_INPUTS:
        command.append("--repeat-inputs")
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=360,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout.strip().splitlines()[-1])
    expected_start_method = "native_threads" if mode == "native_batch" else "spawn"
    assert summary["mode"] == mode
    assert summary["start_method"] == expected_start_method
    assert summary["available_cpu_count"] == available_cpu_count()
    assert summary["worker_count"] == worker_count
    expected_process_count = 1 if mode == "native_batch" else worker_count
    expected_native_workers = worker_count if mode == "native_batch" else 0
    assert summary["process_count"] == expected_process_count
    assert summary["native_batch_worker_count"] == expected_native_workers
    assert summary["requested"] == 1000
    assert summary["repeat_inputs"] is _REPEAT_INPUTS
    assert summary["reported"] == 1000
    assert summary["success"] + summary["failures"] == 1000
    assert summary["unreported"] == 0
    assert not summary["timed_out"]
    assert not summary["crashed_pids"]
    assert not summary["terminated_pids"]
    assert not summary["nonzero_exit_pids"]
    expected_worker_records = 1 if mode == "native_batch" else worker_count
    assert len(summary["workers"]) == expected_worker_records
    assert len({worker["pid"] for worker in summary["workers"]}) == expected_worker_records
    assert summary["items_per_second"] > 0
    for worker in summary["workers"]:
        if mode == "native_batch":
            if worker_count == 1:
                expected_max_threads = 1 if sys.platform == "win32" else None
                assert worker["config"]["max_threads"] == expected_max_threads
                assert worker["config"]["enable_target_bucket_parallelism"]
                assert worker["config"]["target_bucket_parallel_max_threads"] is None
                assert not worker["config"]["enable_candidate_scoring_parallelism"]
            else:
                assert worker["config"] == {
                    "max_threads": 1,
                    "enable_target_bucket_parallelism": False,
                    "target_bucket_parallel_max_threads": 1,
                    "enable_candidate_scoring_parallelism": False,
                }
        else:
            expected_max_threads = 1 if sys.platform == "win32" else None
            assert worker["config"]["max_threads"] == expected_max_threads
            assert worker["config"]["enable_target_bucket_parallelism"]
            assert worker["config"]["target_bucket_parallel_max_threads"] is None
            assert not worker["config"]["enable_candidate_scoring_parallelism"]
            if sys.platform.startswith("linux"):
                assert worker["peak_native_thread_count"] > 1
