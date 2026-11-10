# pyright: reportMissingImports=false

from __future__ import annotations

import csv
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


pytest.importorskip("openbabel")
pytest.importorskip("rdkit")

from openbabel import openbabel as ob
from openbabel import pybel
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDistGeom

from molgr import _core
from molgr.config import CONFIG, MolGRConfig
from molgr.interface import xyz_to_rdmol


RDLogger.DisableLog("rdApp.*")  # type: ignore[arg-type]

_EMBED_SEED = 0xC0FFEE
_RESONANCE_CASE_INDEX = 1


def _load_csv_case(case_index: int) -> tuple[str, int, int]:
    csv_path = Path(__file__).with_name("test_cases.csv")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    row = rows[case_index - 1]
    mol = Chem.MolFromSmiles(row["smiles"].strip())
    assert mol is not None
    mol_h = Chem.AddHs(mol)
    embed_code = rdDistGeom.EmbedMolecule(mol_h, randomSeed=_EMBED_SEED)  # pyright: ignore[reportCallIssue]
    assert int(embed_code) == 0
    total_charge = sum(int(atom.GetFormalCharge()) for atom in mol_h.GetAtoms())
    total_radical_electrons = sum(int(atom.GetNumRadicalElectrons()) for atom in mol_h.GetAtoms())
    return Chem.MolToXYZBlock(mol_h), total_charge, total_radical_electrons


def _cpp_config(*, enable_uff_atom_typing_cache: bool):
    config = MolGRConfig()
    return replace(
        config,
        cpp_backend=replace(
            config.cpp_backend,
            enable_uff_atom_typing_cache=enable_uff_atom_typing_cache,
        ),
    )


def _cpp_resonance_config(*, traversal_score: str):
    config = MolGRConfig()
    return replace(
        config,
        resonance=replace(
            config.resonance,
            traversal_score=traversal_score,
        ),
    )


def test_cpp_config_bridge_requires_current_config_shape() -> None:
    incomplete_config = SimpleNamespace(
        resonance=MolGRConfig().resonance,
    )

    with pytest.raises(AttributeError, match="missing required attribute 'cpp_backend'"):
        _core.dev.pipeline.reconstruct_without_metals.debug_resonance_candidate_summaries(
            "2\nCC\nC 0 0 0\nC 0 0 1.5\n",
            0,
            0,
            config=incomplete_config,
        )


def test_cpp_resonance_traversal_score_accepts_uff_lite_gain() -> None:
    xyz_block, total_charge, total_radical_electrons = _load_csv_case(_RESONANCE_CASE_INDEX)

    uff_lite_summaries = list(
        _core.dev.pipeline.reconstruct_without_metals.debug_resonance_candidate_summaries(
            xyz_block,
            total_charge,
            total_radical_electrons,
            config=_cpp_resonance_config(traversal_score="uff_lite_gain"),
        )
    )

    assert uff_lite_summaries


def test_cpp_resonance_traversal_score_accepts_input_order() -> None:
    xyz_block, total_charge, total_radical_electrons = _load_csv_case(_RESONANCE_CASE_INDEX)

    input_order_summaries = list(
        _core.dev.pipeline.reconstruct_without_metals.debug_resonance_candidate_summaries(
            xyz_block,
            total_charge,
            total_radical_electrons,
            config=_cpp_resonance_config(traversal_score="input_order"),
        )
    )

    assert input_order_summaries


def test_cpp_default_config_enables_target_bucket_thread_parallelism() -> None:
    config = MolGRConfig()

    assert config.cpp_backend.max_threads == (1 if sys.platform == "win32" else None)
    assert config.cpp_backend.enable_target_bucket_parallelism is True
    assert config.cpp_backend.enable_candidate_scoring_parallelism is False
    assert config.cpp_backend.target_bucket_parallel_threshold == 1
    assert config.cpp_backend.target_bucket_parallel_max_threads is None


def test_cpp_xyz_seed_perception_uses_only_the_vendor_helper() -> None:
    source_roots = [Path("src/cpp"), Path("src/bindings")]
    source_files = [
        path
        for root in source_roots
        for path in root.rglob("*")
        if path.suffix in {".cpp", ".h", ".hpp"}
    ]
    direct_connect_the_dots_calls: list[str] = []
    direct_perceive_bond_order_calls: list[str] = []
    for path in source_files:
        source = path.read_text(encoding="utf-8")
        if ".ConnectTheDots(" in source or "OBMol::ConnectTheDots" in source:
            direct_connect_the_dots_calls.append(str(path))
        if ".PerceiveBondOrders(" in source or "OBMol::PerceiveBondOrders" in source:
            direct_perceive_bond_order_calls.append(str(path))

    assert direct_connect_the_dots_calls == []
    assert direct_perceive_bond_order_calls == []


def test_cpp_openbabel_global_state_surfaces_are_confined_to_vendor_adapters() -> None:
    source_files = sorted(
        path
        for root in (Path("src/cpp"), Path("src/bindings"))
        for path in root.rglob("*")
        if path.suffix in {".cpp", ".h", ".hpp"}
    )
    conversion_users = {
        path.as_posix()
        for path in source_files
        if "OpenBabel::OBConversion" in path.read_text(encoding="utf-8")
        or "#include <openbabel/obconversion.h>" in path.read_text(encoding="utf-8")
    }
    assert conversion_users == {"src/cpp/src/vendor/openbabel_conversion.cpp"}

    vendor_uff_source = Path("src/cpp/src/vendor/forcefielduff.cpp").read_text(encoding="utf-8")
    assert "OpenBabel::obErrorLog" not in vendor_uff_source
    assert "OpenBabel::obLocale" not in vendor_uff_source
    assert "SetLocale(" not in vendor_uff_source
    assert "RestoreLocale(" not in vendor_uff_source


def test_cpp_vendor_uff_atom_typing_does_not_call_openbabel_smarts() -> None:
    vendor_uff_source = Path("src/cpp/src/vendor/forcefielduff.cpp").read_text(encoding="utf-8")

    assert "openbabel/parsmart.h" not in vendor_uff_source
    assert "OBSmartsPattern" not in vendor_uff_source
    assert "molgr/utils/smarts.h" not in vendor_uff_source
    assert "molgr::smarts::FindAll" not in vendor_uff_source
    assert "AssignHyb" not in vendor_uff_source
    assert "atomtyper" not in vendor_uff_source


def _pybel_xyz_seed_signature(
    xyz_block: str,
) -> tuple[tuple[tuple[int, int, int, bool], ...], tuple[int, ...]]:
    mol = pybel.readstring("xyz", xyz_block)
    bond_signature = tuple(
        sorted(
            (
                int(bond.GetBeginAtomIdx()),
                int(bond.GetEndAtomIdx()),
                int(bond.GetBondOrder()),
                bool(bond.IsAromatic()),
            )
            for bond in ob.OBMolBondIter(mol.OBMol)
        )
    )
    hybridizations = tuple(int(atom.GetHyb()) for atom in ob.OBMolAtomIter(mol.OBMol))
    return bond_signature, hybridizations


def _cpp_xyz_seed_signature(
    xyz_block: str,
) -> tuple[tuple[tuple[int, int, int, bool], ...], tuple[int, ...]]:
    molecule_data = _core.dev.utils.debug_xyz_seed_molecule_data(xyz_block)
    bond_signature = tuple(
        sorted(
            (
                int(bond.begin_atom_idx),
                int(bond.end_atom_idx),
                int(bond.order),
                bool(bond.aromatic),
            )
            for bond in molecule_data.bonds
        )
    )
    hybridizations = tuple(int(atom.hybridization) for atom in molecule_data.atoms)
    return bond_signature, hybridizations


def test_cpp_xyz_seed_does_not_invent_electron_property_presence() -> None:
    data = _core.dev.utils.debug_xyz_seed_molecule_data("1\nH\nH 0 0 0\n")

    assert len(data.atoms) == 1
    assert data.atoms[0].electron_property_presence == 0


@pytest.mark.parametrize(
    "xyz_block",
    [
        "3\nco2\nO -1.16 0 0\nC 0 0 0\nO 1.16 0 0\n",
        "4\nacetylene\nH -1.66 0 0\nC -0.60 0 0\nC 0.60 0 0\nH 1.66 0 0\n",
        (
            "12\nbenzene\n"
            "C 1.396 0 0\nC 0.698 1.209 0\nC -0.698 1.209 0\n"
            "C -1.396 0 0\nC -0.698 -1.209 0\nC 0.698 -1.209 0\n"
            "H 2.479 0 0\nH 1.240 2.147 0\nH -1.240 2.147 0\n"
            "H -2.479 0 0\nH -1.240 -2.147 0\nH 1.240 -2.147 0\n"
        ),
    ],
)
def test_cpp_vendor_xyz_seed_perception_matches_python_fallback(xyz_block: str) -> None:
    assert _cpp_xyz_seed_signature(xyz_block) == _pybel_xyz_seed_signature(xyz_block)


def test_cpp_vendor_xyz_seed_perception_is_safe_under_parallel_calls() -> None:
    xyz_blocks = [
        "3\nco2\nO -1.16 0 0\nC 0 0 0\nO 1.16 0 0\n",
        "4\nacetylene\nH -1.66 0 0\nC -0.60 0 0\nC 0.60 0 0\nH 1.66 0 0\n",
        (
            "12\nbenzene\n"
            "C 1.396 0 0\nC 0.698 1.209 0\nC -0.698 1.209 0\n"
            "C -1.396 0 0\nC -0.698 -1.209 0\nC 0.698 -1.209 0\n"
            "H 2.479 0 0\nH 1.240 2.147 0\nH -1.240 2.147 0\n"
            "H -2.479 0 0\nH -1.240 -2.147 0\nH 1.240 -2.147 0\n"
        ),
    ]
    expected = {xyz_block: _pybel_xyz_seed_signature(xyz_block) for xyz_block in xyz_blocks}

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [
            executor.submit(_cpp_xyz_seed_signature, xyz_block)
            for _ in range(8)
            for xyz_block in xyz_blocks
        ]

    observed = [future.result() for future in futures]
    expected_sequence = [expected[xyz_block] for _ in range(8) for xyz_block in xyz_blocks]
    assert observed == expected_sequence


def test_cpp_vendor_xyz_writer_round_trips_elements_and_coordinates() -> None:
    xyz_block = """4
plugin-free round trip
Fe 1.25 -2.5 3.75
C -0.125 0.25 -0.5
Cl 10 20 30
H 0 0 0
"""

    serialized = _core.dev.utils.debug_roundtrip_xyz_block(xyz_block)
    lines = serialized.splitlines()

    assert lines[:2] == ["4", "plugin-free round trip"]
    atoms = [line.split() for line in lines[2:]]
    assert [atom[0] for atom in atoms] == ["Fe", "C", "Cl", "H"]
    assert [[float(value) for value in atom[1:]] for atom in atoms] == [
        [1.25, -2.5, 3.75],
        [-0.125, 0.25, -0.5],
        [10.0, 20.0, 30.0],
        [0.0, 0.0, 0.0],
    ]


def test_cpp_force_field_uses_vendor_uff_without_openbabel_plugin_lock() -> None:
    force_field_source = Path("src/cpp/src/utils/force_field.cpp").read_text(encoding="utf-8")

    assert "OBForceField::FindForceField" not in force_field_source
    assert "OpenBabelForceFieldMutex" not in force_field_source
    assert "std::lock_guard<std::mutex>" not in force_field_source
    assert "enable_vendor_uff_force_field" not in force_field_source
    assert "MolgrForceFieldUFF" in force_field_source


def test_cpp_openbabel_thread_local_resources_are_process_lived() -> None:
    smarts_source = Path("src/cpp/src/utils/smarts.cpp").read_text(encoding="utf-8")
    threading_source = Path("src/cpp/src/vendor/openbabel_threading.cpp").read_text(
        encoding="utf-8"
    )
    metal_preparation_source = Path("src/cpp/src/utils/metals/preparation.cpp").read_text(
        encoding="utf-8"
    )
    resonance_source = Path("src/cpp/src/utils/resonance.cpp").read_text(encoding="utf-8")
    force_field_source = Path("src/cpp/src/utils/force_field.cpp").read_text(encoding="utf-8")
    vendor_uff_source = Path("src/cpp/src/vendor/forcefielduff.cpp").read_text(encoding="utf-8")

    assert "thread_local PatternArray *compiled_patterns" in smarts_source
    assert "thread_local std::vector<std::unique_ptr<OpenBabel::OBSmartsPattern>> *patterns" in (
        threading_source
    )
    assert "OpenBabel::OBConversion" not in metal_preparation_source
    assert "WriteXyzBlock" in metal_preparation_source
    assert "OpenBabel::OBConversion" not in resonance_source
    assert "thread_local auto *force_fields = new" in force_field_source
    assert "new std::vector<MolgrCompiledUffAtomTypeRule>" not in vendor_uff_source

    allowed_non_owning_tls_pointers = {
        ("src/cpp/src/utils/perf.cpp", "t_active_run_timing_reducer"),
        ("src/cpp/src/vendor/forcefielduff.cpp", "active_instance_"),
        ("src/cpp/src/utils/force_field.cpp", "force_fields"),
        ("src/cpp/src/utils/smarts.cpp", "compiled_patterns"),
        ("src/cpp/src/vendor/openbabel_threading.cpp", "patterns"),
    }
    observed_non_owning_tls_pointers: set[tuple[str, str]] = set()
    unexpected_tls_pointer_lines: list[str] = []
    source_files = sorted(
        path
        for root in (Path("src/cpp"), Path("src/bindings"))
        for path in root.rglob("*")
        if path.suffix in {".cpp", ".h", ".hpp"}
    )
    for path in source_files:
        source_path = path.as_posix()
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "thread_local" not in line or "*" not in line or "=" not in line:
                continue
            matched_name = next(
                (
                    name
                    for allowed_path, name in allowed_non_owning_tls_pointers
                    if source_path == allowed_path and name in line
                ),
                None,
            )
            if matched_name is None:
                unexpected_tls_pointer_lines.append(f"{source_path}:{line_number}:{line.strip()}")
            else:
                observed_non_owning_tls_pointers.add((source_path, matched_name))

    assert unexpected_tls_pointer_lines == []
    assert observed_non_owning_tls_pointers == allowed_non_owning_tls_pointers


def test_cpp_private_uff_instances_do_not_register_as_openbabel_plugins() -> None:
    vendor_uff_header = Path("src/cpp/include/molgr/vendor/forcefielduff.h").read_text(
        encoding="utf-8"
    )

    assert 'MolgrForceFieldUFF() : OBForceField("", false)' in vendor_uff_header
    assert "MolgrForceFieldUFF(const char*" not in vendor_uff_header


def test_cpp_smiles_conversion_cold_start_is_serialized_across_threads() -> None:
    script = r"""
import threading
from concurrent.futures import ThreadPoolExecutor
from molgr import _core

workers = 16
barrier = threading.Barrier(workers)


def convert():
    barrier.wait()
    return _core.dev.pipeline.resonance.get_radical_resonances_smi("C=C[CH2]")


with ThreadPoolExecutor(max_workers=workers) as executor:
    results = list(executor.map(lambda _: convert(), range(workers)))

assert results == [["[C]=[C][C]"]] * workers
"""
    for _ in range(5):
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert completed.returncode == 0, completed.stderr


def test_cpp_raw_openbabel_pointer_apis_emit_deduplicated_danger_warning() -> None:
    script = r"""
from molgr import _core

data = _core.dev.utils.debug_xyz_seed_molecule_data("1\nH\nH 0 0 0\n")
ptr = _core.utils.molecule_data_to_obmol_ptr(data)
_core.free_obmol_ptr(ptr)
_core.free_obmol_ptr(0)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr.count("[UNSAFE_OPENBABEL]") == 2
    assert "molecule_data_to_obmol_ptr" in completed.stderr
    assert "free_obmol_ptr" in completed.stderr
    assert "not safe to share across threads" in completed.stderr


@pytest.mark.skipif(
    not sys.platform.startswith("linux") or not Path("/proc/self/statm").is_file(),
    reason="native RSS regression requires Linux procfs and malloc_trim",
)
def test_cpp_thread_local_resources_are_bounded_on_reused_batch_workers() -> None:
    script = r'''
import ctypes
import gc
import os
from molgr import _core

xyz = """12
benzene
C 1.396 0 0
C 0.698 1.209 0
C -0.698 1.209 0
C -1.396 0 0
C -0.698 -1.209 0
C 0.698 -1.209 0
H 2.479 0 0
H 1.240 2.147 0
H -1.240 2.147 0
H -2.479 0 0
H -1.240 -2.147 0
H 1.240 -2.147 0
"""
libc = ctypes.CDLL(None)
malloc_trim = getattr(libc, "malloc_trim", None)
if malloc_trim is None:
    raise SystemExit(77)
page_size = os.sysconf("SC_PAGE_SIZE")


def rss_kib():
    with open("/proc/self/statm", encoding="ascii") as handle:
        return int(handle.read().split()[1]) * page_size // 1024


def trim():
    gc.collect()
    malloc_trim(0)


def run_round():
    requests = [(xyz, 0, 0)] * 8
    iterator = _core.pipeline.reconstruct_with_metals.batch_xyz2omol(
        requests, max_workers=8, queue_size=1, ordered=True
    )
    results = list(iterator)
    iterator.close()
    assert all(result["molecule_data"] is not None for result in results)


for _ in range(4):
    run_round()
trim()
before = rss_kib()
for _ in range(32):
    run_round()
trim()
print(before, rss_kib())
'''
    env = os.environ.copy()
    env["MALLOC_ARENA_MAX"] = "2"
    env["PYTHONMALLOC"] = "malloc"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=90,
    )
    if completed.returncode == 77:
        pytest.skip("malloc_trim is unavailable")
    assert completed.returncode == 0, completed.stderr
    before_kib, after_kib = map(int, completed.stdout.strip().split()[-2:])

    assert after_kib - before_kib < 8 * 1024


def test_cpp_config_bridge_reads_current_global_config(monkeypatch: pytest.MonkeyPatch) -> None:
    from openbabel import pybel

    baseline_config = MolGRConfig()
    tuned_config = replace(
        baseline_config,
        organic_topology=replace(
            baseline_config.organic_topology,
            aromatic_stability_ring_size_6_factor=0.50,
            aromatic_stability_hetero_atom_penalty=0.25,
        ),
    )
    mol = pybel.readstring("smi", "n1ccccc1")
    ptr = int(mol.OBMol.this)

    monkeypatch.setattr(CONFIG, "organic_topology", tuned_config.organic_topology)

    metrics = _core.dev.utils.compute_organic_topology_metrics_ptr(ptr)

    assert metrics["aromatic_stability_score"] == pytest.approx(0.50 * 0.75)


def _clear_cpp_scoring_caches(*, clear_uff_atom_typing_cache: bool) -> None:
    _core.dev.pipeline.clear_force_field_evaluation_cache()
    if clear_uff_atom_typing_cache:
        _core.dev.pipeline.clear_uff_atom_typing_cache()


def _candidate_summaries(
    xyz_block: str,
    total_charge: int,
    total_radical_electrons: int,
    *,
    enable_uff_atom_typing_cache: bool,
) -> list[dict[str, Any]]:
    return list(
        _core.dev.pipeline.reconstruct_without_metals.debug_resonance_candidate_summaries(
            xyz_block,
            total_charge,
            total_radical_electrons,
            config=_cpp_config(enable_uff_atom_typing_cache=enable_uff_atom_typing_cache),
        )
    )


def _assert_candidate_summaries_match(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> None:
    assert len(left) == len(right)
    for left_item, right_item in zip(left, right):
        assert left_item["smiles"] == right_item["smiles"]
        assert left_item["resonance_index"] == right_item["resonance_index"]
        assert left_item["aromatic_atom_count"] == right_item["aromatic_atom_count"]
        assert (
            left_item["max_conjugated_component_size"]
            == right_item["max_conjugated_component_size"]
        )
        assert left_item["conjugated_atom_count"] == right_item["conjugated_atom_count"]
        assert left_item["conjugated_bond_count"] == right_item["conjugated_bond_count"]
        assert left_item["score"] == pytest.approx(right_item["score"], abs=1e-12)


def _reconstruction_signature(mol: Chem.Mol) -> tuple[Any, ...]:
    heavy = Chem.RemoveHs(mol)
    canonical_smiles = Chem.MolToSmiles(
        heavy,
        canonical=True,
        isomericSmiles=True,
        allBondsExplicit=True,
        allHsExplicit=True,
    )
    atom_signature = tuple(
        (
            atom.GetIdx(),
            atom.GetAtomicNum(),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
            atom.GetIsAromatic(),
        )
        for atom in mol.GetAtoms()
    )
    bond_signature = tuple(
        sorted(
            (
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                str(bond.GetBondType()),
                bond.GetIsAromatic(),
            )
            for bond in mol.GetBonds()
        )
    )
    return canonical_smiles, atom_signature, bond_signature


def test_cpp_uff_atom_typing_cache_preserves_resonance_candidate_scores() -> None:
    xyz_block, total_charge, total_radical_electrons = _load_csv_case(_RESONANCE_CASE_INDEX)

    _clear_cpp_scoring_caches(clear_uff_atom_typing_cache=True)
    uncached = _candidate_summaries(
        xyz_block,
        total_charge,
        total_radical_electrons,
        enable_uff_atom_typing_cache=False,
    )
    assert uncached
    assert _core.dev.pipeline.get_uff_atom_typing_cache_info()["size"] == 0

    _clear_cpp_scoring_caches(clear_uff_atom_typing_cache=True)
    enabled_cold = _candidate_summaries(
        xyz_block,
        total_charge,
        total_radical_electrons,
        enable_uff_atom_typing_cache=True,
    )
    cold_cache_info = _core.dev.pipeline.get_uff_atom_typing_cache_info()
    assert cold_cache_info["size"] > 0
    assert cold_cache_info["misses"] > 0

    _clear_cpp_scoring_caches(clear_uff_atom_typing_cache=False)
    enabled_warm = _candidate_summaries(
        xyz_block,
        total_charge,
        total_radical_electrons,
        enable_uff_atom_typing_cache=True,
    )
    warm_cache_info = _core.dev.pipeline.get_uff_atom_typing_cache_info()
    assert warm_cache_info["hits"] > cold_cache_info["hits"]

    _assert_candidate_summaries_match(uncached, enabled_cold)
    _assert_candidate_summaries_match(uncached, enabled_warm)


def test_cpp_uff_atom_typing_cache_preserves_public_cpp_reconstruction_result() -> None:
    xyz_block, total_charge, total_radical_electrons = _load_csv_case(_RESONANCE_CASE_INDEX)
    spin_multiplicity = total_radical_electrons + 1

    _clear_cpp_scoring_caches(clear_uff_atom_typing_cache=True)
    uncached = xyz_to_rdmol(
        xyz_block,
        total_charge,
        spin_multiplicity,
        backend="cpp",
        config=_cpp_config(enable_uff_atom_typing_cache=False),
    )

    _clear_cpp_scoring_caches(clear_uff_atom_typing_cache=True)
    cached = xyz_to_rdmol(
        xyz_block,
        total_charge,
        spin_multiplicity,
        backend="cpp",
        config=_cpp_config(enable_uff_atom_typing_cache=True),
    )

    assert _reconstruction_signature(uncached) == _reconstruction_signature(cached)
