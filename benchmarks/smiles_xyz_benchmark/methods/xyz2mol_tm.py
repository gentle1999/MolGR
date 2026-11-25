"""Optional upstream Hückel reconstruction for transition-metal complexes."""

from __future__ import annotations

import sys
import tempfile
import time
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from rdkit import Chem

from benchmarks.smiles_xyz_benchmark.methods.base import BenchmarkMethod, MethodRunOutput


def _load_reconstructor():
    if "rdkit.Chem.MolStandardize.rdMolStandardize" not in sys.modules and (
        "molgr._core.pipeline" in sys.modules or "openbabel._openbabel" in sys.modules
    ):
        raise ImportError(
            "load the benchmark method registry before MolGR/Open Babel in a fresh process "
            "to avoid the RDKit MolStandardize import-order crash"
        )
    return import_module("xyz2mol_tm.huckel_to_smiles.xyz2mol_tmc").get_tmc_mol


@dataclass(frozen=True)
class XYZ2MolTMMethod(BenchmarkMethod):
    method_id: str = "xyz2mol_tm"

    def run(self, case: dict[str, Any]) -> MethodRunOutput:
        xyz = case.get("xyz_block")
        if not isinstance(xyz, str) or not xyz.strip():
            return MethodRunOutput(status="error", error="missing or empty xyz_block")
        try:
            charge = int(case.get("total_charge", 0))
        except (TypeError, ValueError):
            return MethodRunOutput(status="error", error="invalid total_charge")
        try:
            reconstruct = _load_reconstructor()
        except ImportError as exc:
            return MethodRunOutput(status="skipped", error=f"xyz2mol_tm unavailable: {exc}")

        timings: dict[str, float] = {}
        started = time.perf_counter()
        mol = None
        try:
            with tempfile.TemporaryDirectory(prefix="molgr-xyz2mol-tm-") as folder:
                path = Path(folder) / "input.xyz"
                path.write_text(xyz, encoding="utf-8")
                timings["write_xyz_temp_ms"] = (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                # This public API accepts charge, not multiplicity. The final
                # SMILES round trip discards coordinates and atom-index metadata.
                mol = reconstruct(str(path), charge, with_stereo=False)
                timings["xyz2mol_tm_run_ms"] = (time.perf_counter() - started) * 1000
            if mol is None:
                return MethodRunOutput(
                    status="error",
                    error="xyz2mol_tm returned no molecule",
                    timing_ms_breakdown=timings,
                )
            started = time.perf_counter()
            # Keep upstream chemistry and its dative bonds. Coordinate-based
            # postprocessing cannot be applied to this coordinate-free result.
            smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            timings["serialize_ms"] = (time.perf_counter() - started) * 1000
            return MethodRunOutput(
                status="ok",
                predicted_smiles=smiles,
                rdkit_mol=mol,
                timing_ms_breakdown=timings,
            )
        except Exception as exc:  # noqa: BLE001
            # Retain a returned graph if only serialization failed. Upstream
            # exceptions before return expose no intermediate molecule.
            return MethodRunOutput(
                status="error",
                error=f"xyz2mol_tm failed: {exc}",
                rdkit_mol=mol,
                timing_ms_breakdown=timings,
            )
