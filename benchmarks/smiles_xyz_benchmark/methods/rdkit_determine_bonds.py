"""
Author: TMJ
Date: 2026-02-25 14:30:20
LastEditors: TMJ
LastEditTime: 2026-02-25 17:31:53
Description: 请填写简介
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from benchmarks.smiles_xyz_benchmark.methods.base import (
    BenchmarkMethod,
    MethodRunOutput,
    allow_charged_fragments,
)
from benchmarks.smiles_xyz_benchmark.methods.postprocess import finalize_rdmol_with_dative_bonds


@dataclass(frozen=True)
class RDKitDetermineBondsMethod(BenchmarkMethod):
    method_id: str = "rdkit_determine_bonds"

    def run(self, case: dict[str, Any]) -> MethodRunOutput:
        timing_ms_breakdown: dict[str, float] = {}
        warnings: list[str] = []
        failed = False

        xyz_block = case.get("xyz_block")
        if not isinstance(xyz_block, str) or not xyz_block.strip():
            return MethodRunOutput(
                status="error",
                error="missing or empty xyz_block",
                timing_ms_breakdown=timing_ms_breakdown,
            )

        total_charge_value = case.get("total_charge", 0)
        try:
            total_charge = int(total_charge_value)
        except (TypeError, ValueError):
            return MethodRunOutput(
                status="error",
                error=f"invalid total_charge: {total_charge_value!r}",
                timing_ms_breakdown=timing_ms_breakdown,
            )

        xyz_to_mol_started = time.perf_counter()
        mol = Chem.MolFromXYZBlock(xyz_block)
        timing_ms_breakdown["xyz_to_mol_ms"] = (time.perf_counter() - xyz_to_mol_started) * 1000.0

        if mol is None:
            return MethodRunOutput(
                status="error",
                error="Chem.MolFromXYZBlock returned None",
                timing_ms_breakdown=timing_ms_breakdown,
            )

        determine_bonds_started = time.perf_counter()
        try:
            rdDetermineBonds.DetermineBonds(
                mol,
                charge=total_charge,
                allowChargedFragments=allow_charged_fragments(
                    total_charge, int(case.get("total_radical_electrons", 0))
                ),
            )
        except Exception as exc:  # noqa: BLE001
            # The API mutates mol before raising. Retain that graph for independent
            # equivalence checking without concealing the reconstruction error.
            failed = True
            warnings.append(f"DetermineBonds failed: {exc}")
        timing_ms_breakdown["determine_bonds_ms"] = (
            time.perf_counter() - determine_bonds_started
        ) * 1000.0

        postprocess_started = time.perf_counter()
        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"sanitize failed: {exc}")

        try:
            Chem.AssignAtomChiralTagsFromStructure(mol)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"assign_chiral_tags failed: {exc}")

        try:
            Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"assign_stereochemistry failed: {exc}")

        try:
            Chem.rdCIPLabeler.AssignCIPLabels(mol)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"assign_cip_labels failed: {exc}")

        try:
            Chem.Kekulize(mol, clearAromaticFlags=False)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"kekulize failed: {exc}")

        predicted_smiles: str | None = None
        try:
            mol, predicted_smiles = finalize_rdmol_with_dative_bonds(mol)
        except Exception as exc:  # noqa: BLE001
            failed = True
            warnings.append(f"MolToSmiles failed: {exc}")
        timing_ms_breakdown["postprocess_ms"] = (time.perf_counter() - postprocess_started) * 1000.0

        error = "; ".join(warnings) if warnings else None
        return MethodRunOutput(
            status="error" if failed else "ok",
            error=error,
            predicted_smiles=predicted_smiles,
            timing_ms_breakdown=timing_ms_breakdown,
            rdkit_mol=mol,
        )
