from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any

from rdkit import Chem

from benchmarks.smiles_xyz_benchmark.methods.base import BenchmarkMethod, MethodRunOutput
from benchmarks.smiles_xyz_benchmark.methods.postprocess import finalize_rdmol_with_dative_bonds


pt = Chem.GetPeriodicTable()


def _parse_xyz_block(xyz_block: str) -> list[tuple[str, tuple[float, float, float]]]:
    lines = xyz_block.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    if len(lines) < 2:
        raise ValueError("invalid XYZ: too few lines")

    try:
        n_atoms = int(lines[0])
    except ValueError as exc:
        raise ValueError("invalid XYZ: first line must be atom count") from exc

    if n_atoms <= 0:
        raise ValueError("invalid XYZ: atom count must be > 0")

    body_lines = lines[1:]

    def _looks_like_atom_line(line: str) -> bool:
        parts = line.split()
        if len(parts) < 4:
            return False
        try:
            float(parts[1])
            float(parts[2])
            float(parts[3])
        except ValueError:
            return False
        return True

    def _parse_atom_lines(atom_lines: list[str]) -> list[tuple[str, tuple[float, float, float]]]:
        atoms: list[tuple[str, tuple[float, float, float]]] = []
        for i, line in enumerate(atom_lines):
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"invalid XYZ: atom line {i} has <4 columns")
            sym = parts[0]
            try:
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
            except ValueError as exc:
                raise ValueError(f"invalid XYZ: atom line {i} has invalid coordinates") from exc
            atoms.append((sym, (x, y, z)))
        return atoms

    if not body_lines:
        raise ValueError("invalid XYZ: not enough atom lines")

    first_body = body_lines[0]
    parse_modes: list[str]
    if not first_body.strip():
        parse_modes = ["comment"]
    elif _looks_like_atom_line(first_body):
        parse_modes = ["no_comment", "comment"]
    else:
        parse_modes = ["comment", "no_comment"]

    parse_errors: list[ValueError] = []
    for mode in parse_modes:
        atom_lines = body_lines[1 : 1 + n_atoms] if mode == "comment" else body_lines[:n_atoms]

        if len(atom_lines) < n_atoms:
            parse_errors.append(ValueError("invalid XYZ: not enough atom lines"))
            continue

        try:
            return _parse_atom_lines(atom_lines)
        except ValueError as exc:
            parse_errors.append(exc)

    if parse_errors:
        raise parse_errors[0]
    raise ValueError("invalid XYZ: not enough atom lines")


def _graph_to_rdkit_mol(
    graph: Any,
) -> Chem.Mol:
    rw = Chem.RWMol()
    for _, node in graph.nodes.data():
        atom = Chem.Atom(node["symbol"])
        # All nuclei, including H, are already present in the source XYZ.
        atom.SetNoImplicit(True)
        rw.AddAtom(atom)

    for i, j, data in graph.edges(data=True):
        bo = float(data.get("bond_order", 1.0))
        if bo >= 2.5:
            bt = Chem.BondType.TRIPLE
        elif bo >= 1.75:
            bt = Chem.BondType.DOUBLE
        elif 1.4 < bo < 1.6:
            bt = Chem.BondType.AROMATIC
        else:
            bt = Chem.BondType.SINGLE
        rw.AddBond(int(i), int(j), bt)

    conf = Chem.Conformer(len(graph.nodes))
    for idx, node in graph.nodes.data():
        conf.SetAtomPosition(idx, node.get("position", (0.0, 0.0, 0.0)))
        rw.GetAtomWithIdx(idx).SetFormalCharge(
            int(node.get("formal_charge", 0)) or int(node.get("oxidation_state", 0))
        )

    mol = rw.GetMol()
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    return mol


@dataclass(frozen=True)
class XYZGraphCheminfFullMethod(BenchmarkMethod):
    method_id: str = "xyzgraph_cheminf_full"

    def run(self, case: dict[str, Any]) -> MethodRunOutput:
        timing_ms_breakdown: dict[str, float] = {}

        if sys.version_info < (3, 9):
            return MethodRunOutput(
                status="skipped",
                error="xyzgraph requires Python>=3.9",
                timing_ms_breakdown=timing_ms_breakdown,
            )

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
        total_radical_electrons_value = case.get("total_radical_electrons", 0)
        try:
            total_radical_electrons = int(total_radical_electrons_value)
        except (TypeError, ValueError):
            return MethodRunOutput(
                status="error",
                error=f"invalid total_radical_electrons: {total_radical_electrons_value!r}",
                timing_ms_breakdown=timing_ms_breakdown,
            )
        xyz_parse_started = time.perf_counter()
        try:
            atoms = _parse_xyz_block(xyz_block)
        except Exception as exc:  # noqa: BLE001
            timing_ms_breakdown["xyz_parse_ms"] = (time.perf_counter() - xyz_parse_started) * 1000.0
            return MethodRunOutput(
                status="error",
                error=f"invalid xyz_block: {exc}",
                timing_ms_breakdown=timing_ms_breakdown,
            )
        timing_ms_breakdown["xyz_parse_ms"] = (time.perf_counter() - xyz_parse_started) * 1000.0

        build_started = time.perf_counter()
        try:
            from xyzgraph.graph_builders import build_graph  # pyright: ignore[reportMissingImports]

            graph = build_graph(
                atoms=atoms,
                charge=total_charge,
                multiplicity=total_radical_electrons + 1,
                method="cheminf",
                quick=False,
            )
        except Exception as exc:  # noqa: BLE001
            timing_ms_breakdown["xyzgraph_build_graph_ms"] = (
                time.perf_counter() - build_started
            ) * 1000.0
            return MethodRunOutput(
                status="error",
                error=f"xyzgraph build_graph failed: {exc}",
                timing_ms_breakdown=timing_ms_breakdown,
            )
        timing_ms_breakdown["xyzgraph_build_graph_ms"] = (
            time.perf_counter() - build_started
        ) * 1000.0

        to_rdkit_started = time.perf_counter()
        try:
            mol = _graph_to_rdkit_mol(graph=graph)
        except Exception as exc:  # noqa: BLE001
            timing_ms_breakdown["graph_to_rdkit_ms"] = (
                time.perf_counter() - to_rdkit_started
            ) * 1000.0
            return MethodRunOutput(
                status="error",
                error=f"xyzgraph graph->rdkit failed: {exc}",
                timing_ms_breakdown=timing_ms_breakdown,
            )
        timing_ms_breakdown["graph_to_rdkit_ms"] = (time.perf_counter() - to_rdkit_started) * 1000.0

        warnings: list[str] = []
        postprocess_started = time.perf_counter()
        try:
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

            mol, predicted_smiles = finalize_rdmol_with_dative_bonds(mol)
        except Exception as exc:  # noqa: BLE001
            timing_ms_breakdown["postprocess_ms"] = (
                time.perf_counter() - postprocess_started
            ) * 1000.0
            return MethodRunOutput(
                status="error",
                error=f"postprocess failed: {exc}",
                rdkit_mol=mol,
                timing_ms_breakdown=timing_ms_breakdown,
            )
        timing_ms_breakdown["postprocess_ms"] = (time.perf_counter() - postprocess_started) * 1000.0

        error = "; ".join(warnings) if warnings else None
        return MethodRunOutput(
            status="ok",
            error=error,
            predicted_smiles=predicted_smiles,
            rdkit_mol=mol,
            timing_ms_breakdown=timing_ms_breakdown,
        )
