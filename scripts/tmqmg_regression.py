#!/usr/bin/env python3
# pyright: reportCallIssue=false
"""Run tmQMg regression and report organic-only UFF diagnostics.

The tmQMg CSV provides one reference SMILES per entry, while the XYZ atom order
does not match the SMILES atom order. To compare fixed-coordinate organic UFF
energies, this script:

1. reconstructs a MolGR graph from the XYZ geometry,
2. strips metals from both the MolGR result and the reference graph,
3. maps the reference organic heavy-atom skeleton onto the MolGR organic
   skeleton while ignoring bond-order / charge differences, and
4. reuses the MolGR organic coordinates to evaluate UFF on both topologies.

The output CSV is case-wise and the summary JSON aggregates status counts.
Manually reviewed reference issues can be recorded in a whitelist JSON so they
count separately from direct graph-equivalence matches.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from contextlib import suppress
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple, cast

from typing_extensions import Literal, TypeAlias


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openbabel import openbabel as ob
from openbabel import pybel
from rdkit import Chem, RDLogger

from molgr.batch import ReconstructionBatchRequest, iter_xyz_to_rdmol_batch
from molgr.fallback.utils.consts import NON_METAL_DICT
from molgr.fallback.utils.electrons import set_unpaired_electron_count
from molgr.fallback.utils.force_field import force_field_evaluation
from molgr.interface import xyz_to_rdmol
from molgr.utils.equivalence import evaluate_equivalence


RDLogger.DisableLog("rdApp.*")  # type: ignore[arg-type]

NON_METAL_ATOMIC_NUMBERS = frozenset(NON_METAL_DICT)
_DEFAULT_MAX_AUTO_JOBS = 8
BackendName = Literal["cpp", "python"]
PrecomputedReconstruction: TypeAlias = Tuple[Any, Optional[str], Optional[str]]

RESULT_FIELDNAMES = (
    "row_index",
    "id",
    "metal_center",
    "charge",
    "xyz_path",
    "reference_smiles_input",
    "reference_smiles_canonical",
    "molgr_smiles_canonical",
    "equivalent",
    "strict_equivalent",
    "evaluator_decision",
    "equivalence_method",
    "equivalence_reason",
    "spin_source",
    "total_radical_electrons_used",
    "spin_multiplicity_used",
    "reference_parse_status",
    "reference_formula_check_status",
    "reference_formula_match",
    "xyz_atom_count",
    "reference_atom_count_with_h",
    "xyz_formula",
    "reference_formula_with_h",
    "reference_formula_mismatch_detail",
    "molgr_status",
    "reference_answer_wrong",
    "reference_answer_status",
    "reference_answer_reason",
    "manual_whitelist_status",
    "manual_whitelist_reason",
    "effective_equivalent",
    "molgr_organic_smiles",
    "reference_organic_smiles",
    "molgr_organic_atom_count",
    "reference_organic_atom_count",
    "molgr_organic_heavy_atom_count",
    "reference_organic_heavy_atom_count",
    "molgr_organic_uff_status",
    "molgr_organic_uff_kj_mol",
    "reference_organic_mapping_status",
    "reference_organic_uff_status",
    "reference_organic_uff_kj_mol",
    "organic_uff_delta_kj_mol",
    "elapsed_seconds",
    "error",
)


class OrganicCoordinateMappingError(RuntimeError):
    def __init__(self, code: str, detail: str | None = None) -> None:
        self.code = code
        self.detail = detail or code
        super().__init__(self.detail)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="tmQMg metadata CSV path.",
    )
    parser.add_argument(
        "--xyz-dir",
        type=Path,
        required=True,
        help="Directory containing tmQMg XYZ files named <id>.xyz.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="How many CSV rows to process from the top of the file.",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=1,
        help="1-based CSV row index to start processing from. Default: 1.",
    )
    parser.add_argument(
        "--end-row",
        type=int,
        default=None,
        help="Optional 1-based CSV row index to stop at, inclusive.",
    )
    parser.add_argument(
        "--backend",
        choices=("python", "cpp"),
        default="python",
        help="MolGR backend to use for reconstruction. Default keeps the Python reference path.",
    )
    parser.add_argument(
        "--spin-source",
        choices=("reference_smiles", "closed_shell"),
        default="closed_shell",
        help=(
            "How to choose the total radical electron count passed to MolGR. "
            "tmQMg does not expose an independent radical-count column; "
            "'closed_shell' is the intended default for this dataset."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional summary JSON path. Defaults to <out>.summary.json.",
    )
    parser.add_argument(
        "--manual-whitelist",
        type=Path,
        default=Path(__file__).with_name("tmqmg_manual_whitelist.json"),
        help=(
            "Optional JSON file mapping tmQMg ids to manually reviewed accepted divergences. "
            "Default: scripts/tmqmg_manual_whitelist.json"
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print one stderr progress line every N processed entries. Use 0 to silence.",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=0,
        help=(
            "Number of native C++ batch workers. The C++ backend owns this worker pool; "
            "the Python reference backend remains serial. "
            f"Default 0 auto-selects up to {_DEFAULT_MAX_AUTO_JOBS} workers."
        ),
    )
    args = parser.parse_args()
    if args.start_row < 1:
        parser.error("--start-row must be >= 1")
    if args.end_row is not None and args.end_row < args.start_row:
        parser.error("--end-row must be >= --start-row")
    if args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.jobs < 0:
        parser.error("--jobs must be >= 0")
    if args.jobs == 0:
        args.jobs = max(1, min(args.limit, os.cpu_count() or 1, _DEFAULT_MAX_AUTO_JOBS))
    return args


def _summary_path_from_output(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_suffix(f"{output_path.suffix}.summary.json")
    return output_path.with_name(f"{output_path.name}.summary.json")


def _empty_result(row_index: int, row: dict[str, str], xyz_path: Path) -> dict[str, Any]:
    return {
        "row_index": row_index,
        "id": row.get("id", ""),
        "metal_center": row.get("metal_center", ""),
        "charge": row.get("charge", ""),
        "xyz_path": str(xyz_path),
        "reference_smiles_input": row.get("smiles", ""),
        "reference_smiles_canonical": "",
        "molgr_smiles_canonical": "",
        "spin_source": "",
        "total_radical_electrons_used": "",
        "spin_multiplicity_used": "",
        "reference_parse_status": "",
        "reference_formula_check_status": "",
        "reference_formula_match": "",
        "xyz_atom_count": "",
        "reference_atom_count_with_h": "",
        "xyz_formula": "",
        "reference_formula_with_h": "",
        "reference_formula_mismatch_detail": "",
        "molgr_status": "",
        "equivalent": "",
        "strict_equivalent": "",
        "evaluator_decision": "",
        "equivalence_method": "",
        "equivalence_reason": "",
        "reference_answer_wrong": "",
        "reference_answer_status": "",
        "reference_answer_reason": "",
        "manual_whitelist_status": "",
        "manual_whitelist_reason": "",
        "effective_equivalent": "",
        "molgr_organic_smiles": "",
        "reference_organic_smiles": "",
        "molgr_organic_atom_count": "",
        "reference_organic_atom_count": "",
        "molgr_organic_heavy_atom_count": "",
        "reference_organic_heavy_atom_count": "",
        "molgr_organic_uff_status": "",
        "molgr_organic_uff_kj_mol": "",
        "reference_organic_mapping_status": "",
        "reference_organic_uff_status": "",
        "reference_organic_uff_kj_mol": "",
        "organic_uff_delta_kj_mol": "",
        "elapsed_seconds": "",
        "error": "",
    }


def _load_manual_whitelist(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Manual whitelist must be a JSON object keyed by case id: {path}")

    whitelist: dict[str, dict[str, str]] = {}
    for case_id, entry in raw.items():
        if not isinstance(case_id, str):
            raise ValueError(f"Manual whitelist case id must be a string: {case_id!r}")
        if not isinstance(entry, dict):
            raise ValueError(f"Manual whitelist entry for {case_id} must be an object.")
        status = str(entry.get("status", "accepted_reference_issue"))
        reason = str(entry.get("reason", "")).strip()
        whitelist[case_id] = {
            "status": status,
            "reason": reason,
        }
    return whitelist


def _safe_canonical_smiles(mol: Chem.Mol) -> str:
    try:
        heavy_only, _ = _copy_without_hydrogens_with_source_indices(mol)
        return Chem.MolToSmiles(heavy_only, canonical=True)
    except Exception:  # noqa: BLE001
        try:
            clone = Chem.Mol(mol)
            Chem.SanitizeMol(clone)
            heavy_only, _ = _copy_without_hydrogens_with_source_indices(clone)
            return Chem.MolToSmiles(heavy_only, canonical=True)
        except Exception:  # noqa: BLE001
            return ""


def _xyz_element_counts(xyz_block: str) -> Counter[str]:
    lines = xyz_block.splitlines()
    if len(lines) < 2:
        raise ValueError("xyz_too_short")
    try:
        atom_count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError("xyz_invalid_header") from exc

    counts: Counter[str] = Counter()
    coordinate_lines = [line for line in lines[2:] if line.strip()]
    if len(coordinate_lines) < atom_count:
        raise ValueError("xyz_coordinate_count_mismatch")
    for line in coordinate_lines[:atom_count]:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError("xyz_malformed_atom_line")
        counts[parts[0]] += 1
    return counts


def _formula_string(counts: Counter[str]) -> str:
    return " ".join(f"{symbol}:{counts[symbol]}" for symbol in sorted(counts))


def _formula_mismatch_detail(
    xyz_counts: Counter[str],
    reference_counts: Counter[str],
) -> str:
    deltas: list[str] = []
    for symbol in sorted(set(xyz_counts) | set(reference_counts)):
        xyz_count = xyz_counts.get(symbol, 0)
        reference_count = reference_counts.get(symbol, 0)
        if xyz_count == reference_count:
            continue
        deltas.append(f"{symbol}:xyz={xyz_count},ref={reference_count}")
    return "; ".join(deltas)


def _copy_without_metals(mol: Chem.Mol) -> Chem.Mol:
    metal_atom_indices = {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() not in NON_METAL_ATOMIC_NUMBERS
    }
    rwmol = Chem.RWMol(Chem.Mol(mol))
    for atom_idx in sorted(metal_atom_indices, reverse=True):
        rwmol.RemoveAtom(atom_idx)

    stripped = rwmol.GetMol()
    if mol.GetNumConformers():
        conformer = mol.GetConformer()
        stripped_conformer = Chem.Conformer(stripped.GetNumAtoms())
        new_atom_idx = 0
        for atom_idx in range(mol.GetNumAtoms()):
            if atom_idx in metal_atom_indices:
                continue
            stripped_conformer.SetAtomPosition(new_atom_idx, conformer.GetAtomPosition(atom_idx))
            new_atom_idx += 1
        stripped.RemoveAllConformers()
        stripped.AddConformer(stripped_conformer)

    stripped.UpdatePropertyCache(strict=False)
    with suppress(Exception):
        Chem.SanitizeMol(stripped)
    return stripped


def _simplify_connectivity(mol: Chem.Mol) -> Chem.Mol:
    simplified_rw = Chem.RWMol(Chem.Mol(mol))
    for atom in simplified_rw.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNumRadicalElectrons(0)
        atom.SetNoImplicit(True)
        atom.SetIsAromatic(False)
    for bond in simplified_rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetBondDir(Chem.BondDir.NONE)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        bond.SetIsAromatic(False)
    simplified = simplified_rw.GetMol()
    simplified.UpdatePropertyCache(strict=False)
    return simplified


def _heavy_atom_indices(mol: Chem.Mol) -> list[int]:
    return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1]


def _hydrogen_atom_indices(mol: Chem.Mol) -> list[int]:
    return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 1]


def _heavy_atom_count(mol: Chem.Mol) -> int:
    return len(_heavy_atom_indices(mol))


def _copy_without_hydrogens_with_source_indices(mol: Chem.Mol) -> tuple[Chem.Mol, list[int]]:
    heavy_rw = Chem.RWMol()
    source_indices: list[int] = []
    atom_index_map: dict[int, int] = {}
    deferred_bond_stereo: list[tuple[int, Chem.BondStereo, list[int]]] = []

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        source_indices.append(atom.GetIdx())
        atom_index_map[atom.GetIdx()] = heavy_rw.AddAtom(Chem.Atom(atom))

    for bond in mol.GetBonds():
        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        new_begin_atom_idx = atom_index_map.get(begin_atom_idx)
        new_end_atom_idx = atom_index_map.get(end_atom_idx)
        if new_begin_atom_idx is None or new_end_atom_idx is None:
            continue
        heavy_rw.AddBond(new_begin_atom_idx, new_end_atom_idx, bond.GetBondType())
        new_bond = heavy_rw.GetBondBetweenAtoms(new_begin_atom_idx, new_end_atom_idx)
        if new_bond is None:
            continue
        new_bond.SetIsAromatic(bond.GetIsAromatic())
        new_bond.SetBondDir(bond.GetBondDir())
        stereo_atoms = [
            atom_index_map[atom_idx]
            for atom_idx in bond.GetStereoAtoms()
            if atom_idx in atom_index_map
        ]
        deferred_bond_stereo.append((new_bond.GetIdx(), bond.GetStereo(), stereo_atoms))

    heavy = heavy_rw.GetMol()
    # RDKit requires the neighboring bonds referenced by stereo atoms to already
    # exist on the owning molecule before SetStereoAtoms() is called.
    for bond_idx, stereo, stereo_atoms in deferred_bond_stereo:
        heavy_bond = heavy.GetBondWithIdx(bond_idx)
        if len(stereo_atoms) == 2:
            heavy_bond.SetStereoAtoms(*stereo_atoms)
            heavy_bond.SetStereo(stereo)
        else:
            heavy_bond.SetStereo(Chem.BondStereo.STEREONONE)
    if mol.GetNumConformers():
        conformer = mol.GetConformer()
        heavy_conformer = Chem.Conformer(heavy.GetNumAtoms())
        for new_atom_idx, source_atom_idx in enumerate(source_indices):
            heavy_conformer.SetAtomPosition(
                new_atom_idx,
                conformer.GetAtomPosition(source_atom_idx),
            )
        heavy.RemoveAllConformers()
        heavy.AddConformer(heavy_conformer)

    heavy.UpdatePropertyCache(strict=False)
    with suppress(Exception):
        Chem.SanitizeMol(heavy)
    return heavy, source_indices


def _build_reference_organic_with_molgr_coords(
    reference_organic: Chem.Mol,
    molgr_organic: Chem.Mol,
    *,
    prefer_exact_full_match: bool = False,
) -> Chem.Mol:
    if molgr_organic.GetNumConformers() == 0:
        raise OrganicCoordinateMappingError("molgr_organic_missing_conformer")

    if prefer_exact_full_match:
        exact_full_match = molgr_organic.GetSubstructMatch(reference_organic, useChirality=False)
        if exact_full_match and len(exact_full_match) == reference_organic.GetNumAtoms():
            molgr_conformer = molgr_organic.GetConformer()
            aligned_reference_conformer = Chem.Conformer(reference_organic.GetNumAtoms())
            for reference_atom_idx, molgr_atom_idx in enumerate(exact_full_match):
                aligned_reference_conformer.SetAtomPosition(
                    reference_atom_idx,
                    molgr_conformer.GetAtomPosition(molgr_atom_idx),
                )
            aligned_reference_organic = Chem.Mol(reference_organic)
            aligned_reference_organic.RemoveAllConformers()
            aligned_reference_organic.AddConformer(aligned_reference_conformer)
            aligned_reference_organic.UpdatePropertyCache(strict=False)
            return aligned_reference_organic

    reference_heavy, reference_reduced_source_indices = _copy_without_hydrogens_with_source_indices(
        reference_organic
    )
    molgr_heavy, molgr_reduced_source_indices = _copy_without_hydrogens_with_source_indices(
        molgr_organic
    )
    if reference_heavy.GetNumAtoms() != molgr_heavy.GetNumAtoms():
        raise OrganicCoordinateMappingError(
            "heavy_atom_count_mismatch",
            (f"reference={reference_heavy.GetNumAtoms()} molgr={molgr_heavy.GetNumAtoms()}"),
        )

    simplified_reference_heavy = _simplify_connectivity(reference_heavy)
    simplified_molgr_heavy = _simplify_connectivity(molgr_heavy)
    heavy_match = simplified_molgr_heavy.GetSubstructMatch(
        simplified_reference_heavy,
        useChirality=False,
    )
    if not heavy_match or len(heavy_match) != reference_heavy.GetNumAtoms():
        raise OrganicCoordinateMappingError("heavy_connectivity_mismatch")

    reference_hydrogen_indices = _hydrogen_atom_indices(reference_organic)
    molgr_hydrogen_indices = _hydrogen_atom_indices(molgr_organic)

    if len(reference_hydrogen_indices) != len(molgr_hydrogen_indices):
        raise OrganicCoordinateMappingError(
            "hydrogen_atom_count_mismatch",
            (f"reference={len(reference_hydrogen_indices)} molgr={len(molgr_hydrogen_indices)}"),
        )

    molgr_conformer = molgr_organic.GetConformer()
    reference_conformer = Chem.Conformer(reference_organic.GetNumAtoms())

    for reference_heavy_idx, molgr_heavy_simple_idx in enumerate(heavy_match):
        reference_atom_idx = reference_reduced_source_indices[reference_heavy_idx]
        molgr_atom_idx = molgr_reduced_source_indices[molgr_heavy_simple_idx]
        reference_conformer.SetAtomPosition(
            reference_atom_idx,
            molgr_conformer.GetAtomPosition(molgr_atom_idx),
        )

    unused_molgr_hydrogens = set(molgr_hydrogen_indices)
    molgr_parent_to_hydrogens = {
        atom.GetIdx(): [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 1]
        for atom in molgr_organic.GetAtoms()
        if atom.GetAtomicNum() != 1
    }
    deferred_reference_hydrogens: list[int] = []

    for reference_heavy_idx, molgr_heavy_simple_idx in enumerate(heavy_match):
        reference_atom_idx = reference_reduced_source_indices[reference_heavy_idx]
        molgr_atom_idx = molgr_reduced_source_indices[molgr_heavy_simple_idx]
        reference_local_hydrogens = [
            nbr.GetIdx()
            for nbr in reference_organic.GetAtomWithIdx(reference_atom_idx).GetNeighbors()
            if nbr.GetAtomicNum() == 1
        ]
        molgr_local_hydrogens = [
            hydrogen_idx
            for hydrogen_idx in molgr_parent_to_hydrogens.get(molgr_atom_idx, [])
            if hydrogen_idx in unused_molgr_hydrogens
        ]

        assign_count = min(len(reference_local_hydrogens), len(molgr_local_hydrogens))
        for reference_hydrogen_idx, molgr_hydrogen_idx in zip(
            reference_local_hydrogens[:assign_count],
            molgr_local_hydrogens[:assign_count],
        ):
            reference_conformer.SetAtomPosition(
                reference_hydrogen_idx,
                molgr_conformer.GetAtomPosition(molgr_hydrogen_idx),
            )
            unused_molgr_hydrogens.remove(molgr_hydrogen_idx)

        deferred_reference_hydrogens.extend(reference_local_hydrogens[assign_count:])

    deferred_reference_hydrogens.extend(
        hydrogen_idx
        for hydrogen_idx in reference_hydrogen_indices
        if all(
            neighbor.GetAtomicNum() == 1
            for neighbor in reference_organic.GetAtomWithIdx(hydrogen_idx).GetNeighbors()
        )
    )

    for reference_hydrogen_idx in deferred_reference_hydrogens:
        if not unused_molgr_hydrogens:
            raise OrganicCoordinateMappingError("hydrogen_assignment_exhausted")
        reference_heavy_parents = [
            nbr.GetIdx()
            for nbr in reference_organic.GetAtomWithIdx(reference_hydrogen_idx).GetNeighbors()
            if nbr.GetAtomicNum() != 1
        ]
        if reference_heavy_parents:
            parent_position = reference_conformer.GetAtomPosition(reference_heavy_parents[0])
            best_molgr_hydrogen_idx = min(
                unused_molgr_hydrogens,
                key=lambda molgr_hydrogen_idx: (
                    (molgr_conformer.GetAtomPosition(molgr_hydrogen_idx).x - parent_position.x) ** 2
                    + (molgr_conformer.GetAtomPosition(molgr_hydrogen_idx).y - parent_position.y)
                    ** 2
                    + (molgr_conformer.GetAtomPosition(molgr_hydrogen_idx).z - parent_position.z)
                    ** 2
                ),
            )
        else:
            best_molgr_hydrogen_idx = min(unused_molgr_hydrogens)
        reference_conformer.SetAtomPosition(
            reference_hydrogen_idx,
            molgr_conformer.GetAtomPosition(best_molgr_hydrogen_idx),
        )
        unused_molgr_hydrogens.remove(best_molgr_hydrogen_idx)

    aligned_reference_organic = Chem.Mol(reference_organic)
    aligned_reference_organic.RemoveAllConformers()
    aligned_reference_organic.AddConformer(reference_conformer)
    aligned_reference_organic.UpdatePropertyCache(strict=False)
    return aligned_reference_organic


def _rdkit_bond_order_to_openbabel(bond: Chem.Bond) -> int:
    if bond.GetIsAromatic():
        return 5
    bond_order = int(round(float(bond.GetBondTypeAsDouble())))
    if bond_order in {1, 2, 3, 4}:
        return bond_order
    raise ValueError(f"Unsupported RDKit bond type for OpenBabel export: {bond.GetBondType()!r}")


def _rdkit_to_pybel_with_coords(mol: Chem.Mol) -> pybel.Molecule:
    if mol.GetNumConformers() == 0:
        raise ValueError("RDKit molecule is missing 3D coordinates.")
    conformer = mol.GetConformer()
    obmol = ob.OBMol()
    obmol.BeginModify()
    try:
        for atom_idx, atom in enumerate(mol.GetAtoms(), start=1):
            obatom = obmol.NewAtom()
            obatom.SetAtomicNum(atom.GetAtomicNum())
            obatom.SetFormalCharge(atom.GetFormalCharge())
            set_unpaired_electron_count(obatom, atom.GetNumRadicalElectrons())
            position = conformer.GetAtomPosition(atom_idx - 1)
            obatom.SetVector(float(position.x), float(position.y), float(position.z))
        for bond in mol.GetBonds():
            if not obmol.AddBond(
                bond.GetBeginAtomIdx() + 1,
                bond.GetEndAtomIdx() + 1,
                _rdkit_bond_order_to_openbabel(bond),
            ):
                raise RuntimeError(
                    f"Failed to create OpenBabel bond {bond.GetBeginAtomIdx()}-{bond.GetEndAtomIdx()}"
                )
    finally:
        obmol.EndModify()
    return pybel.Molecule(obmol)


def _organic_uff_energy_kj_mol(mol: Chem.Mol) -> float:
    return force_field_evaluation(_rdkit_to_pybel_with_coords(mol)).energy_kj_mol


def _reference_total_radical_electrons(reference_mol: Chem.Mol) -> int:
    return sum(atom.GetNumRadicalElectrons() for atom in reference_mol.GetAtoms())


def _resolve_total_radical_electrons(row: dict[str, str], *, spin_source: str) -> int:
    if spin_source == "closed_shell":
        return 0
    if spin_source == "reference_smiles":
        reference_smiles = row.get("smiles", "").strip()
        if not reference_smiles:
            raise ValueError("missing_reference_smiles")
        reference_mol = Chem.MolFromSmiles(reference_smiles)
        if reference_mol is None:
            raise ValueError("reference_parse_failed")
        return _reference_total_radical_electrons(Chem.AddHs(reference_mol))
    raise ValueError(f"Unsupported spin source: {spin_source!r}")


def _process_row(
    row_index: int,
    row: dict[str, str],
    *,
    xyz_dir: Path,
    backend: BackendName,
    spin_source: str,
    manual_whitelist: dict[str, dict[str, str]],
    precomputed_reconstruction: PrecomputedReconstruction | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    xyz_path = xyz_dir / f"{row['id']}.xyz"
    result = _empty_result(row_index, row, xyz_path)
    result["spin_source"] = spin_source

    reference_mol: Chem.Mol | None = None
    if not row.get("smiles"):
        result["reference_parse_status"] = "missing_reference_smiles"
    else:
        reference_mol = Chem.MolFromSmiles(row["smiles"])
        if reference_mol is None:
            result["reference_parse_status"] = "reference_parse_failed"
        else:
            reference_mol = Chem.AddHs(reference_mol)
            result["reference_parse_status"] = "ok"
            result["reference_smiles_canonical"] = _safe_canonical_smiles(reference_mol)

    if spin_source == "reference_smiles":
        if reference_mol is None:
            result["molgr_status"] = "skipped_missing_reference_radicals"
            result["molgr_organic_uff_status"] = "skipped_missing_molgr_result"
            result["reference_organic_mapping_status"] = "skipped_missing_reference_graph"
            result["reference_organic_uff_status"] = "skipped_missing_reference_graph"
            result["error"] = (
                "Could not derive total radical electrons from the reference SMILES; "
                "tmQMg does not expose a separate radical-count column."
            )
            result["elapsed_seconds"] = round(time.perf_counter() - started_at, 6)
            return result
        total_radical_electrons = _reference_total_radical_electrons(reference_mol)
    elif spin_source == "closed_shell":
        total_radical_electrons = 0
    else:  # pragma: no cover - argparse guards this
        raise ValueError(f"Unsupported spin source: {spin_source}")

    result["total_radical_electrons_used"] = total_radical_electrons
    result["spin_multiplicity_used"] = total_radical_electrons + 1

    if not xyz_path.exists():
        result["molgr_status"] = "xyz_missing"
        result["reference_formula_check_status"] = "xyz_missing"
        result["molgr_organic_uff_status"] = "skipped_missing_molgr_result"
        result["reference_organic_mapping_status"] = "skipped_missing_molgr_result"
        result["reference_organic_uff_status"] = "skipped_missing_molgr_result"
        result["error"] = f"Missing XYZ file: {xyz_path}"
        result["elapsed_seconds"] = round(time.perf_counter() - started_at, 6)
        return result

    xyz_block = xyz_path.read_text()
    try:
        xyz_counts = _xyz_element_counts(xyz_block)
        result["xyz_atom_count"] = sum(xyz_counts.values())
        result["xyz_formula"] = _formula_string(xyz_counts)
    except Exception as exc:  # noqa: BLE001
        result["reference_formula_check_status"] = f"xyz_failed:{type(exc).__name__}"
        result["molgr_status"] = f"failed:{type(exc).__name__}"
        result["molgr_organic_uff_status"] = "skipped_missing_molgr_result"
        result["reference_organic_mapping_status"] = "skipped_missing_molgr_result"
        result["reference_organic_uff_status"] = "skipped_missing_molgr_result"
        result["error"] = str(exc)
        result["elapsed_seconds"] = round(time.perf_counter() - started_at, 6)
        return result

    formula_proves_reference_wrong = False
    formula_wrong_reason = ""
    if reference_mol is None:
        if result["reference_parse_status"] == "missing_reference_smiles":
            result["reference_formula_check_status"] = "missing_reference_smiles"
        elif result["reference_parse_status"] == "reference_parse_failed":
            result["reference_formula_check_status"] = "reference_parse_failed"
    else:
        reference_counts = Counter(atom.GetSymbol() for atom in reference_mol.GetAtoms())
        result["reference_atom_count_with_h"] = sum(reference_counts.values())
        result["reference_formula_with_h"] = _formula_string(reference_counts)
        formula_match = xyz_counts == reference_counts
        result["reference_formula_match"] = formula_match
        if formula_match:
            result["reference_formula_check_status"] = "ok"
        else:
            result["reference_formula_check_status"] = "formula_mismatch"
            result["reference_formula_mismatch_detail"] = _formula_mismatch_detail(
                xyz_counts,
                reference_counts,
            )
            formula_proves_reference_wrong = True
            formula_wrong_reason = "Reference formula does not conserve XYZ atom counts: " + str(
                result["reference_formula_mismatch_detail"]
            )

    if precomputed_reconstruction is None:
        try:
            molgr_mol = xyz_to_rdmol(
                xyz_block,
                total_charge=int(row["charge"]),
                spin_multiplicity=total_radical_electrons + 1,
                backend=backend,
            )
        except Exception as exc:  # noqa: BLE001
            result["molgr_status"] = f"failed:{type(exc).__name__}"
            result["molgr_organic_uff_status"] = "skipped_missing_molgr_result"
            result["reference_organic_mapping_status"] = "skipped_missing_molgr_result"
            result["reference_organic_uff_status"] = "skipped_missing_molgr_result"
            result["error"] = str(exc)
            result["elapsed_seconds"] = round(time.perf_counter() - started_at, 6)
            return result
    else:
        molgr_mol, reconstruction_error, reconstruction_status = precomputed_reconstruction
        if reconstruction_error is not None or molgr_mol is None:
            result["molgr_status"] = f"failed:{reconstruction_status or 'native_batch'}"
            result["molgr_organic_uff_status"] = "skipped_missing_molgr_result"
            result["reference_organic_mapping_status"] = "skipped_missing_molgr_result"
            result["reference_organic_uff_status"] = "skipped_missing_molgr_result"
            result["error"] = reconstruction_error or "native batch returned no molecule"
            result["elapsed_seconds"] = round(time.perf_counter() - started_at, 6)
            return result

    result["molgr_status"] = "ok"
    result["molgr_smiles_canonical"] = _safe_canonical_smiles(molgr_mol)

    if reference_mol is not None:
        try:
            info = evaluate_equivalence(molgr_mol, reference_mol, use_chirality=False)
            decision = info.decision.value
            result["evaluator_decision"] = decision
            result["strict_equivalent"] = (
                True if decision == "equivalent" else False if decision == "not_equivalent" else ""
            )
            result["equivalent"] = (
                True
                if decision == "equivalent"
                else False
                if decision == "not_equivalent"
                else "inconclusive"
            )
            result["equivalence_method"] = info.method.value if info.method is not None else ""
            result["equivalence_reason"] = info.reason
        except Exception as exc:  # noqa: BLE001
            result["equivalent"] = ""
            result["strict_equivalent"] = ""
            result["evaluator_decision"] = ""
            result["equivalence_method"] = ""
            result["equivalence_reason"] = f"equivalence_failed: {type(exc).__name__}: {exc}"

    whitelist_entry = manual_whitelist.get(result["id"])
    whitelist_applied = False
    if formula_proves_reference_wrong:
        result["reference_answer_wrong"] = True
        result["reference_answer_status"] = "formula_mismatch"
        result["reference_answer_reason"] = formula_wrong_reason
        result["equivalent"] = "formula_mismatch"
    elif whitelist_entry is not None and result["evaluator_decision"] == "not_equivalent":
        whitelist_applied = True
        whitelist_status = whitelist_entry["status"]
        result["equivalent"] = whitelist_status
        result["reference_answer_reason"] = whitelist_entry["reason"]
        if whitelist_status == "accepted_reference_issue":
            result["reference_answer_wrong"] = True
            result["reference_answer_status"] = "manual_confirmed_wrong"
        elif whitelist_status == "accepted_ambiguous":
            result["reference_answer_wrong"] = False
            result["reference_answer_status"] = "manual_accepted_ambiguous"
        else:
            result["reference_answer_wrong"] = False
            result["reference_answer_status"] = f"manual_{whitelist_status}"
    else:
        result["reference_answer_wrong"] = False
        result["reference_answer_status"] = "not_flagged"
    if whitelist_applied and whitelist_entry is not None:
        result["manual_whitelist_status"] = whitelist_entry["status"]
        result["manual_whitelist_reason"] = whitelist_entry["reason"]
    result["effective_equivalent"] = bool(
        result["equivalent"] is True or whitelist_applied or formula_proves_reference_wrong
    )

    try:
        molgr_organic = _copy_without_metals(molgr_mol)
        result["molgr_organic_smiles"] = _safe_canonical_smiles(molgr_organic)
        result["molgr_organic_atom_count"] = molgr_organic.GetNumAtoms()
        result["molgr_organic_heavy_atom_count"] = _heavy_atom_count(molgr_organic)
        result["molgr_organic_uff_kj_mol"] = _organic_uff_energy_kj_mol(molgr_organic)
        result["molgr_organic_uff_status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        molgr_organic = None
        result["molgr_organic_uff_status"] = f"failed:{type(exc).__name__}"
        if not result["error"]:
            result["error"] = str(exc)

    if reference_mol is None:
        result["reference_organic_mapping_status"] = "skipped_missing_reference_graph"
        result["reference_organic_uff_status"] = "skipped_missing_reference_graph"
        result["elapsed_seconds"] = round(time.perf_counter() - started_at, 6)
        return result

    try:
        reference_organic = _copy_without_metals(reference_mol)
        result["reference_organic_smiles"] = _safe_canonical_smiles(reference_organic)
        result["reference_organic_atom_count"] = reference_organic.GetNumAtoms()
        result["reference_organic_heavy_atom_count"] = _heavy_atom_count(reference_organic)
    except Exception as exc:  # noqa: BLE001
        result["reference_organic_mapping_status"] = f"failed:{type(exc).__name__}"
        result["reference_organic_uff_status"] = f"failed:{type(exc).__name__}"
        if not result["error"]:
            result["error"] = str(exc)
        result["elapsed_seconds"] = round(time.perf_counter() - started_at, 6)
        return result

    if molgr_organic is None:
        result["reference_organic_mapping_status"] = "skipped_missing_molgr_organic"
        result["reference_organic_uff_status"] = "skipped_missing_molgr_organic"
        result["elapsed_seconds"] = round(time.perf_counter() - started_at, 6)
        return result

    try:
        aligned_reference_organic = _build_reference_organic_with_molgr_coords(
            reference_organic,
            molgr_organic,
            prefer_exact_full_match=(
                result["reference_organic_smiles"] != ""
                and result["reference_organic_smiles"] == result["molgr_organic_smiles"]
            ),
        )
        result["reference_organic_mapping_status"] = "ok"
        result["reference_organic_uff_kj_mol"] = _organic_uff_energy_kj_mol(
            aligned_reference_organic
        )
        result["reference_organic_uff_status"] = "ok"
    except OrganicCoordinateMappingError as exc:
        result["reference_organic_mapping_status"] = exc.code
        result["reference_organic_uff_status"] = exc.code
        if not result["error"]:
            result["error"] = exc.detail
    except Exception as exc:  # noqa: BLE001
        result["reference_organic_mapping_status"] = f"failed:{type(exc).__name__}"
        result["reference_organic_uff_status"] = f"failed:{type(exc).__name__}"
        if not result["error"]:
            result["error"] = str(exc)

    if result["molgr_organic_uff_kj_mol"] != "" and result["reference_organic_uff_kj_mol"] != "":
        result["organic_uff_delta_kj_mol"] = float(result["molgr_organic_uff_kj_mol"]) - float(
            result["reference_organic_uff_kj_mol"]
        )

    result["elapsed_seconds"] = round(time.perf_counter() - started_at, 6)
    return result


def _select_input_rows(
    reader: csv.DictReader,
    *,
    start_row: int,
    end_row: int | None,
    limit: int,
) -> list[tuple[int, dict[str, str]]]:
    selected_rows: list[tuple[int, dict[str, str]]] = []
    for row_index, row in enumerate(reader, start=1):
        if row_index < start_row:
            continue
        if end_row is not None and row_index > end_row:
            break
        if len(selected_rows) >= limit:
            break
        selected_rows.append((row_index, dict(row)))
    return selected_rows


def _unexpected_row_error_result(
    row_index: int,
    row: dict[str, str],
    *,
    xyz_dir: Path,
    spin_source: str,
    started_at: float,
    exc: Exception,
) -> dict[str, Any]:
    xyz_path = xyz_dir / f"{row.get('id', '')}.xyz"
    result = _empty_result(row_index, row, xyz_path)
    result["spin_source"] = spin_source
    result["molgr_status"] = f"failed:{type(exc).__name__}"
    result["molgr_organic_uff_status"] = "skipped_unexpected_error"
    result["reference_organic_mapping_status"] = "skipped_unexpected_error"
    result["reference_organic_uff_status"] = "skipped_unexpected_error"
    result["error"] = str(exc)
    result["elapsed_seconds"] = round(time.perf_counter() - started_at, 6)
    return result


def _process_row_safe(
    task: tuple[int, dict[str, str]],
    *,
    xyz_dir: Path,
    backend: BackendName,
    spin_source: str,
    manual_whitelist: dict[str, dict[str, str]],
) -> dict[str, Any]:
    row_index, row = task
    started_at = time.perf_counter()
    try:
        return _process_row(
            row_index,
            row,
            xyz_dir=xyz_dir,
            backend=backend,
            spin_source=spin_source,
            manual_whitelist=manual_whitelist,
        )
    except Exception as exc:
        return _unexpected_row_error_result(
            row_index,
            row,
            xyz_dir=xyz_dir,
            spin_source=spin_source,
            started_at=started_at,
            exc=exc,
        )


def _native_batch_reconstructions(
    row_tasks: list[tuple[int, dict[str, str]]],
    *,
    xyz_dir: Path,
    spin_source: str,
    max_workers: int,
) -> dict[int, PrecomputedReconstruction]:
    """Reconstruct rows in one native batch and retain row correspondence."""

    prepared: dict[int, PrecomputedReconstruction] = {}
    requests: list[ReconstructionBatchRequest] = []
    request_row_indices: list[int] = []
    for row_index, row in row_tasks:
        xyz_path = xyz_dir / f"{row.get('id', '').strip()}.xyz"
        try:
            xyz_block = xyz_path.read_text(encoding="utf-8")
            total_charge = int(row.get("charge", "0").strip() or 0)
            total_radical_electrons = _resolve_total_radical_electrons(
                row,
                spin_source=spin_source,
            )
        except Exception as exc:  # noqa: BLE001
            prepared[row_index] = (
                None,
                f"{type(exc).__name__}: {exc}",
                type(exc).__name__,
            )
            continue
        requests.append(
            ReconstructionBatchRequest(
                xyz_block=xyz_block,
                total_charge=total_charge,
                spin_multiplicity=total_radical_electrons + 1,
            )
        )
        request_row_indices.append(row_index)

    if not requests:
        return prepared

    try:
        batch_results = iter_xyz_to_rdmol_batch(
            requests,
            backend="cpp",
            max_workers=max_workers,
            ordered=True,
        )
        for row_index, batch_result in zip(request_row_indices, batch_results):
            diagnostics = batch_result.diagnostics
            error = None
            failure_code = None
            if diagnostics is not None:
                failure_code = diagnostics.code.value
                error = f"{diagnostics.code.value} at {diagnostics.stage}: {diagnostics.message}"
            prepared[row_index] = (batch_result.molecule, error, failure_code)
    except Exception as exc:  # noqa: BLE001
        error = f"native batch execution failed: {type(exc).__name__}: {exc}"
        for row_index in request_row_indices:
            prepared[row_index] = (None, error, "BatchExecution")
    return prepared


def _iter_processed_rows(
    row_tasks: list[tuple[int, dict[str, str]]],
    *,
    jobs: int,
    xyz_dir: Path,
    backend: BackendName,
    spin_source: str,
    manual_whitelist: dict[str, dict[str, str]],
) -> Iterator[dict[str, Any]]:
    if backend == "cpp":
        precomputed = _native_batch_reconstructions(
            row_tasks,
            xyz_dir=xyz_dir,
            spin_source=spin_source,
            max_workers=jobs,
        )
        for task in row_tasks:
            row_index, row = task
            if row_index not in precomputed:
                yield _unexpected_row_error_result(
                    row_index,
                    row,
                    xyz_dir=xyz_dir,
                    spin_source=spin_source,
                    started_at=time.perf_counter(),
                    exc=RuntimeError("native batch returned no result for row"),
                )
                continue
            try:
                yield _process_row(
                    row_index,
                    row,
                    xyz_dir=xyz_dir,
                    backend=backend,
                    spin_source=spin_source,
                    manual_whitelist=manual_whitelist,
                    precomputed_reconstruction=precomputed[row_index],
                )
            except Exception as exc:
                yield _unexpected_row_error_result(
                    row_index,
                    row,
                    xyz_dir=xyz_dir,
                    spin_source=spin_source,
                    started_at=time.perf_counter(),
                    exc=exc,
                )
        return

    # The Python reference backend is intentionally serial; it does not own
    # an executor and must not be wrapped in an outer process pool.
    del jobs
    for task in row_tasks:
        yield _process_row_safe(
            task,
            xyz_dir=xyz_dir,
            backend=backend,
            spin_source=spin_source,
            manual_whitelist=manual_whitelist,
        )


def _normalize_backend(backend: str) -> BackendName:
    if backend not in ("cpp", "python"):
        raise ValueError(f"Unsupported backend: {backend}")
    return cast(BackendName, backend)


def _counter_to_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def main() -> int:
    args = _parse_args()
    backend = _normalize_backend(args.backend)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary_out = args.summary_out or _summary_path_from_output(args.out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    manual_whitelist = _load_manual_whitelist(args.manual_whitelist)
    reference_parse_counter: Counter[str] = Counter()
    reference_formula_check_counter: Counter[str] = Counter()
    molgr_status_counter: Counter[str] = Counter()
    equivalence_method_counter: Counter[str] = Counter()
    equivalence_display_counter: Counter[str] = Counter()
    manual_whitelist_counter: Counter[str] = Counter()
    reference_answer_status_counter: Counter[str] = Counter()
    molgr_uff_status_counter: Counter[str] = Counter()
    reference_mapping_counter: Counter[str] = Counter()
    reference_uff_status_counter: Counter[str] = Counter()
    error_counter: Counter[str] = Counter()

    processed = 0
    equivalent_count = 0
    effective_equivalent_count = 0
    uff_pair_count = 0
    organic_uff_delta_sum = 0.0
    organic_uff_abs_delta_sum = 0.0
    started_at = time.perf_counter()

    with args.csv.open(newline="") as input_fh:
        reader = csv.DictReader(input_fh)
        row_tasks = _select_input_rows(
            reader,
            start_row=args.start_row,
            end_row=args.end_row,
            limit=args.limit,
        )

    with args.out.open("w", newline="") as output_fh:
        writer = csv.DictWriter(output_fh, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()

        for result in _iter_processed_rows(
            row_tasks,
            jobs=args.jobs,
            xyz_dir=args.xyz_dir,
            backend=backend,
            spin_source=args.spin_source,
            manual_whitelist=manual_whitelist,
        ):
            writer.writerow(cast(Any, result))
            processed += 1

            reference_parse_counter.update([result["reference_parse_status"] or "missing"])
            reference_formula_check_counter.update(
                [result["reference_formula_check_status"] or "missing"]
            )
            molgr_status_counter.update([result["molgr_status"] or "missing"])
            molgr_uff_status_counter.update([result["molgr_organic_uff_status"] or "missing"])
            reference_mapping_counter.update(
                [result["reference_organic_mapping_status"] or "missing"]
            )
            reference_uff_status_counter.update(
                [result["reference_organic_uff_status"] or "missing"]
            )
            if result["strict_equivalent"] is True:
                equivalent_count += 1
            if result["effective_equivalent"] is True:
                effective_equivalent_count += 1
            if result["equivalent"] != "":
                equivalence_display_counter.update([str(result["equivalent"])])
            if result["equivalence_method"]:
                equivalence_method_counter.update([str(result["equivalence_method"])])
            if result["manual_whitelist_status"]:
                manual_whitelist_counter.update([str(result["manual_whitelist_status"])])
            if result["reference_answer_status"]:
                reference_answer_status_counter.update([str(result["reference_answer_status"])])
            if result["error"]:
                error_counter.update([str(result["error"])])
            if result["organic_uff_delta_kj_mol"] != "":
                delta = float(result["organic_uff_delta_kj_mol"])
                uff_pair_count += 1
                organic_uff_delta_sum += delta
                organic_uff_abs_delta_sum += abs(delta)

            if args.progress_every > 0 and processed % args.progress_every == 0:
                print(
                    f"[tmQMg] processed {processed}/{len(row_tasks)} rows; "
                    f"latest id={result['id']} molgr_status={result['molgr_status']}",
                    file=sys.stderr,
                )

    total_elapsed_seconds = time.perf_counter() - started_at
    summary = {
        "input_csv": str(args.csv),
        "xyz_dir": str(args.xyz_dir),
        "backend": backend,
        "spin_source": args.spin_source,
        "jobs": args.jobs,
        "native_workers": args.jobs if backend == "cpp" else 1,
        "parallel_execution": "native_batch" if backend == "cpp" else "serial",
        "limit": args.limit,
        "start_row": args.start_row,
        "end_row": args.end_row,
        "selected_row_count": len(row_tasks),
        "processed": processed,
        "equivalent_count": equivalent_count,
        "equivalent_fraction": (equivalent_count / processed) if processed else 0.0,
        "effective_equivalent_count": effective_equivalent_count,
        "effective_equivalent_fraction": (
            effective_equivalent_count / processed if processed else 0.0
        ),
        "equivalence_display_counts": _counter_to_dict(equivalence_display_counter),
        "manual_whitelist_path": str(args.manual_whitelist),
        "manual_whitelist_entry_count": len(manual_whitelist),
        "manual_whitelist_applied_status_counts": _counter_to_dict(manual_whitelist_counter),
        "reference_answer_status_counts": _counter_to_dict(reference_answer_status_counter),
        "organic_uff_pair_count": uff_pair_count,
        "organic_uff_delta_mean_kj_mol": (
            organic_uff_delta_sum / uff_pair_count if uff_pair_count else None
        ),
        "organic_uff_abs_delta_mean_kj_mol": (
            organic_uff_abs_delta_sum / uff_pair_count if uff_pair_count else None
        ),
        "reference_parse_status_counts": _counter_to_dict(reference_parse_counter),
        "reference_formula_check_status_counts": _counter_to_dict(reference_formula_check_counter),
        "molgr_status_counts": _counter_to_dict(molgr_status_counter),
        "equivalence_method_counts": _counter_to_dict(equivalence_method_counter),
        "molgr_organic_uff_status_counts": _counter_to_dict(molgr_uff_status_counter),
        "reference_organic_mapping_status_counts": _counter_to_dict(reference_mapping_counter),
        "reference_organic_uff_status_counts": _counter_to_dict(reference_uff_status_counter),
        "error_counts": _counter_to_dict(error_counter),
        "total_elapsed_seconds": round(total_elapsed_seconds, 6),
        "output_csv": str(args.out),
    }

    with summary_out.open("w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(
        f"Wrote {processed} tmQMg regression rows to {args.out} and summary to {summary_out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
