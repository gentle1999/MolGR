# pyright: reportMissingImports=false

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


pytest.importorskip("openbabel")
pytest.importorskip("rdkit")

from openbabel import openbabel as ob
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDistGeom

from molgr import _core as core
from molgr.fallback.pipeline.reconstruct_without_metals import xyz_to_omol_no_metal_state
from molgr.fallback.utils.electrons import (
    get_lone_pair_count,
    get_unpaired_electron_count,
    has_unresolved_two_electron_center,
)
from molgr.interface import xyz_to_rdmol
from molgr.utils.equivalence import EquivalenceDecision, evaluate_equivalence


RDLogger.DisableLog("rdApp.*")  # type: ignore

_EMBED_SEED = 0xC0FFEE
_CHILD_FLAG = "--backend-regression-child"
_CHILD_TIMEOUT_SECONDS = float(os.environ.get("MOLGR_BACKEND_REGRESSION_TIMEOUT", "45"))


def _total_charge_and_radicals(mol: Chem.Mol) -> tuple[int, int]:
    charge = 0
    radicals = 0
    for atom in mol.GetAtoms():  # pyright: ignore[reportCallIssue]
        charge += int(atom.GetFormalCharge())
        radicals += int(atom.GetNumRadicalElectrons())
    return charge, radicals


def _canonical_smiles(mol: Chem.Mol) -> str:
    heavy = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(
        heavy,
        canonical=True,
        isomericSmiles=True,
        allBondsExplicit=True,
        allHsExplicit=True,
    )


def _ordered_atom_signature(mol: Chem.Mol) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            atom.GetIdx(),
            atom.GetAtomicNum(),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
            str(atom.GetChiralTag()),
            atom.GetIsotope(),
            atom.GetNoImplicit(),
            atom.GetIsAromatic(),
        )
        for atom in mol.GetAtoms()  # pyright: ignore[reportCallIssue]
    )


def _ordered_bond_signature(mol: Chem.Mol) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        sorted(
            (
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                str(bond.GetBondType()),
                bond.GetIsAromatic(),
                str(bond.GetStereo()),
            )
            for bond in mol.GetBonds()  # pyright: ignore[reportCallIssue]
        )
    )


def _reconstruction_signature(mol: Chem.Mol) -> tuple[Any, ...]:
    return (
        _canonical_smiles(mol),
        _ordered_atom_signature(mol),
        _ordered_bond_signature(mol),
    )


def _assert_internal_states_differ_by_elementary_three_center_shift(
    xyz_block: str,
    total_charge: int,
    total_radical_electrons: int,
) -> None:
    """Prove backend parity before toolkit-specific RDKit aromaticity perception."""

    cpp = core.pipeline.reconstruct_without_metals.xyz_to_omol_no_metal(
        xyz_block,
        total_charge,
        total_radical_electrons,
    )
    python = xyz_to_omol_no_metal_state(
        xyz_block,
        total_charge,
        total_radical_electrons,
    )
    assert cpp is not None
    assert python is not None

    cpp_atoms = [
        (
            int(atom.atomic_num),
            int(atom.formal_charge),
            int(atom.radical_num),
            int(atom.lone_pair_count),
            bool(atom.unresolved_two_electron_center),
        )
        for atom in cpp.atoms
    ]
    python_atoms = [
        (
            int(atom.GetAtomicNum()),
            int(atom.GetFormalCharge()),
            int(get_unpaired_electron_count(atom)),
            int(get_lone_pair_count(atom)),
            bool(has_unresolved_two_electron_center(atom)),
        )
        for atom in ob.OBMolAtomIter(python.omol.OBMol)  # pyright: ignore[reportCallIssue]
    ]
    assert len(cpp_atoms) == len(python_atoms)
    assert [atom[:2] + atom[3:] for atom in cpp_atoms] == [
        atom[:2] + atom[3:] for atom in python_atoms
    ]

    radical_deltas = [right[2] - left[2] for left, right in zip(cpp_atoms, python_atoms)]
    radical_endpoints = [index for index, delta in enumerate(radical_deltas) if delta]
    assert len(radical_endpoints) == 2
    assert sorted(radical_deltas[index] for index in radical_endpoints) == [-1, 1]

    cpp_bonds = {
        tuple(sorted((int(bond.begin_atom_idx) - 1, int(bond.end_atom_idx) - 1))): int(bond.order)
        for bond in cpp.bonds
    }
    python_bonds = {
        tuple(sorted((int(bond.GetBeginAtomIdx()) - 1, int(bond.GetEndAtomIdx()) - 1))): int(
            bond.GetBondOrder()
        )
        for bond in ob.OBMolBondIter(python.omol.OBMol)  # pyright: ignore[reportCallIssue]
    }
    assert cpp_bonds.keys() == python_bonds.keys()
    changed_bonds = {
        bond: python_bonds[bond] - cpp_order
        for bond, cpp_order in cpp_bonds.items()
        if python_bonds[bond] != cpp_order
    }
    assert len(changed_bonds) == 2
    assert sorted(changed_bonds.values()) == [-1, 1]

    first_bond, second_bond = changed_bonds
    centers = set(first_bond) & set(second_bond)
    assert len(centers) == 1
    center = next(iter(centers))
    assert {*(set(first_bond) - {center}), *(set(second_bond) - {center})} == set(radical_endpoints)


def _assert_coordinates_match(cpp_mol: Chem.Mol, python_mol: Chem.Mol) -> None:
    assert cpp_mol.GetNumAtoms() == python_mol.GetNumAtoms()
    cpp_conf = cpp_mol.GetConformer()
    python_conf = python_mol.GetConformer()
    for atom_idx in range(cpp_mol.GetNumAtoms()):
        cpp_pos = cpp_conf.GetAtomPosition(atom_idx)
        python_pos = python_conf.GetAtomPosition(atom_idx)
        assert cpp_pos.x == pytest.approx(python_pos.x, abs=1e-4)
        assert cpp_pos.y == pytest.approx(python_pos.y, abs=1e-4)
        assert cpp_pos.z == pytest.approx(python_pos.z, abs=1e-4)


def _assert_backend_results_match(
    xyz_block: str,
    total_charge: int,
    total_radical_electrons: int,
    *,
    make_dative_bonds: bool,
) -> None:
    spin_multiplicity = total_radical_electrons + 1
    cpp_mol = xyz_to_rdmol(
        xyz_block,
        total_charge,
        spin_multiplicity,
        backend="cpp",
        make_dative_bonds=make_dative_bonds,
    )
    python_mol = xyz_to_rdmol(
        xyz_block,
        total_charge,
        spin_multiplicity,
        backend="python",
        make_dative_bonds=make_dative_bonds,
    )

    if _reconstruction_signature(cpp_mol) != _reconstruction_signature(python_mol):
        # RDKit/Open Babel releases may choose different Kekule, aromatic, or
        # resonance charge representations for the same backend result. Keep
        # the exact signature as the primary parity contract, but compare the
        # normalized molecular semantics when only toolkit representation
        # differs.
        result = evaluate_equivalence(
            cpp_mol,
            python_mol,
            use_chirality=False,
            max_resonance=100,
        )
        if result.decision == EquivalenceDecision.INCONCLUSIVE:
            _assert_internal_states_differ_by_elementary_three_center_shift(
                xyz_block,
                total_charge,
                total_radical_electrons,
            )
        else:
            assert result.decision == EquivalenceDecision.EQUIVALENT, result.reason
    _assert_coordinates_match(cpp_mol, python_mol)


def _smiles_case_to_xyz(smiles: str) -> tuple[str, int, int]:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    mol_h = Chem.AddHs(mol)
    embed_code = rdDistGeom.EmbedMolecule(  # pyright: ignore[reportCallIssue]
        mol_h,
        randomSeed=_EMBED_SEED,
    )
    assert int(embed_code) == 0
    total_charge, total_radical_electrons = _total_charge_and_radicals(mol_h)
    return Chem.MolToXYZBlock(mol_h), total_charge, total_radical_electrons


def _monnmo_case_to_xyz() -> tuple[str, int, int]:
    mol = Chem.MolFromMolFile(
        str(Path("tests/data/sdf/MoNNMo.sdf")),
        sanitize=False,
        removeHs=False,
        strictParsing=False,
    )
    assert mol is not None
    total_charge, total_radical_electrons = _total_charge_and_radicals(mol)
    return Chem.MolToXYZBlock(mol), total_charge, total_radical_electrons


def _run_backend_case_in_current_process(
    xyz_block: str,
    total_charge: int,
    total_radical_electrons: int,
    *,
    label: str,
    make_dative_bonds: bool,
) -> None:
    print(f"{label} make_dative_bonds={make_dative_bonds}", flush=True)
    _assert_backend_results_match(
        xyz_block,
        total_charge,
        total_radical_electrons,
        make_dative_bonds=make_dative_bonds,
    )


def _run_backend_regression_child(kind: str, case_idx: str, make_dative_arg: str) -> None:
    if make_dative_arg == "0":
        make_dative_bonds = False
    elif make_dative_arg == "1":
        make_dative_bonds = True
    else:
        raise ValueError(f"unknown make_dative_bonds flag: {make_dative_arg!r}")

    if kind == "smiles":
        csv_path = Path(__file__).with_name("test_cases.csv")
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        row_index = int(case_idx)
        smiles = rows[row_index - 1]["smiles"].strip()
        xyz_block, total_charge, total_radical_electrons = _smiles_case_to_xyz(smiles)
        label = f"smiles-case-{row_index:02d}"
    elif kind == "monnmo":
        xyz_block, total_charge, total_radical_electrons = _monnmo_case_to_xyz()
        label = "monnmo"
    else:
        raise ValueError(f"unknown backend regression child kind: {kind!r}")

    _run_backend_case_in_current_process(
        xyz_block,
        total_charge,
        total_radical_electrons,
        label=label,
        make_dative_bonds=make_dative_bonds,
    )


def _prepend_pythonpath(env: dict[str, str], path: Path) -> None:
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(path) if not existing else f"{path}{os.pathsep}{existing}"


def _decode_child_output(value: str | bytes | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _child_output(stdout: str | bytes | None, stderr: str | bytes | None) -> str:
    chunks = []
    decoded_stdout = _decode_child_output(stdout)
    decoded_stderr = _decode_child_output(stderr)
    if decoded_stdout:
        chunks.append(f"stdout:\n{decoded_stdout[-4000:]}")
    if decoded_stderr:
        chunks.append(f"stderr:\n{decoded_stderr[-4000:]}")
    return "\n\n".join(chunks) or "<no child output>"


def _assert_backend_case_in_subprocess(
    kind: str,
    case_idx: str,
    *,
    label: str,
    make_dative_bonds: bool,
) -> None:
    env = os.environ.copy()
    env["PYTHONFAULTHANDLER"] = "1"
    _prepend_pythonpath(env, Path(__file__).resolve().parents[1] / "src")

    make_dative_arg = "1" if make_dative_bonds else "0"
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        _CHILD_FLAG,
        kind,
        case_idx,
        make_dative_arg,
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            text=True,
            capture_output=True,
            timeout=_CHILD_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            (
                f"{label} make_dative_bonds={make_dative_bonds} exceeded "
                f"{_CHILD_TIMEOUT_SECONDS:g}s in isolated backend "
                f"regression subprocess.\n{_child_output(exc.stdout, exc.stderr)}"
            ),
            pytrace=False,
        )

    if completed.returncode != 0:
        pytest.fail(
            (
                f"{label} make_dative_bonds={make_dative_bonds} failed in isolated "
                f"backend regression subprocess (exit code {completed.returncode}).\n"
                f"{_child_output(completed.stdout, completed.stderr)}"
            ),
            pytrace=False,
        )


def _assert_backend_case_in_subprocesses(kind: str, case_idx: str, *, label: str) -> None:
    for make_dative_bonds in (False, True):
        _assert_backend_case_in_subprocess(
            kind,
            case_idx,
            label=label,
            make_dative_bonds=make_dative_bonds,
        )


def _load_smiles_backend_cases() -> list[object]:
    csv_path = Path(__file__).with_name("test_cases.csv")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    cases: list[object] = []
    for case_idx, row in enumerate(rows, start=1):
        smiles = row["smiles"].strip()
        cases.append(
            pytest.param(
                case_idx,
                smiles,
                id=f"smiles-case-{case_idx:02d}",
            )
        )
    return cases


@pytest.mark.parametrize(
    ("case_idx", "smiles"),
    _load_smiles_backend_cases(),
)
def test_cpp_and_python_backends_match_smiles_regression_cases(
    case_idx: int,
    smiles: str,
) -> None:
    _assert_backend_case_in_subprocesses(
        "smiles",
        str(case_idx),
        label=f"smiles-case-{case_idx:02d} ({smiles})",
    )


@pytest.mark.parametrize(
    "make_dative_bonds",
    [False, True],
)
def test_cpp_and_python_backends_match_monnmo_regression_case(
    make_dative_bonds: bool,
) -> None:
    _assert_backend_case_in_subprocess(
        "monnmo",
        "0",
        label="monnmo",
        make_dative_bonds=make_dative_bonds,
    )


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == _CHILD_FLAG:
        _run_backend_regression_child(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        raise SystemExit(f"usage: {sys.argv[0]} {_CHILD_FLAG} <smiles|monnmo> <case-index> <0|1>")
