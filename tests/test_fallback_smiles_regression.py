# pyright: reportMissingImports=false

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest


pytest.importorskip("openbabel")

from rdkit import Chem, RDLogger
from rdkit.Chem import rdDistGeom
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule

from molgr.interface import xyz_to_rdmol
from molgr.utils.equivalence import evaluate_equivalence


RDLogger.DisableLog("rdApp.*")  # type: ignore

_REGRESSION_EMBED_SEED = 0xC0FFEE


@dataclass(frozen=True)
class _RegressionOutcome:
    status: str
    equivalent: bool | None
    predicted_smiles: str | None
    error: str | None
    evaluator_decision: str | None = None


def _load_smiles_regression_cases() -> list[object]:
    csv_path = Path(__file__).with_name("test_cases.csv")
    raw_lines = csv_path.read_text(encoding="utf-8").splitlines()
    smiles_lines = [line.strip() for line in raw_lines[1:] if line.strip()]
    cases = [_build_smiles_case(smiles, idx) for idx, smiles in enumerate(smiles_lines, start=1)]
    return [
        pytest.param(
            case,
            id=f"case-{int(case['case_idx']):02d}",
        )
        for case in cases
    ]


def _total_charge_and_radicals(mol: Chem.Mol) -> tuple[int, int]:
    charge = 0
    radical = 0
    for atom in mol.GetAtoms():  # pyright: ignore[reportCallIssue]
        charge += int(atom.GetFormalCharge())
        radical += int(atom.GetNumRadicalElectrons())
    return charge, radical


def _build_smiles_case(smiles: str, case_idx: int) -> dict[str, object]:
    case: dict[str, object] = {
        "case_idx": case_idx,
        "input_smiles": smiles,
        "ground_truth_rdmol": None,
        "xyz_block": None,
        "total_charge": None,
        "total_radical_electrons": None,
        "provider_error": None,
    }

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("RDKit failed to parse SMILES")

        mol_h = Chem.AddHs(mol)
        embed_code = rdDistGeom.EmbedMolecule(  # pyright: ignore[reportCallIssue]
            mol_h,
            randomSeed=_REGRESSION_EMBED_SEED,
        )
        UFFOptimizeMolecule(mol_h)
        if int(embed_code) != 0:
            raise ValueError(f"RDKit EmbedMolecule failed: code={embed_code}")

        total_charge, total_radical_electrons = _total_charge_and_radicals(mol_h)
        case["ground_truth_rdmol"] = mol
        case["xyz_block"] = Chem.MolToXYZBlock(mol_h)
        case["total_charge"] = total_charge
        case["total_radical_electrons"] = total_radical_electrons
    except Exception as exc:  # noqa: BLE001
        case["provider_error"] = f"{type(exc).__name__}: {exc}"

    return case


def _run_smiles_regression_case(case: dict[str, object]) -> _RegressionOutcome:
    provider_error = case.get("provider_error")
    if provider_error:
        return _RegressionOutcome(
            status="skipped",
            equivalent=None,
            predicted_smiles=None,
            error=str(provider_error),
        )

    xyz_block = case["xyz_block"]
    total_charge = case["total_charge"]
    total_radical_electrons = case["total_radical_electrons"]
    ground_truth_rdmol = case["ground_truth_rdmol"]

    if not isinstance(xyz_block, str):
        raise TypeError("xyz_block must be a string")
    if not isinstance(total_charge, int):
        raise TypeError("total_charge must be an int")
    if not isinstance(total_radical_electrons, int):
        raise TypeError("total_radical_electrons must be an int")

    try:
        rdmol = xyz_to_rdmol(
            xyz_block,
            total_charge,
            total_radical_electrons + 1,
            backend="python",
        )
    except Exception as exc:  # noqa: BLE001
        return _RegressionOutcome(
            status="error",
            equivalent=None,
            predicted_smiles=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    try:
        predicted_rdmol = Chem.RemoveHs(rdmol)
        predicted_smiles = Chem.MolToSmiles(
            predicted_rdmol,
            canonical=True,
            isomericSmiles=True,
        )
    except Exception as exc:  # noqa: BLE001
        return _RegressionOutcome(
            status="error",
            equivalent=None,
            predicted_smiles=None,
            error=f"postprocess failed: {type(exc).__name__}: {exc}",
        )

    if ground_truth_rdmol is None:
        return _RegressionOutcome(
            status="ok",
            equivalent=None,
            predicted_smiles=predicted_smiles,
            error=None,
        )

    try:
        info = evaluate_equivalence(
            ground_truth_rdmol,
            predicted_rdmol,
            use_chirality=True,
            max_resonance=100,
        )
        decision = info.decision.value
    except Exception as exc:  # noqa: BLE001
        return _RegressionOutcome(
            status="error",
            equivalent=None,
            predicted_smiles=predicted_smiles,
            error=f"equivalence check failed: {type(exc).__name__}: {exc}",
        )

    return _RegressionOutcome(
        status="ok",
        equivalent=True if decision == "equivalent" else None,
        predicted_smiles=predicted_smiles,
        error=None,
        evaluator_decision=decision,
    )


@pytest.mark.parametrize("case", _load_smiles_regression_cases())
def test_fallback_smiles_regression_cases(case: dict[str, object]) -> None:
    outcome = _run_smiles_regression_case(case)

    assert outcome.status == "ok", (case["input_smiles"], outcome)
    assert outcome.evaluator_decision in {"equivalent", "inconclusive"}, (
        case["input_smiles"],
        outcome,
    )
