# pyright: reportMissingImports=false

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest
from rdkit import Chem


pytest.importorskip("openbabel")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import tmqmg_regression as regression
from scripts.tmqmg_regression import (
    _copy_without_hydrogens_with_source_indices,
    _heavy_atom_count,
    _safe_canonical_smiles,
)


def _unsanitized_mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    assert mol is not None
    mol.UpdatePropertyCache(strict=False)
    return mol


def _stereo_regression_mol() -> Chem.Mol:
    rw = Chem.RWMol()
    for symbol in ("C", "C", "C", "N"):
        rw.AddAtom(Chem.Atom(symbol))
    # Add the double bond before one substituent bond so SetStereoAtoms() would
    # fail if stereo information is copied before all neighboring bonds exist.
    rw.AddBond(0, 1, Chem.BondType.DOUBLE)
    rw.AddBond(0, 2, Chem.BondType.SINGLE)
    rw.AddBond(1, 3, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    bond = mol.GetBondBetweenAtoms(0, 1)
    assert bond is not None
    bond.SetStereoAtoms(2, 3)
    bond.SetStereo(Chem.BondStereo.STEREOZ)
    mol.UpdatePropertyCache(strict=False)
    return mol


@pytest.mark.parametrize(
    ("smiles", "expected_heavy_atom_count", "expected_smiles"),
    (
        ("[H-].[C-]#[O+]", 2, "[C-]#[O+]"),
        ("[2H]C", 1, "C"),
    ),
)
def test_heavy_atom_helpers_ignore_nonremovable_hydrogens(
    smiles: str,
    expected_heavy_atom_count: int,
    expected_smiles: str,
) -> None:
    mol = _unsanitized_mol(smiles)

    # RDKit RemoveHs() keeps these hydrogens, so it is not a valid heavy-atom count.
    old_remove_hs_count = Chem.RemoveHs(Chem.Mol(mol), sanitize=False).GetNumAtoms()

    heavy_only, source_indices = _copy_without_hydrogens_with_source_indices(mol)

    assert old_remove_hs_count > expected_heavy_atom_count
    assert _heavy_atom_count(mol) == expected_heavy_atom_count
    assert heavy_only.GetNumAtoms() == expected_heavy_atom_count
    assert len(source_indices) == expected_heavy_atom_count
    assert all(atom.GetAtomicNum() != 1 for atom in heavy_only.GetAtoms())
    assert _safe_canonical_smiles(mol) == expected_smiles


def test_copy_without_hydrogens_defers_bond_stereo_until_all_bonds_exist() -> None:
    mol = _stereo_regression_mol()

    heavy_only, source_indices = _copy_without_hydrogens_with_source_indices(mol)

    assert heavy_only.GetNumAtoms() == 4
    assert source_indices == [0, 1, 2, 3]
    bond = heavy_only.GetBondBetweenAtoms(0, 1)
    assert bond is not None
    assert bond.GetStereo() == Chem.BondStereo.STEREOZ
    assert list(bond.GetStereoAtoms()) == [2, 3]


@pytest.mark.parametrize(
    ("candidate", "reference", "entry", "decision", "raw", "status"),
    [
        ("CCO", "CCO", None, "equivalent", True, "valid"),
        ("COC", "CCO", None, "not_equivalent", False, "valid"),
        (
            "[NH3+]CC1=N[N-][N]C1=CO",
            "[NH3+]Cc1[n-]nnc1[CH]O",
            {"status": "accepted_reference_issue", "reason": "preferable"},
            "equivalent",  # Native resonance witness is independently replayed in its tests.
            True,
            "unresolved",
        ),
        (
            "[O][O]",
            "O=O",
            {"status": "accepted_reference_issue", "reason": "preferable"},
            "inconclusive",
            None,
            "unresolved",
        ),
        (
            "CCO",
            "CCO",
            {"status": "accepted_ambiguous", "reason": "ambiguous"},
            "equivalent",
            True,
            "unresolved",
        ),
        (
            "COC",
            "CCO",
            {"status": "accepted_reference_issue", "reason": "preferable"},
            "not_equivalent",
            False,
            "unresolved",
        ),
        (
            "CCO",
            "CCO",
            {
                "status": "accepted_reference_issue",
                "reason": "confirmed",
                "reference_answer_wrong": True,
            },
            "equivalent",
            True,
            "excluded",
        ),
        ("C", "CCO", None, "not_equivalent", False, "excluded"),
    ],
)
def test_reference_validity_never_corrects_raw_decisions(
    tmp_path, monkeypatch, candidate, reference, entry, decision, raw, status
) -> None:
    candidate_mol = Chem.AddHs(Chem.MolFromSmiles(candidate))
    atoms = [atom.GetSymbol() for atom in candidate_mol.GetAtoms()]
    (tmp_path / "case.xyz").write_text(
        f"{len(atoms)}\n\n" + "".join(f"{symbol} 0 0 0\n" for symbol in atoms)
    )

    def skip_uff(mol):
        raise RuntimeError("UFF disabled")

    monkeypatch.setattr(regression, "_organic_uff_energy_kj_mol", skip_uff)
    result = regression._process_row(
        1,
        {"id": "case", "smiles": reference, "charge": "0"},
        xyz_dir=tmp_path,
        backend="python",
        spin_source="closed_shell",
        manual_whitelist={"case": entry} if entry is not None else {},
        precomputed_reconstruction=(candidate_mol, None, None),
    )
    assert result["equivalence_decision"] == decision
    assert result["equivalent"] is raw
    assert result["strict_equivalent"] is raw
    assert result["effective_equivalent"] is raw
    assert result["reference_status"] == status


def test_valid_reference_failure_stays_in_official_denominator(tmp_path) -> None:
    (tmp_path / "case.xyz").write_text("1\n\nHe 0 0 0\n")
    result = regression._process_row(
        1,
        {"id": "case", "smiles": "[He]", "charge": "0"},
        xyz_dir=tmp_path,
        backend="python",
        spin_source="closed_shell",
        manual_whitelist={},
        precomputed_reconstruction=(None, "failed", "solver"),
    )
    counts = Counter()
    regression._update_official_counts(counts, result)
    assert counts["total_count"] == 1
    assert counts["reference_valid_count"] == 1
    assert counts["runtime_failure_count"] == 1
    assert counts["success_count"] == 0


def test_confirmed_csv_reference_exclusion_preserves_raw_equivalence(tmp_path, monkeypatch) -> None:
    molecule = Chem.AddHs(Chem.MolFromSmiles("C"))
    symbols = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    (tmp_path / "case.xyz").write_text(
        f"{len(symbols)}\n\n" + "".join(f"{symbol} 0 0 0\n" for symbol in symbols)
    )

    def skip_uff(mol):
        raise RuntimeError("UFF disabled")

    monkeypatch.setattr(regression, "_organic_uff_energy_kj_mol", skip_uff)
    result = regression._process_row(
        1,
        {
            "id": "case",
            "smiles": "C",
            "charge": "0",
            "reference_answer_wrong": "true",
            "reference_answer_reason": "Confirmed reference-quality record",
        },
        xyz_dir=tmp_path,
        backend="python",
        spin_source="closed_shell",
        manual_whitelist={},
        precomputed_reconstruction=(molecule, None, None),
    )
    counts = Counter()
    regression._update_official_counts(counts, result)
    assert result["equivalence_decision"] == "equivalent"
    assert result["reference_status"] == "excluded"
    assert result["reference_answer_wrong"] is True
    assert counts["reference_excluded_count"] == 1
    assert counts["reference_valid_count"] == 0
    assert counts["success_count"] == 0
