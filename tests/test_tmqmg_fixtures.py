from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

import pytest
from rdkit import Chem

from molgr.interface import xyz_to_rdmol
from molgr.utils.equivalence import (
    EquivalenceDecision,
    EquivalenceMethod,
    EquivalenceRelation,
    evaluate_equivalence,
)
from scripts.tmqmg_reference_formula_check import (
    _read_xyz_element_counts,
    _smiles_element_counts_with_h,
)


_FIXTURE_ROOT = Path(__file__).parent / "data" / "tmqmg"
_EXPECTED_RECONSTRUCTION_SMILES = {
    # This fixture exercises MolGR's radical-to-anion charge-localization path.
    # Its published tmQMg representation is identifier-compatible, but newer
    # RDKit releases do not provide a common resonance form.  Pin the actual
    # reconstruction rather than treating identifier-only evidence as proof.
    "ABELOK": "Cc1cc(C)c(-n2c(-c3ccccc3)[s+][c-](->[Au+]<-[Cl-])c2-c2ccccc2)c(C)c1",
}


def _manifest(group: str) -> list[dict[str, str]]:
    with (_FIXTURE_ROOT / group / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("backend", ["cpp", "python"])
@pytest.mark.parametrize("case", _manifest("reconstruction"), ids=lambda case: case["case_id"])
def test_tmqmg_reconstruction_fixture_matches_reference(
    case: dict[str, str], backend: Literal["cpp", "python"]
) -> None:
    xyz_path = _FIXTURE_ROOT / "reconstruction" / case["xyz_file"]
    xyz_block = xyz_path.read_text(encoding="utf-8")
    reference = Chem.MolFromSmiles(case["reference_smiles"], sanitize=False)
    reference.UpdatePropertyCache(strict=False)
    rebuilt = xyz_to_rdmol(
        xyz_block,
        int(case["charge"]),
        1,
        backend=backend,
        make_dative_bonds=True,
        make_stereochemistry=True,
    )
    result = evaluate_equivalence(rebuilt, reference, use_chirality=False)

    expected_reconstruction = _EXPECTED_RECONSTRUCTION_SMILES.get(case["case_id"])
    if expected_reconstruction is not None:
        observed_reconstruction = Chem.MolToSmiles(
            Chem.RemoveHs(rebuilt),
            canonical=True,
            isomericSmiles=True,
        )
        assert observed_reconstruction == expected_reconstruction
        if result.decision == EquivalenceDecision.INCONCLUSIVE:
            assert result.relation == EquivalenceRelation.IDENTIFIER_EQUIVALENCE
            assert result.method == EquivalenceMethod.INCHI_KEY
            assert result.bounded_search is not None
            assert result.bounded_search.attempted
        else:
            assert result.decision == EquivalenceDecision.EQUIVALENT
            assert result.relation == EquivalenceRelation.RESONANCE_EQUIVALENCE
            assert result.method == EquivalenceMethod.RESONANCE
        return

    assert result.decision == EquivalenceDecision.EQUIVALENT, f"{case['case_id']}: {result.reason}"


def test_tmqmg_source_issue_fixtures_are_separate_and_documented() -> None:
    reconstruction_ids = {case["case_id"] for case in _manifest("reconstruction")}
    source_issues = _manifest("source_issues")

    assert source_issues
    assert reconstruction_ids.isdisjoint(case["case_id"] for case in source_issues)
    assert all(case["reason"].strip() for case in source_issues)
    assert all(
        (_FIXTURE_ROOT / "source_issues" / case["xyz_file"]).exists() for case in source_issues
    )


def test_tmqmg_missing_reference_graph_fixture_is_reproducible() -> None:
    missing_references = [
        case
        for case in _manifest("source_issues")
        if case["classification"] == "missing_reference_graph"
    ]

    assert [case["case_id"] for case in missing_references] == ["ABARAX"]
    assert missing_references[0]["reference_smiles"] == ""


def test_tmqmg_manually_accepted_reference_issues_are_explicit() -> None:
    accepted_ids = {
        case["case_id"]
        for case in _manifest("source_issues")
        if case["classification"] == "accepted_reference_issue"
    }

    assert accepted_ids == {"ABATEC", "ABEGOD", "ABETIK", "ABETOQ"}


@pytest.mark.parametrize(
    "case",
    [
        case
        for case in _manifest("source_issues")
        if case["classification"] == "reference_formula_mismatch"
    ],
    ids=lambda case: case["case_id"],
)
def test_tmqmg_reference_formula_mismatch_fixture_is_reproducible(case: dict[str, str]) -> None:
    xyz_path = _FIXTURE_ROOT / "source_issues" / case["xyz_file"]

    assert _read_xyz_element_counts(xyz_path) != _smiles_element_counts_with_h(
        case["reference_smiles"]
    )
