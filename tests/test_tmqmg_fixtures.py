from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Literal

import pytest
from rdkit import Chem

from molgr import ReconstructionBatchRequest
from molgr.config import MolGRConfig
from molgr.interface import xyz_to_rdmol
from molgr.utils.equivalence import (
    EquivalenceDecision,
    evaluate_equivalence,
)
from scripts.tmqmg_fixture_data import fixture, reference_graph, regression_reference_graph


_FIXTURE_ROOT = Path(__file__).parent / "data" / "tmqmg"
_CONNECTIVITY_EXPECTATIONS = {
    case["id"]: case
    for case in json.loads((_FIXTURE_ROOT / "fixture_sources.json").read_text(encoding="utf-8"))[
        "connectivity"
    ]
}


def _manifest(group: str) -> list[dict[str, str]]:
    with (_FIXTURE_ROOT / group / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("backend", ["cpp", "python"])
def test_abevag_bent_terminal_nitriles(backend: Literal["cpp", "python"]) -> None:
    case = fixture("ABEVAG")
    mol = xyz_to_rdmol(case["xyz"], case["charge"], case["multiplicity"], backend=backend)
    # Three acetonitriles approach La over 60° from their ideal C≡N extensions.
    donors = {a.GetIdx() for a in mol.GetAtomWithIdx(0).GetNeighbors()}
    assert donors == set(range(1, 9))  # Three bromides and five nitrile nitrogens.


@pytest.mark.parametrize("backend", ["cpp", "python"])
def test_acekau_side_on_nitrile(backend: Literal["cpp", "python"]) -> None:
    case = fixture("ACEKAU")
    mol = xyz_to_rdmol(case["xyz"], case["charge"], case["multiplicity"], backend=backend)
    # The nickel is 1.87 Å from the nitrile carbon and 1.93 Å from nitrogen.
    assert mol.GetBondBetweenAtoms(3, 4).GetBondType() == Chem.BondType.TRIPLE
    for donor in (3, 4):
        assert mol.GetBondBetweenAtoms(donor, 0).GetBondType() == Chem.BondType.DATIVE


@pytest.mark.parametrize("backend", ["cpp", "python"])
def test_vivniz_iron_binds_chelate_nitrogens(backend: Literal["cpp", "python"]) -> None:
    case = fixture("VIVNIZ")
    mol = xyz_to_rdmol(case["xyz"], case["charge"], case["multiplicity"], backend=backend)
    # The Fe–N–C=C–N ring contains ligand carbons 10 and 11. Both are
    # equidistant from Fe, which sits in the ligand plane.
    donors = {a.GetIdx() for a in mol.GetAtomWithIdx(49).GetNeighbors()}
    assert donors == {41, 42, 43, 44, 45}
    assert all(mol.GetBondBetweenAtoms(i, 49).GetBondType() == Chem.BondType.DATIVE for i in donors)


@pytest.mark.parametrize("backend", ["cpp", "python"])
@pytest.mark.parametrize(
    "case_id,donor,element,expected",
    [
        ("APUJED", 10, "O", False),  # Ru approaches O from the N–O bond side (40°).
        ("COGXOM", 4, "O", True),  # Short, tilted Ru–O=C contact, 2.02 Å.
        ("DIZMAA", 5, "S", True),  # Mo approaches the terminal S⁻ across C–S (100°).
        ("DIZQIM", 5, "Se", True),  # The Se analogue has an Mo–Se–C angle of 107°.
        ("HOCMAN", 3, "S", True),  # Hg approaches thiocarbonyl sulfur at 89°.
    ],
)
def test_terminal_donor_approach(case_id, donor, element, expected, backend):
    case = fixture(case_id)
    mol = xyz_to_rdmol(case["xyz"], case["charge"], case["multiplicity"], backend=backend)
    assert mol.GetAtomWithIdx(donor).GetSymbol() == element
    assert (mol.GetBondBetweenAtoms(0, donor) is not None) == expected


@pytest.mark.parametrize("case", _manifest("connectivity"), ids=lambda case: case["case_id"])
def test_tmqmg_reviewed_connectivity_and_backend_parity(case: dict[str, str]) -> None:
    """Review only the specified edges, without endorsing source electronic states."""
    xyz = (_FIXTURE_ROOT / "connectivity" / case["xyz_file"]).read_text(encoding="utf-8")
    expected = Chem.MolFromXYZBlock(xyz)
    config = MolGRConfig()
    config.cpp_backend.max_threads = 1
    config.cpp_backend.target_bucket_parallel_max_threads = 1
    connectivity = _CONNECTIVITY_EXPECTATIONS[case["case_id"]]
    outputs = []
    for backend in ("cpp", "python"):
        rebuilt = xyz_to_rdmol(
            xyz, int(case["charge"]), int(case["multiplicity"]), backend=backend, config=config
        )
        assert rebuilt.GetNumAtoms() == expected.GetNumAtoms()
        assert [a.GetAtomicNum() for a in rebuilt.GetAtoms()] == [
            a.GetAtomicNum() for a in expected.GetAtoms()
        ]
        assert all(a.GetNumImplicitHs() == 0 for a in rebuilt.GetAtoms())
        assert Chem.GetFormalCharge(rebuilt) == int(case["charge"])
        assert sum(a.GetNumRadicalElectrons() for a in rebuilt.GetAtoms()) == 0
        for i in range(expected.GetNumAtoms()):
            assert tuple(rebuilt.GetConformer().GetAtomPosition(i)) == pytest.approx(
                tuple(expected.GetConformer().GetAtomPosition(i)), abs=1e-7
            )
            if expected.GetAtomWithIdx(i).GetAtomicNum() == 1:
                # These reviewed inputs have localized, explicit H atoms.
                # Restoring a heavy bond must not make charge recovery detach
                # a methyl hydrogen, as happened initially for FETBOY.
                parent = min(
                    (a.GetIdx() for a in expected.GetAtoms() if a.GetAtomicNum() != 1),
                    key=lambda j: (
                        expected.GetConformer().GetAtomPosition(i)
                        - expected.GetConformer().GetAtomPosition(j)
                    ).Length(),
                )
                assert [a.GetIdx() for a in rebuilt.GetAtomWithIdx(i).GetNeighbors()] == [parent]
        for i, j in connectivity["required_edges"]:
            assert rebuilt.GetBondBetweenAtoms(i, j) is not None, (backend, i, j)
        for i, j in connectivity["forbidden_edges"]:
            assert rebuilt.GetBondBetweenAtoms(i, j) is None, (backend, i, j)
        outputs.append(Chem.MolToSmiles(rebuilt))
    assert outputs[0] == outputs[1]


_SULFUR_REVIEW_EXPECTATIONS = {
    group: {row["id"]: row for row in rows}
    for group, rows in json.loads(
        (_FIXTURE_ROOT / "fixture_sources.json").read_text(encoding="utf-8")
    ).items()
    if group in ("sulfur_electronic_assignment", "electronic_validity")
}


sulfur_cases = pytest.mark.parametrize(
    "group,case,frame",
    [
        pytest.param(
            group,
            case,
            ReconstructionBatchRequest(
                (_FIXTURE_ROOT / group / case["xyz_file"]).read_text(encoding="utf-8"),
                int(case["charge"]),
                int(case["multiplicity"]),
            ),
            id=f"{group}-{case['case_id']}",
        )
        for group, expectations in _SULFUR_REVIEW_EXPECTATIONS.items()
        for case in _manifest(group)
        if expectations[case["case_id"]].get("assert_published_graph")
        or expectations[case["case_id"]].get("assert_reviewed_graph")
    ],
)


@pytest.mark.parametrize("backend", ["cpp", "python"])
@sulfur_cases
def test_sulfur_assignments_match_reference_target_and_local_controls(
    group, case, frame, backend, rebuild_fixture
):
    expected = _SULFUR_REVIEW_EXPECTATIONS[group][case["case_id"]]
    geometry = Chem.MolFromXYZBlock(frame.xyz_block)
    mol = rebuild_fixture(frame, backend)
    perceived = Chem.Mol(mol)
    Chem.SanitizeMol(perceived)
    assert [a.GetAtomicNum() for a in mol.GetAtoms()] == [
        a.GetAtomicNum() for a in geometry.GetAtoms()
    ]
    assert all(a.GetNumImplicitHs() == 0 for a in mol.GetAtoms())
    assert Chem.GetFormalCharge(mol) == int(case["charge"])
    assert sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()) == 0
    for i in range(mol.GetNumAtoms()):
        assert tuple(mol.GetConformer().GetAtomPosition(i)) == pytest.approx(
            tuple(geometry.GetConformer().GetAtomPosition(i)), abs=1e-7
        )
    assert fixture(case["case_id"])["benchmark"]
    reviewed = expected.get("assert_reviewed_graph", False)
    reference = (
        regression_reference_graph(case["case_id"])
        if reviewed
        else reference_graph(case["case_id"])
    )
    comparison = evaluate_equivalence(
        mol,
        reference,
        use_chirality=False,
        ignore_metal_coordination_bonds=not reviewed,
        max_resonance=100000,
    )
    assert comparison.decision is EquivalenceDecision.EQUIVALENT, comparison.reason
    if "metal_formal_charge" in expected:
        metal = next(a for a in mol.GetAtoms() if a.GetSymbol() == case["metal_center"])
        assert metal.GetFormalCharge() == expected["metal_formal_charge"]
        for i, charge in expected["required_atom_charges_1based"]:
            assert mol.GetAtomWithIdx(i - 1).GetFormalCharge() == charge
        for i, j, order in expected["required_bonds_1based"]:
            # Perceive aromatic rings on a copy, preserving the returned Kekulé
            # orders for controls that require a localized single or double bond.
            graph = perceived if order == "AROMATIC" else mol
            bond = graph.GetBondBetweenAtoms(i - 1, j - 1)
            assert bond is not None and str(bond.GetBondType()) == order
        if "metal_neighbors_1based" in expected:
            assert (
                sorted(a.GetIdx() + 1 for a in metal.GetNeighbors())
                == expected["metal_neighbors_1based"]
            )


@sulfur_cases
def test_sulfur_backends_have_identical_canonical_output(group, case, frame, rebuild_fixture):
    """Keep the strict control independent of release-policy graph matching."""
    outputs = []
    for backend in ("cpp", "python"):
        mol = rebuild_fixture(frame, backend)
        for atom in mol.GetAtoms():
            if atom.HasProp("smilesSymbol"):
                atom.ClearProp("smilesSymbol")
        outputs.append(Chem.MolToSmiles(mol, allHsExplicit=True))
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("backend", ["cpp", "python"])
@pytest.mark.parametrize("case", _manifest("reconstruction"), ids=lambda case: case["case_id"])
def test_tmqmg_reconstruction_conserves_input_and_matches_eligible_reference(
    case: dict[str, str], backend: Literal["cpp", "python"]
) -> None:
    xyz_path = _FIXTURE_ROOT / "reconstruction" / case["xyz_file"]
    xyz_block = xyz_path.read_text(encoding="utf-8")
    published = fixture(case["case_id"])
    rebuilt = xyz_to_rdmol(
        xyz_block,
        int(case["charge"]),
        int(case["multiplicity"]),
        backend=backend,
        make_dative_bonds=True,
        make_stereochemistry=True,
    )
    assert rebuilt.GetNumAtoms() == int(xyz_block.splitlines()[0])
    assert all(atom.GetNumImplicitHs() == 0 for atom in rebuilt.GetAtoms())
    assert Chem.GetFormalCharge(rebuilt) == int(case["charge"])
    assert published["benchmark"]
    result = evaluate_equivalence(
        rebuilt,
        reference_graph(case["case_id"]),
        use_chirality=False,
        ignore_metal_coordination_bonds=True,
        max_resonance=100000,
    )
    assert result.decision == EquivalenceDecision.EQUIVALENT, f"{case['case_id']}: {result.reason}"


def test_tmqmg_source_issue_fixtures_are_separate_and_documented() -> None:
    reconstruction_ids = {case["case_id"] for case in _manifest("reconstruction")}
    source_issues = _manifest("source_issues")

    assert source_issues
    assert reconstruction_ids.isdisjoint(case["case_id"] for case in source_issues)
    assert all(case["selection_reason"].strip() for case in source_issues)
    assert all(
        (_FIXTURE_ROOT / "source_issues" / case["xyz_file"]).exists() for case in source_issues
    )


def test_abarax_now_has_a_published_reference() -> None:
    case = fixture("ABARAX")
    assert case["benchmark"]
    assert reference_graph("ABARAX").GetNumAtoms() == int(case["xyz"].splitlines()[0])


@pytest.mark.parametrize("case", _manifest("source_issues"), ids=lambda case: case["case_id"])
def test_historical_source_issues_follow_release_eligibility(case):
    published = fixture(case["case_id"])
    assert (case["benchmark"] == "true") == published["benchmark"]
    assert case["exclusion"] == (published["exclusion"] or "")
    assert published["benchmark"]
    graph = reference_graph(case["case_id"])
    xyz = Chem.MolFromXYZBlock(published["xyz"])
    assert [a.GetAtomicNum() for a in graph.GetAtoms()] == [
        a.GetAtomicNum() for a in xyz.GetAtoms()
    ]


@pytest.mark.parametrize(
    "case", _manifest("gold_electronic_assignment"), ids=lambda c: c["case_id"]
)
def test_tmqmg_reviewed_gold_assignments_and_backend_parity(case: dict[str, str]) -> None:
    """Check local gold chemistry and backend parity, not a whole-target score."""
    cid = case["case_id"]
    expectations = {
        row["id"]: row
        for row in json.loads((_FIXTURE_ROOT / "fixture_sources.json").read_text(encoding="utf-8"))[
            "gold_electronic_assignment"
        ]
    }
    expected = expectations[cid]
    xyz = (_FIXTURE_ROOT / "gold_electronic_assignment" / case["xyz_file"]).read_text(
        encoding="utf-8"
    )
    geometry = Chem.MolFromXYZBlock(xyz)
    config = MolGRConfig()
    config.cpp_backend.max_threads = 1
    config.cpp_backend.target_bucket_parallel_max_threads = 1
    outputs = []
    for backend in ("cpp", "python"):
        mol = xyz_to_rdmol(
            xyz, int(case["charge"]), int(case["multiplicity"]), backend=backend, config=config
        )
        Chem.SanitizeMol(Chem.Mol(mol))
        assert [a.GetAtomicNum() for a in mol.GetAtoms()] == [
            a.GetAtomicNum() for a in geometry.GetAtoms()
        ]
        assert all(a.GetNumImplicitHs() == 0 for a in mol.GetAtoms())
        assert Chem.GetFormalCharge(mol) == int(case["charge"])
        assert sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()) == 0
        for i in range(mol.GetNumAtoms()):
            assert tuple(mol.GetConformer().GetAtomPosition(i)) == pytest.approx(
                tuple(geometry.GetConformer().GetAtomPosition(i)), abs=1e-7
            )
        gold = next(a for a in mol.GetAtoms() if a.GetAtomicNum() == 79)
        assert gold.GetFormalCharge() == expected["gold_formal_charge"]
        assert (
            sorted(a.GetIdx() + 1 for a in gold.GetNeighbors()) == expected["gold_neighbors_1based"]
        )
        if cid in {"CEGPAI", "FETBIS"}:
            borons = [a for a in gold.GetNeighbors() if a.GetAtomicNum() == 5]
            assert len(borons) == 2
            assert all(a.GetFormalCharge() == -1 for a in borons)
            assert all(
                mol.GetBondBetweenAtoms(gold.GetIdx(), a.GetIdx()).GetBondType()
                == Chem.BondType.SINGLE
                for a in borons
            )
        if cid == "RIZHAL":
            assert sorted(a.GetFormalCharge() for a in gold.GetNeighbors()) == [-1, -1, 0, 0]
        if cid == "FATCEJ":
            for sulfur in gold.GetNeighbors():
                assert sulfur.GetAtomicNum() == 16 and sulfur.GetFormalCharge() == -1
                assert all(
                    b.GetBondType() in (Chem.BondType.SINGLE, Chem.BondType.DATIVE)
                    for b in sulfur.GetBonds()
                )
        required_orders = {
            "CIDWEV": [(4, 8, 3)],
            "XOLVOK": [(67, 68, 3), (70, 71, 2)],
            "MIKLOK": [(12, 14, 2)],
            "FETBIS": [(3, 4, 2)],
        }
        for i, j, order in required_orders.get(cid, []):
            bond = mol.GetBondBetweenAtoms(i - 1, j - 1)
            assert bond is not None and bond.GetBondTypeAsDouble() == order
        outputs.append(Chem.MolToSmiles(mol))
    assert outputs[0] == outputs[1]
