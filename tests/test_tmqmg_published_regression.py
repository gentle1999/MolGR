from __future__ import annotations

import hashlib

import pytest
from fixture_reconstruction import fixture_marks
from rdkit import Chem

from molgr import ReconstructionBatchRequest
from molgr.interface import xyz_to_rdmol
from molgr.utils.equivalence import evaluate_equivalence
from scripts.tmqmg_fixture_data import (
    fixture,
    reference_graph,
    regression_case_ids,
    regression_reference_graph,
)


# Published inputs with reviewed local targets where available.
# Strict coordination remains tested separately in backend parity/local invariants.
CASES = [fixture(case_id) for case_id in regression_case_ids()]
RECONSTRUCTION_CASES = [
    case for case in CASES if case["reconstruction_scope"] != "spin_coupling_out_of_scope"
]


def test_coupled_singlet_standards_preserve_local_radicals_and_input():
    coupled = [
        case for case in CASES if case["reconstruction_scope"] == "spin_coupling_out_of_scope"
    ]
    assert len(coupled) == 36
    for case in coupled:
        mol = reference_graph(case["id"])
        xyz = Chem.MolFromXYZBlock(case["xyz"])
        assert case["multiplicity"] == mol.GetIntProp("STANDARD_TOTAL_SPIN_MULTIPLICITY") == 1
        assert sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()) == 2
        coupling = mol.GetProp("STANDARD_SPIN_COUPLING")
        if case["id"] in {"BAFSUY", "BAFTAF", "QURGAN"}:
            # Literature supports two ligand radicals in all three cases;
            # the reported singlet magnetic fit applies to BAFSUY only.
            # Preserve input multiplicity without inventing measured ground
            # states for BAFTAF or QURGAN.
            assert coupling == "two_ligand_radicals_with_fixed_input_singlet_constraint"
            expected_doi = (
                "10.1039/C5SC02214B" if case["id"] == "QURGAN" else "10.1134/S1070328411030092"
            )
            assert mol.GetProp("STANDARD_EVIDENCE_DOI") == expected_doi
        else:
            assert coupling == "antiferromagnetic_ligand_radical_pair"
        assert Chem.GetFormalCharge(mol) == case["charge"] == 0
        assert [a.GetAtomicNum() for a in mol.GetAtoms()] == [
            a.GetAtomicNum() for a in xyz.GetAtoms()
        ]
        assert all(not (a.GetNumImplicitHs() or a.GetNumExplicitHs()) for a in mol.GetAtoms())
        for idx in range(mol.GetNumAtoms()):
            assert tuple(mol.GetConformer().GetAtomPosition(idx)) == pytest.approx(
                tuple(xyz.GetConformer().GetAtomPosition(idx)), abs=1e-8
            )
        metals = [a for a in mol.GetAtoms() if a.GetAtomicNum() in (28, 30, 46, 78)]
        assert len(metals) == 1 and metals[0].GetFormalCharge() == 2
        metal = metals[0]
        cut = Chem.RWMol(mol)
        for neighbor in metal.GetNeighbors():
            cut.RemoveBond(metal.GetIdx(), neighbor.GetIdx())
        ligands = [f for f in Chem.GetMolFrags(cut) if metal.GetIdx() not in f]
        radical_ligands = 0
        for ligand in ligands:
            charge = sum(mol.GetAtomWithIdx(i).GetFormalCharge() for i in ligand)
            radicals = sum(mol.GetAtomWithIdx(i).GetNumRadicalElectrons() for i in ligand)
            if radicals:
                assert charge == -1 and radicals == 1
                radical_ligands += 1
            else:
                assert metal.GetAtomicNum() == 30 and charge == 0
        assert radical_ligands == 2
        assert len(ligands) == (3 if metal.GetAtomicNum() == 30 else 2)
        assert (
            evaluate_equivalence(
                reference_graph(case["id"]),
                mol,
                use_chirality=False,
                ignore_metal_coordination_bonds=False,
                max_resonance=100000,
            ).decision.value
            == "equivalent"
        )


published_cases = pytest.mark.parametrize(
    "case,frame",
    [
        pytest.param(
            case,
            ReconstructionBatchRequest(case["xyz"], case["charge"], case["multiplicity"]),
            id=case["id"],
            marks=fixture_marks(case["id"]),
        )
        for case in RECONSTRUCTION_CASES
    ],
)


@pytest.mark.parametrize("backend", ["cpp", "python"])
@published_cases
def test_reconstruction_preserves_input_and_matches_reference_target(
    case, frame, backend, rebuild_fixture
):
    xyz_text = case["xyz"]
    assert hashlib.sha256(xyz_text.encode()).hexdigest() == case["xyz_sha256"]
    xyz = Chem.MolFromXYZBlock(xyz_text)
    assert xyz is not None
    result = rebuild_fixture(frame, backend)
    assert [a.GetAtomicNum() for a in result.GetAtoms()] == [
        a.GetAtomicNum() for a in xyz.GetAtoms()
    ]
    assert all(a.GetNumImplicitHs() == 0 and a.GetNumExplicitHs() == 0 for a in result.GetAtoms())
    assert Chem.GetFormalCharge(result) == case["charge"]
    for idx in range(xyz.GetNumAtoms()):
        assert tuple(result.GetConformer().GetAtomPosition(idx)) == pytest.approx(
            tuple(xyz.GetConformer().GetAtomPosition(idx)), abs=1e-7
        )
    assert sum(a.GetNumRadicalElectrons() for a in result.GetAtoms()) == (case["multiplicity"] - 1)
    reference = regression_reference_graph(case["id"])
    comparison = evaluate_equivalence(
        reference,
        result,
        use_chirality=False,
        ignore_metal_coordination_bonds=not reference.HasProp("REVIEW_STATUS"),
        max_resonance=100000,
    )
    assert comparison.decision.value == "equivalent", (backend, comparison)


@published_cases
def test_backends_match_with_coordination_bonds(case, frame, rebuild_fixture):
    """A separate control, even when one backend misses the published target."""
    parity = evaluate_equivalence(
        rebuild_fixture(frame, "cpp"),
        rebuild_fixture(frame, "python"),
        use_chirality=False,
        ignore_metal_coordination_bonds=False,
        max_resonance=100000,
    )
    assert parity.decision.value == "equivalent", parity.reason


@pytest.mark.parametrize("case_id", ["BASDAC", "BEBQIK", "YURTOW"])
def test_default_search_finds_a_ligand_resonance_witness(case_id):
    """The public budget must reach an electronic proof for these competing atom mappings."""
    case = fixture(case_id)
    rebuilt = xyz_to_rdmol(case["xyz"], case["charge"], case["multiplicity"], backend="cpp")
    result = evaluate_equivalence(regression_reference_graph(case_id), rebuilt)
    assert result.equivalent, result.reason
    assert result.atom_mapping and result.resonance_witness
