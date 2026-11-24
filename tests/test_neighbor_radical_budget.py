"""Public budget controls for two independent ligand radical pairs."""

from dataclasses import replace

import pytest
from rdkit import Chem

from molgr import MolGRConfig
from molgr.diagnostics import ReconstructionError
from molgr.interface import xyz_to_rdmol
from molgr.utils.equivalence import evaluate_equivalence
from scripts.tmqmg_fixture_data import fixture, reference_graph


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize("neighbor_budget", [-1, 0, 1, 2, 3])
def test_neighbor_budget_controls_two_ligand_recovery(backend, neighbor_budget):
    config = MolGRConfig()
    config.cpp_backend.max_threads = 1
    config.cpp_backend.target_bucket_parallel_max_threads = 1
    config = replace(
        config,
        resonance=replace(
            config.resonance,
            limited_discrepancy_max_discrepancy=1,
            neighbor_radical_max_discrepancy=neighbor_budget,
        ),
    )
    case = fixture("DETCEL")
    args = {
        "total_charge": case["charge"],
        "spin_multiplicity": case["multiplicity"],
        "config": config,
        "backend": backend,
    }
    if neighbor_budget < 1:
        with pytest.raises(ReconstructionError, match="NO_VALID_ORGANIC_CANDIDATE"):
            xyz_to_rdmol(case["xyz"], **args)
        return
    result = xyz_to_rdmol(case["xyz"], **args)
    comparison = evaluate_equivalence(
        reference_graph("DETCEL"),
        result,
        use_chirality=False,
        ignore_metal_coordination_bonds=False,
        max_resonance=100000,
    )
    if neighbor_budget == 1:
        # Restoring the exceptional Zn(+4) branch permits an alternative,
        # but one ligand-recovery step still cannot recover the Zn(II) gold.
        assert comparison.decision.value == "not_equivalent", comparison
        assert [a.GetFormalCharge() for a in result.GetAtoms() if a.GetAtomicNum() == 30] == [4]
    else:
        assert comparison.decision.value == "equivalent", comparison
    xyz = Chem.MolFromXYZBlock(case["xyz"])
    assert [a.GetAtomicNum() for a in result.GetAtoms()] == [
        a.GetAtomicNum() for a in xyz.GetAtoms()
    ]
    assert all(a.GetNumImplicitHs() == a.GetNumExplicitHs() == 0 for a in result.GetAtoms())
    assert Chem.GetFormalCharge(result) == case["charge"]
    assert sum(a.GetNumRadicalElectrons() for a in result.GetAtoms()) == case["multiplicity"] - 1
    for i in range(result.GetNumAtoms()):
        assert tuple(result.GetConformer().GetAtomPosition(i)) == pytest.approx(
            tuple(xyz.GetConformer().GetAtomPosition(i)), abs=1e-7
        )
