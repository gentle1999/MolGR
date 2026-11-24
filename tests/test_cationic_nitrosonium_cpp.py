"""Multiple NO+ ligands must not force reduction of neutral spectator ligands."""

import numpy as np
import pytest
from rdkit import Chem

from molgr import MolGRConfig, _core
from molgr.interface import xyz_to_rdmol
from molgr.utils.converter import get_atom_unpaired_electrons
from scripts.tmqmg_fixture_data import fixture


@pytest.fixture(scope="module")
def inputs():
    return {cid: dict(fixture(cid)) for cid in ("EBAKOJ", "XIJHAB", "NDPECO", "OPUSOJ")}


def config():
    result = MolGRConfig()
    result.cpp_backend.max_threads = 1
    result.cpp_backend.target_bucket_parallel_max_threads = 1
    return result


@pytest.mark.parametrize(
    "case_id, metal_charge, aromatic_atoms",
    [("EBAKOJ", -2, 5), ("XIJHAB", -2, 11), ("NDPECO", -1, 24), ("OPUSOJ", -1, 12)],
)
def test_cpp_cationic_nitrosyl_preserves_neutral_nhc_phosphine_and_amine(
    inputs, case_id, metal_charge, aromatic_atoms
):
    row = inputs[case_id]
    assert row["charge"] == row["multiplicity"] == 1
    mol = xyz_to_rdmol(row["xyz"], 1, 1, backend="cpp", kekulize=False, config=config())
    xyz = Chem.MolFromXYZBlock(row["xyz"])
    assert [a.GetAtomicNum() for a in mol.GetAtoms()] == [a.GetAtomicNum() for a in xyz.GetAtoms()]
    assert mol.GetConformer().GetPositions() == pytest.approx(xyz.GetConformer().GetPositions())
    assert not any(a.GetNumImplicitHs() or a.GetNumExplicitHs() for a in mol.GetAtoms())
    assert Chem.GetFormalCharge(mol) == 1
    assert sum(get_atom_unpaired_electrons(a) for a in mol.GetAtoms()) == 0
    assert mol.GetAtomWithIdx(0).GetFormalCharge() == metal_charge
    assert sum(a.GetIsAromatic() for a in mol.GetAtoms()) == aromatic_atoms
    # The remaining organic ligand is neutral overall, including charge-separated
    # NHC contributors. The formal metal state follows the NO+ convention.
    without_metal = Chem.RWMol(mol)
    without_metal.RemoveAtom(0)
    fragments = Chem.GetMolFrags(without_metal)
    charges = []
    no_count = 0
    for indices in fragments:
        atoms = [without_metal.GetAtomWithIdx(i) for i in indices]
        charge = sum(a.GetFormalCharge() for a in atoms)
        if sorted(a.GetAtomicNum() for a in atoms) == [7, 8]:
            no_count += 1
            assert charge == 1
            assert without_metal.GetBondBetweenAtoms(*indices).GetBondType() == Chem.BondType.TRIPLE
        else:
            charges.append(charge)
    assert no_count == 1 - metal_charge
    assert charges and all(charge == 0 for charge in charges)


@pytest.mark.parametrize("variant, expected_credit", [("original", 2), ("bent", 0), ("long", 0)])
def test_cpp_cationic_nitrosonium_credit_retains_geometry_guards(inputs, variant, expected_credit):
    mol = Chem.MolFromXYZBlock(inputs["EBAKOJ"]["xyz"])
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    for nitrogen, oxygen in ((4, 1), (5, 2), (6, 3)):
        vector = positions[oxygen] - positions[nitrogen]
        if variant == "bent":
            vector = np.cross(positions[0] - positions[nitrogen], [0, 0, 1])
            vector = vector / np.linalg.norm(vector) * 1.16
        elif variant == "long":
            vector = vector / np.linalg.norm(vector) * 1.30
        conf.SetAtomPosition(oxygen, positions[nitrogen] + vector)
    pool = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        Chem.MolToXYZBlock(mol), 1, 0, config=config()
    )["candidates"]
    candidate = next(c for c in pool if c["metal_states"][0]["valence"] == -2)
    assert candidate["metal_discordance_negative_metal_penalty"] == 1
    assert candidate["metal_nitrosonium_formal_charge_exemption"] == expected_credit
    if not expected_credit:
        assert all(c["metal_nitrosonium_formal_charge_exemption"] == 0 for c in pool)


@pytest.mark.parametrize("charge", [-1, 3])
def test_cpp_cationic_no_credit_rejects_uncompensated_charge_constraints(inputs, charge):
    # Probe alternative constraints without changing the stored molecular input.
    # Three NO+ cannot balance a negative metal at charge +3 with neutral ligands;
    # At charge -1, none of this input's spin-compatible Fe candidates has a
    # neutral Fe(NO)3 core; it also has no CO to support the anionic branch.
    pool = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        inputs["EBAKOJ"]["xyz"], charge, 0, config=config()
    )["candidates"]
    assert pool
    assert all(c["metal_nitrosonium_formal_charge_exemption"] == 0 for c in pool)
