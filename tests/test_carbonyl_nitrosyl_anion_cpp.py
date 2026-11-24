"""Anionic CO/NO complexes must not move NO+ compensation into aryl rings."""

import numpy as np
import pytest
from rdkit import Chem

from molgr import MolGRConfig, _core
from molgr.interface import xyz_to_rdmol
from molgr.utils.converter import get_atom_unpaired_electrons
from scripts.tmqmg_fixture_data import fixture


@pytest.fixture(scope="module")
def inputs():
    return {cid: dict(fixture(cid)) for cid in ("JODGER", "JODGIV")}


def config():
    result = MolGRConfig()
    result.cpp_backend.max_threads = 1
    result.cpp_backend.target_bucket_parallel_max_threads = 1
    return result


@pytest.mark.parametrize("case_id", ["JODGER", "JODGIV"])
def test_cpp_nitrosyl_anion_keeps_neutral_phosphine_and_both_aryl_rings(inputs, case_id):
    row = inputs[case_id]
    mol = xyz_to_rdmol(row["xyz"], -1, 1, backend="cpp", kekulize=False, config=config())
    xyz = Chem.MolFromXYZBlock(row["xyz"])
    assert row["charge"] == -1 and row["multiplicity"] == 1
    assert [a.GetAtomicNum() for a in mol.GetAtoms()] == [a.GetAtomicNum() for a in xyz.GetAtoms()]
    assert mol.GetConformer().GetPositions() == pytest.approx(xyz.GetConformer().GetPositions())
    assert not any(a.GetNumImplicitHs() or a.GetNumExplicitHs() for a in mol.GetAtoms())
    assert Chem.GetFormalCharge(mol) == -1
    assert sum(get_atom_unpaired_electrons(a) for a in mol.GetAtoms()) == 0
    # Formal Fe(-2) follows from the requested NO+ convention, not a claim
    # that this is the unique physical oxidation-state assignment of {FeNO}.
    assert mol.GetAtomWithIdx(0).GetFormalCharge() == -2
    assert mol.GetBondBetweenAtoms(5, 6).GetBondType() == Chem.BondType.TRIPLE
    assert mol.GetAtomWithIdx(5).GetFormalCharge() == 0
    assert mol.GetAtomWithIdx(6).GetFormalCharge() == 1
    assert all(a.GetFormalCharge() == 0 for a in list(mol.GetAtoms())[7:])
    assert sum(a.GetIsAromatic() for a in mol.GetAtoms()) == 12
    assert mol.GetAtomWithIdx(7).GetAtomicNum() == 15


@pytest.mark.parametrize(
    "variant, expected_credit",
    [("original", 3), ("one_long_CO", 3), ("both_long_CO", 0), ("both_bent_CO", 0), ("bent_NO", 0)],
)
def test_cpp_anion_credit_requires_short_linear_co_and_no(inputs, variant, expected_credit):
    mol = Chem.MolFromXYZBlock(inputs["JODGER"]["xyz"])
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    for carbon, oxygen in ((1, 2), (3, 4)):
        if variant == "one_long_CO" and carbon != 1:
            continue
        vector = positions[oxygen] - positions[carbon]
        if variant.endswith("long_CO"):
            vector = vector / np.linalg.norm(vector) * 1.31
        elif variant == "both_bent_CO":
            vector = np.cross(positions[0] - positions[carbon], [0, 0, 1])
            vector = vector / np.linalg.norm(vector) * 1.18
        conf.SetAtomPosition(oxygen, positions[carbon] + vector)
    if variant == "bent_NO":
        vector = np.cross(positions[0] - positions[5], [0, 0, 1])
        conf.SetAtomPosition(6, positions[5] + vector / np.linalg.norm(vector) * 1.196)

    pool = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        Chem.MolToXYZBlock(mol), -1, 0, config=config()
    )["candidates"]
    candidate = next(c for c in pool if c["metal_states"][0]["valence"] == -2)
    # Preserve the original costs; the narrowly scoped compensation is explicit.
    assert candidate["metal_discordance_negative_metal_penalty"] == 1
    assert candidate["metal_discordance_inner_visible_same_sign_charge_penalty"] == 2
    assert candidate["metal_nitrosonium_formal_charge_exemption"] == expected_credit
    if not expected_credit:
        assert all(c["metal_nitrosonium_formal_charge_exemption"] == 0 for c in pool)
