"""NO+ bookkeeping must preserve a monoanionic beta-diketiminate ligand."""

import numpy as np
import pytest
from rdkit import Chem

from molgr import MolGRConfig, _core
from molgr.interface import xyz_to_rdmol
from molgr.utils.converter import get_atom_unpaired_electrons
from scripts.tmqmg_fixture_data import fixture


@pytest.fixture(scope="module")
def nucvoz():
    return dict(fixture("NUCVOZ"))


def config():
    result = MolGRConfig()
    result.cpp_backend.max_threads = 1
    result.cpp_backend.target_bucket_parallel_max_threads = 1
    return result


def test_cpp_neutral_nitrosyl_core_preserves_monoanionic_beta_diketiminate(nucvoz):
    assert (nucvoz["charge"], nucvoz["multiplicity"]) == (-1, 1)
    mol = xyz_to_rdmol(nucvoz["xyz"], -1, 1, backend="cpp", config=config())
    xyz = Chem.MolFromXYZBlock(nucvoz["xyz"])
    assert [a.GetAtomicNum() for a in mol.GetAtoms()] == [a.GetAtomicNum() for a in xyz.GetAtoms()]
    assert mol.GetConformer().GetPositions() == pytest.approx(xyz.GetConformer().GetPositions())
    assert Chem.GetFormalCharge(mol) == -1
    assert sum(get_atom_unpaired_electrons(a) for a in mol.GetAtoms()) == 0
    assert not any(a.GetNumImplicitHs() or a.GetNumExplicitHs() for a in mol.GetAtoms())
    assert mol.GetAtomWithIdx(0).GetFormalCharge() == -2
    organic = Chem.RWMol(mol)
    organic.RemoveAtom(0)
    no_count = 0
    ancillary_charges = []
    for indices in Chem.GetMolFrags(organic):
        atoms = [organic.GetAtomWithIdx(i) for i in indices]
        charge = sum(a.GetFormalCharge() for a in atoms)
        if sorted(a.GetAtomicNum() for a in atoms) == [7, 8]:
            no_count += 1
            assert charge == 1
            assert organic.GetBondBetweenAtoms(*indices).GetBondType() == Chem.BondType.TRIPLE
        else:
            ancillary_charges.append(charge)
    assert no_count == 2
    assert ancillary_charges == [-1]
    # Check the ligand's total electron allocation, not a specific resonance
    # contributor or choice of which N carries its negative charge.


@pytest.mark.parametrize("variant", ["original", "bent", "long"])
def test_cpp_neutral_nitrosyl_core_requires_no_geometry_and_rejects_extra_reduction(
    nucvoz, variant
):
    mol = Chem.MolFromXYZBlock(nucvoz["xyz"])
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    for nitrogen, oxygen in ((4, 1), (6, 5)):
        vector = positions[oxygen] - positions[nitrogen]
        if variant == "bent":
            vector = np.cross(positions[0] - positions[nitrogen], [0, 0, 1])
            vector = vector / np.linalg.norm(vector) * 1.20
        elif variant == "long":
            vector = vector / np.linalg.norm(vector) * 1.32
        conf.SetAtomPosition(oxygen, positions[nitrogen] + vector)
    pool = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        Chem.MolToXYZBlock(mol), -1, 0, config=config()
    )["candidates"]
    candidates = {c["metal_states"][0]["valence"]: c for c in pool}
    balanced = candidates[-2]
    assert balanced["metal_discordance_negative_metal_penalty"] == 1
    assert balanced["metal_discordance_inner_visible_same_sign_charge_penalty"] == 1
    assert balanced["metal_nitrosonium_formal_charge_exemption"] == (
        2 if variant == "original" else 0
    )
    # Fe(-4) goes beyond compensation of the two NO+ ligands and must not
    # receive this exemption, even when their geometry remains valid.
    assert candidates[-4]["metal_nitrosonium_formal_charge_exemption"] == 0
    if variant != "original":
        assert all(c["metal_nitrosonium_formal_charge_exemption"] == 0 for c in pool)


def test_cpp_neutral_nitrosyl_core_rejects_cationic_ancillary_component(nucvoz):
    # Alternate constraints are only a boundary probe; the stored input is -1.
    pool = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        nucvoz["xyz"], 1, 0, config=config()
    )["candidates"]
    assert any(c["metal_states"][0]["valence"] == -2 for c in pool)
    assert all(c["metal_nitrosonium_formal_charge_exemption"] == 0 for c in pool)
