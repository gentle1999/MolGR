"""NO+ bookkeeping must also preserve resolved anionic ancillary ligands."""

import pytest
from rdkit import Chem

from molgr import MolGRConfig, _core
from molgr.interface import xyz_to_rdmol
from scripts.tmqmg_fixture_data import fixture


@pytest.fixture(scope="module")
def yerwex():
    return dict(fixture("YERWEX"))


def test_cpp_yerwex_preserves_short_alkene_and_ester_carbonyl(yerwex):
    config = MolGRConfig()
    config.cpp_backend.max_threads = 1
    config.cpp_backend.target_bucket_parallel_max_threads = 1
    mol = xyz_to_rdmol(yerwex["xyz"], 0, 1, backend="cpp", config=config)
    xyz = Chem.MolFromXYZBlock(yerwex["xyz"])
    assert [a.GetAtomicNum() for a in mol.GetAtoms()] == [a.GetAtomicNum() for a in xyz.GetAtoms()]
    assert mol.GetConformer().GetPositions() == pytest.approx(xyz.GetConformer().GetPositions())
    assert not any(a.GetNumImplicitHs() or a.GetNumExplicitHs() for a in mol.GetAtoms())
    assert Chem.GetFormalCharge(mol) == 0
    assert sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()) == 0
    # The 1.366 A C=C and 1.226 A C=O must survive the metal selection.
    assert mol.GetBondBetweenAtoms(5, 6).GetBondType() == Chem.BondType.DOUBLE
    assert mol.GetBondBetweenAtoms(8, 9).GetBondType() == Chem.BondType.DOUBLE
    assert mol.GetAtomWithIdx(5).GetFormalCharge() == -1
    assert mol.GetAtomWithIdx(6).GetFormalCharge() == 0
    assert mol.GetAtomWithIdx(0).GetFormalCharge() == -1
    for n, o in [(1, 2), (3, 4)]:
        assert mol.GetBondBetweenAtoms(n, o).GetBondType() == Chem.BondType.TRIPLE
        assert (
            mol.GetAtomWithIdx(n).GetFormalCharge() + mol.GetAtomWithIdx(o).GetFormalCharge() == 1
        )


@pytest.mark.parametrize("charge", [0, -2])
def test_cpp_anionic_no_compensation_keeps_raw_costs_and_neutral_scope(yerwex, charge):
    config = MolGRConfig()
    config.cpp_backend.max_threads = 1
    config.cpp_backend.target_bucket_parallel_max_threads = 1
    pool = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        yerwex["xyz"], charge, 0, config=config
    )["candidates"]
    assert pool
    key = "metal_nitrosonium_formal_charge_exemption"
    if charge:
        assert all(c[key] == 0 for c in pool)
    else:
        chosen = next(c for c in pool if c["selected"])
        assert chosen["metal_discordance_negative_metal_penalty"] == 0.5
        assert chosen["metal_discordance_inner_visible_same_sign_charge_penalty"] == 1
        assert chosen[key] == 1.5
        assert chosen["metal_discordance_structural_count"] == 0
