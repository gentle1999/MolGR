"""Charge neutralization must not invalidate an adjacent, unchanged N=O bond."""

import pytest
from rdkit import Chem

from molgr import _core
from molgr.utils.converter import rdkit_to_mol_data


@pytest.mark.parametrize(
    "left_cn,right_cn,left_no,translate,protect",
    [
        (1.30, 1.36, 1.40, False, True),
        (1.36, 1.30, 1.40, False, True),
        (1.30, 1.36, 1.40, True, True),
        (1.36, 1.36, 1.40, False, False),
        (1.30, 1.36, 1.25, False, False),
    ],
)
@pytest.mark.parametrize("stage", ["clean_resonances_ptr", "clean_resonances_16_ptr"])
def test_cpp_neutralization_preserves_neighboring_nitrogen_geometry(
    left_cn, right_cn, left_no, translate, protect, stage
):
    mol = Chem.AddHs(Chem.MolFromSmiles("O=[N+]=CC=N[O-]"))
    conf = Chem.Conformer(mol.GetNumAtoms())
    positions = [
        (-left_cn - left_no, 0, 0),
        (-left_cn, 0, 0),
        (0, 0, 0),
        (0.75, 1.30, 0),
        (0.75 + right_cn, 1.30, 0),
        (0.75 + right_cn, 2.70, 0),
        (0.5, -0.9, 0),
        (0.25, 2.20, 0),
    ]
    assert len(positions) == mol.GetNumAtoms()
    for i, p in enumerate(positions):
        conf.SetAtomPosition(i, (p[1] + 30, p[2] - 50, p[0] + 10) if translate else p)
    mol.AddConformer(conf)
    data = rdkit_to_mol_data(mol)
    ptr = _core.utils.molecule_data_to_obmol_ptr(data)
    try:
        assert _core.dev.stages.preprocess.validate_omol_ptr(ptr, 0, 0)
        getattr(_core.dev.stages.clean, stage)(ptr)
        assert _core.dev.stages.preprocess.validate_omol_ptr(ptr, 0, 0)
        result = _core.utils.extract_molecule_data(ptr)
        assert [(a.x, a.y, a.z) for a in result.atoms] == [(a.x, a.y, a.z) for a in data.atoms]
        assert [a.atomic_num for a in result.atoms] == [a.atomic_num for a in data.atoms]
        assert sum(a.formal_charge for a in result.atoms) == 0
        assert all(a.radical_num == 0 for a in result.atoms)
        assert result.atoms[1].formal_charge == (1 if protect else 0)
        assert result.atoms[5].formal_charge == (-1 if protect else 0)
    finally:
        _core.free_obmol_ptr(ptr)
