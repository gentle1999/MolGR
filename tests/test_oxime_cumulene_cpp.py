"""An oxime geometry cannot be resolved as a bent C=N+=O cumulene."""

import math

import pytest
from rdkit import Chem

from molgr import _core
from molgr.utils.converter import rdkit_to_mol_data


@pytest.mark.parametrize(
    "cn,no,angle,allowed",
    [
        (1.30, 1.40, 120, False),
        (1.30, 1.40, 149, False),
        (1.30, 1.40, 151, True),
        (1.30, 1.40, 180, True),
        (1.30, 1.25, 120, True),
        (1.40, 1.40, 120, True),
    ],
)
@pytest.mark.parametrize("rotate", [False, True])
def test_cpp_cumulene_requires_joint_oxime_geometry(cn, no, angle, allowed, rotate):
    mol = Chem.AddHs(Chem.MolFromSmiles("C=[N+]=O"))
    theta = math.radians(angle)
    positions = [
        (cn, 0, 0),
        (0, 0, 0),
        (no * math.cos(theta), no * math.sin(theta), 0),
        (cn + 0.5, 0.9, 0),
        (cn + 0.5, -0.9, 0),
    ]
    conf = Chem.Conformer(len(positions))
    for i, p in enumerate(positions):
        conf.SetAtomPosition(i, (p[1] + 10, p[2] - 30, p[0] + 20) if rotate else p)
    mol.AddConformer(conf)
    data = rdkit_to_mol_data(mol)
    ptr = _core.utils.molecule_data_to_obmol_ptr(data)
    try:
        assert _core.dev.stages.preprocess.validate_omol_ptr(ptr, 1, 0) is allowed
        result = _core.utils.extract_molecule_data(ptr)
        assert [(a.atomic_num, a.formal_charge, a.x, a.y, a.z) for a in result.atoms] == [
            (a.atomic_num, a.formal_charge, a.x, a.y, a.z) for a in data.atoms
        ]
        assert [b.order for b in result.bonds] == [b.order for b in data.bonds]
    finally:
        _core.free_obmol_ptr(ptr)
