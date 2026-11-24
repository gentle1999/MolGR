"""Long tetrahedral C--O geometry must not become an oxocarbenium."""

import math

import pytest
from rdkit import Chem

from molgr import _core
from molgr.fallback.stages.preprocess import validate_omol
from molgr.utils.converter import mol_data_to_pybel, rdkit_to_mol_data


@pytest.mark.parametrize(
    "smiles,co,pyramidal,allowed",
    [
        ("C=[O+]C", 1.38, True, False),
        ("C=[O+]C", 1.40, True, False),
        ("C=[O+]C", 1.38, False, True),
        ("C=[O+]C", 1.28, True, True),
        ("C=[O+]C", 1.28, False, True),
        ("[CH2-]OC", 1.38, True, True),
        ("[CH2-]OC", 1.38, False, True),
        ("C=O", 1.38, True, True),
    ],
)
@pytest.mark.parametrize("rotate", [False, True])
def test_cpp_oxonium_requires_length_and_pyramidal_geometry(smiles, co, pyramidal, allowed, rotate):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    conf = Chem.Conformer(mol.GetNumAtoms())
    positions = {0: (0, 0, 0), 1: (co, 0, 0)}
    hydrogens = [a.GetIdx() for a in mol.GetAtomWithIdx(0).GetNeighbors() if a.GetAtomicNum() == 1]
    assert len(hydrogens) == 2
    if pyramidal:
        vectors = [(-1 / 3, math.sqrt(8 / 9), 0), (-1 / 3, -math.sqrt(2 / 9), math.sqrt(2 / 3))]
    else:
        vectors = [(-0.5, math.sqrt(3) / 2, 0), (-0.5, -math.sqrt(3) / 2, 0)]
    positions.update(zip(hydrogens, vectors))
    if mol.GetNumAtoms() > 4:
        positions[2] = (co + 0.6, 1.3, 0)
        other_h = [
            a.GetIdx() for a in mol.GetAtomWithIdx(2).GetNeighbors() if a.GetAtomicNum() == 1
        ]
        for i, idx in enumerate(other_h):
            theta = 2 * math.pi * i / len(other_h)
            positions[idx] = (co + 0.6 + math.cos(theta), 1.8, math.sin(theta))
    assert len(positions) == mol.GetNumAtoms()
    for idx, p in positions.items():
        conf.SetAtomPosition(idx, (p[1] + 7, p[2] - 11, p[0] + 19) if rotate else p)
    mol.AddConformer(conf)
    data = rdkit_to_mol_data(mol)
    assert validate_omol(mol_data_to_pybel(data), Chem.GetFormalCharge(mol), 0) is allowed
    ptr = _core.utils.molecule_data_to_obmol_ptr(data)
    try:
        assert (
            _core.dev.stages.preprocess.validate_omol_ptr(ptr, Chem.GetFormalCharge(mol), 0)
            is allowed
        )
        result = _core.utils.extract_molecule_data(ptr)
        assert [(a.atomic_num, a.formal_charge, a.x, a.y, a.z) for a in result.atoms] == [
            (a.atomic_num, a.formal_charge, a.x, a.y, a.z) for a in data.atoms
        ]
        assert [b.order for b in result.bonds] == [b.order for b in data.bonds]
    finally:
        _core.free_obmol_ptr(ptr)
