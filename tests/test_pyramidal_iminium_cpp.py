"""Local iminium geometry must preserve supported resonance classes."""

import math

import pytest
from rdkit import Chem

from molgr import MolGRConfig, _core
from molgr.fallback.stages.preprocess import validate_omol
from molgr.interface import xyz_to_rdmol
from molgr.utils.converter import mol_data_to_pybel, rdkit_to_mol_data
from molgr.utils.equivalence import evaluate_equivalence
from scripts.tmqmg_fixture_data import fixture, reference_graph


@pytest.mark.parametrize(
    "smiles,cn,pyramidal,allowed",
    [
        ("C=[N+](C)C", 1.47, True, False),
        ("C=[N+](C)C", 1.43, True, False),
        ("C=[N+](C)C", 1.47, False, True),
        ("C=[N+](C)C", 1.32, True, True),
        ("C=[N+](C)C", 1.32, False, True),
        ("[CH2-]N(C)C", 1.47, True, True),
        ("[CH2-]N(C)C", 1.47, False, True),
        ("C=NC", 1.47, True, True),
    ],
)
@pytest.mark.parametrize("rotate", [False, True])
def test_cpp_iminium_requires_both_long_bond_and_pyramidal_carbon(
    smiles, cn, pyramidal, allowed, rotate
):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    positions = {0: (0, 0, 0), 1: (cn, 0, 0)}
    hydrogens = [a.GetIdx() for a in mol.GetAtomWithIdx(0).GetNeighbors() if a.GetAtomicNum() == 1]
    assert len(hydrogens) == 2
    if pyramidal:
        vectors = [(-1 / 3, math.sqrt(8 / 9), 0), (-1 / 3, -math.sqrt(2 / 9), math.sqrt(2 / 3))]
    else:
        vectors = [(-0.5, math.sqrt(3) / 2, 0), (-0.5, -math.sqrt(3) / 2, 0)]
    positions.update(zip(hydrogens, vectors))
    branches = [a for a in mol.GetAtomWithIdx(1).GetNeighbors() if a.GetIdx() != 0]
    for branch, atom in enumerate(branches):
        x, y = cn + 0.7, 1.2 * (-1 if branch else 1)
        positions[atom.GetIdx()] = (x, y, 0)
        other_h = [a.GetIdx() for a in atom.GetNeighbors() if a.GetAtomicNum() == 1]
        for i, idx in enumerate(other_h):
            theta = 2 * math.pi * i / len(other_h)
            positions[idx] = (x + math.cos(theta), y + 0.5, math.sin(theta))
    assert len(positions) == mol.GetNumAtoms()
    conf = Chem.Conformer(mol.GetNumAtoms())
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


@pytest.mark.parametrize("rotate", [False, True])
@pytest.mark.parametrize(
    "smiles,alternate_planar,allowed",
    [
        ("C=[N+](C)[CH-]C", True, True),
        ("C=[N+](C)[CH-]S", True, True),
        ("[CH2-][N+](C)=CC", True, True),
        ("C=[N+](C)[CH-]C", False, False),
        ("C=[N+](C)CC.[CH3-]", True, False),
    ],
)
def test_cpp_iminium_retains_local_charge_preserving_resonance(
    smiles, alternate_planar, allowed, rotate
):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    conf = Chem.Conformer(mol.GetNumAtoms())
    cn = 1.47
    # C(0)=N+(1)--C-(3), with a tetrahedral first carbon and an independently
    # controlled second carbon. Moving the existing charge preserves the graph.
    positions = {0: (0, 0, 0), 1: (cn, 0, 0), 2: (cn + 0.7, -1.2, 0), 3: (cn + 1.32, 0, 0)}
    terminal_h = [a.GetIdx() for a in mol.GetAtomWithIdx(0).GetNeighbors() if a.GetAtomicNum() == 1]
    positions[terminal_h[0]] = (-1 / 3, math.sqrt(8 / 9), 0)
    positions[terminal_h[1]] = (-1 / 3, -math.sqrt(2 / 9), math.sqrt(2 / 3))
    alternate_h = [
        a.GetIdx() for a in mol.GetAtomWithIdx(3).GetNeighbors() if a.GetAtomicNum() == 1
    ]
    x = cn + 1.32
    if alternate_planar:
        positions[4] = (x + 0.75, 1.5 * math.sqrt(3) / 2, 0)
        positions[alternate_h[0]] = (x + 0.5, -math.sqrt(3) / 2, 0)
    else:
        positions[4] = (x + 0.5, 1.5 * math.sqrt(8 / 9), 0)
        positions[alternate_h[0]] = (x + 1 / 3, -math.sqrt(2 / 9), math.sqrt(2 / 3))
    for a in mol.GetAtoms():
        if a.GetIdx() in positions:
            continue
        if a.GetAtomicNum() == 6:
            positions[a.GetIdx()] = (8, 8, 8)  # Disconnected charge control.
        else:
            parent = a.GetNeighbors()[0].GetIdx()
            p = positions[parent]
            theta = a.GetIdx() * 2.1
            positions[a.GetIdx()] = (p[0] + math.cos(theta), p[1] + math.sin(theta), p[2] + 0.7)
    for idx, p in positions.items():
        conf.SetAtomPosition(idx, (p[1] + 7, p[2] - 11, p[0] + 19) if rotate else p)
    mol.AddConformer(conf)
    data = rdkit_to_mol_data(mol)
    assert validate_omol(mol_data_to_pybel(data), Chem.GetFormalCharge(mol), 0) is allowed
    ptr = _core.utils.molecule_data_to_obmol_ptr(data)
    try:
        assert _core.dev.stages.preprocess.validate_omol_ptr(ptr, 0, 0) is allowed
    finally:
        _core.free_obmol_ptr(ptr)


@pytest.mark.parametrize("rigid_motion", [False, True])
@pytest.mark.parametrize(
    "case_id,metal,charge",
    [
        ("XEZKIZ", "Os", 2),
        ("ROLDOL", "W", 2),
        ("BAYWAZ", "Fe", 2),
        ("QOKMOS", "Cr", 0),
        ("QOKMUY", "Cr", 0),
        ("OFOBEQ", "Rh", 1),
    ],
)
def test_cpp_iminium_geometry_preserves_supported_electronic_assignments(
    rigid_motion, case_id, metal, charge
):

    case = dict(fixture(case_id))
    xyz = Chem.MolFromXYZBlock(case["xyz"])
    text = case["xyz"]
    if rigid_motion:
        for i, pos in enumerate(xyz.GetConformer().GetPositions()):
            xyz.GetConformer().SetAtomPosition(i, pos[[1, 2, 0]] + (15, -20, 10))
        text = Chem.MolToXYZBlock(xyz)
    cfg = MolGRConfig()
    cfg.cpp_backend.max_threads = 1
    cfg.cpp_backend.target_bucket_parallel_max_threads = 1
    mol = xyz_to_rdmol(text, case["charge"], case["multiplicity"], backend="cpp", config=cfg)
    assert [
        (a.GetSymbol(), a.GetFormalCharge()) for a in mol.GetAtoms() if a.GetSymbol() == metal
    ] == [(metal, charge)]
    assert [a.GetAtomicNum() for a in mol.GetAtoms()] == [a.GetAtomicNum() for a in xyz.GetAtoms()]
    assert mol.GetConformer().GetPositions() == pytest.approx(xyz.GetConformer().GetPositions())
    assert Chem.GetFormalCharge(mol) == case["charge"]
    assert sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()) == case["multiplicity"] - 1
    assert not any(a.GetNumImplicitHs() or a.GetNumExplicitHs() for a in mol.GetAtoms())
    assert (
        evaluate_equivalence(
            reference_graph(case_id),
            mol,
            use_chirality=False,
            ignore_metal_coordination_bonds=True,
            max_resonance=100000,
        ).decision.value
        == "equivalent"
    )


@pytest.mark.parametrize("rotate", [False, True])
@pytest.mark.parametrize(
    "angle,cn,allowed",
    [(125, 1.43, False), (180, 1.43, True), (160, 1.43, True), (125, 1.32, True)],
)
def test_cpp_keteniminium_uses_joint_bond_length_and_angle_evidence(angle, cn, allowed, rotate):
    mol = Chem.AddHs(Chem.MolFromSmiles("C=C=[N+](C)C"))
    theta = math.radians(angle)
    positions = {
        0: (1.35 * math.cos(theta), 1.35 * math.sin(theta), 0),
        1: (0, 0, 0),
        2: (cn, 0, 0),
        3: (cn + 0.7, 1.2, 0),
        4: (cn + 0.7, -1.2, 0),
    }
    terminal_h = [a.GetIdx() for a in mol.GetAtomWithIdx(0).GetNeighbors() if a.GetAtomicNum() == 1]
    for idx, offset in zip(terminal_h, [-math.pi / 3, math.pi / 3]):
        positions[idx] = (
            positions[0][0] + math.cos(theta + offset),
            positions[0][1] + math.sin(theta + offset),
            0,
        )
    for a in mol.GetAtoms():
        if a.GetIdx() not in positions:
            p = positions[a.GetNeighbors()[0].GetIdx()]
            positions[a.GetIdx()] = (p[0] + math.cos(a.GetIdx()), p[1] + math.sin(a.GetIdx()), 0.7)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for idx, p in positions.items():
        conf.SetAtomPosition(idx, (p[1] + 7, p[2] - 11, p[0] + 19) if rotate else p)
    mol.AddConformer(conf)
    if rotate:
        mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    data = rdkit_to_mol_data(mol)
    assert validate_omol(mol_data_to_pybel(data), Chem.GetFormalCharge(mol), 0) is allowed
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
