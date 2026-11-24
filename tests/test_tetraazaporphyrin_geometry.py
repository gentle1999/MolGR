"""The four bridging nitrogens of a C8N8 circuit are not metal donors."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
from openbabel import openbabel as ob
from openbabel import pybel

from molgr.fallback.pipeline import reconstruct_with_metals
from molgr.fallback.utils.bond_geometry import planar_tetraazaporphyrin_circuits
from molgr.fallback.utils.metals import scoring
from scripts.tmqmg_fixture_data import fixture


@pytest.fixture(scope="module")
def candidates():

    case = fixture("RILVOX")
    captured = []
    original = scoring.select_best_candidate

    def capture(pool, **kwargs):
        captured[:] = pool
        return original(pool, **kwargs)

    with patch.object(scoring, "select_best_candidate", capture):
        result = reconstruct_with_metals.xyz2omol_state(
            case["xyz"], case["charge"], case["multiplicity"] - 1
        )
    assert result is not None
    return {c.metal_states[0].valence: c for c in captured if c.metal_states[0].valence in (1, 3)}


def _copy(candidate):
    return replace(
        candidate,
        metadata=dict(candidate.metadata),
        no_metal_state=replace(
            candidate.no_metal_state,
            omol=pybel.Molecule(ob.OBMol(candidate.no_metal_state.omol.OBMol)),
        ),
    )


def test_tetraaza_circuit_compares_16_and_18_electrons(candidates):
    for valence, expected in [(1, 16), (3, 18)]:
        candidate = _copy(candidates[valence])
        mol = candidate.no_metal_state.omol.OBMol
        [(ids, donors)] = list(planar_tetraazaporphyrin_circuits(mol))
        assert len(ids) == 16 and len(donors) == 4
        assert sum(mol.GetAtom(i).GetAtomicNum() == 7 for i in ids) == 8
        candidate.metadata["metal_visible_atom_groups"] = ",".join(map(str, sorted(donors)))
        assert scoring._macrocycle_electron_sites(candidate) == {tuple(sorted(ids)): expected}
        for atom in ob.OBMolAtomIter(mol):
            atom.SetVector(atom.GetY() + 100, atom.GetZ() - 50, atom.GetX() + 20)
        # Apply the same rigid transform to the metal and ligand.
        metal = candidate.metal_states[0]
        shifted_metal = replace(
            metal,
            position_x=metal.position_y + 100,
            position_y=metal.position_z - 50,
            position_z=metal.position_x + 20,
        )
        candidate = replace(candidate, metal_states=(shifted_metal,))
        assert scoring._macrocycle_electron_sites(candidate) == {tuple(sorted(ids)): expected}
        candidate.metadata["metal_visible_atom_groups"] = ",".join(map(str, sorted(donors)[1:]))
        assert scoring._macrocycle_electron_sites(candidate) == {}


@pytest.mark.parametrize(("height", "expected_conflict"), [(0.7, 1), (0.9, 0), (-0.9, 0)])
def test_only_in_plane_tetraaza_pairs_receive_electron_count_conflict(
    candidates, height, expected_conflict
):
    from molgr import core
    from molgr.config import MolGRConfig

    mol = candidates[1].no_metal_state.omol.OBMol
    [(_, donors)] = list(planar_tetraazaporphyrin_circuits(mol))
    atoms = [mol.GetAtom(i) for i in sorted(donors)]
    points = np.array([(atom.GetX(), atom.GetY(), atom.GetZ()) for atom in atoms])
    center = points.mean(axis=0)
    normal = np.linalg.svd(points - center)[2][-1]
    x, y, z = center + height * normal

    pool = []
    for valence in (1, 3):
        candidate = _copy(candidates[valence])
        metal = replace(candidate.metal_states[0], position_x=x, position_y=y, position_z=z)
        pool.append(replace(candidate, metal_states=(metal,)))
    scoring._annotate_macrocycle_alternatives(pool)
    key = "metal_macrocycle_electron_conflict_count"
    assert [c.metadata[key] for c in pool] == [expected_conflict, 0]

    case = fixture("RILVOX")
    lines = case["xyz"].splitlines()
    idx = next(i for i, line in enumerate(lines[2:], 2) if line.split()[0] == "Ir")
    lines[idx] = f"Ir {x} {y} {z}"
    result = core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        "\n".join(lines) + "\n", case["charge"], case["multiplicity"] - 1, config=MolGRConfig()
    )
    conflicts = {c["metal_states"][0]["valence"]: c[key] for c in result["candidates"]}
    assert [conflicts[1], conflicts[3]] == [expected_conflict, 0]


@pytest.mark.parametrize(
    "defect", ["implicit_h", "element", "length", "nonplanar", "missing", "nan", "fused_ring"]
)
def test_tetraaza_geometry_requires_the_complete_planar_topology(candidates, defect):
    mol = _copy(candidates[3]).no_metal_state.omol.OBMol
    [(ids, _)] = list(planar_tetraazaporphyrin_circuits(mol))
    atom = mol.GetAtom(ids[0])
    if defect == "implicit_h":
        atom.SetImplicitHCount(1)
    elif defect == "element":
        atom.SetAtomicNum(15)
    elif defect == "length":
        for atom in ob.OBMolAtomIter(mol):
            atom.SetVector(atom.GetX() * 1.2, atom.GetY() * 1.2, atom.GetZ() * 1.2)
    elif defect == "nonplanar":
        atom.SetVector(atom.GetX(), atom.GetY(), atom.GetZ() + 0.8)
    elif defect in ("missing", "nan"):
        for atom in ob.OBMolAtomIter(mol):
            atom.SetVector(0, 0, 0 if defect == "missing" else float("nan"))
    else:
        fused = next(
            r for r in mol.GetSSSR() if len(r._path) == 5 and len(set(r._path) & set(ids)) == 3
        )
        outer = next(i for i in fused._path if i not in ids)
        mol.GetAtom(outer).SetAtomicNum(8)
    assert list(planar_tetraazaporphyrin_circuits(mol)) == []


def test_tetraaza_single_candidate_does_not_impose_a_metal_state(candidates):
    candidate = _copy(candidates[1])
    scoring._annotate_macrocycle_alternatives([candidate])
    assert candidate.metadata["metal_macrocycle_electron_conflict_count"] == 0
    assert candidate.metal_states[0].valence == 1
