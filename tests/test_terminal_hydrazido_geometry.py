"""A metal-supported redox branch must retain neutral and bent alternatives."""

import math
from dataclasses import replace

import pytest
from rdkit import Chem


# Load RDKit before OpenBabel to avoid shared-library initialization conflicts.
# isort: split
from openbabel import openbabel as ob
from openbabel import pybel

from molgr.fallback.state import ReconstructionState, make_metal_candidate_state
from molgr.fallback.utils.coordination_seeds import expand_hydrazido_seeds
from molgr.fallback.utils.dataclasses import MetalAtomPosition
from molgr.fallback.utils.electrons import (
    set_unpaired_electron_count,
    set_unresolved_two_electron_center,
)
from molgr.fallback.utils.metals import hydrazido_geometry as geometry
from molgr.fallback.utils.resonance import build_resonance_state_key


def _state(*, charge=-2, angle=170, distance=1.82, nn=1.33):
    mol = ob.OBMol()
    for z, q, xyz in [
        (7, -1, (0, 0, 0)),
        (7, 1, (nn, 0, 0)),
        (6, 0, (nn + 0.8, 1.2, 0)),
        (6, 0, (nn + 0.8, -1.2, 0)),
    ]:
        a = mol.NewAtom()
        a.SetAtomicNum(z)
        a.SetFormalCharge(q)
        a.SetImplicitHCount(0)
        a.SetVector(*xyz)
    mol.AddBond(1, 2, 2)
    mol.AddBond(2, 3, 1)
    mol.AddBond(2, 4, 1)
    metal = MetalAtomPosition(
        1,
        "Ta",
        73,
        3,
        0,
        distance * math.cos(math.radians(angle)),
        distance * math.sin(math.radians(angle)),
        0,
    )
    return ReconstructionState(pybel.Molecule(mol), charge, charge, 0), metal


@pytest.mark.parametrize(
    "options,expected",
    [
        ({}, 2),
        ({"charge": -1}, 1),
        ({"charge": 0}, 1),
        ({"angle": 159.9}, 1),
        ({"angle": 160.1}, 2),
        ({"nn": 1.199}, 1),
        ({"nn": 1.201}, 2),
        ({"nn": 1.449}, 2),
        ({"nn": 1.451}, 1),
        ({"distance": 0}, 1),
        ({"distance": 3}, 1),
    ],
)
def test_supported_branch_retains_original_and_conserves_charge(options, expected):
    state, metal = _state(**options)
    before = build_resonance_state_key(state.omol)
    seeds = expand_hydrazido_seeds(state, [metal.site])
    assert len(seeds) == expected
    assert seeds[0] is state
    assert build_resonance_state_key(state.omol) == before
    for seed in seeds:
        assert (
            sum(a.OBAtom.GetFormalCharge() for a in seed.omol.atoms) + seed.given_charge
            == state.total_charge
        )
    if expected == 2:
        assert seeds[1].omol.OBMol.GetAtom(1).GetFormalCharge() == -2
        assert seeds[1].omol.OBMol.GetAtom(2).GetFormalCharge() == 0
        assert seeds[1].omol.OBMol.GetBond(1, 2).GetBondOrder() == 1


@pytest.mark.parametrize(
    "condition",
    ["no_metal", "two_metals", "radical", "unresolved", "implicit_h", "hetero_substitution"],
)
def test_unsupported_context_does_not_add_seed(condition):
    state, metal = _state()
    metals = [metal]
    mol = state.omol.OBMol
    if condition == "no_metal":
        metals = []
    elif condition == "two_metals":
        metals.append(replace(metal, idx=2))
    elif condition == "radical":
        set_unpaired_electron_count(mol.GetAtom(1), 1)
    elif condition == "unresolved":
        set_unresolved_two_electron_center(mol.GetAtom(1), True)
    elif condition == "implicit_h":
        mol.GetAtom(1).SetImplicitHCount(1)
    else:
        mol.GetAtom(3).SetAtomicNum(8)
    assert expand_hydrazido_seeds(state, [m.site for m in metals]) == [state]


def _candidates():
    state, metal = _state()
    seeds = expand_hydrazido_seeds(state, [metal.site])
    result = []
    for i, seed in enumerate(seeds):
        c = make_metal_candidate_state(
            (), (replace(metal, valence=3 + i * 2),), -i * 2, 0, combination_index=i
        )
        c.no_metal_state = seed
        c.metadata.update(metal_discordance_count=0.0, metal_discordance_structural_count=0.0)
        result.append(c)
    return result


def test_existing_counterpart_is_required_and_annotations_are_idempotent():
    neutral, reduced = _candidates()
    geometry.annotate([neutral])
    assert neutral.metadata[geometry.KEY] == 0
    for _ in range(2):
        geometry.annotate([neutral, reduced])
        assert neutral.metadata[geometry.KEY] == neutral.metadata["metal_discordance_count"] == 1
        assert reduced.metadata[geometry.KEY] == reduced.metadata["metal_discordance_count"] == 0
    geometry.clear([neutral, reduced])
    assert neutral.metadata["metal_discordance_count"] == 0


@pytest.mark.parametrize(
    "condition",
    ["other_charge", "other_bond", "other_radical", "metal_spin", "metal_charge", "other_conflict"],
)
def test_pairwise_preference_cannot_hide_unrelated_changes(condition):
    neutral, reduced = _candidates()
    mol = reduced.no_metal_state.omol.OBMol
    if condition == "other_charge":
        mol.GetAtom(3).SetFormalCharge(-1)
    elif condition == "other_bond":
        mol.GetBond(2, 3).SetBondOrder(2)
    elif condition == "other_radical":
        set_unpaired_electron_count(mol.GetAtom(3), 1)
    elif condition in ("metal_spin", "metal_charge"):
        reduced.metal_states = (
            replace(
                reduced.metal_states[0],
                **({"radical_num": 2} if condition == "metal_spin" else {"valence": 4}),
            ),
        )
    else:
        reduced.metadata["metal_discordance_structural_count"] = 1.0
    geometry.annotate([neutral, reduced])
    assert neutral.metadata[geometry.KEY] == 0


def _mixed_nitrosyl_hydrazido_seed():
    """Synthetic explicit-H search control, not an experimental assignment."""
    state, metal = _state(charge=-1, angle=180)
    mol = state.omol.OBMol
    for z, charge, x in [(7, -1, -3.82), (8, 0, -4.97)]:
        atom = mol.NewAtom()
        atom.SetAtomicNum(z)
        atom.SetFormalCharge(charge)
        atom.SetImplicitHCount(0)
        atom.SetVector(x, 0, 0)
    mol.AddBond(5, 6, 2)
    for index, direction in [(3, 1), (4, -1)]:
        carbon = mol.GetAtom(index)
        for dx, dy, dz in [
            (1, 0, 0),
            (-0.25, 0.7 * direction, 0.8),
            (-0.25, 0.7 * direction, -0.8),
        ]:
            hydrogen = mol.NewAtom()
            hydrogen.SetAtomicNum(1)
            hydrogen.SetImplicitHCount(0)
            hydrogen.SetVector(carbon.GetX() + dx, carbon.GetY() + dy, dz)
            mol.AddBond(index, hydrogen.GetIdx(), 1)
    return replace(state, given_charge=0), metal


@pytest.mark.parametrize("include_metal", [False, True])
def test_ligand_search_combines_geometry_seeds_before_search(monkeypatch, include_metal):
    """The same search validates ordinary and combined NO/NN occupations."""
    from molgr.fallback.pipeline import reconstruct_without_metals as pipeline

    state, metal = _mixed_nitrosyl_hydrazido_seed()
    before = build_resonance_state_key(state.omol)
    monkeypatch.setattr(pipeline, "_run_linear_preparation", lambda seed: seed)
    candidates = []
    result = pipeline._run_no_metal_pipeline_from_state(
        state,
        candidate_pool=candidates,
        metal_sites=(metal.site,) if include_metal else (),
    )
    assert result is not None
    assert build_resonance_state_key(state.omol) == before
    assignments = set()
    for candidate in candidates:
        graph = candidate.omol.OBMol
        assert graph.NumAtoms() == 12
        assert sum(atom.GetFormalCharge() for atom in ob.OBMolAtomIter(graph)) == -1
        assert candidate.given_charge == 0
        assert candidate.total_radical_electrons == 0
        assert all(atom.GetImplicitHCount() == 0 for atom in ob.OBMolAtomIter(graph))
        assignments.add(tuple(graph.GetAtom(i).GetFormalCharge() for i in (1, 2, 5, 6)))
    assert (-1, 1, -1, 0) in assignments  # Original valid candidate is retained.
    assert ((-2, 0, 0, 1) in assignments) is include_metal


def test_ordinary_success_does_not_gate_coordination_seeds(monkeypatch):
    from molgr.fallback.pipeline import reconstruct_without_metals as pipeline

    state, metal = _mixed_nitrosyl_hydrazido_seed()
    monkeypatch.setattr(pipeline, "_run_linear_preparation", lambda seed: seed)
    searched_assignments = []

    def successful_search(seed, *, config):
        mol = seed.omol.OBMol
        searched_assignments.append(
            (mol.GetAtom(1).GetFormalCharge(), mol.GetAtom(6).GetFormalCharge())
        )
        # Isolate orchestration: even a first successful search must leave the
        # other geometry-supported seeds on the same search schedule.
        return [state]

    monkeypatch.setattr(pipeline, "_find_no_metal_candidates", successful_search)
    pipeline._run_no_metal_pipeline_from_state(state, metal_sites=(metal.site,))
    assert searched_assignments == [(-1, 0), (-1, 1), (-2, 1)]


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_combined_geometry_search_is_reachable_from_xyz(backend):
    """Both pipelines must reach the combined branch without orchestration mocks."""
    from molgr.interface import xyz_to_rdmol

    state, metal = _mixed_nitrosyl_hydrazido_seed()
    lines = [
        "13",
        "Synthetic combined geometry search control",
        f"Ta {metal.position_x:.12f} {metal.position_y:.12f} {metal.position_z:.12f}",
    ]
    lines.extend(
        f"{ob.GetSymbol(a.GetAtomicNum())} {a.GetX()} {a.GetY()} {a.GetZ()}"
        for a in ob.OBMolAtomIter(state.omol.OBMol)
    )
    xyz = "\n".join(lines) + "\n"
    result = xyz_to_rdmol(xyz, 4, 1, backend=backend)
    original = Chem.MolFromXYZBlock(xyz)
    assert [a.GetAtomicNum() for a in result.GetAtoms()] == [
        a.GetAtomicNum() for a in original.GetAtoms()
    ]
    assert result.GetConformer().GetPositions() == pytest.approx(
        original.GetConformer().GetPositions(), abs=1e-8
    )
    assert [result.GetAtomWithIdx(i).GetFormalCharge() for i in (0, 1, 2, 5, 6)] == [5, -2, 0, 0, 1]
    assert result.GetBondBetweenAtoms(1, 2).GetBondType() == Chem.BondType.SINGLE
    assert result.GetBondBetweenAtoms(5, 6).GetBondType() == Chem.BondType.TRIPLE
    assert Chem.GetFormalCharge(result) == 4
    assert sum(a.GetNumRadicalElectrons() for a in result.GetAtoms()) == 0
    assert all(a.GetNumImplicitHs() == a.GetNumExplicitHs() == 0 for a in result.GetAtoms())
