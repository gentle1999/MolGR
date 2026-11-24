"""Geometry preferences must compare compatible, fully resolved local graphs."""

from dataclasses import replace
from unittest.mock import patch

import pytest
from openbabel import openbabel as ob
from openbabel import pybel
from rdkit import Chem

from molgr import MolGRConfig
from molgr.fallback.pipeline import reconstruct_with_metals
from molgr.fallback.utils.electrons import (
    set_unpaired_electron_count,
    set_unresolved_two_electron_center,
)
from molgr.fallback.utils.metals import conjugated_geometry as geometry
from molgr.fallback.utils.metals import scoring
from molgr.interface import xyz_to_rdmol
from molgr.utils.converter import pybel_to_rdmol
from molgr.utils.equivalence import evaluate_equivalence
from scripts.tmqmg_fixture_data import fixture


QUINODIIMINES = ["CAHWIR", "ALEHOO", "MUTCIP", "MUTCOV", "ENAVUL", "COGXOM"]


def _candidates(case_id, *, include_ligand_alternatives=False):
    row = fixture(case_id)
    captured = []
    selection = (
        "select_ligand_alternatives" if include_ligand_alternatives else "select_best_candidate"
    )
    original = getattr(scoring, selection)

    def capture(pool, **kwargs):
        result = original(pool, **kwargs)
        if include_ligand_alternatives:
            # A local rule can be exercised on a validated contributor even when
            # another contributor subsequently wins for its metal assignment.
            retained_ids = {id(candidate) for candidate in result}
            captured[:] = [*result, *(c for c in pool if id(c) not in retained_ids)]
        else:
            captured[:] = pool
        return result

    with patch.object(scoring, selection, capture):
        result = reconstruct_with_metals.xyz2omol_state(
            row["xyz"], row["charge"], row["multiplicity"] - 1
        )
    assert result is not None
    return captured


def _copy(candidate):
    result = replace(
        candidate,
        metadata=dict(candidate.metadata),
        key_cache={},
        no_metal_state=replace(
            candidate.no_metal_state,
            metadata=dict(candidate.no_metal_state.metadata),
            omol=pybel.Molecule(ob.OBMol(candidate.no_metal_state.omol.OBMol)),
        ),
    )
    geometry.clear_pyridyl([result])
    geometry.clear_paired_diimine([result])
    geometry.clear_polyenamine([result])
    geometry.clear_quinodiimine([result])
    return result


def _pair(candidates, annotate, key):
    for candidate in candidates:
        for partner in candidates:
            first, second = _copy(candidate), _copy(partner)
            annotate([first, second])
            if first.metadata[key]:
                return _copy(candidate), _copy(partner)
    pytest.fail("No compatible geometry-supported candidate pair")


@pytest.fixture(scope="module", params=QUINODIIMINES)
def quinodiimine_candidates(request):
    return _candidates(request.param, include_ligand_alternatives=True)


@pytest.fixture(scope="module")
def polyenamine_candidates():
    return _candidates("MULTAR")


@pytest.fixture(scope="module")
def quinodiimine_pair_template(quinodiimine_candidates):
    return _pair(quinodiimine_candidates, geometry.annotate_quinodiimine, geometry.QUINODIIMINE_KEY)


@pytest.fixture
def quinodiimine_pair(quinodiimine_pair_template):
    return tuple(_copy(c) for c in quinodiimine_pair_template)


@pytest.fixture
def polyenamine_pair(polyenamine_candidates):
    return _pair(polyenamine_candidates, geometry.annotate_polyenamine, geometry.POLYENAMINE_KEY)


def _changed_sites(first, second):
    return geometry.quinodiimine_sites(first)[1] & geometry.quinodiimine_sites(second)[0]


PARTNER_DEFECTS = [
    "metal",
    "delta",
    "metal_spin",
    "ligand_spin",
    "no",
    "independent",
    "outside_atom",
    "boundary_bond",
]


def _assert_incompatible_partner(kind, defect, pair):
    first, second = pair
    annotate = getattr(geometry, "annotate_" + kind)
    key = getattr(geometry, kind.upper() + "_KEY")
    if kind == "quinodiimine":
        core = {i for site in _changed_sites(first, second) for i in site}
    else:
        core = set(next(iter(geometry.polyenamine_sites(first))))
    if defect in ("metal", "delta", "metal_spin"):
        attr, value = {
            "metal": ("idx", 1000),
            "delta": ("valence", second.metal_states[0].valence + 2),
            "metal_spin": ("radical_num", second.metal_states[0].radical_num + 2),
        }[defect]
        second.metal_states = (replace(second.metal_states[0], **{attr: value}),)
    elif defect == "ligand_spin":
        second.no_metal_radical_target += 2
    elif defect == "no":
        second.metadata["nitrosyl_geometry_penalty"] += 1
    elif defect == "independent":
        second.metadata["metal_discordance_structural_count"] = (
            first.metadata["metal_discordance_structural_count"] + 1
        )
    else:
        mol = second.no_metal_state.omol.OBMol
        hydrogen = next(
            a
            for a in ob.OBMolAtomIter(mol)
            if a.GetAtomicNum() == 1 and next(ob.OBAtomAtomIter(a)).GetIdx() in core
        )
        if defect == "outside_atom":
            hydrogen.SetFormalCharge(1)
        else:
            next(ob.OBAtomBondIter(hydrogen)).SetBondOrder(2)
    annotate([first, second])
    assert first.metadata[key] == 0


@pytest.mark.parametrize("defect", PARTNER_DEFECTS)
def test_quinodiimine_partner_abstains(quinodiimine_pair, defect):
    _assert_incompatible_partner("quinodiimine", defect, quinodiimine_pair)


@pytest.mark.parametrize("defect", PARTNER_DEFECTS)
def test_polyenamine_partner_abstains(polyenamine_pair, defect):
    _assert_incompatible_partner("polyenamine", defect, polyenamine_pair)


@pytest.mark.parametrize(
    "defect", ["radical", "deferred", "implicit_h", "missing", "nan", "distorted", "visibility"]
)
def test_quinodiimine_geometry_abstains(quinodiimine_pair, defect):
    first, second = quinodiimine_pair
    site = next(iter(_changed_sites(first, second)))
    mol = second.no_metal_state.omol.OBMol
    atom = mol.GetAtom(site[-1])
    if defect == "radical":
        set_unpaired_electron_count(atom, 1)
    elif defect == "deferred":
        set_unresolved_two_electron_center(atom, True)
    elif defect == "implicit_h":
        atom.SetImplicitHCount(1)
    elif defect == "visibility":
        second.metadata["metal_visible_atom_groups"] = ""
    elif defect == "distorted":
        atom.SetVector(atom.GetX() + 10, atom.GetY(), atom.GetZ())
    else:
        for atom in ob.OBMolAtomIter(mol):
            atom.SetVector(0, 0, 0 if defect == "missing" else float("nan"))
    geometry.annotate_quinodiimine([first, second])
    assert first.metadata[geometry.QUINODIIMINE_KEY] == 0


def test_quinodiimine_partner_removal_and_repeat(quinodiimine_pair):
    first, second = quinodiimine_pair
    before = [geometry.outside_signature(c, set()) for c in (first, second)]
    for pool in ([first, second], [second, first], [first, second]):
        geometry.annotate_quinodiimine(pool)
        assert first.metadata[geometry.QUINODIIMINE_KEY] > 0
        assert second.metadata[geometry.QUINODIIMINE_KEY] == 0
    geometry.annotate_quinodiimine([first])
    assert first.metadata[geometry.QUINODIIMINE_KEY] == 0
    assert before == [geometry.outside_signature(c, set()) for c in (first, second)]


@pytest.mark.parametrize("transform", ["rigid_motion", "atom_order"])
def test_quinodiimine_frame_invariance(quinodiimine_pair, transform):
    first, second = quinodiimine_pair
    for candidate in (first, second):
        source = candidate.no_metal_state.omol.OBMol
        if transform == "rigid_motion":
            for atom in ob.OBMolAtomIter(source):
                atom.SetVector(atom.GetY() + 30, atom.GetZ() - 50, atom.GetX() + 10)
            continue
        mol, mapping = ob.OBMol(), {}
        for idx in range(source.NumAtoms(), 0, -1):
            mol.AddAtom(source.GetAtom(idx))
            mapping[idx] = mol.NumAtoms()
        for bond in ob.OBMolBondIter(source):
            mol.AddBond(
                mapping[bond.GetBeginAtomIdx()], mapping[bond.GetEndAtomIdx()], bond.GetBondOrder()
            )
        candidate.no_metal_state = replace(candidate.no_metal_state, omol=pybel.Molecule(mol))
        groups = candidate.metadata["metal_visible_atom_groups"].split(";")
        candidate.metadata["metal_visible_atom_groups"] = ";".join(
            ",".join(str(mapping[int(i)]) for i in group.split(",") if i) for group in groups
        )
    geometry.annotate_quinodiimine([first, second])
    assert first.metadata[geometry.QUINODIIMINE_KEY] > 0


@pytest.mark.parametrize(
    "budget,value", [("MAX_PATH_STATES", 0), ("MAX_PATH_ATOMS", 1), ("MAX_VARIANTS", 0)]
)
def test_quinodiimine_budget_abstains_without_partial_traversal(quinodiimine_pair, budget, value):
    first, second = quinodiimine_pair
    changed = _changed_sites(first, second)
    needs_transfer = any(
        second.no_metal_state.omol.OBMol.GetAtom(i).GetFormalCharge() == 0
        for site in changed
        for i in site[-2:]
    )
    with patch.object(geometry, budget, value):
        geometry.annotate_quinodiimine([first, second])
    assert bool(first.metadata[geometry.QUINODIIMINE_KEY]) == (
        not needs_transfer and budget != "MAX_VARIANTS"
    )


def test_attached_arm_normalization_is_native_resonance(quinodiimine_candidates):
    checks = 0
    for candidate in quinodiimine_candidates:
        reduced, _ = geometry.quinodiimine_sites(candidate)
        if not reduced:
            continue
        source = pybel_to_rdmol(candidate.no_metal_state.omol, sanitize=False, kekulize=False)
        for normalized in geometry.normalized_variants(candidate, reduced):
            target = pybel_to_rdmol(pybel.Molecule(normalized), sanitize=False, kekulize=False)
            result = evaluate_equivalence(source, target, max_resonance=100000)
            assert result.decision.value == "equivalent", result.reason
            checks += 1
    assert checks > 0


def test_polyenamine_accepts_resonance_contributor(polyenamine_pair):
    first, second = polyenamine_pair
    site = next(iter(geometry.polyenamine_sites(first)))
    mol = second.no_metal_state.omol.OBMol
    original = pybel_to_rdmol(second.no_metal_state.omol, sanitize=False, kekulize=False)
    mol.GetAtom(site[-1]).SetFormalCharge(0)
    mol.GetAtom(site[-3]).SetFormalCharge(-1)
    mol.GetBond(site[-1], site[-2]).SetBondOrder(2)
    mol.GetBond(site[-2], site[-3]).SetBondOrder(1)
    changed = pybel_to_rdmol(second.no_metal_state.omol, sanitize=False, kekulize=False)
    assert (
        evaluate_equivalence(original, changed, max_resonance=100000).decision.value == "equivalent"
    )
    geometry.annotate_polyenamine([first, second])
    assert first.metadata[geometry.POLYENAMINE_KEY] == 1
    geometry.annotate_polyenamine([first])
    assert first.metadata[geometry.POLYENAMINE_KEY] == 0


@pytest.mark.parametrize(
    "defect",
    ["radical", "deferred", "implicit_h", "twist", "equal_lengths", "missing", "nan", "visibility"],
)
def test_polyenamine_geometry_abstains(polyenamine_pair, defect):
    first, _ = polyenamine_pair
    site = next(iter(geometry.polyenamine_sites(first)))
    mol = first.no_metal_state.omol.OBMol
    atom = mol.GetAtom(site[3])
    if defect == "radical":
        set_unpaired_electron_count(atom, 1)
    elif defect == "deferred":
        set_unresolved_two_electron_center(atom, True)
    elif defect == "implicit_h":
        atom.SetImplicitHCount(1)
    elif defect == "twist":
        atom.SetVector(atom.GetX(), atom.GetY(), atom.GetZ() + 1)
    elif defect == "visibility":
        first.metadata["metal_visible_atom_groups"] = ""
    elif defect == "equal_lengths":
        import math

        for i, idx in enumerate(site):
            mol.GetAtom(idx).SetVector(
                i * 1.4 * math.cos(math.pi / 6), (i % 2) * 1.4 * math.sin(math.pi / 6), 0
            )
    else:
        for atom in ob.OBMolAtomIter(mol):
            atom.SetVector(0, 0, 0 if defect == "missing" else float("nan"))
    assert not geometry.polyenamine_sites(first)


@pytest.mark.parametrize("case_id", QUINODIIMINES + ["MULTAR", "BEXZEJ", "BADQAA", "CAFWUC"])
def test_conjugated_geometry_backend_parity_and_conservation(case_id):
    """Backend agreement is separate from the pending chemical standard decisions."""
    row = fixture(case_id)
    config = MolGRConfig()
    config.cpp_backend.max_threads = 1
    config.cpp_backend.target_bucket_parallel_max_threads = 1
    xyz = Chem.MolFromXYZBlock(row["xyz"])
    signatures = []
    for backend in ("python", "cpp"):
        mol = xyz_to_rdmol(
            row["xyz"],
            total_charge=row["charge"],
            spin_multiplicity=row["multiplicity"],
            backend=backend,
            config=config,
        )
        assert [a.GetAtomicNum() for a in mol.GetAtoms()] == [
            a.GetAtomicNum() for a in xyz.GetAtoms()
        ]
        assert Chem.GetFormalCharge(mol) == row["charge"]
        assert sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()) == row["multiplicity"] - 1
        assert all(
            a.GetNoImplicit() and a.GetNumImplicitHs() == 0 and a.GetNumExplicitHs() == 0
            for a in mol.GetAtoms()
        )
        assert mol.GetConformer().GetPositions() == pytest.approx(
            xyz.GetConformer().GetPositions(), abs=1e-8
        )
        atoms = [
            (a.GetAtomicNum(), a.GetIsotope(), a.GetFormalCharge(), a.GetNumRadicalElectrons())
            for a in mol.GetAtoms()
        ]
        bonds = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if bond.GetBondType() != Chem.BondType.DATIVE:
                i, j = sorted((i, j))
            bonds.append((i, j, str(bond.GetBondType())))
        signatures.append((atoms, sorted(bonds)))
    assert signatures[0] == signatures[1]


@pytest.mark.parametrize("case_id", QUINODIIMINES + ["MULTAR", "BEXZEJ", "BADQAA", "CAFWUC"])
def test_geometry_diagnostics_backend_parity(case_id):
    from molgr import _core

    row = fixture(case_id)
    config = MolGRConfig()
    config.cpp_backend.max_threads = 1
    config.cpp_backend.target_bucket_parallel_max_threads = 1
    native = _core.dev.pipeline.reconstruct_with_metals.debug_scored_candidate_summaries(
        row["xyz"], row["charge"], row["multiplicity"] - 1, config=config
    )
    fallback = _candidates(case_id)
    assert len(native["candidates"]) == len(fallback)
    for cpp, python in zip(native["candidates"], fallback):
        assert cpp["combination_index"] == python.metadata["combination_index"]
        for key in (
            geometry.QUINODIIMINE_KEY,
            geometry.POLYENAMINE_KEY,
            geometry.PAIRED_DIIMINE_KEY,
        ):
            assert cpp[key] == python.metadata[key]


@pytest.fixture(scope="module")
def paired_diimine_candidates():
    return {c.metal_states[0].valence: c for c in _candidates("BADQAA")}


def test_paired_diimine_supported_allocation(paired_diimine_candidates):
    candidates = [_copy(paired_diimine_candidates[q]) for q in (0, 2, 4)]
    before = [geometry.outside_signature(c, set()) for c in candidates]
    for order in (candidates, candidates[::-1], [candidates[i] for i in (1, 0, 2)]):
        geometry.annotate_paired_diimine(order)
        assert [c.metadata[geometry.PAIRED_DIIMINE_KEY] for c in candidates] == [1, 0, 1]
    assert before == [geometry.outside_signature(c, set()) for c in candidates]
    # The geometry must not force an allocation absent from the candidate pool.
    geometry.annotate_paired_diimine([candidates[0], candidates[2]])
    assert not any(c.metadata[geometry.PAIRED_DIIMINE_KEY] for c in (candidates[0], candidates[2]))


@pytest.mark.parametrize("charge", [0, 4])
@pytest.mark.parametrize(
    "defect",
    [
        "metal",
        "metal_spin",
        "delta",
        "ligand_spin",
        "no",
        "independent",
        "outside_atom",
        "boundary_bond",
        "multiple_metals",
        "unknown_state",
        "visibility",
        "radical",
        "deferred",
        "implicit_h",
        "missing",
        "nan",
        "long_cc",
        "short_cn",
        "translation",
    ],
)
def test_paired_diimine_incompatible_support(paired_diimine_candidates, charge, defect):
    import numpy as np

    first, second = [_copy(paired_diimine_candidates[q]) for q in (charge, 2)]
    core = next(iter(geometry.paired_diimine_sites(first)))
    mol = second.no_metal_state.omol.OBMol
    n = next(mol.GetAtom(i) for i in core if mol.GetAtom(i).GetAtomicNum() == 7)
    carbon = next(a for a in ob.OBAtomAtomIter(n) if a.GetIdx() in core)
    if defect in ("metal", "metal_spin", "delta"):
        attr, value = {
            "metal": ("idx", 999),
            "metal_spin": ("radical_num", 2),
            "delta": ("valence", 8),
        }[defect]
        second.metal_states = (replace(second.metal_states[0], **{attr: value}),)
    elif defect == "ligand_spin":
        second.no_metal_radical_target += 2
    elif defect == "no":
        second.metadata["nitrosyl_geometry_penalty"] += 1
    elif defect == "independent":
        second.metadata["metal_discordance_structural_count"] = (
            first.metadata["metal_discordance_structural_count"] + 1
        )
    elif defect == "multiple_metals":
        second.metal_states = (*second.metal_states, second.metal_states[0])
    elif defect == "unknown_state":
        second.no_metal_state = None
    elif defect == "visibility":
        second.metadata["metal_visible_atom_groups"] = str(n.GetIdx())
    elif defect == "radical":
        set_unpaired_electron_count(n, 1)
    elif defect == "deferred":
        set_unresolved_two_electron_center(n, True)
    elif defect == "implicit_h":
        n.SetImplicitHCount(1)
    elif defect in ("missing", "nan", "translation"):
        for atom in ob.OBMolAtomIter(mol):
            if defect == "translation":
                atom.SetVector(atom.GetX() + 1, atom.GetY(), atom.GetZ())
            else:
                atom.SetVector(0, 0, float("nan") if defect == "nan" else 0)
    elif defect in ("outside_atom", "boundary_bond"):
        outside = next(a for a in ob.OBAtomAtomIter(n) if a.GetIdx() not in core)
        if defect == "outside_atom":
            outside.SetFormalCharge(1)
        else:
            n.GetBond(outside).SetBondOrder(2)
    else:
        target = (
            n
            if defect == "short_cn"
            else next(
                a for a in ob.OBAtomAtomIter(carbon) if a.GetIdx() in core and a.GetAtomicNum() == 6
            )
        )
        origin = np.array([carbon.GetX(), carbon.GetY(), carbon.GetZ()])
        delta = np.array([target.GetX(), target.GetY(), target.GetZ()]) - origin
        target.SetVector(
            *(origin + delta / np.linalg.norm(delta) * (1.25 if defect == "short_cn" else 1.52))
        )
    geometry.annotate_paired_diimine([first, second])
    assert first.metadata[geometry.PAIRED_DIIMINE_KEY] == 0


@pytest.mark.parametrize("transform", ["rigid", "reindex", "trans_diimine"])
def test_paired_diimine_geometry_scope(paired_diimine_candidates, transform):
    import numpy as np

    candidates = [_copy(paired_diimine_candidates[q]) for q in (0, 2, 4)]
    for candidate in candidates:
        core = next(iter(geometry.paired_diimine_sites(candidate)))
        mol = candidate.no_metal_state.omol.OBMol
        if transform == "rigid":
            for atom in ob.OBMolAtomIter(mol):
                atom.SetVector(atom.GetZ() + 20, atom.GetX() - 30, atom.GetY() + 40)
        elif transform == "reindex":
            new, mapping = ob.OBMol(), {}
            for i in range(mol.NumAtoms(), 0, -1):
                new.AddAtom(mol.GetAtom(i))
                mapping[i] = new.NumAtoms()
            for bond in ob.OBMolBondIter(mol):
                new.AddBond(
                    mapping[bond.GetBeginAtomIdx()],
                    mapping[bond.GetEndAtomIdx()],
                    bond.GetBondOrder(),
                )
            candidate.no_metal_state = replace(candidate.no_metal_state, omol=pybel.Molecule(new))
            candidate.metadata["metal_visible_atom_groups"] = ";".join(
                ",".join(str(mapping[int(i)]) for i in group.split(",") if i)
                for group in candidate.metadata["metal_visible_atom_groups"].split(";")
            )
        else:
            # Preserve N-C-C lengths/angle while rotating the terminal N to trans.
            n = next(mol.GetAtom(i) for i in core if mol.GetAtom(i).GetAtomicNum() == 7)
            carbon = next(a for a in ob.OBAtomAtomIter(n) if a.GetIdx() in core)
            other = next(
                a for a in ob.OBAtomAtomIter(carbon) if a.GetIdx() in core and a.GetAtomicNum() == 6
            )
            origin = np.array([carbon.GetX(), carbon.GetY(), carbon.GetZ()])
            axis = np.array([other.GetX(), other.GetY(), other.GetZ()]) - origin
            axis /= np.linalg.norm(axis)
            delta = np.array([n.GetX(), n.GetY(), n.GetZ()]) - origin
            n.SetVector(*(origin + 2 * axis * np.dot(axis, delta) - delta))
    geometry.annotate_paired_diimine(candidates)
    assert [c.metadata[geometry.PAIRED_DIIMINE_KEY] for c in candidates] == (
        [0, 0, 0] if transform == "trans_diimine" else [1, 0, 1]
    )


@pytest.fixture(scope="module")
def pyridyl_candidates():
    return _candidates("MUTSOL")


@pytest.fixture
def pyridyl_pair(pyridyl_candidates):
    return _pair(pyridyl_candidates, geometry.annotate_pyridyl, geometry.PYRIDYL_KEY)


def test_pyridyl_preference_is_scoped_and_does_not_mutate(pyridyl_pair):
    first, second = pyridyl_pair
    before = [geometry.outside_signature(c, ()) for c in (first, second)]
    for pool in ([first, second], [second, first], [first, second]):
        geometry.annotate_pyridyl(pool)
        assert first.metadata[geometry.PYRIDYL_KEY] == 1
        assert second.metadata[geometry.PYRIDYL_KEY] == 0
    assert before == [geometry.outside_signature(c, ()) for c in (first, second)]
    geometry.annotate_pyridyl([first])
    assert first.metadata[geometry.PYRIDYL_KEY] == 0


@pytest.mark.parametrize("defect", PARTNER_DEFECTS)
def test_pyridyl_incompatible_partner_abstains(pyridyl_pair, defect):
    first, second = pyridyl_pair
    core = set(next(iter(geometry.pyridyl_sites(first)))[0])
    if defect in ("metal", "delta", "metal_spin"):
        attr, value = {
            "metal": ("idx", 1000),
            "delta": ("valence", second.metal_states[0].valence + 2),
            "metal_spin": ("radical_num", second.metal_states[0].radical_num + 2),
        }[defect]
        second.metal_states = (replace(second.metal_states[0], **{attr: value}),)
    elif defect == "ligand_spin":
        second.no_metal_radical_target += 2
    elif defect == "no":
        second.metadata["nitrosyl_geometry_penalty"] += 1
    elif defect == "independent":
        second.metadata["metal_discordance_structural_count"] = (
            first.metadata["metal_discordance_structural_count"] + 1
        )
    else:
        mol = second.no_metal_state.omol.OBMol
        hydrogen = next(
            a
            for a in ob.OBMolAtomIter(mol)
            if a.GetAtomicNum() == 1 and next(ob.OBAtomAtomIter(a)).GetIdx() in core
        )
        if defect == "outside_atom":
            hydrogen.SetFormalCharge(1)
        else:
            next(ob.OBAtomBondIter(hydrogen)).SetBondOrder(2)
    geometry.annotate_pyridyl([first, second])
    assert first.metadata[geometry.PYRIDYL_KEY] == 0


@pytest.mark.parametrize(
    "defect",
    [
        "toward_bond",
        "coincident",
        "nan",
        "infinite",
        "reverse",
        "hidden",
        "implicit_h",
        "radical",
        "deferred",
    ],
)
def test_pyridyl_invalid_donor_direction_abstains(pyridyl_pair, defect):
    first, second = pyridyl_pair
    for candidate in (first, second):
        mol = candidate.no_metal_state.omol.OBMol
        center, _, _ = next(geometry.pyramidal_pyridyl_centers(mol))
        metal = candidate.metal_states[0]
        position = (center.GetX(), center.GetY(), center.GetZ())
        if defect == "hidden":
            candidate.metadata["metal_visible_atom_groups"] = ""
            continue
        if defect == "implicit_h":
            center.SetImplicitHCount(1)
            continue
        if defect == "radical":
            set_unpaired_electron_count(center, 1)
            continue
        if defect == "deferred":
            set_unresolved_two_electron_center(center, True)
            continue
        if defect == "toward_bond":
            neighbor = next(ob.OBAtomAtomIter(center))
            xyz = (neighbor.GetX(), neighbor.GetY(), neighbor.GetZ())
        elif defect == "coincident":
            xyz = position
        elif defect == "reverse":
            xyz = tuple(
                2 * x - y
                for x, y in zip(position, (metal.position_x, metal.position_y, metal.position_z))
            )
        else:
            xyz = (
                float("nan") if defect == "nan" else float("inf"),
                metal.position_y,
                metal.position_z,
            )
        candidate.metal_states = (
            replace(metal, position_x=xyz[0], position_y=xyz[1], position_z=xyz[2]),
        )
    geometry.annotate_pyridyl([first, second])
    assert first.metadata[geometry.PYRIDYL_KEY] == 0


def test_pyridyl_direction_is_rigid_transform_invariant(pyridyl_pair):
    first, second = pyridyl_pair
    for candidate in (first, second):
        for atom in ob.OBMolAtomIter(candidate.no_metal_state.omol.OBMol):
            atom.SetVector(atom.GetZ() + 20, atom.GetX() - 30, atom.GetY() + 40)
        candidate.metal_states = tuple(
            replace(
                m,
                position_x=m.position_z + 20,
                position_y=m.position_x - 30,
                position_z=m.position_y + 40,
            )
            for m in candidate.metal_states
        )
    geometry.annotate_pyridyl([first, second])
    assert first.metadata[geometry.PYRIDYL_KEY] == 1


def test_pyridyl_anion_resonance_does_not_pin_charge(pyridyl_pair):
    first, second = pyridyl_pair
    ring, _, center = next(iter(geometry.pyridyl_sites(second)))
    mol = second.no_metal_state.omol.OBMol
    adjacent = next(a for a in ob.OBAtomAtomIter(mol.GetAtom(center)) if a.GetIdx() in ring)
    other = next(
        a
        for a in ob.OBAtomAtomIter(adjacent)
        if a.GetIdx() in ring and adjacent.GetBond(a).GetBondOrder() == 2
    )
    before = pybel_to_rdmol(second.no_metal_state.omol, sanitize=False, kekulize=False)
    mol.GetAtom(center).SetFormalCharge(0)
    other.SetFormalCharge(-1)
    mol.GetBond(center, adjacent.GetIdx()).SetBondOrder(2)
    adjacent.GetBond(other).SetBondOrder(1)
    after = pybel_to_rdmol(second.no_metal_state.omol, sanitize=False, kekulize=False)
    assert evaluate_equivalence(before, after, max_resonance=100000).decision.value == "equivalent"
    geometry.annotate_pyridyl([first, second])
    assert first.metadata[geometry.PYRIDYL_KEY] == 1


@pytest.mark.parametrize("case_id", ["DOQPOP", "HEXPUW", "QOXNID"])
def test_pyridyl_eta2_approach_does_not_imply_sigma_reduction(case_id):
    candidates = _candidates(case_id)
    assert any(
        list(geometry.pyramidal_pyridyl_centers(c.no_metal_state.omol.OBMol)) for c in candidates
    )
    geometry.annotate_pyridyl(candidates)
    assert all(c.metadata[geometry.PYRIDYL_KEY] == 0 for c in candidates)


@pytest.fixture(scope="module")
def pyramidal_ps_candidates():
    return _candidates("AKAGEZ")


@pytest.fixture
def pyramidal_ps_pair(pyramidal_ps_candidates):
    by_valence = {c.metal_states[0].valence: _copy(c) for c in pyramidal_ps_candidates}
    pair = by_valence[0], by_valence[2]
    geometry.clear_pyramidal_ps(pair)
    return pair


def test_pyramidal_ps_pair_requires_partner_and_is_idempotent(pyramidal_ps_pair):
    first, partner = pyramidal_ps_pair
    before = [c.no_metal_state.omol.write("mol") for c in (first, partner)]
    geometry.annotate_pyramidal_ps([first, partner])
    assert first.metadata[geometry.PYRAMIDAL_PS_KEY] == 1
    assert partner.metadata[geometry.PYRAMIDAL_PS_KEY] == 0
    count = first.metadata["metal_discordance_structural_count"]
    geometry.annotate_pyramidal_ps([first, partner])
    assert first.metadata["metal_discordance_structural_count"] == count
    assert before == [c.no_metal_state.omol.write("mol") for c in (first, partner)]
    geometry.annotate_pyramidal_ps([first])
    assert first.metadata[geometry.PYRAMIDAL_PS_KEY] == 0
    assert first.metadata["metal_discordance_structural_count"] == count - 1


@pytest.mark.parametrize(
    "defect",
    [
        "metal",
        "delta",
        "metal_spin",
        "ligand_spin",
        "no",
        "independent",
        "outside_atom",
        "boundary_bond",
        "invisible",
        "planar",
        "missing_geometry",
        "radical",
        "unresolved",
        "implicit_hydrogen",
        "nonterminal_sulfur",
    ],
)
def test_pyramidal_ps_abstains_on_incompatible_evidence(pyramidal_ps_pair, defect):
    first, partner = pyramidal_ps_pair
    site = next(iter(geometry.pyramidal_ps_sites(first)))
    mol = partner.no_metal_state.omol.OBMol
    p, sulfur = (mol.GetAtom(i) for i in site)
    neighbors = list(ob.OBAtomAtomIter(p))
    carbon = next(a for a in neighbors if a.GetAtomicNum() == 6)
    if defect in ("metal", "delta", "metal_spin"):
        attr, value = {
            "metal": ("idx", 1000),
            "delta": ("valence", 4),
            "metal_spin": ("radical_num", 2),
        }[defect]
        partner.metal_states = (replace(partner.metal_states[0], **{attr: value}),)
    elif defect == "ligand_spin":
        partner.no_metal_radical_target += 2
    elif defect == "no":
        partner.metadata["nitrosyl_geometry_penalty"] = 100
    elif defect == "independent":
        partner.metadata["metal_discordance_structural_count"] = (
            first.metadata["metal_discordance_structural_count"] + 1
        )
    elif defect == "outside_atom":
        carbon.SetFormalCharge(carbon.GetFormalCharge() - 1)
    elif defect == "boundary_bond":
        p.GetBond(carbon).SetBondOrder(2)
    elif defect == "invisible":
        partner.metadata["metal_visible_atom_indices"] = ""
    elif defect in ("planar", "missing_geometry"):
        import math

        p.SetVector(0, 0, 0)
        for i, atom in enumerate(neighbors):
            angle = 2 * math.pi * i / 3
            atom.SetVector(math.cos(angle), math.sin(angle), 0)
            if defect == "missing_geometry":
                atom.SetVector(0, 0, 0)
    elif defect == "radical":
        set_unpaired_electron_count(sulfur, 1)
    elif defect == "unresolved":
        set_unresolved_two_electron_center(sulfur, True)
    elif defect == "implicit_hydrogen":
        sulfur.SetImplicitHCount(1)
    else:
        mol.AddBond(sulfur.GetIdx(), carbon.GetIdx(), 1)
    geometry.annotate_pyramidal_ps([first, partner])
    assert first.metadata[geometry.PYRAMIDAL_PS_KEY] == 0


@pytest.mark.parametrize("scale", [0.4, 1, 3])
def test_pyramidal_ps_geometry_is_frame_invariant(pyramidal_ps_pair, scale):
    first, partner = pyramidal_ps_pair
    for candidate in (first, partner):
        for atom in ob.OBMolAtomIter(candidate.no_metal_state.omol.OBMol):
            x, y, z = atom.GetX(), atom.GetY(), atom.GetZ()
            atom.SetVector(3 + scale * y, 4 - scale * x, 5 + scale * z)
    geometry.annotate_pyramidal_ps([partner, first])
    assert first.metadata[geometry.PYRAMIDAL_PS_KEY] == 1


@pytest.fixture(scope="module")
def pyramidal_pn_candidates():
    return _candidates("BUTMIM")


@pytest.fixture
def pyramidal_pn_pair(pyramidal_pn_candidates):
    by_valence = {c.metal_states[0].valence: _copy(c) for c in pyramidal_pn_candidates}
    pair = by_valence[0], by_valence[2]
    geometry.clear_pyramidal_pn(pair)
    return pair


def test_pyramidal_pn_requires_complete_partner_and_preserves_graph(pyramidal_pn_pair):
    first, partner = pyramidal_pn_pair
    before = [c.no_metal_state.omol.write("mol") for c in (first, partner)]
    geometry.annotate_pyramidal_pn([partner, first])
    assert first.metadata[geometry.PYRAMIDAL_PN_KEY] == 1
    assert partner.metadata[geometry.PYRAMIDAL_PN_KEY] == 0
    count = first.metadata["metal_discordance_structural_count"]
    geometry.annotate_pyramidal_pn([first, partner])
    assert first.metadata["metal_discordance_structural_count"] == count
    assert before == [c.no_metal_state.omol.write("mol") for c in (first, partner)]
    geometry.annotate_pyramidal_pn([first])
    assert first.metadata[geometry.PYRAMIDAL_PN_KEY] == 0
    assert first.metadata["metal_discordance_structural_count"] == count - 1


@pytest.mark.parametrize(
    "defect",
    [
        "metal",
        "delta",
        "metal_spin",
        "ligand_spin",
        "no",
        "independent",
        "outside_atom",
        "boundary_bond",
        "invisible",
        "planar",
        "missing_geometry",
        "radical",
        "unresolved",
        "implicit_hydrogen",
        "short_pn",
        "long_pn",
        "non_phosphorus",
        "pi_active_carbon",
    ],
)
def test_pyramidal_pn_abstains_on_incompatible_evidence(pyramidal_pn_pair, defect):
    first, partner = pyramidal_pn_pair
    site = next(iter(geometry.pyramidal_pn_sites(first)))
    mol = partner.no_metal_state.omol.OBMol
    p, n = (mol.GetAtom(i) for i in site)
    neighbors = list(ob.OBAtomAtomIter(p))
    carbon = next(a for a in neighbors if a.GetAtomicNum() == 6)
    if defect in ("metal", "delta", "metal_spin"):
        attr, value = {
            "metal": ("idx", 1000),
            "delta": ("valence", 4),
            "metal_spin": ("radical_num", 2),
        }[defect]
        partner.metal_states = (replace(partner.metal_states[0], **{attr: value}),)
    elif defect == "ligand_spin":
        partner.no_metal_radical_target += 2
    elif defect == "no":
        partner.metadata["nitrosyl_geometry_penalty"] = 100
    elif defect == "independent":
        partner.metadata["metal_discordance_structural_count"] = (
            first.metadata["metal_discordance_structural_count"] + 1
        )
    elif defect == "outside_atom":
        carbon.SetFormalCharge(carbon.GetFormalCharge() - 1)
    elif defect == "boundary_bond":
        p.GetBond(carbon).SetBondOrder(2)
    elif defect == "invisible":
        partner.metadata["metal_visible_atom_indices"] = ""
    elif defect in ("planar", "missing_geometry"):
        import math

        p.SetVector(0, 0, 0)
        for i, a in enumerate(neighbors):
            angle = 2 * math.pi * i / 3
            a.SetVector(1.7 * math.cos(angle), 1.7 * math.sin(angle), 0)
            if defect == "missing_geometry":
                a.SetVector(0, 0, 0)
    elif defect in ("short_pn", "long_pn"):
        length = 1.5 if defect == "short_pn" else 2.0
        ratio = length / p.GetDistance(n)
        n.SetVector(
            p.GetX() + ratio * (n.GetX() - p.GetX()),
            p.GetY() + ratio * (n.GetY() - p.GetY()),
            p.GetZ() + ratio * (n.GetZ() - p.GetZ()),
        )
    elif defect == "radical":
        set_unpaired_electron_count(n, 1)
    elif defect == "unresolved":
        set_unresolved_two_electron_center(n, True)
    elif defect == "implicit_hydrogen":
        n.SetImplicitHCount(1)
    elif defect == "pi_active_carbon":
        carbon.SetFormalCharge(-1)
    else:
        next(a for a in ob.OBAtomAtomIter(n) if a.GetIdx() != p.GetIdx()).SetAtomicNum(6)
    geometry.annotate_pyramidal_pn([first, partner])
    assert first.metadata[geometry.PYRAMIDAL_PN_KEY] == 0


def test_pyramidal_pn_is_frame_invariant(pyramidal_pn_pair):
    for c in pyramidal_pn_pair:
        for a in ob.OBMolAtomIter(c.no_metal_state.omol.OBMol):
            x, y, z = a.GetX(), a.GetY(), a.GetZ()
            a.SetVector(3 + y, 4 - x, 5 + z)
    geometry.annotate_pyramidal_pn(list(reversed(pyramidal_pn_pair)))
    assert pyramidal_pn_pair[0].metadata[geometry.PYRAMIDAL_PN_KEY] == 1


def test_pyramidal_pn_excludes_coupled_carbon_redox_control():
    candidates = _candidates("COLPOI")
    assert all(not geometry.pyramidal_pn_sites(c) for c in candidates)


@pytest.fixture(scope="module")
def ortho_dioxygen_candidates():
    return _candidates("REDKOD")


@pytest.fixture
def ortho_dioxygen_pair(ortho_dioxygen_candidates):
    return _pair(
        ortho_dioxygen_candidates,
        geometry.annotate_ortho_dioxygen,
        geometry.ORTHO_DIOXYGEN_KEY,
    )


def _dioxygen_core(first, second):
    own, other = geometry.ortho_dioxygen_sites(first), geometry.ortho_dioxygen_sites(second)
    return next(site for site, reduced in own.items() if not reduced and other.get(site) is True)


def test_ortho_dioxygen_requires_partner_and_preserves_graphs(ortho_dioxygen_pair):
    first, second = ortho_dioxygen_pair
    before = [geometry.outside_signature(c, set()) for c in (first, second)]
    for pool in ([first, second], [second, first], [first, second]):
        geometry.annotate_ortho_dioxygen(pool)
        assert first.metadata[geometry.ORTHO_DIOXYGEN_KEY] == 1
        assert second.metadata[geometry.ORTHO_DIOXYGEN_KEY] == 0
    geometry.annotate_ortho_dioxygen([first])
    assert first.metadata[geometry.ORTHO_DIOXYGEN_KEY] == 0
    assert before == [geometry.outside_signature(c, set()) for c in (first, second)]


@pytest.mark.parametrize("defect", PARTNER_DEFECTS)
def test_ortho_dioxygen_incompatible_partner_abstains(ortho_dioxygen_pair, defect):
    first, second = ortho_dioxygen_pair
    core = set(_dioxygen_core(first, second))
    if defect in ("metal", "delta", "metal_spin"):
        attr, value = {
            "metal": ("idx", 1000),
            "delta": ("valence", second.metal_states[0].valence + 2),
            "metal_spin": ("radical_num", second.metal_states[0].radical_num + 2),
        }[defect]
        second.metal_states = (replace(second.metal_states[0], **{attr: value}),)
    elif defect == "ligand_spin":
        second.no_metal_radical_target += 2
    elif defect == "no":
        second.metadata["nitrosyl_geometry_penalty"] += 1
    elif defect == "independent":
        second.metadata["metal_discordance_structural_count"] = (
            first.metadata["metal_discordance_structural_count"] + 1
        )
    else:
        mol = second.no_metal_state.omol.OBMol
        hydrogen = next(
            a
            for a in ob.OBMolAtomIter(mol)
            if a.GetAtomicNum() == 1 and next(ob.OBAtomAtomIter(a)).GetIdx() in core
        )
        if defect == "outside_atom":
            hydrogen.SetFormalCharge(1)
        else:
            next(ob.OBAtomBondIter(hydrogen)).SetBondOrder(2)
    geometry.annotate_ortho_dioxygen([first, second])
    assert first.metadata[geometry.ORTHO_DIOXYGEN_KEY] == 0


@pytest.mark.parametrize(
    "defect",
    [
        "radical",
        "deferred",
        "implicit_h",
        "missing",
        "nan",
        "visibility",
        "short_co",
        "long_co",
        "nonplanar",
        "unequal_ring",
    ],
)
def test_ortho_dioxygen_geometry_abstains(ortho_dioxygen_pair, defect):
    first, second = ortho_dioxygen_pair
    site = _dioxygen_core(first, second)
    for candidate in (first, second):
        mol = candidate.no_metal_state.omol.OBMol
        oxygen = mol.GetAtom(site[-1])
        carbon = next(a for a in ob.OBAtomAtomIter(oxygen) if a.GetIdx() in site[:6])
        if defect == "radical":
            set_unpaired_electron_count(oxygen, 1)
        elif defect == "deferred":
            set_unresolved_two_electron_center(oxygen, True)
        elif defect == "implicit_h":
            oxygen.SetImplicitHCount(1)
        elif defect == "visibility":
            candidate.metadata["metal_visible_atom_groups"] = ""
        elif defect in ("short_co", "long_co"):
            scale = (1.23 if defect == "short_co" else 1.55) / oxygen.GetDistance(carbon)
            oxygen.SetVector(
                carbon.GetX() + (oxygen.GetX() - carbon.GetX()) * scale,
                carbon.GetY() + (oxygen.GetY() - carbon.GetY()) * scale,
                carbon.GetZ() + (oxygen.GetZ() - carbon.GetZ()) * scale,
            )
        elif defect in ("nonplanar", "unequal_ring"):
            atom = mol.GetAtom(site[2])
            delta = 1.0 if defect == "nonplanar" else 3.0
            atom.SetVector(atom.GetX(), atom.GetY(), atom.GetZ() + delta)
        else:
            for atom in ob.OBMolAtomIter(mol):
                atom.SetVector(0, 0, 0 if defect == "missing" else float("nan"))
    geometry.annotate_ortho_dioxygen([first, second])
    assert first.metadata[geometry.ORTHO_DIOXYGEN_KEY] == 0


def test_ortho_dioxygen_rigid_motion_invariance(ortho_dioxygen_pair):
    first, second = ortho_dioxygen_pair
    for candidate in (first, second):
        for atom in ob.OBMolAtomIter(candidate.no_metal_state.omol.OBMol):
            atom.SetVector(atom.GetY() + 30, atom.GetZ() - 50, atom.GetX() + 10)
    geometry.annotate_ortho_dioxygen([first, second])
    assert first.metadata[geometry.ORTHO_DIOXYGEN_KEY] == 1
    assert second.metadata[geometry.ORTHO_DIOXYGEN_KEY] == 0


@pytest.fixture(
    scope="module", params=[("aryl_amido_nhc", "LUQXAZ"), ("aryl_phenoxide_nhc", "APOVOS")]
)
def aryl_donor_nhc_candidates(request):
    kind, case_id = request.param
    return kind, _candidates(case_id)


@pytest.fixture
def aryl_donor_nhc_pair(aryl_donor_nhc_candidates):
    kind, candidates = aryl_donor_nhc_candidates
    return kind, _pair(
        candidates, getattr(geometry, "annotate_" + kind), getattr(geometry, kind.upper() + "_KEY")
    )


def _donor_nhc_core(kind, first, second):
    sites = getattr(geometry, kind + "_sites")
    own, other = sites(first), sites(second)
    return next(site for site, reduced in own.items() if not reduced and other.get(site) is True)


def test_aryl_donor_nhc_requires_partner_and_preserves_graphs(aryl_donor_nhc_pair):
    kind, (first, second) = aryl_donor_nhc_pair
    annotate = getattr(geometry, "annotate_" + kind)
    key = getattr(geometry, kind.upper() + "_KEY")
    before = [geometry.outside_signature(c, set()) for c in (first, second)]
    for pool in ([first, second], [second, first], [first, second]):
        annotate(pool)
        assert first.metadata[key] == 1
        assert second.metadata[key] == 0
    annotate([first])
    assert first.metadata[key] == 0
    assert before == [geometry.outside_signature(c, set()) for c in (first, second)]


@pytest.mark.parametrize("defect", PARTNER_DEFECTS)
def test_aryl_donor_nhc_incompatible_partner_abstains(aryl_donor_nhc_pair, defect):
    kind, (first, second) = aryl_donor_nhc_pair
    annotate = getattr(geometry, "annotate_" + kind)
    key = getattr(geometry, kind.upper() + "_KEY")
    core = set(_donor_nhc_core(kind, first, second))
    if defect in ("metal", "delta", "metal_spin"):
        attr, value = {
            "metal": ("idx", 1000),
            "delta": ("valence", second.metal_states[0].valence + 2),
            "metal_spin": ("radical_num", second.metal_states[0].radical_num + 2),
        }[defect]
        second.metal_states = (replace(second.metal_states[0], **{attr: value}),)
    elif defect == "ligand_spin":
        second.no_metal_radical_target += 2
    elif defect == "no":
        second.metadata["nitrosyl_geometry_penalty"] += 1
    elif defect == "independent":
        second.metadata["metal_discordance_structural_count"] = (
            first.metadata["metal_discordance_structural_count"] + 1
        )
    else:
        mol = second.no_metal_state.omol.OBMol
        hydrogen = next(
            a
            for a in ob.OBMolAtomIter(mol)
            if a.GetAtomicNum() == 1 and next(ob.OBAtomAtomIter(a)).GetIdx() in core
        )
        if defect == "outside_atom":
            hydrogen.SetFormalCharge(1)
        else:
            next(ob.OBAtomBondIter(hydrogen)).SetBondOrder(2)
    annotate([first, second])
    assert first.metadata[key] == 0


@pytest.mark.parametrize(
    "defect",
    [
        "radical",
        "deferred",
        "implicit_h",
        "missing",
        "nan",
        "visibility",
        "split_visibility",
        "short_cn",
        "long_cn",
        "short_arm_cn",
        "pyramidal_nhc_n",
        "nonplanar_aryl",
        "nonplanar_nhc",
        "carbene_charge",
        "backbone_order",
        "nhc_element",
    ],
)
def test_aryl_donor_nhc_invalid_local_motif_abstains(aryl_donor_nhc_pair, defect):
    kind, (first, second) = aryl_donor_nhc_pair
    annotate = getattr(geometry, "annotate_" + kind)
    key = getattr(geometry, kind.upper() + "_KEY")
    site = _donor_nhc_core(kind, first, second)
    for candidate in (first, second):
        mol = candidate.no_metal_state.omol.OBMol
        amido = mol.GetAtom(site[-1])
        carbon = next(a for a in ob.OBAtomAtomIter(amido) if a.GetIdx() in site[:6])
        azole = [mol.GetAtom(i) for i in site[6:11]]
        carbene = next(a for a in azole if a.GetAtomicNum() == 6 and a.GetExplicitDegree() == 2)
        backbone = [a for a in azole if a.GetAtomicNum() == 6 and a.GetIdx() != carbene.GetIdx()]
        if defect == "radical":
            set_unpaired_electron_count(carbene, 1)
        elif defect == "deferred":
            set_unresolved_two_electron_center(carbene, True)
        elif defect == "implicit_h":
            carbene.SetImplicitHCount(1)
        elif defect == "visibility":
            candidate.metadata["metal_visible_atom_groups"] = ""
        elif defect == "split_visibility":
            candidate.metadata["metal_visible_atom_groups"] = f"{amido.GetIdx()};{carbene.GetIdx()}"
        elif defect in ("short_cn", "long_cn"):
            scale = (1.25 if defect == "short_cn" else 1.6) / amido.GetDistance(carbon)
            amido.SetVector(
                carbon.GetX() + (amido.GetX() - carbon.GetX()) * scale,
                carbon.GetY() + (amido.GetY() - carbon.GetY()) * scale,
                carbon.GetZ() + (amido.GetZ() - carbon.GetZ()) * scale,
            )
        elif defect in ("short_arm_cn", "pyramidal_nhc_n"):
            arm = next(
                a
                for a in azole
                if a.GetAtomicNum() == 7
                and any(n.GetIdx() in site[:6] for n in ob.OBAtomAtomIter(a))
            )
            aryl = next(n for n in ob.OBAtomAtomIter(arm) if n.GetIdx() in site[:6])
            if defect == "short_arm_cn":
                scale = 1.25 / arm.GetDistance(aryl)
                arm.SetVector(
                    aryl.GetX() + (arm.GetX() - aryl.GetX()) * scale,
                    aryl.GetY() + (arm.GetY() - aryl.GetY()) * scale,
                    aryl.GetZ() + (arm.GetZ() - aryl.GetZ()) * scale,
                )
            else:
                import numpy as np

                neighbors = list(ob.OBAtomAtomIter(arm))
                positions = np.array([(a.GetX(), a.GetY(), a.GetZ()) for a in neighbors])
                normal = np.cross(positions[1] - positions[0], positions[2] - positions[0])
                normal /= np.linalg.norm(normal)
                moved = positions.mean(axis=0) + 1.0 * normal
                arm.SetVector(*moved)
        elif defect in ("nonplanar_aryl", "nonplanar_nhc"):
            atom = mol.GetAtom(site[2]) if defect == "nonplanar_aryl" else backbone[0]
            atom.SetVector(atom.GetX(), atom.GetY(), atom.GetZ() + 1.0)
        elif defect == "carbene_charge":
            carbene.SetFormalCharge(0)
        elif defect == "backbone_order":
            bond = mol.GetBond(*backbone)
            bond.SetBondOrder(3 - bond.GetBondOrder())
        elif defect == "nhc_element":
            backbone[0].SetAtomicNum(14)
        else:
            for atom in ob.OBMolAtomIter(mol):
                atom.SetVector(0, 0, 0 if defect == "missing" else float("nan"))
    annotate([first, second])
    assert first.metadata[key] == 0


def test_aryl_donor_nhc_accepts_both_charge_separated_contributors(aryl_donor_nhc_pair):
    kind, (first, second) = aryl_donor_nhc_pair
    annotate = getattr(geometry, "annotate_" + kind)
    key = getattr(geometry, kind.upper() + "_KEY")
    site = _donor_nhc_core(kind, first, second)
    mol = second.no_metal_state.omol.OBMol
    carbene = next(
        mol.GetAtom(i)
        for i in site[6:11]
        if mol.GetAtom(i).GetAtomicNum() == 6 and mol.GetAtom(i).GetExplicitDegree() == 2
    )
    for n in ob.OBAtomAtomIter(carbene):
        bond = carbene.GetBond(n)
        bond.SetBondOrder(3 - bond.GetBondOrder())
        n.SetFormalCharge(1 - n.GetFormalCharge())
    assert carbene.GetFormalCharge() == -1
    assert getattr(geometry, kind + "_sites")(second)[site] is True
    annotate([first, second])
    assert first.metadata[key] == 1


def test_aryl_donor_nhc_rigid_motion_invariance(aryl_donor_nhc_pair):
    kind, (first, second) = aryl_donor_nhc_pair
    annotate = getattr(geometry, "annotate_" + kind)
    key = getattr(geometry, kind.upper() + "_KEY")
    for candidate in (first, second):
        for atom in ob.OBMolAtomIter(candidate.no_metal_state.omol.OBMol):
            atom.SetVector(atom.GetY() + 30, atom.GetZ() - 50, atom.GetX() + 10)
    annotate([second, first])
    assert first.metadata[key] == 1
    assert second.metadata[key] == 0


def test_aryl_donor_nhc_atom_order_invariance(aryl_donor_nhc_pair):
    kind, (first, second) = aryl_donor_nhc_pair
    annotate = getattr(geometry, "annotate_" + kind)
    key = getattr(geometry, kind.upper() + "_KEY")
    for candidate in (first, second):
        mol = candidate.no_metal_state.omol.OBMol
        new, mapping = ob.OBMol(), {}
        for i in range(mol.NumAtoms(), 0, -1):
            new.AddAtom(mol.GetAtom(i))
            mapping[i] = new.NumAtoms()
        for bond in ob.OBMolBondIter(mol):
            new.AddBond(
                mapping[bond.GetBeginAtomIdx()], mapping[bond.GetEndAtomIdx()], bond.GetBondOrder()
            )
        candidate.no_metal_state = replace(candidate.no_metal_state, omol=pybel.Molecule(new))
        candidate.metadata["metal_visible_atom_groups"] = ";".join(
            ",".join(str(mapping[int(i)]) for i in group.split(",") if i)
            for group in candidate.metadata["metal_visible_atom_groups"].split(";")
        )
    annotate([first, second])
    assert first.metadata[key] == 1
    assert second.metadata[key] == 0


@pytest.fixture(scope="module")
def amino_c5_candidates():
    return _candidates("FUWZII")


@pytest.fixture
def amino_c5_pair(amino_c5_candidates):
    return _pair(
        amino_c5_candidates,
        geometry.annotate_amino_c5,
        geometry.AMINO_C5_KEY,
    )


def _amino_c5_core(first, second):
    own, other = geometry.amino_c5_sites(first), geometry.amino_c5_sites(second)
    return next(site for site, reduced in own.items() if not reduced and other.get(site) is True)


def test_amino_c5_requires_partner_and_preserves_graphs(amino_c5_pair):
    first, second = amino_c5_pair
    before = [geometry.outside_signature(c, set()) for c in (first, second)]
    for pool in ([first, second], [second, first], [first, second]):
        geometry.annotate_amino_c5(pool)
        assert first.metadata[geometry.AMINO_C5_KEY] == 1
        assert second.metadata[geometry.AMINO_C5_KEY] == 0
    geometry.annotate_amino_c5([first])
    assert first.metadata[geometry.AMINO_C5_KEY] == 0
    assert before == [geometry.outside_signature(c, set()) for c in (first, second)]


@pytest.mark.parametrize("defect", PARTNER_DEFECTS)
def test_amino_c5_incompatible_partner_abstains(amino_c5_pair, defect):
    first, second = amino_c5_pair
    core = set(_amino_c5_core(first, second))
    if defect in ("metal", "delta", "metal_spin"):
        attr, value = {
            "metal": ("idx", 1000),
            "delta": ("valence", second.metal_states[0].valence + 2),
            "metal_spin": ("radical_num", second.metal_states[0].radical_num + 2),
        }[defect]
        second.metal_states = (replace(second.metal_states[0], **{attr: value}),)
    elif defect == "ligand_spin":
        second.no_metal_radical_target += 2
    elif defect == "no":
        second.metadata["nitrosyl_geometry_penalty"] += 1
    elif defect == "independent":
        second.metadata["metal_discordance_structural_count"] = (
            first.metadata["metal_discordance_structural_count"] + 1
        )
    else:
        mol = second.no_metal_state.omol.OBMol
        hydrogen = next(
            a
            for a in ob.OBMolAtomIter(mol)
            if a.GetAtomicNum() == 1 and next(ob.OBAtomAtomIter(a)).GetIdx() in core
        )
        if defect == "outside_atom":
            hydrogen.SetFormalCharge(1)
        else:
            next(ob.OBAtomBondIter(hydrogen)).SetBondOrder(2)
    geometry.annotate_amino_c5([first, second])
    assert first.metadata[geometry.AMINO_C5_KEY] == 0


def test_amino_c5_atom_order_invariance(amino_c5_pair):
    first, second = amino_c5_pair
    for candidate in (first, second):
        mol = candidate.no_metal_state.omol.OBMol
        new, mapping = ob.OBMol(), {}
        for i in range(mol.NumAtoms(), 0, -1):
            new.AddAtom(mol.GetAtom(i))
            mapping[i] = new.NumAtoms()
        for bond in ob.OBMolBondIter(mol):
            new.AddBond(
                mapping[bond.GetBeginAtomIdx()], mapping[bond.GetEndAtomIdx()], bond.GetBondOrder()
            )
        candidate.no_metal_state = replace(candidate.no_metal_state, omol=pybel.Molecule(new))
        candidate.metadata["metal_visible_atom_groups"] = ";".join(
            ",".join(str(mapping[int(i)]) for i in group.split(",") if i)
            for group in candidate.metadata["metal_visible_atom_groups"].split(";")
        )
    geometry.annotate_amino_c5([first, second])
    assert first.metadata[geometry.AMINO_C5_KEY] == 1
    assert second.metadata[geometry.AMINO_C5_KEY] == 0


@pytest.mark.parametrize(
    "defect",
    [
        "radical",
        "deferred",
        "implicit_h",
        "missing",
        "nan",
        "visibility",
        "split_visibility",
        "short_cn",
        "long_cn",
        "nonplanar_c5",
        "nonplanar_c6",
        "amine_pyramidal",
        "unsaturated_substituent",
        "heteroaryl",
        "fused_ring_charge",
        "ring_charge",
    ],
)
def test_amino_c5_invalid_local_motif_abstains(amino_c5_pair, defect):
    first, second = amino_c5_pair
    site = _amino_c5_core(first, second)
    for candidate in (first, second):
        mol = candidate.no_metal_state.omol.OBMol
        nitrogen = mol.GetAtom(site[-1])
        carbon = next(a for a in ob.OBAtomAtomIter(nitrogen) if a.GetIdx() in site[:-1])
        five = next(list(r._path) for r in mol.GetSSSR() if len(r._path) == 5)
        fused = next(i for i in site[:-1] if i not in five)
        if defect == "radical":
            set_unpaired_electron_count(nitrogen, 1)
        elif defect == "deferred":
            set_unresolved_two_electron_center(nitrogen, True)
        elif defect == "implicit_h":
            nitrogen.SetImplicitHCount(1)
        elif defect == "visibility":
            candidate.metadata["metal_visible_atom_groups"] = ""
        elif defect == "split_visibility":
            candidate.metadata["metal_visible_atom_groups"] = ";".join(str(i) for i in five)
        elif defect in ("short_cn", "long_cn"):
            scale = (1.25 if defect == "short_cn" else 1.6) / nitrogen.GetDistance(carbon)
            nitrogen.SetVector(
                carbon.GetX() + (nitrogen.GetX() - carbon.GetX()) * scale,
                carbon.GetY() + (nitrogen.GetY() - carbon.GetY()) * scale,
                carbon.GetZ() + (nitrogen.GetZ() - carbon.GetZ()) * scale,
            )
        elif defect in ("nonplanar_c5", "nonplanar_c6", "amine_pyramidal"):
            atom = (
                mol.GetAtom(five[0] if defect == "nonplanar_c5" else fused)
                if defect != "amine_pyramidal"
                else nitrogen
            )
            atom.SetVector(atom.GetX(), atom.GetY(), atom.GetZ() + 1.0)
        elif defect == "unsaturated_substituent":
            neighbor = next(a for a in ob.OBAtomAtomIter(nitrogen) if a.GetIdx() not in site)
            neighbor.SetFormalCharge(-1)
        elif defect == "heteroaryl":
            mol.GetAtom(fused).SetAtomicNum(7)
        elif defect == "fused_ring_charge":
            mol.GetAtom(fused).SetFormalCharge(-1)
        elif defect == "ring_charge":
            carbon.SetFormalCharge(1)
        else:
            for atom in ob.OBMolAtomIter(mol):
                atom.SetVector(0, 0, 0 if defect == "missing" else float("nan"))
    geometry.annotate_amino_c5([first, second])
    assert first.metadata[geometry.AMINO_C5_KEY] == 0


def test_amino_c5_rigid_motion_invariance(amino_c5_pair):
    first, second = amino_c5_pair
    for candidate in (first, second):
        for atom in ob.OBMolAtomIter(candidate.no_metal_state.omol.OBMol):
            atom.SetVector(atom.GetY() + 30, atom.GetZ() - 50, atom.GetX() + 10)
    geometry.annotate_amino_c5([second, first])
    assert first.metadata[geometry.AMINO_C5_KEY] == 1
    assert second.metadata[geometry.AMINO_C5_KEY] == 0


def test_amino_c5_also_accepts_unfused_cyclopentadienyl(amino_c5_pair):
    """The preference is a ring/amine redox comparison, not an indenyl ID rule."""
    import math

    first, second = amino_c5_pair
    for candidate, reduced in ((first, False), (second, True)):
        mol = ob.OBMol()
        radius = 1.42 / (2 * math.sin(math.pi / 5))
        # C1 is the anionic contributor; C3 carries NMe2. Rotate C3 outward
        # along +x to give an exactly planar tertiary nitrogen.
        for i in range(5):
            angle = 2 * math.pi * (i - 2) / 5
            atom = mol.NewAtom()
            atom.SetAtomicNum(6)
            atom.SetFormalCharge(-1 if reduced and i == 0 else 0)
            atom.SetVector(radius * math.cos(angle), radius * math.sin(angle), 0)
        nitrogen = mol.NewAtom()
        nitrogen.SetAtomicNum(7)
        nitrogen.SetFormalCharge(0 if reduced else 1)
        nitrogen.SetVector(radius + 1.40, 0, 0)
        orders = [1, 2, 1, 2, 1] if reduced else [2, 1, 1, 2, 1]
        for i, order in enumerate(orders):
            mol.AddBond(i + 1, (i + 1) % 5 + 1, order)
        mol.AddBond(3, 6, 1 if reduced else 2)
        for i in (0, 1, 3, 4):
            angle = 2 * math.pi * (i - 2) / 5
            atom = mol.NewAtom()
            atom.SetAtomicNum(1)
            atom.SetVector((radius + 1.09) * math.cos(angle), (radius + 1.09) * math.sin(angle), 0)
            mol.AddBond(i + 1, atom.GetIdx(), 1)
        for sign in (-1, 1):
            carbon = mol.NewAtom()
            carbon.SetAtomicNum(6)
            carbon.SetVector(radius + 1.40 + 0.735, sign * 1.47 * math.sqrt(3) / 2, 0)
            mol.AddBond(6, carbon.GetIdx(), 1)
            for j in range(3):
                hydrogen = mol.NewAtom()
                hydrogen.SetAtomicNum(1)
                hydrogen.SetVector(carbon.GetX() + 1, carbon.GetY() + j * 0.4, sign * 0.3)
                mol.AddBond(carbon.GetIdx(), hydrogen.GetIdx(), 1)
        candidate.no_metal_state = replace(candidate.no_metal_state, omol=pybel.Molecule(mol))
        candidate.metadata["metal_visible_atom_groups"] = "1,2,3,4,5"
        assert geometry.amino_c5_sites(candidate) == {(1, 2, 3, 4, 5, 6): reduced}
    geometry.annotate_amino_c5([first, second])
    assert first.metadata[geometry.AMINO_C5_KEY] == 1
    assert second.metadata[geometry.AMINO_C5_KEY] == 0
