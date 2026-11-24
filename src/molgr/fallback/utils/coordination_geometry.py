"""State-independent contact geometry used during ligand preparation and metal scoring."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import DefaultDict, cast

import numpy as np
from openbabel import openbabel as ob
from openbabel import pybel

from molgr.config import MetalScoringConfig
from molgr.fallback.utils.dataclasses import MetalSite
from molgr.fallback.utils.ob_graph import atom_neighbors, mol_atoms
from molgr.utils.coordination import coordination_distance_cutoff
from molgr.utils.coordination_visibility import (
    CoordinationBlockerArrays,
    Point3D,
    coordination_visibility_mask,
    empty_coordination_blocker_arrays,
)


def _atom_coordinates(atom: ob.OBAtom) -> Point3D:
    """Return Cartesian coordinates in ångströms for the metal–donor visibility test."""
    return float(atom.GetX()), float(atom.GetY()), float(atom.GetZ())


def build_coordination_blocker_arrays(
    obmol: ob.OBMol,
    *,
    metal_scoring_config: MetalScoringConfig,
) -> CoordinationBlockerArrays:
    """Represent ligand atoms as spheres for the metal-to-donor visibility test."""
    blocker_indices: list[int] = []
    blocker_coordinates: list[Point3D] = []
    blocker_radii: list[float] = []
    for blocker in mol_atoms(obmol):
        blocker_radius = (
            metal_scoring_config.metal_access_radius_scale
            * float(ob.GetCovalentRad(int(blocker.GetAtomicNum())))
            + metal_scoring_config.metal_access_clearance_angstrom
        )
        if blocker_radius <= 0.0:
            continue
        blocker_indices.append(int(blocker.GetIdx()))
        blocker_coordinates.append(_atom_coordinates(blocker))
        blocker_radii.append(blocker_radius)
    if not blocker_indices:
        return empty_coordination_blocker_arrays()
    return (
        np.asarray(blocker_indices, dtype=np.int64),
        np.asarray(blocker_coordinates, dtype=np.float64),
        np.asarray(blocker_radii, dtype=np.float64),
    )


def inner_visible_atoms_to_metal(
    atoms: Sequence[ob.OBAtom],
    metal_state: MetalSite,
    *,
    metal_scoring_config: MetalScoringConfig,
    blocker_arrays: CoordinationBlockerArrays,
) -> tuple[ob.OBAtom, ...]:
    """Select donors inside the element-pair cutoff whose approach to the metal is unobstructed."""
    if not atoms:
        return ()

    atom_indices = np.asarray([int(atom.GetIdx()) for atom in atoms], dtype=np.int64)
    atom_coordinates = np.asarray([_atom_coordinates(atom) for atom in atoms], dtype=np.float64)
    metal_coordinates = np.asarray(
        (
            float(metal_state.position_x),
            float(metal_state.position_y),
            float(metal_state.position_z),
        ),
        dtype=np.float64,
    )
    cutoff = np.asarray(
        [
            coordination_distance_cutoff(
                int(metal_state.element_idx),
                int(atom.GetAtomicNum()),
                radius_scale=metal_scoring_config.metal_access_radius_scale,
                extra_tolerance_angstrom=(
                    metal_scoring_config.metal_coordination_extra_tolerance_angstrom
                ),
            )
            for atom in atoms
        ],
        dtype=np.float64,
    )
    delta = atom_coordinates - metal_coordinates
    distance_sq = np.einsum("ij,ij->i", delta, delta)
    inner_mask = (distance_sq > 0.0) & (distance_sq <= cutoff * cutoff)
    if not np.any(inner_mask):
        return ()

    inner_indices = atom_indices[inner_mask]
    inner_coordinates = atom_coordinates[inner_mask]
    visible_mask = coordination_visibility_mask(
        inner_indices,
        inner_coordinates,
        metal_coordinates,
        blocker_arrays,
    )
    inner_atoms = [atom for atom, is_inner in zip(atoms, inner_mask) if is_inner]
    return tuple(atom for atom, is_visible in zip(inner_atoms, visible_mask) if is_visible)


def metal_contact_distances(
    mol: ob.OBMol, sites: Sequence[MetalSite], config: MetalScoringConfig
) -> tuple[int, ...]:
    """Nearest visible inner-sphere contact in milliångströms, indexed like OBMol."""
    atoms = list(mol_atoms(mol))
    blockers = build_coordination_blocker_arrays(mol, metal_scoring_config=config)
    missing = 2**31 - 1
    distances = [missing] * (len(atoms) + 1)
    for site in sites:
        visible = inner_visible_atoms_to_metal(
            atoms, site, metal_scoring_config=config, blocker_arrays=blockers
        )
        point = (site.position_x, site.position_y, site.position_z)
        for atom in visible:
            distance = math.dist(_atom_coordinates(atom), point)
            distances[atom.GetIdx()] = min(distances[atom.GetIdx()], int(distance * 1000 + 0.5))
    return tuple(distances) if any(d < missing for d in distances) else ()


def metal_donor_angle(site: MetalSite, donor: ob.OBAtom, terminal: ob.OBAtom) -> float:
    """M–donor–terminal angle in degrees for a metal outside the ligand OBMol."""
    center = _atom_coordinates(donor)
    metal = (site.position_x, site.position_y, site.position_z)
    approach = tuple(x - y for x, y in zip(metal, center))
    bond = tuple(x - y for x, y in zip(_atom_coordinates(terminal), center))
    denominator = math.sqrt(sum(x * x for x in approach)) * math.sqrt(sum(x * x for x in bond))
    if not math.isfinite(denominator) or denominator == 0:
        return 0.0
    cosine = sum(a * b for a, b in zip(approach, bond)) / denominator
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def terminal_nitrosyl_contacts(
    obmol: ob.OBMol,
    metal_sites: Sequence[MetalSite],
    visible_atoms_by_metal: Sequence[Sequence[ob.OBAtom]],
) -> Iterator[tuple[ob.OBAtom, ob.OBAtom, float]]:
    """Yield short terminal N-bound contacts with their M–N–O geometry weights."""
    visible_sites: DefaultDict[int, list[int]] = defaultdict(list)
    for site, atoms in enumerate(visible_atoms_by_metal):
        for atom in atoms:
            visible_sites[int(atom.GetIdx())].append(site)
    for nitrogen in mol_atoms(obmol):
        if (
            nitrogen.GetAtomicNum() != 7
            or nitrogen.GetExplicitDegree() != 1
            or nitrogen.GetImplicitHCount() != 0
            or len(visible_sites[int(nitrogen.GetIdx())]) != 1
        ):
            continue
        oxygen = next(atom_neighbors(nitrogen))
        if (
            oxygen.GetAtomicNum() != 8
            or oxygen.GetExplicitDegree() != 1
            or oxygen.GetImplicitHCount() != 0
            or visible_sites[int(oxygen.GetIdx())]
        ):
            continue
        metal = metal_sites[visible_sites[int(nitrogen.GetIdx())][0]]
        no_length = math.dist(_atom_coordinates(nitrogen), _atom_coordinates(oxygen))
        if not 1.0 < no_length < 1.25:
            continue
        angle = metal_donor_angle(metal, nitrogen, oxygen)
        if angle <= 160.0:
            continue
        weight = min(1.0, (angle - 160.0) / 20.0) * min(1.0, (1.25 - no_length) / 0.10)
        yield nitrogen, oxygen, weight


def detach_carbonyl_hydride_contacts(omol: pybel.Molecule, metals: Sequence[MetalSite]) -> None:
    """Remove a stretched formyl C-H only with a compact hydride and a linear M-C-O."""
    mol = cast(ob.OBMol, omol.OBMol)
    for h in mol_atoms(mol):
        if h.GetAtomicNum() != 1 or h.GetExplicitDegree() != 1:
            continue
        carbon = next(atom_neighbors(h))
        if (
            carbon.GetAtomicNum() != 6
            or carbon.GetExplicitDegree() != 2
            or h.GetDistance(carbon) <= ob.GetCovalentRad(1) + ob.GetCovalentRad(6) + 0.20
        ):
            continue
        oxygen = next(
            (
                a
                for a in atom_neighbors(carbon)
                if a.GetAtomicNum() == 8 and a.GetExplicitDegree() == 1
            ),
            None,
        )
        if oxygen is None or carbon.GetDistance(oxygen) > 1.25:
            continue
        for metal in metals:
            point = (metal.position_x, metal.position_y, metal.position_z)
            radius = ob.GetCovalentRad(metal.element_idx)
            if (
                math.dist(point, _atom_coordinates(h)) <= radius + ob.GetCovalentRad(1)
                and math.dist(point, _atom_coordinates(carbon)) <= radius + ob.GetCovalentRad(6)
                and metal_donor_angle(metal, carbon, oxygen) > 160.0
            ):
                mol.DeleteBond(h.GetBond(carbon))
                break
