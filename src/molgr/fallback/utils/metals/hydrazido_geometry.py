"""Retain terminal hydrazido alternatives supported by metal--N--N geometry.

The detached ligand is also a valid neutral isodiazene. Explore both electronic
states; only compare a local two-electron transfer when every other ligand
label and bond is identical. Geometry is evidence for this representation
preference, not a general oxidation-state assignment rule.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Tuple, cast

from openbabel import openbabel as ob

from molgr.fallback.state import MetalCandidateState
from molgr.fallback.utils.coordination_seeds import hydrazido_sites, supports_hydrazido
from molgr.fallback.utils.electrons import (
    get_lone_pair_count,
    get_unpaired_electron_count,
    has_unresolved_two_electron_center,
)
from molgr.fallback.utils.ob_graph import mol_atoms, mol_bonds


AtomLabels = Tuple[int, int, Optional[int], int, int, bool, int, float, float, float]
BondLabels = Tuple[int, int, Optional[int]]
LigandEnvironment = Tuple[Tuple[AtomLabels, ...], Tuple[BondLabels, ...]]

KEY = "metal_terminal_hydrazido_geometry_conflict_count"
_COUNTS = ("metal_discordance_count", "metal_discordance_structural_count")


def _features(
    candidate: MetalCandidateState,
) -> tuple[dict[tuple[int, int], tuple[int, int, int]], LigandEnvironment | None]:
    """Mask only the formal charges and N–N orders at recognized sites when comparing the remaining ligand."""
    if len(candidate.metal_states) != 1 or candidate.no_metal_state is None:
        return {}, None
    mol = cast(ob.OBMol, candidate.no_metal_state.omol.OBMol)
    sites = {
        (a.GetIdx(), b.GetIdx()): state
        for a, b, state in hydrazido_sites(mol)
        if supports_hydrazido(a, b, candidate.metal_states[0].site)
    }
    if not sites:
        return sites, None
    masked = {i for pair in sites for i in pair}
    pairs = {tuple(sorted(pair)) for pair in sites}
    atoms = tuple(
        (
            a.GetAtomicNum(),
            a.GetIsotope(),
            None if a.GetIdx() in masked else a.GetFormalCharge(),
            get_unpaired_electron_count(a),
            get_lone_pair_count(a),
            has_unresolved_two_electron_center(a),
            a.GetImplicitHCount(),
            a.GetX(),
            a.GetY(),
            a.GetZ(),
        )
        for a in mol_atoms(mol)
    )
    bonds = []
    for bond in mol_bonds(mol):
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        pair = (min(begin, end), max(begin, end))
        bonds.append((*pair, None if pair in pairs else bond.GetBondOrder()))
    return sites, (atoms, tuple(sorted(bonds)))


def clear(candidates: Sequence[MetalCandidateState]) -> None:
    """Remove the hydrazido contribution before comparing a new set of metal assignments."""
    for candidate in candidates:
        previous = candidate.metadata.get(KEY, 0)
        for key in _COUNTS:
            candidate.metadata[key] = candidate.metadata.get(key, 0.0) - previous
        candidate.metadata[KEY] = 0


def annotate(candidates: Sequence[MetalCandidateState]) -> None:
    """Count neutral isodiazene assignments with a geometry-matched, charge-balanced hydrazido partner."""
    clear(candidates)
    features = [_features(c) for c in candidates]
    counts = [c.metadata.get("metal_discordance_structural_count", 0.0) for c in candidates]
    for candidate, (sites, signature), count in zip(candidates, features, counts):
        if not sites:
            continue
        conflicts = set()
        for partner, (other_sites, other_signature), other_count in zip(
            candidates, features, counts
        ):
            if (
                signature != other_signature
                or sites.keys() != other_sites.keys()
                or other_count > count
                or candidate.no_metal_radical_target != partner.no_metal_radical_target
            ):
                continue
            changed = {i for i in sites if sites[i] == (-1, 1, 2) and other_sites[i] == (-2, 0, 1)}
            if not changed or any(sites[i] != other_sites[i] for i in sites.keys() - changed):
                continue
            metal, other = candidate.metal_states[0], partner.metal_states[0]
            if (
                metal.idx != other.idx
                or metal.element_idx != other.element_idx
                or metal.radical_num != other.radical_num
                or (metal.position_x, metal.position_y, metal.position_z)
                != (other.position_x, other.position_y, other.position_z)
                or other.valence - metal.valence != 2 * len(changed)
            ):
                continue
            conflicts.update(changed)
        candidate.metadata[KEY] = len(conflicts)
        for key in _COUNTS:
            candidate.metadata[key] += len(conflicts)
