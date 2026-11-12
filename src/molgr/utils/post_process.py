"""
Author: TMJ
Date: 2026-02-27 23:35:43
LastEditors: TMJ
LastEditTime: 2026-06-18 19:53:00
Description: 请填写简介
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from rdkit import Chem

from molgr.config import CONFIG, MolGRConfig
from molgr.fallback.utils.consts import NON_METAL_DICT
from molgr.fallback.utils.electrons import LONE_PAIR_COUNT_PROP, UNPAIRED_ELECTRON_COUNT_PROP
from molgr.utils.converter import (
    get_atom_lone_pair_count,
    get_atom_unpaired_electrons,
    has_atom_unresolved_two_electron_center,
    refresh_atom_electron_property_lists,
    refresh_rdkit_aromaticity,
)
from molgr.utils.coordination import coordination_distance_cutoff
from molgr.utils.coordination_visibility import (
    CoordinationBlockerArrays,
    Point3D,
    coordination_visibility_mask,
    empty_coordination_blocker_arrays,
)
from molgr.utils.rdkit_stereo import apply_3d_stereo_workarounds


pt = Chem.GetPeriodicTable()


def _has_limited_coordination_valence(atomic_num: int) -> bool:
    """Select metal elements with a finite valence table in this RDKit version.

    A -1 entry permits unrestricted valence (e.g. K, transition metals), so
    those elements retain dative bonds. Scanning all MolGR metals confirms the
    finite-table elements are exactly those exhibiting acceptor-valence failures.
    """
    if atomic_num == 0 or atomic_num in NON_METAL_DICT:
        return False
    valences = tuple(pt.GetValenceList(atomic_num))
    return bool(valences) and -1 not in valences


def _normalize_limited_valence_coordination(rwmol: Chem.RWMol) -> None:
    """Express donation to valence-limited metals as covalent bonds and charges.

    Each unit of bond order transfers one formal-charge unit from acceptor to
    donor. Anionic donors use their negative charge to form up to triple bonds;
    a bridging donor first allocates one bond to each acceptor. This is formal
    charge bookkeeping, not reassignment of the metal oxidation state. It does
    not change connectivity, ligand-internal bonds or radical annotations.
    """
    by_donor: dict[int, list[Chem.Bond]] = {}
    for bond in rwmol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DATIVE and _has_limited_coordination_valence(
            bond.GetEndAtom().GetAtomicNum()
        ):
            by_donor.setdefault(bond.GetBeginAtomIdx(), []).append(bond)
    conformer = rwmol.GetConformer() if rwmol.GetNumConformers() else None
    for donor_idx, bonds in by_donor.items():
        donor = rwmol.GetAtomWithIdx(donor_idx)
        # Allocate additional order to shorter contacts first. All contacts are
        # selected by the existing coordination search, not by this conversion.
        bonds.sort(
            key=lambda bond: (
                (
                    conformer.GetAtomPosition(donor_idx)
                    - conformer.GetAtomPosition(bond.GetEndAtomIdx())
                ).LengthSq()
                if conformer is not None
                else 0.0,
                bond.GetEndAtomIdx(),
            )
        )
        extra_order = max(0, -donor.GetFormalCharge() - len(bonds))
        for bond in bonds:
            order = 1 + min(2, extra_order)
            extra_order -= order - 1
            bond.SetBondType(
                (Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)[order - 1]
            )
            donor.SetFormalCharge(donor.GetFormalCharge() + order)
            metal = bond.GetEndAtom()
            metal.SetFormalCharge(metal.GetFormalCharge() - order)
        _consume_active_lone_pair(donor)


def _is_explicit_singlet_two_electron_center(atom: Chem.Atom) -> bool:
    """Return whether a resolved C/N/P lone pair fills a two-unit valence deficit."""

    atomic_num = int(atom.GetAtomicNum())
    element_info = NON_METAL_DICT.get(atomic_num)
    if (
        atomic_num not in (6, 7, 15)
        or element_info is None
        or atom.GetFormalCharge() != 0
        or get_atom_unpaired_electrons(atom) != 0
        or get_atom_lone_pair_count(atom) != 1
        or has_atom_unresolved_two_electron_center(atom)
    ):
        return False
    bond_valence = sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())
    return abs(float(element_info.default_valence) - bond_valence - 2.0) <= 1e-12


def _is_explicit_carbyne_center(atom: Chem.Atom) -> bool:
    """Return whether a resolved neutral carbon has a three-unit valence deficit."""

    if (
        atom.GetAtomicNum() != 6
        or atom.GetFormalCharge() != 0
        or get_atom_unpaired_electrons(atom) != 3
        or get_atom_lone_pair_count(atom) != 0
        or has_atom_unresolved_two_electron_center(atom)
    ):
        return False
    bond_valence = sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())
    return abs(float(NON_METAL_DICT[6].default_valence) - bond_valence - 3.0) <= 1e-12


def _consume_active_lone_pair(atom: Chem.Atom) -> None:
    if atom.HasProp(LONE_PAIR_COUNT_PROP):
        atom.SetIntProp(LONE_PAIR_COUNT_PROP, 0)


def _rdkit_atom_coordinates(conformer: Chem.Conformer, atom_idx: int) -> Point3D:
    position = conformer.GetAtomPosition(atom_idx)
    return float(position.x), float(position.y), float(position.z)


def _build_rdkit_coordination_blocker_arrays(
    rdmol: Chem.Mol,
    conformer: Chem.Conformer,
    *,
    access_radius_scale: float,
    access_clearance_angstrom: float,
) -> CoordinationBlockerArrays:
    blocker_indices: list[int] = []
    blocker_coordinates: list[Point3D] = []
    blocker_radii: list[float] = []
    for atom_idx in range(rdmol.GetNumAtoms()):
        atom = rdmol.GetAtomWithIdx(atom_idx)
        atomic_num = atom.GetAtomicNum()
        if atomic_num not in NON_METAL_DICT:
            continue
        blocker_radius = (
            access_radius_scale * pt.GetRcovalent(atomic_num) + access_clearance_angstrom
        )
        if blocker_radius <= 0.0:
            continue
        atom_idx = int(atom.GetIdx())
        blocker_indices.append(atom_idx)
        blocker_coordinates.append(_rdkit_atom_coordinates(conformer, atom_idx))
        blocker_radii.append(blocker_radius)
    if not blocker_indices:
        return empty_coordination_blocker_arrays()
    return (
        np.asarray(blocker_indices, dtype=np.int64),
        np.asarray(blocker_coordinates, dtype=np.float64),
        np.asarray(blocker_radii, dtype=np.float64),
    )


def _visible_atom_mask_to_metal(
    atom_indices: list[int],
    metal_atom_idx: int,
    conformer: Chem.Conformer,
    blocker_arrays: CoordinationBlockerArrays,
) -> np.ndarray:
    if not atom_indices:
        return np.empty((0,), dtype=bool)
    atom_coordinates = np.asarray(
        [_rdkit_atom_coordinates(conformer, atom_idx) for atom_idx in atom_indices],
        dtype=np.float64,
    )
    return coordination_visibility_mask(
        np.asarray(atom_indices, dtype=np.int64),
        atom_coordinates,
        np.asarray(_rdkit_atom_coordinates(conformer, metal_atom_idx), dtype=np.float64),
        blocker_arrays,
    )


def make_dative_bond(
    rdmol: Chem.Mol,
    extra_tolerance: Optional[float] = None,
    *,
    config: MolGRConfig | None = None,
) -> Chem.Mol:
    resolved_config = CONFIG if config is None else config
    if extra_tolerance is None:
        extra_tolerance = resolved_config.metal_scoring.metal_coordination_extra_tolerance_angstrom
    pi_dative_distance_difference_tolerance = (
        resolved_config.metal_scoring.pi_dative_distance_difference_tolerance_angstrom
    )
    metal_atom_ids = [
        atom_id
        for atom_id in range(rdmol.GetNumAtoms())
        if rdmol.GetAtomWithIdx(atom_id).GetAtomicNum() not in NON_METAL_DICT
    ]
    if not metal_atom_ids:
        return rdmol

    rwmol = Chem.RWMol(rdmol)
    distance_matrix = Chem.Get3DDistanceMatrix(rwmol)
    conformer = rwmol.GetConformer()
    blocker_arrays = _build_rdkit_coordination_blocker_arrays(
        rwmol,
        conformer,
        access_radius_scale=resolved_config.metal_scoring.metal_access_radius_scale,
        access_clearance_angstrom=resolved_config.metal_scoring.metal_access_clearance_angstrom,
    )

    for metal_atom_id in metal_atom_ids:
        metal_atom = rwmol.GetAtomWithIdx(metal_atom_id)
        distance_to_metal_atoms = distance_matrix[metal_atom_id]
        close_non_metal_atom_ids = np.argsort(distance_to_metal_atoms)
        close_candidate_atom_ids: list[int] = []

        for close_non_metal_atom_id in close_non_metal_atom_ids:
            non_metal_atom = rwmol.GetAtomWithIdx(int(close_non_metal_atom_id))
            if non_metal_atom.GetAtomicNum() not in NON_METAL_DICT:
                continue
            coordination_cutoff = coordination_distance_cutoff(
                int(metal_atom.GetAtomicNum()),
                int(non_metal_atom.GetAtomicNum()),
                radius_scale=resolved_config.metal_scoring.metal_access_radius_scale,
                extra_tolerance_angstrom=extra_tolerance,
            )
            if distance_to_metal_atoms[close_non_metal_atom_id] > coordination_cutoff:
                # out of bond distance, skip
                break
            close_candidate_atom_ids.append(int(close_non_metal_atom_id))

        visible_mask = _visible_atom_mask_to_metal(
            close_candidate_atom_ids,
            int(metal_atom_id),
            conformer,
            blocker_arrays,
        )
        visible_candidate_atom_ids = {
            atom_id
            for atom_id, is_visible in zip(close_candidate_atom_ids, visible_mask)
            if is_visible
        }
        close_non_metal_atom_ids_set = set()

        for close_non_metal_atom_id in close_candidate_atom_ids:
            if close_non_metal_atom_id not in visible_candidate_atom_ids:
                continue
            if rwmol.GetBondBetweenAtoms(close_non_metal_atom_id, int(metal_atom_id)) is not None:
                # already bonded, skip
                continue
            non_metal_atom = rwmol.GetAtomWithIdx(close_non_metal_atom_id)
            if (
                pt.GetNOuterElecs(non_metal_atom.GetAtomicNum()) == 3
                and non_metal_atom.GetFormalCharge() < 0
            ):
                # BR4- type, skip
                continue
            close_non_metal_atom_ids_set.add(close_non_metal_atom_id)
            if _has_limited_coordination_valence(
                metal_atom.GetAtomicNum()
            ) and _is_explicit_singlet_two_electron_center(non_metal_atom):
                # A lone-pair donor to a valence-limited acceptor uses the same
                # charge-separated single-bond representation as other donors.
                rwmol.AddBond(close_non_metal_atom_id, int(metal_atom_id), Chem.BondType.DATIVE)
            elif _is_explicit_carbyne_center(non_metal_atom):
                rwmol.AddBond(
                    close_non_metal_atom_id,
                    int(metal_atom_id),
                    Chem.BondType.TRIPLE,
                )
                non_metal_atom.SetNumRadicalElectrons(0)
                if non_metal_atom.HasProp(UNPAIRED_ELECTRON_COUNT_PROP):
                    non_metal_atom.SetIntProp(UNPAIRED_ELECTRON_COUNT_PROP, 0)
            elif _is_explicit_singlet_two_electron_center(non_metal_atom):
                rwmol.AddBond(
                    close_non_metal_atom_id,
                    int(metal_atom_id),
                    Chem.BondType.DOUBLE,
                )
                _consume_active_lone_pair(non_metal_atom)
            elif non_metal_atom.GetFormalCharge() < 0:
                if non_metal_atom.GetAtomicNum() == 1:
                    # make metal-H bond
                    rwmol.AddBond(close_non_metal_atom_id, int(metal_atom_id), Chem.BondType.SINGLE)
                    non_metal_atom.SetFormalCharge(0)
                    metal_atom.SetFormalCharge(metal_atom.GetFormalCharge() - 1)
                else:
                    rwmol.AddBond(close_non_metal_atom_id, int(metal_atom_id), Chem.BondType.DATIVE)
            elif (
                pt.GetNOuterElecs(non_metal_atom.GetAtomicNum()) in (5, 6, 7)
                and non_metal_atom.GetFormalCharge() == 0
            ):
                rwmol.AddBond(close_non_metal_atom_id, int(metal_atom_id), Chem.BondType.DATIVE)
        for bond_idx in range(rwmol.GetNumBonds()):
            rd_bond = rwmol.GetBondWithIdx(bond_idx)
            # find pi dative bond
            if rd_bond.GetBondType() not in (
                Chem.BondType.DOUBLE,
                Chem.BondType.TRIPLE,
                Chem.BondType.AROMATIC,
            ):
                continue
            # make sure atoms in bond are closed
            if (
                rd_bond.GetBeginAtomIdx() not in close_non_metal_atom_ids_set
                or rd_bond.GetEndAtomIdx() not in close_non_metal_atom_ids_set
            ):
                continue
            if rwmol.GetBondBetweenAtoms(
                rd_bond.GetBeginAtomIdx(), int(metal_atom_id)
            ) or rwmol.GetBondBetweenAtoms(rd_bond.GetEndAtomIdx(), int(metal_atom_id)):
                continue
            distance_1 = distance_matrix[rd_bond.GetBeginAtomIdx()][int(metal_atom_id)]
            distance_2 = distance_matrix[rd_bond.GetEndAtomIdx()][int(metal_atom_id)]
            if abs(distance_1 - distance_2) > pi_dative_distance_difference_tolerance:
                continue
            rwmol.AddBond(rd_bond.GetBeginAtomIdx(), int(metal_atom_id), Chem.BondType.DATIVE)
            rwmol.AddBond(rd_bond.GetEndAtomIdx(), int(metal_atom_id), Chem.BondType.DATIVE)

    _normalize_limited_valence_coordination(rwmol)
    result = rwmol.GetMol()
    refresh_atom_electron_property_lists(result)
    return result


def make_stereochemistry(rdmol: Chem.Mol) -> Chem.Mol:
    rdmol.UpdatePropertyCache(strict=False)
    refresh_rdkit_aromaticity(rdmol)
    # This already perceives bond directions and atom tags, then cleans stereo.
    # Keep its directional single bonds: STEREONONE on a single bond does not
    # mean its direction is unused (it may encode E/Z on an adjacent double).
    # UNKNOWN directions also carry explicit uncertainty across repeated calls.
    apply_3d_stereo_workarounds(rdmol, perceive=True)
    return rdmol
