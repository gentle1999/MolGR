"""
Author: TMJ
Date: 2026-02-27 23:35:43
LastEditors: TMJ
LastEditTime: 2026-06-18 19:53:00
Description: 请填写简介
"""

from __future__ import annotations

from itertools import product
from math import sqrt
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D as RDPoint3D

from molgr.config import CONFIG, MolGRConfig
from molgr.fallback.utils.consts import NON_METAL_DICT
from molgr.fallback.utils.electrons import (
    LONE_PAIR_COUNT_PROP,
    UNPAIRED_ELECTRON_COUNT_PROP,
    has_cationic_lewis_pair,
)
from molgr.utils.converter import (
    get_atom_lone_pair_count,
    get_atom_unpaired_electrons,
    has_atom_unresolved_two_electron_center,
    refresh_atom_electron_property_lists,
    refresh_rdkit_aromaticity,
)
from molgr.utils.coordination import coordination_distance_cutoff, coordination_surface_distance
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
    for i in range(rwmol.GetNumBonds()):
        bond = rwmol.GetBondWithIdx(i)
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
    """Return whether a resolved C/N/Si/P lone pair fills a two-unit valence deficit."""

    atomic_num = int(atom.GetAtomicNum())
    element_info = NON_METAL_DICT.get(atomic_num)
    if (
        atomic_num not in (6, 7, 14, 15)
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


def _has_sigma_donor_lone_pair(atom: Chem.Atom) -> bool:
    """Check the ligand valence budget for a neutral or cationic sigma donor.

    Sulfoxide and sulfite S+ retain a pair; tetravalent P+ does not. Count only
    ligand bonds, since existing outgoing dative contacts do not consume valence.
    """
    # Adding the first bridge leg can invalidate RDKit's local valence cache
    # before a second metal is considered for the same donor.
    if atom.NeedsUpdatePropertyCache():
        atom.UpdatePropertyCache(strict=False)
    valence = (
        sum(
            bond.GetBondTypeAsDouble()
            for bond in atom.GetBonds()
            if bond.GetBondType() != Chem.BondType.DATIVE
        )
        + atom.GetNumExplicitHs()
        + atom.GetNumImplicitHs()
    )
    outer = pt.GetNOuterElecs(atom.GetAtomicNum())
    if atom.GetFormalCharge() == 0:
        # Aromatic bond orders may be fractional. P(V) and S(VI) already
        # use all their valence electrons even when their formal charge is zero.
        return outer - valence - get_atom_unpaired_electrons(atom) >= 2
    return has_cationic_lewis_pair(
        outer,
        atom.GetFormalCharge(),
        valence,
        get_atom_unpaired_electrons(atom),
        has_atom_unresolved_two_electron_center(atom),
    )


def _carbonyl_plane_tangent(
    atom: Chem.Atom, axis: RDPoint3D, conf: Chem.Conformer
) -> RDPoint3D | None:
    """Resolve the plane of a terminal pi donor from its trigonal ligand partner."""
    partner = atom.GetNeighbors()[0]
    if partner.GetHybridization() != Chem.HybridizationType.SP2 or partner.GetDegree() != 3:
        return None
    # In a carboxylate either terminal oxygen can carry the negative charge.
    # Both Lewis forms must allow the same range of metal approaches.
    if any(a.GetAtomicNum() == 8 and a.GetFormalCharge() < 0 for a in partner.GetNeighbors()):
        return None
    left, right = (
        conf.GetAtomPosition(a.GetIdx())
        for a in partner.GetNeighbors()
        if a.GetIdx() != atom.GetIdx()
    )
    span = left - right
    tangent = span - axis * span.DotProduct(axis)
    if tangent.Length() < 1e-8:
        return None
    tangent.Normalize()
    return tangent


def _directional_sigma_contacts(
    atom: Chem.Atom, conf: Chem.Conformer, metals: list[int], existing: set[int]
) -> set[int] | None:
    """Infer allowed sigma contacts from ligand bonds, hybridization and Lewis occupancy.

    None leaves the site to the distance and valence rules (e.g. free halide).
    A single localized pair selects one acceptor. Two localized pairs on a
    neutral bent chalcogen or carbonyl can point to separate acceptors. Anionic
    donors may bridge through a shared pair. With just one bond and no plane
    to fix the azimuth, screen the approach to the bond, without choosing an
    arbitrary torsion for the equivalent lone pairs.
    """
    group = pt.GetNOuterElecs(atom.GetAtomicNum())
    charge = atom.GetFormalCharge()
    degree = atom.GetDegree()
    hybrid = atom.GetHybridization()
    sp, sp2, sp3 = Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3
    if atom.GetIsAromatic() and group == 5 and charge == 0 and atom.GetTotalDegree() == 3:
        return set()
    if degree == 0 or degree > 3 or degree != atom.GetTotalDegree():
        return None
    electrons = group - charge - sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())
    if electrons < 2 or (charge < 0 and atom.GetIsAromatic()):
        return None
    if group == 4 and charge < 0 and hybrid != sp:
        # A localized charge in an allylic or ylidic Lewis form does not
        # identify a unique sigma direction at the charged atom.
        return None
    origin = conf.GetAtomPosition(atom.GetIdx())
    outward = [origin - conf.GetAtomPosition(a.GetIdx()) for a in atom.GetNeighbors()]
    for v in outward:
        if v.Length() < 1e-8:
            return set()
        v.Normalize()

    axes: list[RDPoint3D]
    min_cosine = 0.5
    shared = charge < 0
    if degree == 1:
        axis = outward[0]
        if hybrid == sp:
            axes = [axis]
            min_cosine = 1e-12  # End-on donation can bend into the outward hemisphere.
        elif electrons >= 4:
            multiple_bond = atom.GetBonds()[0].GetBondType() == Chem.BondType.DOUBLE
            if multiple_bond and atom.GetAtomicNum() > 10:
                # A heavy terminal pi donor (e.g. C=S) also permits side-on
                # approaches. Its bond axis alone cannot exclude a contact.
                return None
            tangent = (
                _carbonyl_plane_tangent(atom, axis, conf)
                if atom.GetAtomicNum() == 8 and charge == 0 and multiple_bond and hybrid == sp2
                else None
            )
            if tangent is None:
                # R–X, alkoxide, thiolate or a terminal donor without a pi plane.
                # Exclude only the 60° cone pointing towards the substituent.
                axes, min_cosine, shared = [axis], -0.5, True
            else:
                axes = [axis * 0.5 + tangent * sign for sign in (-sqrt(0.75), sqrt(0.75))]
                # Allow tilted carbonyl coordination while screening approaches
                # normal to the pi plane (zero projection on both sigma axes).
                min_cosine = 0.25
        else:
            return None
    elif degree == 2 and hybrid in (sp2, sp3):
        a, b = outward
        axis = a + b
        if axis.Length() < 1e-8:
            return set() if group == 5 and charge == 0 else None
        axis.Normalize()
        if hybrid == sp2 or _is_explicit_singlet_two_electron_center(atom):
            # An imine, furan-like O or explicitly resolved singlet carbene
            # has one in-plane sigma pair along the exterior bisector.
            axes = [axis]
            if atom.GetAtomicNum() > 10:
                min_cosine = 1e-12  # A heavy bent donor can bind across the open hemisphere.
        elif electrons >= 4:
            normal = a.CrossProduct(b)
            if normal.Length() < 1e-8:
                return None
            normal.Normalize()
            # The two missing tetrahedral directions subtend 109.47 degrees.
            axes = [axis * sqrt(1 / 3) + normal * sign for sign in (-sqrt(2 / 3), sqrt(2 / 3))]
        else:
            return None
    elif degree == 3 and hybrid == sp3:
        a, b, c = outward
        axis = (b - a).CrossProduct(c - a)
        if axis.Length() < 1e-8:
            return set()
        axis.Normalize()
        height = axis.DotProduct(a)
        if abs(height) < 0.1:
            return set()
        axes = [axis if height > 0 else axis * -1.0]
    else:
        return None
    return _rank_sigma_contacts(axes, origin, conf, metals, existing, min_cosine, shared)


def _rank_sigma_contacts(
    axes: list[RDPoint3D],
    origin: RDPoint3D,
    conf: Chem.Conformer,
    metals: list[int],
    existing: set[int],
    min_cosine: float,
    shared: bool,
) -> set[int]:
    """Assign visible metals to one or two axes, reserving any existing contacts.

    Maximize occupied directions, then total alignment and finally proximity.
    Equally ranked assignments contribute only their common metals. For a
    shared or azimuthally unresolved pair, apply the angular screen alone.
    """
    if not shared and len(existing) >= len(axes):
        return set()
    options: list[dict[int | None, tuple[float, float]]] = [{None: (0.0, 0.0)} for _ in axes]
    for metal in set(metals) | existing:
        v = conf.GetAtomPosition(metal) - origin
        distance = v.Length()
        if distance < 1e-8:
            continue
        alignments = [round(axis.DotProduct(v) / distance, 12) for axis in axes]
        closest = max(alignments)
        for cosine, choices in zip(alignments, options):
            # Metals approaching the same lobe compete for that pair even
            # where the angular tolerances for the two lobes overlap.
            if cosine == closest and (metal in existing or cosine >= min_cosine):
                choices[metal] = (cosine, -distance)
    if shared:
        return {m for choices in options for m in choices if m is not None}
    best = None
    selected: set[int] = set()
    for assignment in product(*options):
        occupied = {m for m in assignment if m is not None}
        if len(occupied) != len(assignment) - assignment.count(None) or not existing <= occupied:
            continue
        scores = [choices[m] for choices, m in zip(options, assignment)]
        rank = (
            len(occupied),
            round(sum(score[0] for score in scores), 12),
            round(sum(score[1] for score in scores), 10),
        )
        if best is None or rank > best:
            best, selected = rank, occupied
        elif rank == best:
            selected.intersection_update(occupied)
    return selected


def _sigma_acceptors(
    mol: Chem.Mol, visible_candidates: dict[int, list[int]]
) -> dict[int, set[int]]:
    """Perceive donor orbitals on a ligand-only copy, preserving the input's Lewis graph."""
    metals_by_donor: dict[int, list[int]] = {}
    for metal, candidates in visible_candidates.items():
        for idx in candidates:
            atom = mol.GetAtomWithIdx(idx)
            group = pt.GetNOuterElecs(atom.GetAtomicNum())
            if group in (5, 6, 7) or (
                group == 4 and (atom.GetFormalCharge() < 0 or get_atom_lone_pair_count(atom))
            ):
                metals_by_donor.setdefault(idx, []).append(metal)
    if not metals_by_donor:
        return {}

    ligand = Chem.RWMol(mol)
    for metal in visible_candidates:
        for neighbor in mol.GetAtomWithIdx(metal).GetNeighbors():
            if neighbor.GetAtomicNum() in NON_METAL_DICT:
                ligand.RemoveBond(metal, neighbor.GetIdx())
    ligand.UpdatePropertyCache(strict=False)
    flags = Chem.SanitizeFlags
    Chem.SanitizeMol(
        ligand,
        sanitizeOps=(
            flags.SANITIZE_SYMMRINGS
            | flags.SANITIZE_SETAROMATICITY
            | flags.SANITIZE_SETCONJUGATION
            | flags.SANITIZE_SETHYBRIDIZATION
        ),
    )
    conformer = mol.GetConformer()
    acceptors: dict[int, set[int]] = {}
    for donor, metals in metals_by_donor.items():
        existing = {
            a.GetIdx()
            for a in mol.GetAtomWithIdx(donor).GetNeighbors()
            if a.GetAtomicNum() not in NON_METAL_DICT
        }
        allowed = _directional_sigma_contacts(
            ligand.GetAtomWithIdx(donor), conformer, metals, existing
        )
        if allowed is not None:
            acceptors[donor] = allowed
    return acceptors


def _add_hydride_contacts(
    mol: Chem.RWMol, hydride_idx: int, metals: list[int], distances: np.ndarray, radius_scale: float
) -> None:
    """Assign a visible hydride after collecting candidate metals from the whole molecule.

    Two short contacts to dative acceptors can share the hydride pair. Use
    outgoing H⁻ dative bonds. Subsequent finite-valence normalization may
    convert one of these to a single bond, which exhausts hydrogen's covalent
    valence. Use a terminal single bond for a lone acceptor or when two
    acceptors would both require conversion.
    """
    hydride = mol.GetAtomWithIdx(hydride_idx)
    existing = {atom.GetIdx() for atom in hydride.GetNeighbors()}
    short_contacts = [
        i
        for i in metals
        if distances[i]
        <= coordination_distance_cutoff(
            mol.GetAtomWithIdx(i).GetAtomicNum(),
            1,
            radius_scale=radius_scale,
            extra_tolerance_angstrom=0.20,
        )
    ]
    if (
        len(short_contacts) >= 2
        and existing.issubset(short_contacts)
        and sum(
            _has_limited_coordination_valence(mol.GetAtomWithIdx(i).GetAtomicNum())
            for i in short_contacts
        )
        <= 1
    ):
        for i in short_contacts:
            if i not in existing:
                mol.AddBond(hydride_idx, i, Chem.BondType.DATIVE)
    elif not existing:
        nearest = min(metals, key=lambda i: (distances[i], mol.GetAtomWithIdx(i).GetAtomicNum(), i))
        mol.AddBond(hydride_idx, nearest, Chem.BondType.SINGLE)
        hydride.SetFormalCharge(0)
        metal = mol.GetAtomWithIdx(nearest)
        metal.SetFormalCharge(metal.GetFormalCharge() - 1)


def _is_bent_cumulene_donor(atom: Chem.Atom, conformer: Chem.Conformer, metal_idx: int) -> bool:
    if (
        atom.GetAtomicNum() != 6
        or atom.GetFormalCharge() != 0
        or atom.GetDegree() != 2
        or atom.IsInRing()
        or any(b.GetBondType() != Chem.BondType.DOUBLE for b in atom.GetBonds())
        or any(a.GetAtomicNum() != 6 for a in atom.GetNeighbors())
    ):
        return False
    center = np.asarray(conformer.GetAtomPosition(atom.GetIdx()))
    ends = [np.asarray(conformer.GetAtomPosition(a.GetIdx())) - center for a in atom.GetNeighbors()]
    lengths = [np.linalg.norm(v) for v in ends]
    if min(lengths) <= 0:
        return False
    angle_cos = float(np.dot(ends[0], ends[1]) / (lengths[0] * lengths[1]))
    metal = atom.GetOwningMol().GetAtomWithIdx(metal_idx)
    distance = np.linalg.norm(np.asarray(conformer.GetAtomPosition(metal_idx)) - center)
    return bool(
        angle_cos > np.cos(np.deg2rad(150))
        and distance
        <= coordination_distance_cutoff(
            metal.GetAtomicNum(),
            6,
            radius_scale=1.0,
            extra_tolerance_angstrom=0.20,
        )
    )


def _can_complete_pi_contact(
    ligand_bond: Chem.Bond, contact: Chem.Bond, metal_idx: int, input_mol: Chem.Mol
) -> bool:
    """Decide whether one outgoing dative edge can be extended to the other pi endpoint.

    Aromatic hapticity can span an anionic donor. A neutral nitrile can bind
    side-on even after N was selected for sigma donation. For a neutral C=C or
    C≡C unit, complete a partial input contact while preserving independently
    assigned sigma donors such as bent cumulenes. Distance and visibility of
    both endpoints are checked by the caller.
    """
    if contact.GetBondType() != Chem.BondType.DATIVE or contact.GetEndAtomIdx() != metal_idx:
        return False
    if ligand_bond.GetIsAromatic():
        return True
    ends = (ligand_bond.GetBeginAtom(), ligand_bond.GetEndAtom())
    if any(
        a.GetFormalCharge() or (a.GetAtomicNum() == 6 and get_atom_lone_pair_count(a)) for a in ends
    ):
        return False
    elements = {a.GetAtomicNum() for a in ends}
    if elements == {6, 7} and ligand_bond.GetBondType() == Chem.BondType.TRIPLE:
        return True
    return (
        elements == {6}
        and input_mol.GetBondBetweenAtoms(contact.GetBeginAtomIdx(), metal_idx) is not None
        and not any(
            b.GetIdx() != ligand_bond.GetIdx() and b.GetBondType() == Chem.BondType.DOUBLE
            for b in contact.GetBeginAtom().GetBonds()
        )
    )


def _is_edge_on_chelate_bond(
    bond: Chem.Bond, conformer: Chem.Conformer, metal_idx: int, bound_donors: set[int]
) -> bool:
    """Identify an edge-on pi bond between two independently bound chelate donors.

    In a donor–X=Y–donor path, each flanking donor defines a local plane
    with X and Y. Within 30 degrees of both planes the metal approaches the
    edge of the linker. A triple bond has two pi directions. Use the
    conformer of the bond's molecule and its metal atom index. The donor
    indices are the metal's neighbors before inferring any pi contacts.
    Return False when the flanking atoms cannot define both planes.
    """
    if bond.GetBondType() == Chem.BondType.TRIPLE:
        return False
    begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
    left = {a.GetIdx() for a in begin.GetNeighbors()} & bound_donors
    right = {a.GetIdx() for a in end.GetNeighbors()} & bound_donors
    left.discard(end.GetIdx())
    right.discard(begin.GetIdx())
    if not left or not right or len(left | right) < 2:
        return False
    first = conformer.GetAtomPosition(begin.GetIdx())
    last = conformer.GetAtomPosition(end.GetIdx())
    axis = last - first
    approach = conformer.GetAtomPosition(metal_idx) - (first + last) * 0.5
    for donors, position in ((left, first), (right, last)):
        for donor in donors:
            substituent = conformer.GetAtomPosition(donor) - position
            normal = axis.CrossProduct(substituent)
            # sin²(angle) <= 1/4 excludes substituents within 30° of the axis.
            if normal.LengthSq() <= 0.25 * axis.LengthSq() * substituent.LengthSq():
                return False
            if normal.DotProduct(approach) ** 2 > 0.25 * normal.LengthSq() * approach.LengthSq():
                return False
    return True


def make_dative_bond(
    rdmol: Chem.Mol,
    extra_tolerance: Optional[float] = None,
    *,
    config: MolGRConfig | None = None,
) -> Chem.Mol:
    """Infer metal–ligand contacts from a reconstructed Lewis graph and 3D geometry.

    Parameters
    ----------
    rdmol : Chem.Mol
        Graph containing formal charges, unpaired-electron assignments and a
        conformer in angstroms. Existing bonds and donor valence constrain
        which nearby atoms and pi systems can coordinate a metal. For a
        chelate with an unsaturated linker, the metal's approach to the local
        ligand planes helps distinguish pi binding from proximity to the linker.
    extra_tolerance : float, optional
        Additional distance tolerance in angstroms for the coordination cutoff.
        None uses ``config.metal_scoring.metal_coordination_extra_tolerance_angstrom``.
    config : MolGRConfig, optional
        Controls metal accessibility and coordination geometry. None uses CONFIG.

    Returns
    -------
    Chem.Mol
        Graph with the inferred coordination bonds. For metal-containing inputs
        the edits are made on a copy. Metal-free inputs return the original object.

    Notes
    -----
    Supply hydrogens as atoms with coordinates. For localized sigma donors
    the ligand hybridization and explicit bond directions constrain the
    metal's approach. Nitriles and the carbon end of CO use the outward
    bond extension. Amines and phosphines use the open face of the pyramid.
    Bent ethers and thioethers can donate along either of two directions.
    A carbonyl oxygen uses the plane of its trigonal partner when available.
    Visible metals competing for one direction are ranked by alignment, then
    distance. Free halides have no ligand axis and can bridge several metals.
    Side-on pi coordination is assessed separately.

    A visible hydride within the scaled covalent-radius sum plus 0.20 Å of
    multiple metals can form a bridge. The graph uses outgoing H⁻ dative
    contacts, with at most one converted to a single bond for a valence-limited
    acceptor. When several acceptors require single bonds, hydrogen's valence
    permits only a terminal contact to the nearest eligible metal.
    """
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
    if rwmol.NeedsUpdatePropertyCache():
        rwmol.UpdatePropertyCache(strict=False)
    distance_matrix = Chem.Get3DDistanceMatrix(rwmol)
    conformer = rwmol.GetConformer()
    blocker_arrays = _build_rdkit_coordination_blocker_arrays(
        rwmol,
        conformer,
        access_radius_scale=resolved_config.metal_scoring.metal_access_radius_scale,
        access_clearance_angstrom=resolved_config.metal_scoring.metal_access_clearance_angstrom,
    )

    visible_candidates: dict[int, list[int]] = {}
    for metal_atom_id in metal_atom_ids:
        metal_atom = rwmol.GetAtomWithIdx(metal_atom_id)
        distance_to_metal_atoms = distance_matrix[metal_atom_id]
        # Subtract the donor radius before sorting. The remaining threshold is
        # the same scaled metal radius plus tolerance for every donor element.
        close_non_metal_atom_ids = sorted(
            (
                i
                for i in range(rwmol.GetNumAtoms())
                if rwmol.GetAtomWithIdx(i).GetAtomicNum() in NON_METAL_DICT
            ),
            key=lambda i: (
                coordination_surface_distance(
                    distance_to_metal_atoms[i],
                    rwmol.GetAtomWithIdx(i).GetAtomicNum(),
                    radius_scale=resolved_config.metal_scoring.metal_access_radius_scale,
                ),
                i,
            ),
        )
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
                # Surface-distance ordering makes this early stop valid.
                break
            close_candidate_atom_ids.append(int(close_non_metal_atom_id))

        visible_mask = _visible_atom_mask_to_metal(
            close_candidate_atom_ids,
            int(metal_atom_id),
            conformer,
            blocker_arrays,
        )
        visible_candidates[metal_atom_id] = [
            atom_id
            for atom_id, is_visible in zip(close_candidate_atom_ids, visible_mask)
            if is_visible
        ]

    sigma_acceptors = _sigma_acceptors(rwmol, visible_candidates)
    hydride_candidates: dict[int, list[int]] = {}
    for metal_atom_id, visible_atom_ids in visible_candidates.items():
        metal_atom = rwmol.GetAtomWithIdx(metal_atom_id)
        distance_to_metal_atoms = distance_matrix[metal_atom_id]
        close_non_metal_atom_ids_set = set()

        for close_non_metal_atom_id in visible_atom_ids:
            non_metal_atom = rwmol.GetAtomWithIdx(close_non_metal_atom_id)
            if (
                pt.GetNOuterElecs(non_metal_atom.GetAtomicNum()) == 3
                and non_metal_atom.GetFormalCharge() < 0
            ):
                # BR4- type, skip
                continue
            # A radical is not a paired-electron donor. Exclude it from both
            # atom donation and the pi-contact pass. Resolved carbynes follow
            # the separate covalent triple-bond path below, consuming their spin.
            if has_atom_unresolved_two_electron_center(non_metal_atom) or (
                get_atom_unpaired_electrons(non_metal_atom) > 0
                and not _is_explicit_carbyne_center(non_metal_atom)
            ):
                continue
            if (
                non_metal_atom.GetAtomicNum() in (9, 17, 35, 53)
                and non_metal_atom.GetFormalCharge() == 0
                and non_metal_atom.GetDegree() == 1
                and non_metal_atom.GetNeighbors()[0].GetAtomicNum() == 5
                and rwmol.GetBondBetweenAtoms(
                    non_metal_atom.GetNeighbors()[0].GetIdx(), metal_atom_id
                )
                is not None
                and distance_to_metal_atoms[close_non_metal_atom_id]
                > coordination_distance_cutoff(
                    metal_atom.GetAtomicNum(),
                    non_metal_atom.GetAtomicNum(),
                    radius_scale=resolved_config.metal_scoring.metal_access_radius_scale,
                    extra_tolerance_angstrom=min(0.20, extra_tolerance),
                )
            ):
                # Do not turn a remote B-X substituent into a second donor
                # when the borane itself already binds this metal.
                continue
            if (
                non_metal_atom.GetAtomicNum() == 16
                and non_metal_atom.GetFormalCharge() == 0
                and len(list(non_metal_atom.GetBonds())) == 2
                and all(b.GetBondType() == Chem.BondType.SINGLE for b in non_metal_atom.GetBonds())
                and distance_to_metal_atoms[close_non_metal_atom_id]
                > coordination_distance_cutoff(
                    metal_atom.GetAtomicNum(),
                    16,
                    radius_scale=resolved_config.metal_scoring.metal_access_radius_scale,
                    extra_tolerance_angstrom=min(0.20, extra_tolerance),
                )
                and any(
                    a.GetAtomicNum() == 6
                    and a.GetFormalCharge() < 0
                    and sum(n.GetAtomicNum() == 16 for n in a.GetNeighbors()) == 3
                    and all(
                        b.GetBondType() in (Chem.BondType.SINGLE, Chem.BondType.DATIVE)
                        for b in a.GetBonds()
                    )
                    and rwmol.GetBondBetweenAtoms(a.GetIdx(), metal_atom_id) is not None
                    for a in non_metal_atom.GetNeighbors()
                )
            ):
                # Remote thioethers on a bound tris(thio)carbanion are not
                # additional donors. Keep short C/S chelates and anionic S.
                continue
            close_non_metal_atom_ids_set.add(close_non_metal_atom_id)
            if non_metal_atom.GetAtomicNum() == 1 and non_metal_atom.GetFormalCharge() == -1:
                hydride_candidates.setdefault(close_non_metal_atom_id, []).append(metal_atom_id)
                continue
            if rwmol.GetBondBetweenAtoms(close_non_metal_atom_id, metal_atom_id) is not None:
                # An existing contact can still be one end of an incomplete pi bond.
                continue
            if (
                close_non_metal_atom_id in sigma_acceptors
                and metal_atom_id not in sigma_acceptors[close_non_metal_atom_id]
            ):
                # This rejects only atom donation; the independent pi pass can
                # still use this atom when both ligand endpoints support it.
                continue
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
            elif (
                metal_atom.GetAtomicNum() == 79
                and metal_atom.GetFormalCharge() < 0
                and non_metal_atom.GetAtomicNum() == 5
                and non_metal_atom.GetFormalCharge() == 0
                and sum(b.GetBondTypeAsDouble() for b in non_metal_atom.GetBonds()) == 3
            ):
                # Z-type donation is metal -> B. Normalize after every contact
                # has been selected, since neutral tetravalent B is invalid in RDKit.
                rwmol.AddBond(int(metal_atom_id), close_non_metal_atom_id, Chem.BondType.DATIVE)
            elif (
                non_metal_atom.GetFormalCharge() < 0
                or _is_bent_cumulene_donor(non_metal_atom, conformer, metal_atom_id)
                or (
                    pt.GetNOuterElecs(non_metal_atom.GetAtomicNum()) in (5, 6, 7)
                    and _has_sigma_donor_lone_pair(non_metal_atom)
                )
            ):
                rwmol.AddBond(close_non_metal_atom_id, int(metal_atom_id), Chem.BondType.DATIVE)
        # Snapshot atom donors before adding pi contacts: haptic bonds added
        # below cannot establish an independent chelate at another bond.
        bound_donors = {
            a.GetIdx() for a in metal_atom.GetNeighbors() if a.GetAtomicNum() in NON_METAL_DICT
        }
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
            begin_coordination = rwmol.GetBondBetweenAtoms(
                rd_bond.GetBeginAtomIdx(), int(metal_atom_id)
            )
            end_coordination = rwmol.GetBondBetweenAtoms(
                rd_bond.GetEndAtomIdx(), int(metal_atom_id)
            )
            if begin_coordination is not None and end_coordination is not None:
                continue
            existing = begin_coordination if begin_coordination is not None else end_coordination
            if existing is not None and not _can_complete_pi_contact(
                rd_bond, existing, metal_atom_id, rdmol
            ):
                continue
            distance_1 = distance_matrix[rd_bond.GetBeginAtomIdx()][int(metal_atom_id)]
            distance_2 = distance_matrix[rd_bond.GetEndAtomIdx()][int(metal_atom_id)]
            if abs(distance_1 - distance_2) > pi_dative_distance_difference_tolerance:
                continue
            if _is_edge_on_chelate_bond(rd_bond, conformer, metal_atom_id, bound_donors):
                continue
            if begin_coordination is None:
                rwmol.AddBond(rd_bond.GetBeginAtomIdx(), int(metal_atom_id), Chem.BondType.DATIVE)
            if end_coordination is None:
                rwmol.AddBond(rd_bond.GetEndAtomIdx(), int(metal_atom_id), Chem.BondType.DATIVE)

    for hydride_idx, metals in hydride_candidates.items():
        _add_hydride_contacts(
            rwmol,
            hydride_idx,
            metals,
            distance_matrix[hydride_idx],
            resolved_config.metal_scoring.metal_access_radius_scale,
        )

    for i in range(rwmol.GetNumBonds()):
        bond = rwmol.GetBondWithIdx(i)
        if (
            bond.GetBondType() == Chem.BondType.DATIVE
            and bond.GetBeginAtom().GetAtomicNum() == 79
            and bond.GetEndAtom().GetAtomicNum() == 5
        ):
            bond.SetBondType(Chem.BondType.SINGLE)
            bond.GetBeginAtom().SetFormalCharge(bond.GetBeginAtom().GetFormalCharge() + 1)
            bond.GetEndAtom().SetFormalCharge(bond.GetEndAtom().GetFormalCharge() - 1)

    _normalize_limited_valence_coordination(rwmol)
    result = rwmol.GetMol()
    refresh_atom_electron_property_lists(result)
    return result


def make_stereochemistry(rdmol: Chem.Mol, *, kekulize: bool = False) -> Chem.Mol:
    """Assign RDKit stereo tags from the reconstructed 3D conformer in place.

    Parameters
    ----------
    rdmol : Chem.Mol
        Graph with bond orders, coordination contacts and Cartesian coordinates.
        Bond directions and atom stereo tags are derived from this geometry.
    kekulize : bool, default False
        Use localized integer bond orders when refreshing ring properties.
        The single-frame reconstruction interface passes True by default.

    Returns
    -------
    Chem.Mol
        The same molecule with refreshed aromaticity, conjugation and stereo.
    """
    rdmol.UpdatePropertyCache(strict=False)
    refresh_rdkit_aromaticity(rdmol, kekulize=kekulize)
    # This already perceives bond directions and atom tags, then cleans stereo.
    # Keep its directional single bonds: STEREONONE on a single bond does not
    # mean its direction is unused (it may encode E/Z on an adjacent double).
    # UNKNOWN directions also carry explicit uncertainty across repeated calls.
    apply_3d_stereo_workarounds(rdmol, perceive=True)
    return rdmol
