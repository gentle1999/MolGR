"""Early structure preparation and validation stages for fallback."""

from __future__ import annotations

from typing import Callable, Optional, Tuple, cast

from openbabel import openbabel as ob
from openbabel import pybel

from molgr.fallback.utils import consts, smarts
from molgr.fallback.utils.bond_geometry import (
    carbon_oxygen_order_allowed,
    cationic_carbon_double_bond_allowed,
    nitrilium_geometry_allowed,
    nitrogen_nitrogen_order_allowed,
    nitrogen_oxygen_order_allowed,
    phosphorus_carbon_order_allowed,
    silicon_carbon_order_allowed,
    twisted_ring_link_order_allowed,
)
from molgr.fallback.utils.electrons import (
    get_unpaired_electron_count,
    has_unresolved_two_electron_center,
    set_lone_pair_count,
    set_unpaired_electron_count,
    set_unresolved_two_electron_center,
)
from molgr.fallback.utils.ob_graph import atom_neighbors, mol_atoms, mol_bonds, neighbor_bond_order
from molgr.fallback.utils.search_constraints import search_constraints_satisfied


def validate_omol(
    omol: pybel.Molecule, total_charge: int = 0, total_radical_electrons: int = 0
) -> bool:
    """Resolve deferred two-electron centers against the global radical budget.

    Fixed atoms contribute only their real unpaired-electron counts; active lone
    pairs do not enter the spin budget. If ``T - R`` is an admissible even number,
    the first required unresolved centers become triplets ``(2, 0)`` and the rest
    singlets ``(0, 1)`` in stable atom order. An impossible budget returns false
    without modifying any marker or occupancy field.
    """

    # Screen candidates before resolving any deferred electronic labels. Invalid
    # intermediates may still be explored, but cannot terminate the search.
    obmol = cast(ob.OBMol, omol.OBMol)
    if not search_constraints_satisfied(obmol):
        return False
    for bond in mol_bonds(obmol):
        a = cast(ob.OBAtom, bond.GetBeginAtom())
        b = cast(ob.OBAtom, bond.GetEndAtom())
        order = int(bond.GetBondOrder())
        if (
            order > 3
            and a.GetAtomicNum() in consts.NON_METAL_DICT
            and b.GetAtomicNum() in consts.NON_METAL_DICT
        ):
            return False
        if not nitrogen_nitrogen_order_allowed(a, b, order):
            return False
        if order not in (2, 3):
            continue
        if order == 2 and {a.GetAtomicNum(), b.GetAtomicNum()} == {6, 7}:
            carbon, nitrogen = (a, b) if a.GetAtomicNum() == 6 else (b, a)
            if not cationic_carbon_double_bond_allowed(carbon, nitrogen):
                return False
        if (
            not nitrilium_geometry_allowed(a, b, order)
            or not phosphorus_carbon_order_allowed(a, b, order)
            or not carbon_oxygen_order_allowed(a, b, order)
            or not silicon_carbon_order_allowed(a, b, order)
            or not nitrogen_oxygen_order_allowed(a, b, order)
            or not twisted_ring_link_order_allowed(a, b, order)
        ):
            return False

    formal_charge_sum = sum(atom.GetFormalCharge() for atom in mol_atoms(obmol))
    unresolved_atoms = [
        atom for atom in mol_atoms(obmol) if has_unresolved_two_electron_center(atom)
    ]
    radical_sum = sum(
        get_unpaired_electron_count(atom)
        for atom in mol_atoms(obmol)
        if not has_unresolved_two_electron_center(atom)
    )

    # A low-coordinate carbon can legitimately be represented as ``C-`` with
    # one radical electron. Fresh preparation defers this two-electron center;
    # resolve that specific one-unit charge/open-shell target here, where both
    # global budgets are available, instead of teaching generic charge cleanup
    # to partially consume unresolved markers.
    if (
        total_charge - formal_charge_sum == -1
        and len(unresolved_atoms) >= 1
        and total_radical_electrons - radical_sum == 1
    ):
        atom = unresolved_atoms[0]
        set_unresolved_two_electron_center(atom, False)
        set_unpaired_electron_count(atom, 1)
        set_lone_pair_count(atom, 0)
        atom.SetFormalCharge(-1)
        formal_charge_sum -= 1
        radical_sum += 1
        unresolved_atoms.remove(atom)

    if formal_charge_sum != total_charge:
        return False
    if any(bond.GetBondOrder() > 5 or bond.GetBondOrder() < 0 for bond in mol_bonds(obmol)):
        return False
    required_unpaired_electrons = total_radical_electrons - radical_sum
    if (
        required_unpaired_electrons < 0
        or required_unpaired_electrons % 2
        or required_unpaired_electrons > 2 * len(unresolved_atoms)
    ):
        return False

    triplet_centers = required_unpaired_electrons // 2
    for index, atom in enumerate(unresolved_atoms):
        set_unresolved_two_electron_center(atom, False)
        if index < triplet_centers:
            set_unpaired_electron_count(atom, 2)
            set_lone_pair_count(atom, 0)
        else:
            set_unpaired_electron_count(atom, 0)
            set_lone_pair_count(atom, 1)
    return True


def _remove_one_long_seed_bond(omol: pybel.Molecule, pattern: pybel.Smarts) -> bool:
    """Delete one SMARTS-matched bond longer than 1.15 times the summed covalent radii."""
    mol = cast(ob.OBMol, omol.OBMol)
    for first_idx, second_idx, *_ in pattern.findall(omol):
        first, second = (
            cast(ob.OBAtom, mol.GetAtom(first_idx)),
            cast(ob.OBAtom, mol.GetAtom(second_idx)),
        )
        radii = ob.GetCovalentRad(first.GetAtomicNum()) + ob.GetCovalentRad(second.GetAtomicNum())
        if first.GetDistance(second) > 1.15 * radii:
            return mol.DeleteBond(mol.GetBond(first, second))
    return False


def _fills_octahedral_vacancy(mol: ob.OBMol, center: ob.OBAtom, partner: ob.OBAtom) -> bool:
    """Check a prospective sixth P bond for four cis neighbors and one trans neighbor."""
    distance = center.GetDistance(partner)
    cis = trans = 0
    similar_bond = False
    for neighbor in atom_neighbors(center):
        if neighbor_bond_order(center, neighbor) != 1:
            return False
        angle = mol.GetAngle(partner, center, neighbor)
        if 70.0 <= angle <= 110.0:
            cis += 1
        if angle >= 160.0:
            trans += 1
        if (
            neighbor.GetAtomicNum() == partner.GetAtomicNum()
            and abs(center.GetDistance(neighbor) - distance) <= 0.10
        ):
            similar_bond = True
    return cis == 4 and trans == 1 and similar_bond


def _try_add_one_connection(
    omol: pybel.Molecule,
    rules: tuple[tuple[pybel.Smarts, pybel.Smarts], ...],
    extra_tolerance_angstrom: float,
    geometry_allowed: Optional[Callable[[ob.OBMol, ob.OBAtom, ob.OBAtom], bool]] = None,
) -> bool:
    """Reconnect the first eligible donor to its nearest compatible partner within the covalent cutoff."""
    obmol = cast(ob.OBMol, omol.OBMol)
    partners_by_donor = {}
    for donor_pattern, partner_pattern in rules:
        donors = donor_pattern.findall(omol)
        if donors:
            matches = [
                cast(ob.OBAtom, obmol.GetAtom(idx)) for (idx,) in partner_pattern.findall(omol)
            ]
            for (idx,) in donors:
                partners_by_donor[idx] = matches

    # Each donor matches one rule per group. Atom order gives donors first choice.
    for idx in sorted(partners_by_donor):
        donor = cast(ob.OBAtom, obmol.GetAtom(idx))
        partners = (atom for atom in partners_by_donor[idx] if atom.GetIdx() != idx)
        donor_radius = ob.GetCovalentRad(donor.GetAtomicNum())
        for partner in sorted(partners, key=donor.GetDistance):
            partner_radius = ob.GetCovalentRad(partner.GetAtomicNum())
            if (
                donor.GetDistance(partner)
                >= donor_radius + partner_radius + extra_tolerance_angstrom
            ):
                continue
            bond = cast(Optional[ob.OBBond], obmol.GetBond(donor, partner))
            if bond and bond.GetBondOrder() != 0:
                continue
            if geometry_allowed and not geometry_allowed(obmol, donor, partner):
                continue
            if bond:
                bond.SetBondOrder(1)
            else:
                obmol.AddBond(idx, partner.GetIdx(), 1)
            return True
    return False


def make_connections(
    omol: pybel.Molecule,
    extra_tolerance_angstrom: float = 0.15,
) -> tuple[pybel.Molecule, bool]:
    """Remove overlong seed bonds and reconnect partners using their updated valences."""
    changed = False
    for pattern in smarts.LONG_SEED_BONDS:
        # Rematch after deletion: bonds sharing an endpoint may no longer qualify.
        while _remove_one_long_seed_bond(omol, pattern):
            changed = True

    # The geometry check permits a further 0.10 Å for the sixth P bond.
    while (
        _try_add_one_connection(
            omol,
            smarts.OCTAHEDRAL_PHOSPHORUS_RULES,
            extra_tolerance_angstrom + 0.10,
            _fills_octahedral_vacancy,
        )
        or _try_add_one_connection(omol, smarts.PRIMARY_CONNECTION_RULES, extra_tolerance_angstrom)
        or _try_add_one_connection(
            omol, smarts.SECONDARY_CONNECTION_RULES, extra_tolerance_angstrom
        )
    ):
        changed = True
    return omol, changed


def _remove_one_long_ring_diagonal(omol: pybel.Molecule) -> bool:
    """Remove an overlong C–N contact across a four-membered ring.

    The SMARTS orders the atoms as bridgehead C, outer C, bridgehead N,
    outer C. A strained bicyclic structure can have a real shared C–N bond.
    Allow the same 15% stretch as for other seed bonds, relative to both the
    covalent radii and the longest perimeter bond. Rematch after each edit
    because rings can overlap.
    """
    mol = cast(ob.OBMol, omol.OBMol)
    for match in smarts.PRE_CLEAN_BCP_RING_4.findall(omol):
        carbon, first, nitrogen, second = (cast(ob.OBAtom, mol.GetAtom(idx)) for idx in match)
        radii = ob.GetCovalentRad(carbon.GetAtomicNum()) + ob.GetCovalentRad(
            nitrogen.GetAtomicNum()
        )
        longest_side = max(
            end.GetDistance(outer) for end in (carbon, nitrogen) for outer in (first, second)
        )
        if carbon.GetDistance(nitrogen) > 1.15 * max(radii, longest_side):
            return mol.DeleteBond(cast(ob.OBBond, mol.GetBond(carbon, nitrogen)))
    return False


def pre_clean(omol: pybel.Molecule) -> tuple[pybel.Molecule, bool]:
    """Apply cheap structural fixes that simplify the later heuristic stages."""

    obmol = cast(ob.OBMol, omol.OBMol)
    hit = False

    while res := smarts.PRE_CLEAN_HYPERVALENT.findall(omol):
        idxs_1 = cast(Tuple[int, int], res.pop(0))
        obbond = cast(ob.OBBond, obmol.GetBond(idxs_1[0], idxs_1[1]))
        obbond.SetBondOrder(obbond.GetBondOrder() - 1)
        hit = True

    def neutral_carbon_monochalcogenide(bond: ob.OBBond) -> bool:
        atoms = (cast(ob.OBAtom, bond.GetBeginAtom()), cast(ob.OBAtom, bond.GetEndAtom()))
        numbers = {atom.GetAtomicNum() for atom in atoms}
        return numbers in ({6, 8}, {6, 16}) and all(
            atom.GetTotalDegree() == 1 and atom.GetFormalCharge() == 0 for atom in atoms
        )

    # These matches depend only on element and bond order. A single pass lets
    # isolated CS retain the same carbene/donor precursor as CO.
    for idxs_2 in smarts.PRE_CLEAN_HYPER_PI_BOND.findall(omol):
        obbond = cast(ob.OBBond, obmol.GetBond(idxs_2[0], idxs_2[1]))
        if neutral_carbon_monochalcogenide(obbond):
            continue
        obbond.SetBondOrder(1)
        hit = True

    for bond in mol_bonds(obmol):
        if neutral_carbon_monochalcogenide(bond) and bond.GetBondOrder() in (1, 3):
            # Fresh electron inference marks neutral C=X as an unresolved
            # carbon center. The shared donor rule then yields C-#X+.
            bond.SetBondOrder(2)
            hit = True

    while res := smarts.PRE_CLEAN_BCP_RING_5.findall(omol):
        idxs_3 = cast(Tuple[int, int, int, int, int], res.pop(0))
        bcp_n: Optional[int] = None
        bcp_c: Optional[int] = None
        for idx in idxs_3:
            indexs = set(idxs_3) - {idx}
            if all(cast(ob.OBBond, obmol.GetBond(idx, idx_3)) for idx_3 in indexs):
                if cast(ob.OBAtom, obmol.GetAtom(idx)).GetAtomicNum() == 7:
                    bcp_n = idx
                if cast(ob.OBAtom, obmol.GetAtom(idx)).GetAtomicNum() == 6:
                    bcp_c = idx
        if bcp_n is not None and bcp_c is not None:
            obmol.DeleteBond(cast(ob.OBBond, obmol.GetBond(bcp_n, bcp_c)))
            hit = True

    while _remove_one_long_ring_diagonal(omol):
        hit = True

    return omol, hit


__all__ = ["make_connections", "pre_clean", "validate_omol"]
