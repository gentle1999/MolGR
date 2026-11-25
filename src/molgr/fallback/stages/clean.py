"""Radical/resonance cleanup stages for fallback."""

from __future__ import annotations

from itertools import combinations
from typing import List, Tuple, cast

from openbabel import openbabel as ob
from openbabel import pybel

from molgr.fallback.stages.fresh import assign_charge_radical_for_atom
from molgr.fallback.utils import consts, smarts
from molgr.fallback.utils.bond_geometry import (
    nitrogen_nitrogen_order_allowed,
    nitrogen_oxygen_order_allowed,
    protect_short_carbonyl,
    would_form_bent_ring_allene,
)
from molgr.fallback.utils.electrons import (
    get_lone_pair_count,
    get_unpaired_electron_count,
    has_unresolved_two_electron_center,
    set_lone_pair_count,
    set_unpaired_electron_count,
    set_unresolved_two_electron_center,
)
from molgr.fallback.utils.ob_graph import (
    atom_neighbors,
    mol_atoms,
    mol_bonds,
    neighbor_bond_order,
    ring_paths,
)


def clean_carbene_neighbor_unsaturated(omol: pybel.Molecule) -> tuple[pybel.Molecule, bool]:
    """Move an unresolved two-electron center through an adjacent pi bond.

    ``A*-B=C`` becomes ``A=B-C``. The unresolved marker at A is consumed and
    replaced by one real unpaired electron at A plus one at C, representing the
    homolytic redistribution. Existing active lone pairs at C are preserved.
    """

    obmol = cast(ob.OBMol, omol.OBMol)
    hit = False
    while True:
        # A bond edit can expose another path for an unresolved center. Requery
        # after each applied move, in the same priority order as the C++ stage.
        for idxs in smarts.CLEAN_CARBENE_NEIGHBOR_UNSAT.findall(omol):
            atom1 = cast(ob.OBAtom, obmol.GetAtom(idxs[0]))
            atom3 = cast(ob.OBAtom, obmol.GetAtom(idxs[2]))
            bond12 = cast(ob.OBBond, obmol.GetBond(idxs[0], idxs[1]))
            bond23 = cast(ob.OBBond, obmol.GetBond(idxs[1], idxs[2]))
            if (
                has_unresolved_two_electron_center(atom1)
                and get_unpaired_electron_count(atom3) == 0
                and not has_unresolved_two_electron_center(atom3)
                and bond12.GetBondOrder() == 1
                and bond23.GetBondOrder() == 2
                and not protect_short_carbonyl(obmol.GetAtom(idxs[1]), atom3, atom1)
            ):
                bond23.SetBondOrder(int(bond23.GetBondOrder() - 1))
                bond12.SetBondOrder(int(bond12.GetBondOrder() + 1))
                set_unresolved_two_electron_center(atom1, False)
                set_lone_pair_count(atom1, 0)
                set_unpaired_electron_count(atom1, 1)
                set_unpaired_electron_count(atom3, get_unpaired_electron_count(atom3) + 1)
                hit = True
                break
        else:
            return omol, hit


def clean_possible_1_3_dipole(
    omol: pybel.Molecule,
    given_charge: int,
    total_radical_electrons: int,
) -> tuple[pybel.Molecule, bool]:
    """Convert excess terminal radicals into a neutral 1,3-dipole.

    A conversion is available while the current real-unpaired-electron count,
    after reserving the requested system radicals and pending formal-charge
    assignments, contains at least two excess electrons. The two matched terminal
    atoms each consume one real unpaired electron. The terminal with the lower
    average explicit bond order forms the additional bond to the middle atom;
    ties are resolved by atom index. The middle atom becomes ``+1`` and the
    opposite terminal becomes ``-1``.
    """

    obmol = cast(ob.OBMol, omol.OBMol)
    hit = False

    def available_unpaired_electrons() -> int:
        return (
            sum(get_unpaired_electron_count(atom) for atom in mol_atoms(obmol))
            - total_radical_electrons
            - abs(given_charge)
        )

    matches: List[Tuple[int, int, int]] = list(smarts.CLEAN_POSSIBLE_1_3_DIPOLE.findall(omol))
    for idxs in matches:
        if available_unpaired_electrons() < 2:
            break

        atom1 = cast(ob.OBAtom, obmol.GetAtom(idxs[0]))
        atom2 = cast(ob.OBAtom, obmol.GetAtom(idxs[1]))
        atom3 = cast(ob.OBAtom, obmol.GetAtom(idxs[2]))
        bond12 = cast(ob.OBBond, obmol.GetBond(idxs[0], idxs[1]))
        bond23 = cast(ob.OBBond, obmol.GetBond(idxs[1], idxs[2]))
        if not all((atom1, atom2, atom3, bond12, bond23)):
            continue
        if any(atom.GetFormalCharge() != 0 for atom in (atom1, atom2, atom3)):
            continue
        if (
            get_unpaired_electron_count(atom1) < 1
            or get_unpaired_electron_count(atom3) < 1
            or has_unresolved_two_electron_center(atom1)
            or has_unresolved_two_electron_center(atom3)
        ):
            continue

        degree1 = int(atom1.GetExplicitDegree())
        degree3 = int(atom3.GetExplicitDegree())
        if degree1 <= 0 or degree3 <= 0:
            continue
        valence1 = int(atom1.GetExplicitValence())
        valence3 = int(atom3.GetExplicitValence())
        left_average_key = valence1 * degree3
        right_average_key = valence3 * degree1
        add_left_bond = left_average_key < right_average_key or (
            left_average_key == right_average_key and atom1.GetIdx() < atom3.GetIdx()
        )
        bond_to_increase = bond12 if add_left_bond else bond23
        negative_atom = atom3 if add_left_bond else atom1
        if bond_to_increase.GetBondOrder() >= 3:
            continue

        bond_to_increase.SetBondOrder(bond_to_increase.GetBondOrder() + 1)
        atom2.SetFormalCharge(1)
        negative_atom.SetFormalCharge(-1)
        set_unpaired_electron_count(atom1, get_unpaired_electron_count(atom1) - 1)
        set_unpaired_electron_count(atom3, get_unpaired_electron_count(atom3) - 1)
        hit = True

    return omol, hit


def clean_neighbor_radicals(
    omol: pybel.Molecule,
    given_charge: int,
    total_radical_electrons: int,
) -> tuple[pybel.Molecule, bool]:
    """Resolve adjacent radical-compatible electron states.

    Each endpoint contributes either its real unpaired electrons or the two
    electrons represented by an unresolved center. A matched pair consumes one
    electron from each endpoint and increases the bond by one order. The
    operation is limited by the excess-electron budget, so two unresolved
    centers can consume one pair and become a double bond while their remaining
    electrons are classified on the next pass. A real radical adjacent to an
    unresolved center therefore preserves the real-radical count after one
    pair is consumed, while two real radicals require two excess electrons.
    """

    obmol = cast(ob.OBMol, omol.OBMol)

    def available_electron_budget() -> int:
        return (
            sum(
                get_unpaired_electron_count(atom)
                + (2 if has_unresolved_two_electron_center(atom) else 0)
                for atom in mol_atoms(obmol)
            )
            - total_radical_electrons
            - abs(given_charge)
        )

    hit = False
    for bond in list(mol_bonds(obmol)):
        begin_atom = cast(ob.OBAtom, bond.GetBeginAtom())
        end_atom = cast(ob.OBAtom, bond.GetEndAtom())
        begin_unpaired = get_unpaired_electron_count(begin_atom)
        end_unpaired = get_unpaired_electron_count(end_atom)
        begin_unresolved = has_unresolved_two_electron_center(begin_atom)
        end_unresolved = has_unresolved_two_electron_center(end_atom)

        begin_capacity = 2 if begin_unresolved else begin_unpaired
        end_capacity = 2 if end_unresolved else end_unpaired
        if begin_capacity <= 0 or end_capacity <= 0:
            continue

        available = available_electron_budget()
        if available < 2:
            continue
        bond_to_add = min(begin_capacity, end_capacity, available // 2, 3 - bond.GetBondOrder())
        if bond_to_add <= 0:
            continue

        bond.SetBondOrder(bond.GetBondOrder() + bond_to_add)
        for atom, unresolved, capacity in (
            (begin_atom, begin_unresolved, begin_capacity),
            (end_atom, end_unresolved, end_capacity),
        ):
            remaining = capacity - bond_to_add
            if unresolved:
                set_unresolved_two_electron_center(atom, False)
                set_lone_pair_count(atom, 0)
                set_unpaired_electron_count(atom, remaining)
                # A partial consumption leaves one real radical at this
                # endpoint. Re-running the local classifier would see the
                # post-bond two-electron deficit and mark it unresolved again.
                if remaining == 0:
                    assign_charge_radical_for_atom(atom)
            else:
                set_unpaired_electron_count(atom, remaining)
                assign_charge_radical_for_atom(atom)
        hit = True
    return omol, hit


def _pi_cleanup_preserves_nitrogen_geometry(
    mol: ob.OBMol,
    bonds: tuple[ob.OBBond, ...],
    charge_changes: tuple[tuple[int, int], ...] = (),
) -> bool:
    """Check the proposed pi pattern before consuming terminal electrons.

    Cleanup must not collapse a viable hydrazone or oximate contributor into
    an azo or nitroso pattern that final validation rejects. Apply all coupled bond changes on
    a copy: checking the unshifted neighbors misses the short single C--N
    bonds in the proposed product. Charge-shifting normalization must also
    supply the endpoint changes so charge-sensitive geometry checks see the
    complete product. Reuse the existing geometry predicate.
    """
    nitrogens = {
        atom.GetIdx()
        for bond in bonds
        for atom in (cast(ob.OBAtom, bond.GetBeginAtom()), cast(ob.OBAtom, bond.GetEndAtom()))
        if atom.GetAtomicNum() == 7
    }
    nitrogens.update(idx for idx, _ in charge_changes if mol.GetAtom(idx).GetAtomicNum() == 7)
    targets = [
        (idx, a.GetIdx())
        for idx in nitrogens
        for a in atom_neighbors(mol.GetAtom(idx))
        if a.GetAtomicNum() in (7, 8)
    ]
    if not targets:
        return True
    proposed = ob.OBMol(mol)
    for atom_idx, delta in charge_changes:
        atom = cast(ob.OBAtom, proposed.GetAtom(atom_idx))
        atom.SetFormalCharge(atom.GetFormalCharge() + delta)
    for index, bond in enumerate(bonds):
        proposed.GetBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()).SetBondOrder(
            2 if index % 2 == 0 else 1
        )
    for n_idx, other_idx in targets:
        nitrogen, other = (
            cast(ob.OBAtom, proposed.GetAtom(n_idx)),
            cast(ob.OBAtom, proposed.GetAtom(other_idx)),
        )
        order = neighbor_bond_order(nitrogen, other)
        if not (
            nitrogen_nitrogen_order_allowed(nitrogen, other, order)
            and nitrogen_oxygen_order_allowed(nitrogen, other, order)
        ):
            return False
    return True


def clean_1_4_radicals(
    omol: pybel.Molecule,
    given_charge: int,
    total_radical_electrons: int,
) -> tuple[pybel.Molecule, bool]:
    """Resolve separated terminal radicals or unresolved electron centers.

    For ``A-B=C-D``, consume one eligible electron state at each terminal and
    shift the middle pi bond to the two outer bonds. Real radicals obey the
    same global excess-electron budget as :func:`clean_neighbor_radicals`;
    unresolved two-electron centers are consumed directly.
    """

    obmol = cast(ob.OBMol, omol.OBMol)

    def available_unpaired_electrons() -> int:
        return (
            sum(get_unpaired_electron_count(atom) for atom in mol_atoms(obmol))
            - total_radical_electrons
            - abs(given_charge)
        )

    hit = False
    for idxs in list(smarts.CLEAN_1_4_RADICALS.findall(omol)):
        atom1 = cast(ob.OBAtom, obmol.GetAtom(idxs[0]))
        atom2 = cast(ob.OBAtom, obmol.GetAtom(idxs[1]))
        atom3 = cast(ob.OBAtom, obmol.GetAtom(idxs[2]))
        atom4 = cast(ob.OBAtom, obmol.GetAtom(idxs[3]))
        bond1 = cast(ob.OBBond, obmol.GetBond(idxs[0], idxs[1]))
        bond2 = cast(ob.OBBond, obmol.GetBond(idxs[1], idxs[2]))
        bond3 = cast(ob.OBBond, obmol.GetBond(idxs[2], idxs[3]))
        if not all((atom1, atom2, atom3, atom4, bond1, bond2, bond3)):
            continue
        if bond1.GetBondOrder() != 1 or bond2.GetBondOrder() != 2 or bond3.GetBondOrder() != 1:
            continue

        endpoints = (atom1, atom4)
        endpoint_radicals = [get_unpaired_electron_count(atom) for atom in endpoints]
        endpoint_unresolved = [has_unresolved_two_electron_center(atom) for atom in endpoints]
        if not all(
            radical > 0 or unresolved
            for radical, unresolved in zip(endpoint_radicals, endpoint_unresolved)
        ):
            continue
        if (
            sum(radical > 0 for radical in endpoint_radicals) == 2
            and available_unpaired_electrons() < 2
        ):
            continue

        if not _pi_cleanup_preserves_nitrogen_geometry(obmol, (bond1, bond2, bond3)):
            continue

        bond1.SetBondOrder(2)
        bond2.SetBondOrder(1)
        bond3.SetBondOrder(2)
        for atom, radical, unresolved in zip(endpoints, endpoint_radicals, endpoint_unresolved):
            if radical > 0:
                set_unpaired_electron_count(atom, radical - 1)
            if unresolved:
                set_unresolved_two_electron_center(atom, False)
                set_lone_pair_count(atom, 0)
            assign_charge_radical_for_atom(atom)
        hit = True
    return omol, hit


def clean_1_6_radicals(
    omol: pybel.Molecule,
    given_charge: int,
    total_radical_electrons: int,
) -> tuple[pybel.Molecule, bool]:
    """Resolve terminal electron states across an alternating six-atom path.

    For ``A-B=C-D=E-F``, consume one eligible state at A and F and shift the
    two existing pi bonds outward, producing ``A=B-C=D-E=F``. Endpoint and
    global-electron conditions are identical to :func:`clean_1_4_radicals`.
    """

    obmol = cast(ob.OBMol, omol.OBMol)

    def available_unpaired_electrons() -> int:
        return (
            sum(get_unpaired_electron_count(atom) for atom in mol_atoms(obmol))
            - total_radical_electrons
            - abs(given_charge)
        )

    hit = False
    for idxs in list(smarts.CLEAN_1_6_RADICALS.findall(omol)):
        atoms = tuple(cast(ob.OBAtom, obmol.GetAtom(idx)) for idx in idxs)
        bonds = tuple(
            cast(ob.OBBond, obmol.GetBond(begin_idx, end_idx))
            for begin_idx, end_idx in zip(idxs, idxs[1:])
        )
        if not all((*atoms, *bonds)):
            continue
        if tuple(bond.GetBondOrder() for bond in bonds) != (1, 2, 1, 2, 1):
            continue

        endpoints = (atoms[0], atoms[-1])
        endpoint_radicals = [get_unpaired_electron_count(atom) for atom in endpoints]
        endpoint_unresolved = [has_unresolved_two_electron_center(atom) for atom in endpoints]
        if not all(
            radical > 0 or unresolved
            for radical, unresolved in zip(endpoint_radicals, endpoint_unresolved)
        ):
            continue
        if (
            sum(radical > 0 for radical in endpoint_radicals) == 2
            and available_unpaired_electrons() < 2
        ):
            continue

        if not _pi_cleanup_preserves_nitrogen_geometry(obmol, bonds):
            continue

        for bond, order in zip(bonds, (2, 1, 2, 1, 2)):
            bond.SetBondOrder(order)
        for atom, radical, unresolved in zip(endpoints, endpoint_radicals, endpoint_unresolved):
            if radical > 0:
                set_unpaired_electron_count(atom, radical - 1)
            if unresolved:
                set_unresolved_two_electron_center(atom, False)
                set_lone_pair_count(atom, 0)
            assign_charge_radical_for_atom(atom)
        hit = True
    return omol, hit


def clean_resonances_0(omol: pybel.Molecule) -> tuple[pybel.Molecule, bool]:
    """Neutralize opposite terminal charges by shifting three consecutive bond orders."""
    obmol = cast(ob.OBMol, omol.OBMol)
    res: List[Tuple[int, int, int, int]] = list(smarts.CLEAN_RESONANCE_0.findall(omol))
    hit = False
    while len(res):
        idxs = res.pop(0)
        obatom1 = cast(ob.OBAtom, obmol.GetAtom(idxs[0]))
        obatom4 = cast(ob.OBAtom, obmol.GetAtom(idxs[-1]))
        obbond1 = cast(ob.OBBond, obmol.GetBond(idxs[0], idxs[1]))
        obbond2 = cast(ob.OBBond, obmol.GetBond(idxs[1], idxs[2]))
        obbond3 = cast(ob.OBBond, obmol.GetBond(idxs[2], idxs[3]))
        if (
            obatom1.GetFormalCharge() == -1
            and obatom4.GetFormalCharge() == 1
            and obbond1.GetBondOrder() == 1
            and obbond2.GetBondOrder() == 2
            and consts.NON_METAL_DICT[obatom1.GetAtomicNum()].default_valence
            > obatom1.GetTotalValence()
            and consts.NON_METAL_DICT[obatom4.GetAtomicNum()].default_valence
            > obatom4.GetTotalValence()
        ):
            obbond1.SetBondOrder(obbond1.GetBondOrder() + 1)
            obbond2.SetBondOrder(obbond2.GetBondOrder() - 1)
            obbond3.SetBondOrder(obbond3.GetBondOrder() + 1)
            obatom1.SetFormalCharge(obatom1.GetFormalCharge() + 1)
            obatom4.SetFormalCharge(obatom4.GetFormalCharge() - 1)
            hit = True
    return omol, hit


def clean_resonances_1(omol: pybel.Molecule) -> tuple[pybel.Molecule, bool]:
    """Convert an anion–cation cumulene to a triple/single pair with the anion at the opposite end."""
    obmol = cast(ob.OBMol, omol.OBMol)
    res: List[Tuple[int, int, int]] = list(smarts.CLEAN_RESONANCE_1.findall(omol))
    hit = False
    while len(res):
        idxs = res.pop(0)
        obatom1 = cast(ob.OBAtom, obmol.GetAtom(idxs[0]))
        obatom2 = cast(ob.OBAtom, obmol.GetAtom(idxs[1]))
        obatom3 = cast(ob.OBAtom, obmol.GetAtom(idxs[-1]))
        obbond1 = cast(ob.OBBond, obmol.GetBond(idxs[0], idxs[1]))
        obbond2 = cast(ob.OBBond, obmol.GetBond(idxs[1], idxs[2]))
        if (
            obatom1.GetFormalCharge() == -1
            and obatom2.GetFormalCharge() == 1
            and obatom3.GetFormalCharge() == 0
            and obbond1.GetBondOrder() == 2
            and obbond2.GetBondOrder() == 2
        ):
            obbond1.SetBondOrder(obbond1.GetBondOrder() + 1)
            obbond2.SetBondOrder(obbond2.GetBondOrder() - 1)
            obatom1.SetFormalCharge(obatom1.GetFormalCharge() + 1)
            obatom3.SetFormalCharge(obatom3.GetFormalCharge() - 1)
            hit = True
    return omol, hit


def clean_resonances_2(omol: pybel.Molecule) -> tuple[pybel.Molecule, bool]:
    """Shift a conjugated terminal C or N anion onto the carbonyl oxygen through four bonds."""
    obmol = cast(ob.OBMol, omol.OBMol)
    res: List[Tuple[int, int, int, int, int, int]] = list(smarts.CLEAN_RESONANCE_2.findall(omol))
    hit = False
    while len(res):
        idxs = res.pop(0)
        obatom1 = cast(ob.OBAtom, obmol.GetAtom(idxs[0]))
        obatom5 = cast(ob.OBAtom, obmol.GetAtom(idxs[-1]))
        obbond1 = cast(ob.OBBond, obmol.GetBond(idxs[0], idxs[1]))
        obbond2 = cast(ob.OBBond, obmol.GetBond(idxs[1], idxs[3]))
        obbond3 = cast(ob.OBBond, obmol.GetBond(idxs[3], idxs[4]))
        obbond4 = cast(ob.OBBond, obmol.GetBond(idxs[4], idxs[5]))
        obatom5_room = consts.NON_METAL_DICT[obatom5.GetAtomicNum()].default_valence - (
            obatom5.GetTotalValence() + 1
        )
        if (
            obbond4.GetBondOrder() == 1
            and obbond3.GetBondOrder() == 2
            and obbond2.GetBondOrder() == 1
            and obbond1.GetBondOrder() == 2
            and obatom1.GetFormalCharge() == 0
            and obatom5.GetFormalCharge() == -1
            and obatom5_room >= 0
            and not would_form_bent_ring_allene(obatom5, obmol.GetAtom(idxs[4]))
        ):
            if not _pi_cleanup_preserves_nitrogen_geometry(
                obmol,
                (obbond4, obbond3, obbond2, obbond1),
                ((obatom1.GetIdx(), -1), (obatom5.GetIdx(), 1)),
            ):
                continue
            obbond4.SetBondOrder(obbond4.GetBondOrder() + 1)
            obbond3.SetBondOrder(obbond3.GetBondOrder() - 1)
            obbond2.SetBondOrder(obbond2.GetBondOrder() + 1)
            obbond1.SetBondOrder(obbond1.GetBondOrder() - 1)
            obatom1.SetFormalCharge(obatom1.GetFormalCharge() - 1)
            obatom5.SetFormalCharge(obatom5.GetFormalCharge() + 1)
            hit = True
    return omol, hit


def clean_resonances_3(omol: pybel.Molecule) -> tuple[pybel.Molecule, bool]:
    """Shift the rule-3 charge/pi pattern, then rebuild its two endpoints.

    Internal atoms exchange one bond-order unit on each side and keep the same
    total valence. Only the two charged endpoints change total valence, so only
    they are allowed to replace unpaired, active-lone-pair, or unresolved fields.
    """

    obmol = cast(ob.OBMol, omol.OBMol)
    res: List[Tuple[int, int, int, int, int]] = list(smarts.CLEAN_RESONANCE_3.findall(omol))
    hit = False
    while len(res):
        idxs = res.pop(0)
        obatom1 = cast(ob.OBAtom, obmol.GetAtom(idxs[0]))
        obatom5 = cast(ob.OBAtom, obmol.GetAtom(idxs[-1]))
        obbond1 = cast(ob.OBBond, obmol.GetBond(idxs[0], idxs[1]))
        obbond2 = cast(ob.OBBond, obmol.GetBond(idxs[1], idxs[2]))
        obbond3 = cast(ob.OBBond, obmol.GetBond(idxs[2], idxs[3]))
        obbond4 = cast(ob.OBBond, obmol.GetBond(idxs[3], idxs[4]))
        if (
            obatom1.GetFormalCharge() == 1
            and obatom5.GetFormalCharge() == -1
            and obbond1.GetBondOrder() == 2
            and obbond2.GetBondOrder() == 1
            and obbond3.GetBondOrder() == 2
            and obbond4.GetBondOrder() == 1
        ):
            if not _pi_cleanup_preserves_nitrogen_geometry(
                obmol,
                (obbond4, obbond3, obbond2, obbond1),
                ((obatom1.GetIdx(), -1), (obatom5.GetIdx(), 1)),
            ):
                continue
            obbond1.SetBondOrder(obbond1.GetBondOrder() - 1)
            obbond2.SetBondOrder(obbond2.GetBondOrder() + 1)
            obbond3.SetBondOrder(obbond3.GetBondOrder() - 1)
            obbond4.SetBondOrder(obbond4.GetBondOrder() + 1)
            obatom1.SetFormalCharge(obatom1.GetFormalCharge() - 1)
            obatom5.SetFormalCharge(obatom5.GetFormalCharge() + 1)
            assign_charge_radical_for_atom(obatom1)
            assign_charge_radical_for_atom(obatom5)
            hit = True
    return omol, hit


def clean_resonances_4(omol: pybel.Molecule) -> tuple[pybel.Molecule, bool]:
    """Neutralize a cationic N/O and a 1,3-related anion by moving their intervening pi bond."""
    obmol = cast(ob.OBMol, omol.OBMol)
    res: List[Tuple[int, int, int]] = list(smarts.CLEAN_RESONANCE_4.findall(omol))
    hit = False
    while len(res):
        idxs = res.pop(0)
        obatom1 = cast(ob.OBAtom, obmol.GetAtom(idxs[0]))
        obatom3 = cast(ob.OBAtom, obmol.GetAtom(idxs[-1]))
        obbond1 = cast(ob.OBBond, obmol.GetBond(idxs[0], idxs[1]))
        obbond2 = cast(ob.OBBond, obmol.GetBond(idxs[1], idxs[2]))
        if (
            obatom1.GetFormalCharge() == 1
            and obatom3.GetFormalCharge() == -1
            and obbond1.GetBondOrder() == 2
            and obbond2.GetBondOrder() == 1
        ):
            obbond2.SetBondOrder(obbond2.GetBondOrder() + 1)
            obbond1.SetBondOrder(obbond1.GetBondOrder() - 1)
            obatom1.SetFormalCharge(obatom1.GetFormalCharge() - 1)
            obatom3.SetFormalCharge(obatom3.GetFormalCharge() + 1)
            hit = True
    return omol, hit


def clean_resonances_5(omol: pybel.Molecule) -> tuple[pybel.Molecule, bool]:
    """Move an allylic anion from C/N onto a conjugated neutral N, O or S."""
    obmol = cast(ob.OBMol, omol.OBMol)
    res: List[Tuple[int, int, int]] = list(smarts.CLEAN_RESONANCE_5.findall(omol))
    hit = False
    while len(res):
        idxs = res.pop(0)
        obatom1 = cast(ob.OBAtom, obmol.GetAtom(idxs[0]))
        obatom3 = cast(ob.OBAtom, obmol.GetAtom(idxs[-1]))
        obbond1 = cast(ob.OBBond, obmol.GetBond(idxs[0], idxs[1]))
        obbond2 = cast(ob.OBBond, obmol.GetBond(idxs[1], idxs[2]))
        if (
            obbond1.GetBondOrder() == 2
            and obbond2.GetBondOrder() == 1
            and obatom3.GetFormalCharge() == -1
            and obatom1.GetFormalCharge() == 0
            and not would_form_bent_ring_allene(obatom3, obmol.GetAtom(idxs[1]))
            and not protect_short_carbonyl(obmol.GetAtom(idxs[1]), obatom1, obatom3)
        ):
            obbond2.SetBondOrder(obbond2.GetBondOrder() + 1)
            obbond1.SetBondOrder(obbond1.GetBondOrder() - 1)
            obatom1.SetFormalCharge(obatom1.GetFormalCharge() - 1)
            obatom3.SetFormalCharge(obatom3.GetFormalCharge() + 1)
            hit = True
    return omol, hit


def clean_resonances_6(omol: pybel.Molecule) -> tuple[pybel.Molecule, bool]:
    """Localize an anionic cumulene as a triple bond with the opposite carbon carrying the charge."""
    obmol = cast(ob.OBMol, omol.OBMol)
    res: List[Tuple[int, int, int]] = list(smarts.CLEAN_RESONANCE_6.findall(omol))
    hit = False
    while len(res):
        idxs = res.pop(0)
        obatom1 = cast(ob.OBAtom, obmol.GetAtom(idxs[0]))
        obatom3 = cast(ob.OBAtom, obmol.GetAtom(idxs[-1]))
        obbond1 = cast(ob.OBBond, obmol.GetBond(idxs[0], idxs[1]))
        obbond2 = cast(ob.OBBond, obmol.GetBond(idxs[1], idxs[2]))
        if (
            obatom1.GetFormalCharge() == 0
            and obatom3.GetFormalCharge() == -1
            and obbond1.GetBondOrder() == 2
            and obbond2.GetBondOrder() == 2
        ):
            obbond2.SetBondOrder(obbond2.GetBondOrder() + 1)
            obbond1.SetBondOrder(obbond1.GetBondOrder() - 1)
            obatom1.SetFormalCharge(obatom1.GetFormalCharge() - 1)
            obatom3.SetFormalCharge(obatom3.GetFormalCharge() + 1)
            hit = True
    return omol, hit


def clean_resonances_7(omol: pybel.Molecule) -> tuple[pybel.Molecule, bool]:
    """Move a ring anion onto its exocyclic neighbor while restoring an endocyclic double bond."""
    obmol = cast(ob.OBMol, omol.OBMol)
    res: List[Tuple[int, int, int, int, int, int, int]] = list(
        smarts.CLEAN_RESONANCE_7.findall(omol)
    )
    hit = False
    while len(res):
        idxs = res.pop(0)
        obatom1 = cast(ob.OBAtom, obmol.GetAtom(idxs[0]))
        obatom3 = cast(ob.OBAtom, obmol.GetAtom(idxs[2]))
        obbond1 = cast(ob.OBBond, obmol.GetBond(idxs[0], idxs[1]))
        obbond2 = cast(ob.OBBond, obmol.GetBond(idxs[1], idxs[2]))
        obatom1_room = consts.NON_METAL_DICT[obatom1.GetAtomicNum()].default_valence - (
            obatom1.GetTotalValence() + 1
        )
        if (
            obbond1.GetBondOrder() == 1
            and obbond2.GetBondOrder() == 2
            and obatom1.GetFormalCharge() == -1
            and obatom3.GetFormalCharge() == 0
            and obatom1_room >= 0
        ):
            obbond1.SetBondOrder(obbond1.GetBondOrder() + 1)
            obbond2.SetBondOrder(obbond2.GetBondOrder() - 1)
            obatom1.SetFormalCharge(obatom1.GetFormalCharge() + 1)
            obatom3.SetFormalCharge(obatom3.GetFormalCharge() - 1)
            hit = True
    return omol, hit


def clean_resonances_8(omol: pybel.Molecule) -> tuple[pybel.Molecule, bool]:
    """Shift the rule-8 charge/pi pattern and rebuild both endpoint states.

    Endpoint refresh may produce unpaired electrons, active lone pairs, or a new
    unresolved center from the changed local valence; no old field is blindly
    carried across the bond/charge change.
    """

    obmol = cast(ob.OBMol, omol.OBMol)
    obmol.SetAromaticPerceived(False)
    res = smarts.CLEAN_RESONANCE_8.findall(omol)
    hit = False
    while len(res):
        idxs = res.pop(0)
        obatom1 = cast(ob.OBAtom, obmol.GetAtom(idxs[0]))
        obatom5 = cast(ob.OBAtom, obmol.GetAtom(idxs[4]))
        obbond1 = cast(ob.OBBond, obmol.GetBond(idxs[0], idxs[1]))
        obbond2 = cast(ob.OBBond, obmol.GetBond(idxs[1], idxs[2]))
        obbond3 = cast(ob.OBBond, obmol.GetBond(idxs[2], idxs[3]))
        obbond4 = cast(ob.OBBond, obmol.GetBond(idxs[3], idxs[4]))
        if (
            obatom1.GetFormalCharge() == -1
            and obatom5.GetFormalCharge() == 0
            and obbond1.GetBondOrder() == 1
            and obbond2.GetBondOrder() == 2
            and obbond3.GetBondOrder() == 1
            and obbond4.GetBondOrder() == 2
        ):
            obbond1.SetBondOrder(obbond1.GetBondOrder() + 1)
            obbond2.SetBondOrder(obbond2.GetBondOrder() - 1)
            obbond3.SetBondOrder(obbond3.GetBondOrder() + 1)
            obbond4.SetBondOrder(obbond4.GetBondOrder() - 1)
            obatom1.SetFormalCharge(obatom1.GetFormalCharge() + 1)
            obatom5.SetFormalCharge(obatom5.GetFormalCharge() - 1)
            assign_charge_radical_for_atom(obatom1)
            assign_charge_radical_for_atom(obatom5)
            hit = True
    return omol, hit


def clean_resonances_9(omol: pybel.Molecule) -> tuple[pybel.Molecule, bool]:
    """Increase bond order to neutralize adjacent opposite charges, up to a triple bond."""
    obmol = cast(ob.OBMol, omol.OBMol)
    res: List[Tuple[int, int]] = list(smarts.CLEAN_RESONANCE_9.findall(omol))
    hit = False
    while len(res):
        idxs = res.pop(0)
        obatom1 = cast(ob.OBAtom, obmol.GetAtom(idxs[0]))
        obatom2 = cast(ob.OBAtom, obmol.GetAtom(idxs[1]))
        obbond1 = cast(ob.OBBond, obmol.GetBond(idxs[0], idxs[1]))
        if (
            obatom1.GetFormalCharge() > 0
            and obatom2.GetFormalCharge() < 0
            and consts.NON_METAL_DICT[obatom1.GetAtomicNum()].default_valence
            - obatom1.GetTotalValence()
            >= 1
            and consts.NON_METAL_DICT[obatom2.GetAtomicNum()].default_valence
            - obatom2.GetTotalValence()
            >= 1
            and obbond1.GetBondOrder() in (1, 2)
        ):
            bond_to_add = min(
                3 - obbond1.GetBondOrder(),
                consts.NON_METAL_DICT[obatom1.GetAtomicNum()].default_valence
                - obatom1.GetTotalValence(),
                consts.NON_METAL_DICT[obatom2.GetAtomicNum()].default_valence
                - obatom2.GetTotalValence(),
            )
            obbond1.SetBondOrder(obbond1.GetBondOrder() + bond_to_add)
            obatom1.SetFormalCharge(obatom1.GetFormalCharge() - bond_to_add)
            obatom2.SetFormalCharge(obatom2.GetFormalCharge() + bond_to_add)
            hit = True
    return omol, hit


def clean_resonances_10(omol: pybel.Molecule) -> tuple[pybel.Molecule, bool]:
    """Consume two terminal monoradicals while redistributing one middle pi bond.

    Each terminal must carry exactly one real unpaired electron. Both are consumed
    to form the two new bond-order units; lone pairs and unresolved centers cannot
    substitute for either radical.
    """

    obmol = cast(ob.OBMol, omol.OBMol)
    res: List[Tuple[int, int, int, int]] = list(smarts.CLEAN_RESONANCE_10.findall(omol))
    hit = False
    while len(res):
        idxs = res.pop(0)
        obatom1 = cast(ob.OBAtom, obmol.GetAtom(idxs[0]))
        obatom3 = cast(ob.OBAtom, obmol.GetAtom(idxs[-1]))
        obbond1 = cast(ob.OBBond, obmol.GetBond(idxs[0], idxs[1]))
        obbond2 = cast(ob.OBBond, obmol.GetBond(idxs[1], idxs[2]))
        obbond3 = cast(ob.OBBond, obmol.GetBond(idxs[2], idxs[3]))
        if get_unpaired_electron_count(obatom1) == 1 and get_unpaired_electron_count(obatom3) == 1:
            if not (
                obbond1.GetBondOrder() == 1
                and obbond2.GetBondOrder() in (2, 3)
                and obbond3.GetBondOrder() == 1
            ):
                continue
            obbond2.SetBondOrder(obbond2.GetBondOrder() - 1)
            obbond1.SetBondOrder(obbond1.GetBondOrder() + 1)
            obbond3.SetBondOrder(obbond3.GetBondOrder() + 1)
            set_unpaired_electron_count(obatom1, get_unpaired_electron_count(obatom1) - 1)
            set_unpaired_electron_count(obatom3, get_unpaired_electron_count(obatom3) - 1)
            hit = True
    return omol, hit


def clean_resonances_11(omol: pybel.Molecule) -> tuple[pybel.Molecule, bool]:
    """Donate a heteroatom lone pair into an adjacent cation when the new bond geometry permits it."""
    obmol = cast(ob.OBMol, omol.OBMol)
    res: List[Tuple[int, int]] = list(smarts.CLEAN_RESONANCE_11.findall(omol))
    hit = False
    while len(res):
        idxs = res.pop(0)
        obatom1 = cast(ob.OBAtom, obmol.GetAtom(idxs[0]))
        obatom2 = cast(ob.OBAtom, obmol.GetAtom(idxs[1]))
        obbond1 = cast(ob.OBBond, obmol.GetBond(idxs[0], idxs[1]))
        if (
            obatom1.GetFormalCharge() == 0
            and obatom2.GetFormalCharge() == 1
            and consts.NON_METAL_DICT[obatom2.GetAtomicNum()].default_valence
            - obatom2.GetTotalValence()
            >= 1
            and (obbond1.GetBondOrder() == 1 or obbond1.GetBondOrder() == 2)
        ):
            if obbond1.GetBondOrder() == 1 and not _pi_cleanup_preserves_nitrogen_geometry(
                obmol, (obbond1,), ((obatom1.GetIdx(), 1), (obatom2.GetIdx(), -1))
            ):
                continue
            obbond1.SetBondOrder(obbond1.GetBondOrder() + 1)
            obatom1.SetFormalCharge(obatom1.GetFormalCharge() + 1)
            obatom2.SetFormalCharge(obatom2.GetFormalCharge() - 1)
            hit = True
    return omol, hit


def clean_resonances_12(omol: pybel.Molecule) -> tuple[pybel.Molecule, bool]:
    """Relay heteroatom donation through an alternating four-atom path to a terminal cation."""
    obmol = cast(ob.OBMol, omol.OBMol)
    res: List[Tuple[int, int, int, int]] = list(smarts.CLEAN_RESONANCE_12.findall(omol))
    hit = False
    while len(res):
        idxs = res.pop(0)
        obatom1 = cast(ob.OBAtom, obmol.GetAtom(idxs[0]))
        obatom4 = cast(ob.OBAtom, obmol.GetAtom(idxs[3]))
        obbond1 = cast(ob.OBBond, obmol.GetBond(idxs[0], idxs[1]))
        obbond2 = cast(ob.OBBond, obmol.GetBond(idxs[1], idxs[2]))
        obbond3 = cast(ob.OBBond, obmol.GetBond(idxs[2], idxs[3]))
        if (
            obatom1.GetFormalCharge() == 0
            and obatom4.GetFormalCharge() == 1
            and consts.NON_METAL_DICT[obatom4.GetAtomicNum()].default_valence
            - obatom4.GetTotalValence()
            >= 1
            and (
                obbond1.GetBondOrder() == 1
                and obbond2.GetBondOrder() == 2
                and obbond3.GetBondOrder() == 1
            )
        ):
            obbond1.SetBondOrder(obbond1.GetBondOrder() + 1)
            obbond2.SetBondOrder(obbond2.GetBondOrder() - 1)
            obbond3.SetBondOrder(obbond3.GetBondOrder() + 1)
            obatom1.SetFormalCharge(obatom1.GetFormalCharge() + 1)
            obatom4.SetFormalCharge(obatom4.GetFormalCharge() - 1)
            hit = True
    return omol, hit


def clean_resonances_13(omol: pybel.Molecule) -> tuple[pybel.Molecule, bool]:
    """Move a ring anion onto an exocyclic imine N or carbonyl O by shifting the pi bond."""
    obmol = cast(ob.OBMol, omol.OBMol)
    res: List[Tuple[int, int, int]] = list(smarts.CLEAN_RESONANCE_13.findall(omol))
    hit = False
    while len(res):
        idxs = res.pop(0)
        obatom1 = cast(ob.OBAtom, obmol.GetAtom(idxs[0]))
        obatom3 = cast(ob.OBAtom, obmol.GetAtom(idxs[2]))
        obbond1 = cast(ob.OBBond, obmol.GetBond(idxs[0], idxs[1]))
        obbond2 = cast(ob.OBBond, obmol.GetBond(idxs[1], idxs[2]))
        if (
            obatom1.GetFormalCharge() == -1
            and obatom3.GetFormalCharge() == 0
            and consts.NON_METAL_DICT[obatom1.GetAtomicNum()].default_valence
            - obatom1.GetTotalValence()
            >= 1
            and obbond1.GetBondOrder() == 1
            and obbond2.GetBondOrder() == 2
        ):
            obbond1.SetBondOrder(obbond1.GetBondOrder() + 1)
            obbond2.SetBondOrder(obbond2.GetBondOrder() - 1)
            obatom1.SetFormalCharge(obatom1.GetFormalCharge() + 1)
            obatom3.SetFormalCharge(obatom3.GetFormalCharge() - 1)
            hit = True
    return omol, hit


def clean_resonances_14(omol: pybel.Molecule) -> tuple[pybel.Molecule, bool]:
    """Neutralize the rule-14 triple-bond ion pair and rebuild both endpoints.

    Lowering the bond and neutralizing charges can expose new unpaired/lone-pair
    occupancy. Endpoint-local assignment replaces stale explicit fields and may
    create an unresolved C/N/Si/P two-electron center.
    """

    obmol = cast(ob.OBMol, omol.OBMol)
    res: List[Tuple[int, int]] = list(smarts.CLEAN_RESONANCE_14.findall(omol))
    hit = False
    while len(res):
        idxs = res.pop(0)
        obatom1 = cast(ob.OBAtom, obmol.GetAtom(idxs[0]))
        obatom2 = cast(ob.OBAtom, obmol.GetAtom(idxs[1]))
        obbond1 = cast(ob.OBBond, obmol.GetBond(idxs[0], idxs[1]))
        if (
            obatom1.GetFormalCharge() == -1
            and obatom2.GetFormalCharge() == 1
            and obbond1.GetBondOrder() == 3
        ):
            obbond1.SetBondOrder(obbond1.GetBondOrder() - 1)
            obatom1.SetFormalCharge(obatom1.GetFormalCharge() + 1)
            obatom2.SetFormalCharge(obatom2.GetFormalCharge() - 1)
            assign_charge_radical_for_atom(obatom1)
            assign_charge_radical_for_atom(obatom2)
            hit = True
    return omol, hit


def clean_resonances_16(omol: pybel.Molecule) -> tuple[pybel.Molecule, bool]:
    """Neutralize the rule-16 conjugated ion pair and rebuild both endpoints.

    The input requires zero explicit unpaired and active-lone-pair counts at both
    charged endpoints. After charge/pi migration, endpoint-local assignment
    derives any newly exposed electron state, including an unresolved center.
    """

    obmol = cast(ob.OBMol, omol.OBMol)
    res: List[Tuple[int, int, int, int, int]] = list(smarts.CLEAN_RESONANCE_16.findall(omol))
    hit = False
    while len(res):
        idxs = res.pop(0)
        obatom1 = cast(ob.OBAtom, obmol.GetAtom(idxs[0]))
        obatom5 = cast(ob.OBAtom, obmol.GetAtom(idxs[4]))
        obbond1 = cast(ob.OBBond, obmol.GetBond(idxs[0], idxs[1]))
        obbond2 = cast(ob.OBBond, obmol.GetBond(idxs[1], idxs[2]))
        obbond3 = cast(ob.OBBond, obmol.GetBond(idxs[2], idxs[3]))
        obbond4 = cast(ob.OBBond, obmol.GetBond(idxs[3], idxs[4]))

        if (
            obatom1.GetFormalCharge() == -1
            and get_unpaired_electron_count(obatom1) == 0
            and get_lone_pair_count(obatom1) == 0
            and obatom5.GetFormalCharge() == 1
            and get_unpaired_electron_count(obatom5) == 0
            and get_lone_pair_count(obatom5) == 0
            and obbond1.GetBondOrder() == 1
            and obbond2.GetBondOrder() == 2
            and obbond3.GetBondOrder() == 1
            and obbond4.GetBondOrder() == 2
        ):
            if not _pi_cleanup_preserves_nitrogen_geometry(
                obmol,
                (obbond1, obbond2, obbond3, obbond4),
                ((obatom1.GetIdx(), 1), (obatom5.GetIdx(), -1)),
            ):
                continue
            obbond1.SetBondOrder(obbond1.GetBondOrder() + 1)
            obbond2.SetBondOrder(obbond2.GetBondOrder() - 1)
            obbond3.SetBondOrder(obbond3.GetBondOrder() + 1)
            obbond4.SetBondOrder(obbond4.GetBondOrder() - 1)
            obatom1.SetFormalCharge(obatom1.GetFormalCharge() + 1)
            obatom5.SetFormalCharge(obatom5.GetFormalCharge() - 1)
            assign_charge_radical_for_atom(obatom1)
            assign_charge_radical_for_atom(obatom5)
            hit = True
    return omol, hit


def clean_resonances_17(omol: pybel.Molecule) -> tuple[pybel.Molecule, bool]:
    """Replace a ring allene with its charge-shifted conjugated resonance form.

    ``A(-)-B=C=D`` becomes ``A=B-C(-)=D`` when B, C, and D are ring atoms.
    The transformation conserves total charge and each atom's electron count;
    only the location of one negative charge and one pi bond changes.
    """

    obmol = cast(ob.OBMol, omol.OBMol)
    res: List[Tuple[int, int, int, int]] = list(smarts.CLEAN_RESONANCE_17.findall(omol))
    hit = False
    while len(res):
        idxs = res.pop(0)
        atom1 = cast(ob.OBAtom, obmol.GetAtom(idxs[0]))
        atom3 = cast(ob.OBAtom, obmol.GetAtom(idxs[2]))
        bond1 = cast(ob.OBBond, obmol.GetBond(idxs[0], idxs[1]))
        bond2 = cast(ob.OBBond, obmol.GetBond(idxs[1], idxs[2]))
        bond3 = cast(ob.OBBond, obmol.GetBond(idxs[2], idxs[3]))
        atom1_room = consts.NON_METAL_DICT[atom1.GetAtomicNum()].default_valence - (
            atom1.GetTotalValence() + 1
        )
        if (
            atom1.GetFormalCharge() == -1
            and atom3.GetFormalCharge() == 0
            and bond1.GetBondOrder() == 1
            and bond2.GetBondOrder() == 2
            and bond3.GetBondOrder() == 2
            and atom1_room >= 0
        ):
            bond1.SetBondOrder(2)
            bond2.SetBondOrder(1)
            atom1.SetFormalCharge(0)
            atom3.SetFormalCharge(-1)
            hit = True
    return omol, hit


def clean_resonances_18(omol: pybel.Molecule) -> tuple[pybel.Molecule, bool]:
    """Convert an unresolved terminal diazene fragment to a charge-separated azide.

    ``[*]-N=N-[N]`` becomes ``[*]-N=[N+]=[N-]`` when the terminal nitrogen is
    an unresolved two-electron center. The transformation consumes that deferred
    pair as the additional N-N pi bond and does not alter the system radical
    budget.
    """

    obmol = cast(ob.OBMol, omol.OBMol)
    res: List[Tuple[int, int, int, int]] = list(smarts.CLEAN_RESONANCE_18.findall(omol))
    hit = False
    while len(res):
        idxs = res.pop(0)
        atom2 = cast(ob.OBAtom, obmol.GetAtom(idxs[1]))
        atom3 = cast(ob.OBAtom, obmol.GetAtom(idxs[2]))
        atom4 = cast(ob.OBAtom, obmol.GetAtom(idxs[3]))
        bond1 = cast(ob.OBBond, obmol.GetBond(idxs[0], idxs[1]))
        bond2 = cast(ob.OBBond, obmol.GetBond(idxs[1], idxs[2]))
        bond3 = cast(ob.OBBond, obmol.GetBond(idxs[2], idxs[3]))
        if not all((atom2, atom3, atom4, bond1, bond2, bond3)):
            continue
        if (
            atom2.GetFormalCharge() != 0
            or atom3.GetFormalCharge() != 0
            or atom4.GetFormalCharge() != 0
            or get_unpaired_electron_count(atom2) != 0
            or get_unpaired_electron_count(atom3) != 0
            or get_unpaired_electron_count(atom4) != 0
            or has_unresolved_two_electron_center(atom2)
            or has_unresolved_two_electron_center(atom3)
            or not has_unresolved_two_electron_center(atom4)
            or bond1.GetBondOrder() != 1
            or bond2.GetBondOrder() != 2
            or bond3.GetBondOrder() != 1
        ):
            continue

        bond3.SetBondOrder(bond3.GetBondOrder() + 1)
        atom3.SetFormalCharge(1)
        atom4.SetFormalCharge(-1)
        set_unresolved_two_electron_center(atom4, False)
        set_lone_pair_count(atom4, 0)
        set_unpaired_electron_count(atom4, 0)
        hit = True
    return omol, hit


def clean_short_diazo_resonance(omol: pybel.Molecule) -> tuple[pybel.Molecule, bool]:
    """C#N+-N(2-) -> C(2-)-N+#N when the terminal N-N is geometrically short.

    Transfer existing charge and pi bonds; do not seed a new diazo charge state
    or overwrite a valid neutral-carbene/diazonium contributor.
    """
    mol = cast(ob.OBMol, omol.OBMol)
    hit = False
    for terminal in mol_atoms(mol):
        if (
            terminal.GetAtomicNum() != 7
            or terminal.GetExplicitDegree() != 1
            or terminal.GetFormalCharge() != -2
        ):
            continue
        middle = next(atom_neighbors(terminal))
        if (
            middle.GetAtomicNum() != 7
            or middle.GetFormalCharge() != 1
            or middle.GetExplicitDegree() != 2
        ):
            continue
        carbon = next(a for a in atom_neighbors(middle) if a.GetIdx() != terminal.GetIdx())
        nn = cast(ob.OBBond, mol.GetBond(terminal, middle))
        cn = cast(ob.OBBond, mol.GetBond(middle, carbon))
        if (
            carbon.GetAtomicNum() == 6
            and carbon.GetFormalCharge() == 0
            and carbon.GetTotalValence() == 4
            and carbon.GetExplicitDegree() == 2
            and nn.GetBondOrder() == 1
            and cn.GetBondOrder() == 3
            and 1.05 < terminal.GetDistance(middle) < 1.25
            and 1.20 < middle.GetDistance(carbon) < 1.45
            and all(
                not get_unpaired_electron_count(a) and not has_unresolved_two_electron_center(a)
                for a in (terminal, middle, carbon)
            )
        ):
            nn.SetBondOrder(3)
            cn.SetBondOrder(1)
            terminal.SetFormalCharge(0)
            carbon.SetFormalCharge(-2)
            assign_charge_radical_for_atom(terminal)
            assign_charge_radical_for_atom(carbon)
            hit = True
    return omol, hit


def clean_stretched_dithiolate_resonance(omol: pybel.Molecule) -> tuple[pybel.Molecule, bool]:
    """Prefer two S- termini when both long C-S and short exocyclic C-C agree.

    Move an existing carbanion through a bounded alternating carbon path. This
    is a charge-conserving resonance change, not ligand reduction or cleavage.
    """
    mol = cast(ob.OBMol, omol.OBMol)
    hit = False
    for sulfur in mol_atoms(mol):
        if (
            sulfur.GetAtomicNum() != 16
            or sulfur.GetExplicitDegree() != 1
            or sulfur.GetFormalCharge() != 0
        ):
            continue
        center = next(atom_neighbors(sulfur))
        bond = cast(ob.OBBond, mol.GetBond(center, sulfur))
        neighbors = list(atom_neighbors(center))
        if (
            center.GetAtomicNum() != 6
            or center.GetExplicitDegree() != 3
            or bond.GetBondOrder() != 2
        ):
            continue
        others = [a for a in neighbors if a.GetIdx() != sulfur.GetIdx()]
        if sorted(a.GetAtomicNum() for a in others) != [6, 16]:
            continue
        other_s = next(a for a in others if a.GetAtomicNum() == 16)
        carbon = next(a for a in others if a.GetAtomicNum() == 6)
        if not (
            other_s.GetFormalCharge() == -1
            and other_s.GetExplicitDegree() == 1
            and 1.73 < sulfur.GetDistance(center) < 1.90
            and 1.73 < other_s.GetDistance(center) < 1.90
            and 1.35 < carbon.GetDistance(center) < 1.43
        ):
            continue
        paths = [[sulfur.GetIdx(), center.GetIdx()]]
        selected = None
        while paths and selected is None:
            path = paths.pop(0)
            end = cast(ob.OBAtom, mol.GetAtom(path[-1]))
            if (
                len(path) > 2
                and len(path) % 2 == 1
                and end.GetFormalCharge() == -1
                and end.GetTotalValence() == 3
            ):
                selected = path
                break
            if len(path) >= 7 or end.GetFormalCharge() != 0:
                continue
            for neighbor in atom_neighbors(end):
                if (
                    neighbor.GetIdx() not in path
                    and neighbor.GetAtomicNum() == 6
                    and neighbor_bond_order(end, neighbor) == (1 if len(path) % 2 == 0 else 2)
                    and not get_unpaired_electron_count(neighbor)
                    and not has_unresolved_two_electron_center(neighbor)
                ):
                    paths.append(path + [neighbor.GetIdx()])
        if selected is None:
            continue
        for i, (a, b) in enumerate(zip(selected, selected[1:])):
            mol.GetBond(a, b).SetBondOrder(1 + i % 2)
        start = cast(ob.OBAtom, mol.GetAtom(selected[-1]))
        start.SetFormalCharge(0)
        sulfur.SetFormalCharge(-1)
        assign_charge_radical_for_atom(start)
        assign_charge_radical_for_atom(sulfur)
        mol.SetAromaticPerceived(False)
        hit = True
    return omol, hit


def clean_stretched_silyloxonium_resonance(omol: pybel.Molecule) -> tuple[pybel.Molecule, bool]:
    """Move C- through C-O+=Si when short C-O and long Si-O support C=O+-Si-.

    This closed-shell resonance move preserves charge and all sigma bonds.
    It uses only the ligand geometry; it neither reads nor assigns metal states.
    Short Si=O, long C-O, hypercoordination and active electrons abstain.
    """
    mol = cast(ob.OBMol, omol.OBMol)
    hit = False
    for oxygen in mol_atoms(mol):
        if (
            oxygen.GetAtomicNum() != 8
            or oxygen.GetFormalCharge() != 1
            or oxygen.GetExplicitDegree() != 2
        ):
            continue
        neighbors = list(atom_neighbors(oxygen))
        carbon = next((a for a in neighbors if a.GetAtomicNum() == 6), None)
        silicon = next((a for a in neighbors if a.GetAtomicNum() == 14), None)
        if carbon is None or silicon is None:
            continue
        if (
            carbon.GetFormalCharge() != -1
            or silicon.GetFormalCharge() != 0
            or carbon.GetExplicitDegree() != 3
            or silicon.GetExplicitDegree() != 3
            or carbon.GetTotalValence() != 3
            or silicon.GetTotalValence() != 4
            or any(
                a.GetImplicitHCount()
                or get_unpaired_electron_count(a)
                or get_lone_pair_count(a)
                or has_unresolved_two_electron_center(a)
                for a in (carbon, oxygen, silicon)
            )
        ):
            continue
        co = cast(ob.OBBond, mol.GetBond(carbon, oxygen))
        so = cast(ob.OBBond, mol.GetBond(silicon, oxygen))
        if (
            co.GetBondOrder() != 1
            or so.GetBondOrder() != 2
            or not 1.20 < carbon.GetDistance(oxygen) < 1.35
            or not 1.75 < silicon.GetDistance(oxygen) < 2.15
        ):
            continue
        co.SetBondOrder(2)
        so.SetBondOrder(1)
        carbon.SetFormalCharge(0)
        silicon.SetFormalCharge(-1)
        assign_charge_radical_for_atom(carbon)
        assign_charge_radical_for_atom(silicon)
        mol.SetAromaticPerceived(False)
        hit = True
    return omol, hit


def clean_phosphorus_cumulene(omol: pybel.Molecule) -> tuple[pybel.Molecule, bool]:
    """Resolve C=P+=C-N into C=P-C=N+ when short C-N supports N donation."""
    mol = cast(ob.OBMol, omol.OBMol)
    hit = False
    for phosphorus in mol_atoms(mol):
        if (
            phosphorus.GetAtomicNum() != 15
            or phosphorus.GetFormalCharge() != 1
            or phosphorus.GetExplicitDegree() != 2
            or phosphorus.GetTotalValence() != 4
        ):
            continue
        for carbon in atom_neighbors(phosphorus):
            pc = cast(ob.OBBond, mol.GetBond(phosphorus, carbon))
            if (
                carbon.GetAtomicNum() != 6
                or carbon.GetFormalCharge() != 0
                or carbon.GetTotalValence() != 4
                or pc.GetBondOrder() != 2
                or phosphorus.GetDistance(carbon) < 1.70
            ):
                continue
            for nitrogen in atom_neighbors(carbon):
                cn = cast(ob.OBBond, mol.GetBond(carbon, nitrogen))
                if (
                    nitrogen.GetAtomicNum() == 7
                    and nitrogen.GetFormalCharge() == 0
                    and nitrogen.GetExplicitDegree() == 3
                    and nitrogen.GetTotalValence() == 3
                    and cn.GetBondOrder() == 1
                    and 1.25 < carbon.GetDistance(nitrogen) < 1.43
                    and all(
                        not get_unpaired_electron_count(a)
                        and not has_unresolved_two_electron_center(a)
                        for a in (phosphorus, carbon, nitrogen)
                    )
                ):
                    pc.SetBondOrder(1)
                    cn.SetBondOrder(2)
                    phosphorus.SetFormalCharge(0)
                    nitrogen.SetFormalCharge(1)
                    mol.SetAromaticPerceived(False)
                    hit = True
                    break
            if phosphorus.GetFormalCharge() == 0:
                break
    return omol, hit


def clean_pyrrolide_resonance(omol: pybel.Molecule) -> tuple[pybel.Molecule, bool]:
    """Localize a five-membered pyrrolide anion on its unsubstituted N.

    Transfer an existing charge through alternating bonds only. No charge,
    spin or connectivity is created; two-N carbene rings are outside this rule.
    """
    mol = cast(ob.OBMol, omol.OBMol)
    hit = False
    for ids in ring_paths(mol):
        if len(ids) != 5:
            continue
        atoms = [cast(ob.OBAtom, mol.GetAtom(i)) for i in ids]
        if sorted(a.GetAtomicNum() for a in atoms) != [6, 6, 6, 6, 7]:
            continue
        if sorted(a.GetFormalCharge() for a in atoms) != [-1, 0, 0, 0, 0]:
            continue
        nitrogen: ob.OBAtom = next(a for a in atoms if a.GetAtomicNum() == 7)
        start: ob.OBAtom = next(a for a in atoms if a.GetFormalCharge() == -1)
        if (
            start.GetAtomicNum() != 6
            or start.GetTotalValence() != 3
            or nitrogen.GetExplicitDegree() != 2
            or nitrogen.GetTotalValence() != 3
        ):
            continue
        if any(
            get_unpaired_electron_count(a) or has_unresolved_two_electron_center(a) for a in atoms
        ):
            continue
        offset = ids.index(start.GetIdx())
        for direction in (1, -1):
            path = [ids[offset]]
            for step in range(1, 5):
                path.append(ids[(offset + step * direction) % 5])
                if path[-1] == nitrogen.GetIdx():
                    break
            bonds = [cast(ob.OBBond, mol.GetBond(a, b)) for a, b in zip(path, path[1:])]
            if len(bonds) % 2 or any(b.GetBondOrder() != 1 + (i % 2) for i, b in enumerate(bonds)):
                continue
            for i, bond in enumerate(bonds):
                bond.SetBondOrder(2 - (i % 2))
            start.SetFormalCharge(0)
            nitrogen.SetFormalCharge(-1)
            assign_charge_radical_for_atom(start)
            assign_charge_radical_for_atom(nitrogen)
            mol.SetAromaticPerceived(False)
            hit = True
            break
    return omol, hit


def clean_pyramidal_allyl_resonance(omol: pybel.Molecule) -> tuple[pybel.Molecule, bool]:
    """Place an existing allylic anion on the pyramidal CH2 end of a long C–C bond."""
    mol = cast(ob.OBMol, omol.OBMol)
    hit = False
    for first in mol_atoms(mol):
        if (
            first.GetAtomicNum() != 6
            or first.GetFormalCharge()
            or first.IsInRing()
            or first.GetExplicitDegree() != 3
            or first.GetTotalValence() != 4
        ):
            continue
        neighbors = list(atom_neighbors(first))
        if sorted(a.GetAtomicNum() for a in neighbors) != [1, 1, 6]:
            continue
        middle = next(a for a in neighbors if a.GetAtomicNum() == 6)
        long_cc = first.GetDistance(middle)
        if (
            middle.IsInRing()
            or middle.GetFormalCharge()
            or middle.GetExplicitDegree() != 3
            or middle.GetTotalValence() != 4
            or neighbor_bond_order(first, middle) != 2
            or not 1.44 < long_cc < 1.60
        ):
            continue
        angles = [mol.GetAngle(a, first, b) for a, b in combinations(neighbors, 2)]
        if not (all(90 < angle < 130 for angle in angles) and 300 < sum(angles) < 345):
            continue
        for last in atom_neighbors(middle):
            if (
                last.GetAtomicNum() != 6
                or last.GetFormalCharge() != -1
                or last.IsInRing()
                or last.GetExplicitDegree() != 3
                or last.GetTotalValence() != 3
                or neighbor_bond_order(middle, last) != 1
            ):
                continue
            short_cc = middle.GetDistance(last)
            if not (1.28 < short_cc < 1.40 and long_cc - short_cc > 0.08):
                continue
            if any(
                a.GetImplicitHCount()
                or get_unpaired_electron_count(a)
                or get_lone_pair_count(a)
                or has_unresolved_two_electron_center(a)
                or any(n.GetAtomicNum() not in (1, 6) for n in atom_neighbors(a))
                for a in (first, middle, last)
            ):
                continue
            first.GetBond(middle).SetBondOrder(1)
            middle.GetBond(last).SetBondOrder(2)
            first.SetFormalCharge(-1)
            last.SetFormalCharge(0)
            assign_charge_radical_for_atom(first)
            assign_charge_radical_for_atom(last)
            mol.SetAromaticPerceived(False)
            hit = True
            break
    return omol, hit


def clean_resonances(omol: pybel.Molecule) -> tuple[pybel.Molecule, bool]:
    """Run the ordered resonance normalization rule set after candidate generation."""

    hit = False
    omol, stage_hit = clean_resonances_11(omol)
    hit = stage_hit or hit
    omol, stage_hit = clean_resonances_0(omol)
    hit = stage_hit or hit
    omol, stage_hit = clean_resonances_1(omol)
    hit = stage_hit or hit
    omol, stage_hit = clean_resonances_2(omol)
    hit = stage_hit or hit
    omol, stage_hit = clean_resonances_3(omol)
    hit = stage_hit or hit
    omol, stage_hit = clean_resonances_4(omol)
    hit = stage_hit or hit
    omol, stage_hit = clean_resonances_9(omol)
    hit = stage_hit or hit
    omol, stage_hit = clean_resonances_5(omol)
    hit = stage_hit or hit
    omol, stage_hit = clean_resonances_6(omol)
    hit = stage_hit or hit
    omol, stage_hit = clean_resonances_7(omol)
    hit = stage_hit or hit
    omol, stage_hit = clean_resonances_8(omol)
    hit = stage_hit or hit
    omol, stage_hit = clean_resonances_9(omol)
    hit = stage_hit or hit
    omol, stage_hit = clean_resonances_10(omol)
    hit = stage_hit or hit
    omol, stage_hit = clean_resonances_12(omol)
    hit = stage_hit or hit
    omol, stage_hit = clean_resonances_13(omol)
    hit = stage_hit or hit
    omol, stage_hit = clean_resonances_14(omol)
    hit = stage_hit or hit
    omol, stage_hit = clean_resonances_16(omol)
    hit = stage_hit or hit
    omol, stage_hit = clean_resonances_17(omol)
    hit = stage_hit or hit
    omol, stage_hit = clean_resonances_18(omol)
    hit = stage_hit or hit
    omol, stage_hit = clean_short_diazo_resonance(omol)
    hit = stage_hit or hit
    omol, stage_hit = clean_stretched_dithiolate_resonance(omol)
    hit = stage_hit or hit
    omol, stage_hit = clean_stretched_silyloxonium_resonance(omol)
    hit = stage_hit or hit
    omol, stage_hit = clean_phosphorus_cumulene(omol)
    hit = stage_hit or hit
    omol, stage_hit = clean_pyrrolide_resonance(omol)
    hit = stage_hit or hit
    omol, stage_hit = clean_pyramidal_allyl_resonance(omol)
    hit = stage_hit or hit
    return omol, hit


__all__ = [
    "clean_carbene_neighbor_unsaturated",
    "clean_1_4_radicals",
    "clean_1_6_radicals",
    "clean_neighbor_radicals",
    "clean_possible_1_3_dipole",
    "clean_resonances",
    "clean_resonances_18",
]
