"""Bond-breaking heuristics used when the deterministic cleanup path still stalls."""

from __future__ import annotations

from typing import List, Tuple, cast

from openbabel import openbabel as ob
from openbabel import pybel

from molgr.fallback.utils import smarts
from molgr.fallback.utils.electrons import (
    get_unpaired_electron_count,
    has_unresolved_two_electron_center,
    set_unpaired_electron_count,
)
from molgr.fallback.utils.ob_graph import mol_atoms, mol_bonds


def _bond_breaking_electron_budget(obmol: ob.OBMol) -> int:
    """Count electrons already represented before a recovery bond cleavage."""

    return sum(
        get_unpaired_electron_count(atom) + (2 if has_unresolved_two_electron_center(atom) else 0)
        for atom in mol_atoms(obmol)
    )


def break_deformed_ene(
    omol: pybel.Molecule,
    given_charge: int = 0,
    given_radical: int = 0,
    tolerance: float = 10,
) -> tuple[pybel.Molecule, bool]:
    """Homolytically reduce deformed multiple bonds by one bond-order unit.

    Each reduction creates one real unpaired electron on each endpoint. Deferred
    two-electron centers count toward the stopping budget so their electrons are
    not created again by another bond cleavage; their markers remain unresolved.
    """

    possible_ene_pairs: List[Tuple[int, int, float]] = []
    obmol = cast(ob.OBMol, omol.OBMol)
    matches: List[Tuple[int, int, int, int]] = list(smarts.BREAK_DEFORMED_ENE_A.findall(omol))
    while len(matches):
        idxs = matches.pop(0)
        bond2_1 = cast(ob.OBBond, obmol.GetBond(idxs[1], idxs[2]))
        if bond2_1.IsRotor() or bond2_1.GetBondOrder() == 1:
            continue
        torsion_angle: float = abs(obmol.GetTorsion(*idxs))
        torsion_angle = min(torsion_angle, 180 - torsion_angle)
        if torsion_angle > tolerance:
            possible_ene_pairs.append((idxs[1], idxs[2], torsion_angle))

    matches = list(smarts.BREAK_DEFORMED_ENE_B.findall(omol))
    while len(matches):
        idxs = matches.pop(0)
        bond2_2 = cast(ob.OBBond, obmol.GetBond(idxs[1], idxs[2]))
        if bond2_2.IsRotor() or bond2_2.GetBondOrder() == 1:
            continue
        torsion_angle = abs(obmol.GetTorsion(*idxs))
        torsion_angle = min(torsion_angle, 180 - torsion_angle)
        if torsion_angle > tolerance:
            possible_ene_pairs.append((idxs[1], idxs[2], torsion_angle))

    possible_ene_pairs.sort(key=lambda x: x[2], reverse=True)
    hit = False
    for idx1, idx2, _ in possible_ene_pairs:
        if _bond_breaking_electron_budget(obmol) >= abs(given_charge) + given_radical:
            return omol, hit
        bond = cast(ob.OBBond, obmol.GetBond(idx1, idx2))
        if bond.IsRotor() or bond.GetBondOrder() == 1:
            continue
        bond.SetBondOrder(bond.GetBondOrder() - 1)
        begin_atom = cast(ob.OBAtom, bond.GetBeginAtom())
        end_atom = cast(ob.OBAtom, bond.GetEndAtom())
        set_unpaired_electron_count(begin_atom, get_unpaired_electron_count(begin_atom) + 1)
        set_unpaired_electron_count(end_atom, get_unpaired_electron_count(end_atom) + 1)
        hit = True
    return omol, hit


def break_one_bond(
    omol: pybel.Molecule,
    given_charge: int = 0,
    given_radical: int = 0,
) -> tuple[pybel.Molecule, int, bool]:
    """Apply last-resort homolytic and charge-transfer bond-breaking templates.

    Multiple/aromatic bond reduction and single-bond deletion create one real
    unpaired electron at each endpoint. The cation template instead creates one
    radical only at the neutral endpoint while lowering the positive endpoint's
    formal charge, an explicit heterolytic/one-electron heuristic. Deferred
    two-electron centers count toward the stopping budget without being classified
    or consumed; active lone pairs do not count toward that budget.

    Recover overbonded N/P cations before reducing neutral multiple bonds.
    This can satisfy the electron deficit without breaking an intact carbonyl.
    For an all-single-bond XYZ graph, require geometric evidence of stretching
    before deleting a sigma bond. An electronic target alone must not split a
    compact C-H bond into an isolated hydride and a carbon anion.
    """

    obmol = cast(ob.OBMol, omol.OBMol)
    hit = False
    while res := smarts.BREAK_ONE_BOND_CATION.findall(omol):
        if _bond_breaking_electron_budget(obmol) >= abs(given_charge) + given_radical:
            return omol, given_charge, hit
        idxs = res.pop(0)
        bond2 = cast(ob.OBBond, obmol.GetBond(idxs[0], idxs[1]))
        bond2.SetBondOrder(bond2.GetBondOrder() - 1)
        # SMARTS identifies the charged endpoint; bond storage orientation is
        # unrelated to the direction of this charge-transfer rule.
        begin_atom2 = cast(ob.OBAtom, obmol.GetAtom(idxs[0]))
        end_atom2 = cast(ob.OBAtom, obmol.GetAtom(idxs[1]))
        set_unpaired_electron_count(end_atom2, get_unpaired_electron_count(end_atom2) + 1)
        begin_atom2.SetFormalCharge(int(begin_atom2.GetFormalCharge() - 1))
        given_charge += 1
        hit = True

    while res := smarts.BREAK_ONE_BOND_MULTIPLE.findall(omol):
        if _bond_breaking_electron_budget(obmol) >= abs(given_charge) + given_radical:
            return omol, given_charge, hit
        idxs = res.pop(0)
        bond = cast(ob.OBBond, obmol.GetBond(idxs[0], idxs[1]))
        bond.SetBondOrder(bond.GetBondOrder() - 1)
        begin_atom = cast(ob.OBAtom, bond.GetBeginAtom())
        end_atom = cast(ob.OBAtom, bond.GetEndAtom())
        set_unpaired_electron_count(begin_atom, get_unpaired_electron_count(begin_atom) + 1)
        set_unpaired_electron_count(end_atom, get_unpaired_electron_count(end_atom) + 1)
        hit = True

    res = list(smarts.BREAK_ONE_BOND_AROMATIC.findall(omol))
    if res:
        if _bond_breaking_electron_budget(obmol) >= abs(given_charge) + given_radical:
            return omol, given_charge, hit
        for idxs in res:
            bond3 = cast(ob.OBBond, obmol.GetBond(idxs[0], idxs[1]))
            if bond3.GetBondOrder() == 1:
                continue
            bond3.SetBondOrder(bond3.GetBondOrder() - 1)
            begin_atom3 = cast(ob.OBAtom, bond3.GetBeginAtom())
            end_atom3 = cast(ob.OBAtom, bond3.GetEndAtom())
            set_unpaired_electron_count(begin_atom3, get_unpaired_electron_count(begin_atom3) + 1)
            set_unpaired_electron_count(end_atom3, get_unpaired_electron_count(end_atom3) + 1)
            hit = True
            break

    if all(bond.GetBondOrder() == 1 for bond in mol_bonds(obmol)):
        single_bonds = list(mol_bonds(obmol))
        if obmol.GetDimension() == 3:
            # Compare excess length relative to both atomic radii, so a normal
            # bond between large atoms is not preferred over a stretched C-H.
            def excess_length(bond: ob.OBBond) -> float:
                return (
                    bond.GetLength()
                    - ob.GetCovalentRad(bond.GetBeginAtom().GetAtomicNum())
                    - ob.GetCovalentRad(bond.GetEndAtom().GetAtomicNum())
                )

            single_bonds = sorted(
                (bond for bond in single_bonds if excess_length(bond) > 0.20),
                key=excess_length,
                reverse=True,
            )
        if single_bonds:
            if _bond_breaking_electron_budget(obmol) >= abs(given_charge) + given_radical:
                return omol, given_charge, hit
            single_bond = single_bonds[0]
            single_begin_atom = cast(ob.OBAtom, single_bond.GetBeginAtom())
            single_end_atom = cast(ob.OBAtom, single_bond.GetEndAtom())
            set_unpaired_electron_count(
                single_begin_atom, get_unpaired_electron_count(single_begin_atom) + 1
            )
            set_unpaired_electron_count(
                single_end_atom, get_unpaired_electron_count(single_end_atom) + 1
            )
            obmol.DeleteBond(single_bond)
            hit = True
    return omol, given_charge, hit


__all__ = ["break_deformed_ene", "break_one_bond"]
