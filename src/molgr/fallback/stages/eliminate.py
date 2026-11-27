"""Charge-elimination heuristics shared by the linear and resonance cleanup paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, cast

from openbabel import openbabel as ob
from openbabel import pybel

from molgr.fallback.utils import consts, smarts
from molgr.fallback.utils.bond_geometry import (
    nitrogen_nitrogen_order_allowed,
    nitrogen_oxygen_order_allowed,
    phosphorus_carbon_order_allowed,
    restores_azinium_ring,
)
from molgr.fallback.utils.electrons import (
    get_lone_pair_count,
    get_unpaired_electron_count,
    has_unresolved_two_electron_center,
    set_lone_pair_count,
    set_unpaired_electron_count,
    set_unresolved_two_electron_center,
)
from molgr.fallback.utils.ob_graph import atom_neighbors, indexed_atoms


@dataclass(frozen=True)
class _ChargeAssignmentAction:
    """A candidate charge edit and the unpaired electrons it consumes or creates."""

    atom_idx: int
    formal_charge: int
    spin_consumed: int
    consume_unresolved_center: bool
    charge_delta: int
    score_key: Tuple[int, ...]
    bond_idx: int | None = None
    charge_atom_idx: int | None = None
    spin_created: int = 0


# The single-atom anion patterns rank a neutral radical by element and total
# valence (SMARTS v). Carbon and hydrogen tiers require a negative charge budget.
_ANION_RADICAL_TIERS = {
    (9, 0): 10,
    (8, 0): 20,
    (8, 1): 21,
    (17, 0): 30,
    (7, 0): 40,
    (7, 1): 41,
    (7, 2): 42,
    (35, 0): 50,
    (53, 0): 60,
    (16, 0): 70,
    (16, 1): 71,
    (34, 0): 80,
    (34, 1): 81,
    (15, 0): 90,
    (15, 1): 91,
    (15, 2): 92,
    (5, 0): 95,
    (5, 1): 96,
    (5, 2): 97,
    (6, 3): 100,
    (6, 0): 110,
    (6, 1): 110,
    (6, 2): 110,
    (1, 0): 120,
}
_BORON_REDUCTION_TIER = 98
_GENERIC_ANION_TIER = 1000


@dataclass(frozen=True)
class _AnionSite:
    """Neutral atom that can accept charge before the next bond-order change."""

    atom_idx: int
    atomic_num: int
    tier: int
    unpaired: int

    def action(
        self, charge: int, real_unpaired: int, radical_target: int | None
    ) -> _ChargeAssignmentAction | None:
        """Evaluate this site's charge and spin costs against the current budgets."""
        if charge == 0 and self.tier >= _BORON_REDUCTION_TIER:
            return None
        creates_spin = self.tier == _BORON_REDUCTION_TIER
        consumes_center = self.unpaired == 0 and not creates_spin
        consumed_spin = 0
        if creates_spin:
            if radical_target is None or real_unpaired >= radical_target:
                return None
            amount = 1
        elif consumes_center:
            if charge > -2:
                return None
            amount = 2
        else:
            consumed_spin = min(self.unpaired, max(1, -charge))
            amount = consumed_spin
        after = charge + amount
        return _ChargeAssignmentAction(
            atom_idx=self.atom_idx,
            formal_charge=-amount,
            spin_consumed=consumed_spin,
            spin_created=int(creates_spin),
            consume_unresolved_center=consumes_center,
            charge_delta=amount,
            score_key=(self.tier, abs(after), max(after, 0), self.atomic_num, self.atom_idx),
        )


def _is_closed_shell_trivalent_boron(atom: ob.OBAtom) -> bool:
    return (
        atom.GetAtomicNum() == 5
        and atom.GetTotalValence() == 3
        and atom.GetFormalCharge() == 0
        and get_unpaired_electron_count(atom) == 0
        and get_lone_pair_count(atom) == 0
        and not has_unresolved_two_electron_center(atom)
    )


def _apply_charge_assignment_action(
    obmol: ob.OBMol,
    action: _ChargeAssignmentAction,
) -> bool:
    """Encode one explicitly identified electron source as formal charge.

    Radical actions consume the requested number of real unpaired electrons. An
    unresolved action instead consumes one pure deferred two-electron marker
    atomically and assigns ``+2`` or ``-2`` without first creating two radicals.
    A trivalent-boron reduction instead creates one real unpaired electron.
    Active lone pairs are never interchangeable inputs here.
    """

    atom = cast(ob.OBAtom, obmol.GetAtom(action.atom_idx))
    if atom is None:
        return False
    if action.bond_idx is not None and action.charge_atom_idx is not None:
        charge_atom = cast(ob.OBAtom, obmol.GetAtom(action.charge_atom_idx))
        bond = cast(ob.OBBond, obmol.GetBond(action.atom_idx, action.charge_atom_idx))
        if (
            charge_atom is None
            or bond is None
            or get_unpaired_electron_count(atom) < action.spin_consumed
            or has_unresolved_two_electron_center(atom)
        ):
            return False
        set_unpaired_electron_count(atom, get_unpaired_electron_count(atom) - action.spin_consumed)
        bond.SetBondOrder(bond.GetBondOrder() + 1)
        charge_atom.SetFormalCharge(charge_atom.GetFormalCharge() + 1)
        return True
    if atom.GetFormalCharge() != 0:
        return False
    if action.spin_created:
        if (
            action.spin_created != 1
            or action.spin_consumed != 0
            or action.consume_unresolved_center
            or action.formal_charge != -1
            or not _is_closed_shell_trivalent_boron(atom)
        ):
            return False
        set_unpaired_electron_count(atom, 1)
    elif action.consume_unresolved_center:
        if (
            action.spin_consumed != 0
            or abs(action.formal_charge) != 2
            or not has_unresolved_two_electron_center(atom)
            or get_unpaired_electron_count(atom) != 0
            or get_lone_pair_count(atom) != 0
        ):
            return False
        set_unresolved_two_electron_center(atom, False)
    else:
        if action.spin_consumed <= 0 or has_unresolved_two_electron_center(atom):
            return False
        if get_unpaired_electron_count(atom) < action.spin_consumed:
            return False
        set_unpaired_electron_count(atom, get_unpaired_electron_count(atom) - action.spin_consumed)
    atom.SetFormalCharge(action.formal_charge)
    return True


def _atom_idx(atom: ob.OBAtom) -> int:
    return int(atom.GetIdx())


def _positive_charge_assignment_actions(
    omol: pybel.Molecule,
    obmol: ob.OBMol,
    given_charge: int,
    total_radical_electrons: int = 0,
) -> list[_ChargeAssignmentAction]:
    """Rank cation actions from radicals or a pure unresolved 2e center.

    Radical actions preserve their existing motif priority. When the remaining
    deficit is at least two, the generic tier also admits a pure unresolved center
    as an atomic ``+2`` action. Active lone pairs remain excluded.
    """

    actions: list[_ChargeAssignmentAction] = []
    seen: set[tuple[int, int, int]] = set()

    def append_action(atom: ob.OBAtom, *, tier: int, match_order: int, amount: int) -> None:
        if atom is None or amount <= 0 or has_unresolved_two_electron_center(atom):
            return
        atom_idx = _atom_idx(atom)
        key = (atom_idx, tier, amount)
        if key in seen:
            return
        seen.add(key)
        charge_after = given_charge - amount
        atomic_num = int(atom.GetAtomicNum())
        actions.append(
            _ChargeAssignmentAction(
                atom_idx=atom_idx,
                formal_charge=amount,
                spin_consumed=amount,
                consume_unresolved_center=False,
                charge_delta=-amount,
                score_key=(
                    tier,
                    abs(charge_after),
                    max(charge_after, 0),
                    atomic_num,
                    atom_idx,
                    match_order,
                ),
            )
        )

    def append_unresolved_action(atom: ob.OBAtom, *, match_order: int) -> None:
        if (
            atom is None
            or given_charge < 2
            or atom.GetFormalCharge() != 0
            or not has_unresolved_two_electron_center(atom)
            or get_unpaired_electron_count(atom) != 0
            or get_lone_pair_count(atom) != 0
        ):
            return
        atom_idx = _atom_idx(atom)
        charge_after = given_charge - 2
        actions.append(
            _ChargeAssignmentAction(
                atom_idx=atom_idx,
                formal_charge=2,
                spin_consumed=0,
                consume_unresolved_center=True,
                charge_delta=-2,
                score_key=(
                    100,
                    abs(charge_after),
                    max(charge_after, 0),
                    int(atom.GetAtomicNum()),
                    atom_idx,
                    match_order,
                ),
            )
        )

    for match_order, n_idxs in enumerate(
        cast(List[Tuple[int, int]], smarts.ELIM_POSITIVE_N.findall(omol))
    ):
        atom = cast(ob.OBAtom, obmol.GetAtom(n_idxs[1]))
        if (
            given_charge > 0
            and atom.GetFormalCharge() == 0
            and get_unpaired_electron_count(atom) >= 1
        ):
            append_action(atom, tier=0, match_order=match_order, amount=1)

    for match_order, c_h_idxs in enumerate(
        cast(List[Tuple[int, int, int]], smarts.ELIM_POSITIVE_C_H.findall(omol))
    ):
        atom = cast(ob.OBAtom, obmol.GetAtom(c_h_idxs[0]))
        if (
            given_charge > 0
            and atom.GetFormalCharge() == 0
            and get_unpaired_electron_count(atom) >= 1
        ):
            append_action(atom, tier=10, match_order=match_order, amount=1)

    dipole_budget_checked = False
    for match_order, dipole_idxs in enumerate(
        cast(List[Tuple[int, int]], smarts.ELIM_POSITIVE_DIPOLE.findall(omol))
    ):
        radical_atom = cast(ob.OBAtom, obmol.GetAtom(dipole_idxs[0]))
        charge_atom = cast(ob.OBAtom, obmol.GetAtom(dipole_idxs[1]))
        bond = cast(ob.OBBond, obmol.GetBond(dipole_idxs[0], dipole_idxs[1]))
        if (
            radical_atom.GetFormalCharge() != 0
            or get_unpaired_electron_count(radical_atom) != 1
            or has_unresolved_two_electron_center(radical_atom)
            or bond is None
        ):
            continue
        if not dipole_budget_checked:
            # The budget is shared by all dipoles in this unmodified candidate.
            electrons = sum(
                get_unpaired_electron_count(atom) + 2 * has_unresolved_two_electron_center(atom)
                for atom in indexed_atoms(obmol)
            )
            if abs(given_charge - 1) > electrons - 1:
                break
            dipole_budget_checked = True
        actions.append(
            _ChargeAssignmentAction(
                atom_idx=_atom_idx(radical_atom),
                formal_charge=0,
                spin_consumed=1,
                consume_unresolved_center=False,
                charge_delta=-1,
                score_key=(
                    4 if restores_azinium_ring(charge_atom, radical_atom) else 5,
                    abs(given_charge - 1),
                    max(given_charge - 1, 0),
                    _atom_idx(radical_atom),
                    match_order,
                ),
                bond_idx=bond.GetIdx(),
                charge_atom_idx=_atom_idx(charge_atom),
            )
        )

    if given_charge > 0:
        for match_order, atom in enumerate(indexed_atoms(obmol)):
            if atom.GetFormalCharge() == 0 and get_unpaired_electron_count(atom) >= 1:
                append_action(
                    atom,
                    tier=100,
                    match_order=match_order,
                    amount=min(get_unpaired_electron_count(atom), given_charge),
                )
            append_unresolved_action(atom, match_order=match_order)

    return sorted(actions, key=lambda action: action.score_key)


def _negative_charge_sites(
    obmol: ob.OBMol, given_charge: int, radical_target: int | None
) -> tuple[dict[int, _AnionSite], int]:
    """Index neutral acceptors and count all real radicals, including those on charged atoms.

    Anion assignment changes only the selected neutral atom's charge and
    electronic labels. Its neighbours retain their valences and priorities.
    The index is private to one elimination pass and is discarded before any
    later resonance stage can change a bond.
    """
    sites: dict[int, _AnionSite] = {}
    real_unpaired = 0
    for atom in indexed_atoms(obmol):
        spin = get_unpaired_electron_count(atom)
        real_unpaired += spin
        if (spin == 0 and given_charge == 0) or atom.GetFormalCharge() != 0:
            continue
        element = int(atom.GetAtomicNum())
        if spin > 0:
            if has_unresolved_two_electron_center(atom):
                continue
            tier = _ANION_RADICAL_TIERS.get((element, atom.GetTotalValence()), _GENERIC_ANION_TIER)
        elif (
            given_charge <= -2
            and spin == 0
            and has_unresolved_two_electron_center(atom)
            and not get_lone_pair_count(atom)
        ):
            tier = _GENERIC_ANION_TIER
        elif element == 5 and radical_target is not None and _is_closed_shell_trivalent_boron(atom):
            tier = _BORON_REDUCTION_TIER
        else:
            continue
        index = _atom_idx(atom)
        sites[index] = _AnionSite(index, element, tier, spin)
    return sites, real_unpaired


def eliminate_high_positive_charge_atoms(
    omol: pybel.Molecule, given_charge: int, metal_distances: tuple[int, ...] = ()
) -> tuple[pybel.Molecule, int, bool]:
    """Neutralize positive centers from their most electronegative neighbors.

    Eligible neighboring monoradicals are considered in descending
    electronegativity order. Each accepted neighbor consumes one real unpaired
    electron and becomes ``-1``. Assignment stops as soon as the existing and
    newly assigned neighboring negative charges balance the center's positive
    formal charge. Lone pairs and unresolved centers are not consumed.
    """

    obmol = cast(ob.OBMol, omol.OBMol)
    hit = False
    neighbors_by_center: dict[int, set[int]] = {}
    for center_idx, neighbor_idx in cast(
        List[Tuple[int, int]], smarts.ELIM_HIGH_POSITIVE.findall(omol)
    ):
        neighbors_by_center.setdefault(center_idx, set()).add(neighbor_idx)

    for center_idx in sorted(neighbors_by_center):
        center = cast(ob.OBAtom, obmol.GetAtom(center_idx))
        remaining_positive_charge = center.GetFormalCharge() + sum(
            min(0, neighbor.GetFormalCharge()) for neighbor in atom_neighbors(center)
        )
        if remaining_positive_charge <= 0:
            continue

        neighbor_indices = sorted(
            neighbors_by_center[center_idx],
            key=lambda idx: (
                -float(ob.GetElectroNeg(cast(ob.OBAtom, obmol.GetAtom(idx)).GetAtomicNum())),
                metal_distances[idx] if metal_distances else 0,
                idx,
            ),
        )
        for neighbor_idx in neighbor_indices:
            if remaining_positive_charge <= 0:
                break
            neighbor = cast(ob.OBAtom, obmol.GetAtom(neighbor_idx))
            neighbor_has_pending_electrons = any(
                adjacent.GetIdx() != center_idx
                and (
                    get_unpaired_electron_count(adjacent) > 0
                    or has_unresolved_two_electron_center(adjacent)
                )
                for adjacent in atom_neighbors(neighbor)
            )
            if (
                neighbor.GetFormalCharge() != 0
                or get_unpaired_electron_count(neighbor) != 1
                or has_unresolved_two_electron_center(neighbor)
                or neighbor_has_pending_electrons
            ):
                continue
            set_unpaired_electron_count(neighbor, 0)
            neighbor.SetFormalCharge(-1)
            given_charge += 1
            remaining_positive_charge -= 1
            hit = True
    return omol, given_charge, hit


def _interchangeable_carboxyl_oxygens(
    carbon: ob.OBAtom, radical_oxygen: ob.OBAtom, carbonyl_oxygen: ob.OBAtom
) -> bool:
    """Check whether initial bond perception could equally put C=O on either oxygen."""
    first = carbon.GetDistance(radical_oxygen)
    second = carbon.GetDistance(carbonyl_oxygen)
    return (
        carbon.GetExplicitDegree() == 3
        and carbon.GetFormalCharge() == 0
        and any(a.GetAtomicNum() in (1, 6) for a in atom_neighbors(carbon))
        and carbonyl_oxygen.GetFormalCharge() == 0
        and not get_unpaired_electron_count(carbonyl_oxygen)
        and not get_unpaired_electron_count(carbon)
        and all(
            a.GetExplicitDegree() == 1 and not a.GetImplicitHCount()
            for a in (radical_oxygen, carbonyl_oxygen)
        )
        and all(
            not get_lone_pair_count(a) and not has_unresolved_two_electron_center(a)
            for a in (radical_oxygen, carbon, carbonyl_oxygen)
        )
        and 1.20 <= first <= 1.34
        and 1.20 <= second <= 1.34
        and abs(first - second) <= 0.03
    )


def eliminate_carboxyl(
    omol: pybel.Molecule, given_charge: int, metal_distances: tuple[int, ...] = ()
) -> tuple[pybel.Molecule, int, bool]:
    """Convert a carboxyl-like oxygen monoradical into an ``O-`` center.

    This one-electron localization requires and consumes exactly one real
    unpaired electron, then records it through a one-unit formal charge.
    """

    obmol = cast(ob.OBMol, omol.OBMol)
    res: List[Tuple[int, int, int]] = list(smarts.ELIM_CARBOXYL.findall(omol))
    hit = False
    while len(res):
        idxs = res.pop(0)
        atom_1 = cast(ob.OBAtom, obmol.GetAtom(idxs[0]))
        if get_unpaired_electron_count(atom_1) != 1:
            continue
        target = atom_1
        if metal_distances and metal_distances[idxs[2]] < metal_distances[idxs[0]]:
            carbon, other = (
                cast(ob.OBAtom, obmol.GetAtom(idxs[1])),
                cast(ob.OBAtom, obmol.GetAtom(idxs[2])),
            )
            if _interchangeable_carboxyl_oxygens(carbon, atom_1, other):
                carbon.GetBond(atom_1).SetBondOrder(2)
                carbon.GetBond(other).SetBondOrder(1)
                target = other
        set_unpaired_electron_count(atom_1, 0)
        target.SetFormalCharge(-1)
        given_charge += 1
        hit = True
    return omol, given_charge, hit


def eliminate_carbene_neighbor_heteroatom(
    omol: pybel.Molecule, given_charge: int
) -> tuple[pybel.Molecule, int, bool]:
    """Close an unresolved two-electron center with a heteroatom donor.

    Raising the bond order consumes the center marker and encodes its electrons
    as ``center- / donor+``. If the donor carries an explicitly tracked active
    lone pair, one pair is consumed; a zero count means the conventional donor
    pair is represented implicitly by valence and formal charge. Real unpaired
    electrons cannot drive this closed-shell transformation.
    """

    def possible_carbene_atom_checker(obatom: ob.OBAtom) -> bool:
        """Accept only an unresolved center, never an inferred spin/lone-pair surrogate."""

        return has_unresolved_two_electron_center(obatom)

    hit = False
    for obatom in indexed_atoms(cast(ob.OBMol, omol.OBMol)):
        if possible_carbene_atom_checker(obatom):
            if any(
                get_unpaired_electron_count(neighbor) == 1 for neighbor in atom_neighbors(obatom)
            ):
                continue
            for neighbor in atom_neighbors(obatom):
                if (
                    neighbor.GetAtomicNum() in consts.HETEROATOM
                    and neighbor.GetFormalCharge() == 0
                    and get_unpaired_electron_count(neighbor) == 0
                    and not has_unresolved_two_electron_center(neighbor)
                    # A zero active-pair count does not establish a conventional
                    # donor pair. Neutral five-bond P, for example, has none.
                    and consts.NON_METAL_DICT[neighbor.GetAtomicNum()].num_outer_electrons
                    - neighbor.GetFormalCharge()
                    - neighbor.GetTotalValence()
                    >= 2
                ):
                    donor = neighbor
                    bond = cast(ob.OBBond, obatom.GetBond(donor))
                    bond.SetBondOrder(bond.GetBondOrder() + 1)
                    set_lone_pair_count(obatom, 0)
                    set_unpaired_electron_count(obatom, 0)
                    set_unresolved_two_electron_center(obatom, False)
                    obatom.SetFormalCharge(obatom.GetFormalCharge() - 1)
                    if get_lone_pair_count(donor) > 0:
                        set_lone_pair_count(donor, get_lone_pair_count(donor) - 1)
                    donor.SetFormalCharge(donor.GetFormalCharge() + 1)
                    hit = True
                    break
    return omol, given_charge, hit


def eliminate_NNN(
    omol: pybel.Molecule, given_charge: int, positive: bool = False
) -> tuple[pybel.Molecule, int, bool]:
    """Close N-N-N motifs using explicitly classified endpoint electrons.

    In the negative branch each terminal N must supply one classified
    two-electron state: an unresolved marker, two real unpaired electrons, or
    one active lone pair. The middle N must supply exactly one real unpaired
    electron. The positive branch also requires one middle-N monoradical and
    a nonnegative remaining charge budget. With a negative budget, leave that
    radical available for anionic localization instead of creating a cation
    that would force unrelated bond cleavage during recovery.
    Missing or mixed labels reject the rule before any state is changed.
    """

    obmol = cast(ob.OBMol, omol.OBMol)
    hit = False

    def has_consumable_two_electron_center(atom: ob.OBAtom) -> bool:
        """Recognize one pure unresolved, triplet, or active-singlet two-electron state.

        Mixed marker/occupancy states and a singlet accompanied by any real
        unpaired electrons are rejected before the N-N-N graph is changed.
        """

        unresolved = has_unresolved_two_electron_center(atom)
        unpaired = get_unpaired_electron_count(atom)
        lone_pairs = get_lone_pair_count(atom)
        return (
            (unresolved and unpaired == 0 and lone_pairs == 0)
            or (not unresolved and unpaired == 2 and lone_pairs == 0)
            or (not unresolved and unpaired == 0 and lone_pairs >= 1)
        )

    def consume_two_electron_center(atom: ob.OBAtom) -> None:
        """Remove exactly one prevalidated two-electron state from its own field.

        No parity conversion is performed: unresolved clears the marker, triplet
        clears two unpaired electrons, and singlet subtracts one active lone pair.
        """

        if has_unresolved_two_electron_center(atom):
            set_unresolved_two_electron_center(atom, False)
        elif get_unpaired_electron_count(atom) == 2:
            set_unpaired_electron_count(atom, 0)
        else:
            set_lone_pair_count(atom, get_lone_pair_count(atom) - 1)

    if not positive:
        while res := smarts.ELIM_NNN_NEGATIVE.findall(omol):
            idxs = cast(List[Tuple[int, int, int]], res.pop(0))
            atom1 = cast(ob.OBAtom, obmol.GetAtom(idxs[0]))
            atom2 = cast(ob.OBAtom, obmol.GetAtom(idxs[1]))
            atom3 = cast(ob.OBAtom, obmol.GetAtom(idxs[2]))
            bond1 = cast(ob.OBBond, obmol.GetBond(idxs[0], idxs[1]))
            if (
                not has_consumable_two_electron_center(atom1)
                or get_unpaired_electron_count(atom2) != 1
                or not has_consumable_two_electron_center(atom3)
            ):
                break
            bond1.SetBondOrder(bond1.GetBondOrder() + 1)
            bond2 = cast(ob.OBBond, obmol.GetBond(idxs[1], idxs[2]))
            bond2.SetBondOrder(bond2.GetBondOrder() + 1)
            consume_two_electron_center(atom1)
            atom1.SetFormalCharge(atom1.GetFormalCharge() - 1)
            set_unpaired_electron_count(atom2, get_unpaired_electron_count(atom2) - 1)
            atom2.SetFormalCharge(atom2.GetFormalCharge() + 1)
            consume_two_electron_center(atom3)
            atom3.SetFormalCharge(atom3.GetFormalCharge() - 1)
            given_charge += 1
            hit = True
    else:
        while given_charge >= 0 and (res := smarts.ELIM_NNN_POSITIVE.findall(omol)):
            idxs = cast(List[Tuple[int, int, int]], res.pop(0))
            atom1 = cast(ob.OBAtom, obmol.GetAtom(idxs[0]))
            atom2 = cast(ob.OBAtom, obmol.GetAtom(idxs[1]))
            atom3 = cast(ob.OBAtom, obmol.GetAtom(idxs[2]))
            bond1 = cast(ob.OBBond, obmol.GetBond(idxs[0], idxs[1]))
            if get_unpaired_electron_count(atom2) != 1:
                break
            bond1.SetBondOrder(bond1.GetBondOrder() + 1)
            atom1.SetFormalCharge(atom1.GetFormalCharge() + 1)
            set_unpaired_electron_count(atom2, get_unpaired_electron_count(atom2) - 1)
            given_charge -= 1
            hit = True
    return omol, given_charge, hit


def eliminate_1_3_dipole_postive(
    omol: pybel.Molecule, given_charge: int
) -> tuple[pybel.Molecule, int, bool]:
    """Use positive-charge budget to close a bond in a 1,3-dipole motif.

    Forming the extra bond consumes exactly one real unpaired electron and shifts
    one unit of formal charge to the middle atom. The rule runs while
    ``given_charge >= 0``; lone pairs and unresolved centers do not satisfy it.
    """

    obmol = cast(ob.OBMol, omol.OBMol)
    hit = False
    if given_charge < 0:
        return omol, given_charge, hit

    res: List[Tuple[int, int, int]] = list(smarts.ELIM_1_3_DIPOLE_POSTIVE.findall(omol))
    while given_charge >= 0 and len(res):
        idxs = res.pop(0)
        atom2 = cast(ob.OBAtom, obmol.GetAtom(idxs[1]))
        atom3 = cast(ob.OBAtom, obmol.GetAtom(idxs[2]))
        bond2 = cast(ob.OBBond, obmol.GetBond(idxs[1], idxs[2]))
        if (
            get_unpaired_electron_count(atom3) == 1
            and consts.NON_METAL_DICT[atom2.GetAtomicNum()].num_outer_electrons
            + atom2.GetTotalValence()
            == 8
        ):
            atom2.SetFormalCharge(atom2.GetFormalCharge() + 1)
            bond2.SetBondOrder(int(bond2.GetBondOrder() + 1))
            set_unpaired_electron_count(atom3, get_unpaired_electron_count(atom3) - 1)
            given_charge -= 1
            hit = True
    return omol, given_charge, hit


def eliminate_possible_cp_like_radical_anion(
    omol: pybel.Molecule,
    given_charge: int,
    total_radical_electrons: int,
    metal_distances: tuple[int, ...] = (),
) -> tuple[pybel.Molecule, int, bool]:
    """Localize an excess cyclopentadienyl-like radical as an aromatic anion.

    One real unpaired electron is consumed only while the global electron pool
    exceeds the requested radical count and pending charge budget. Diradicals,
    active lone pairs, and unresolved centers are not collapsed into this state.
    """

    obmol = cast(ob.OBMol, omol.OBMol)
    hit = False

    def available_unpaired_electrons() -> int:
        return (
            sum(get_unpaired_electron_count(atom) for atom in indexed_atoms(obmol))
            - total_radical_electrons
            - abs(given_charge)
        )

    while res := smarts.ELIM_NEGATIVE_CP.findall(omol):
        if available_unpaired_electrons() < 1:
            break
        idxs = res[0]
        atom = cast(ob.OBAtom, obmol.GetAtom(idxs[4]))
        if atom.GetFormalCharge() != 0:
            break
        if get_unpaired_electron_count(atom) != 1:
            break
        if metal_distances and len(res) > 1:
            alternatives = (cast(ob.OBAtom, obmol.GetAtom(match[4])) for match in res)
            atom = min(
                (
                    a
                    for a in alternatives
                    if a.GetFormalCharge() == 0 and get_unpaired_electron_count(a) == 1
                ),
                key=lambda a: metal_distances[a.GetIdx()],
            )
        set_unpaired_electron_count(atom, 0)
        atom.SetFormalCharge(-1)
        given_charge += 1
        hit = True
    return omol, given_charge, hit


def eliminate_positive_charges(
    omol: pybel.Molecule,
    given_charge: int,
    total_radical_electrons: int = 0,
) -> tuple[pybel.Molecule, int, bool]:
    """Consume classified electron sources into positive formal charge.

    Radical actions can consume several explicit unpaired electrons at one atom.
    Separately, a pure unresolved center may satisfy two remaining charge units by
    becoming ``+2`` and clearing its marker; it is not treated as a diradical.
    """

    obmol = cast(ob.OBMol, omol.OBMol)
    hit = False

    while True:
        actions = _positive_charge_assignment_actions(
            omol, obmol, given_charge, total_radical_electrons
        )
        if not actions:
            break
        action = next(
            (action for action in actions if _positive_charge_geometry_allowed(obmol, action)),
            None,
        )
        if action is None:
            break
        if not _apply_charge_assignment_action(obmol, action):
            break
        given_charge += action.charge_delta
        hit = True
    return omol, given_charge, hit


def _positive_charge_geometry_allowed(obmol: ob.OBMol, action: _ChargeAssignmentAction) -> bool:
    """Skip N--N/N--O donations already excluded by final geometry validation.

    Probe the resulting charge and electron labels on a copy, because these
    screens distinguish resolved cumulenes from radicals and deferred centers.
    A rejected first action must leave the other ranked choices available.
    """
    if action.bond_idx is None or action.charge_atom_idx is None:
        return True
    pair = sorted(
        obmol.GetAtom(i).GetAtomicNum() for i in (action.atom_idx, action.charge_atom_idx)
    )
    if pair not in ([7, 7], [7, 8], [6, 15]):
        return True
    trial = ob.OBMol(obmol)
    if not _apply_charge_assignment_action(trial, action):
        return False
    bond = cast(ob.OBBond, trial.GetBond(action.atom_idx, action.charge_atom_idx))
    atoms_and_order = (
        cast(ob.OBAtom, bond.GetBeginAtom()),
        cast(ob.OBAtom, bond.GetEndAtom()),
        bond.GetBondOrder(),
    )
    return (
        nitrogen_nitrogen_order_allowed(*atoms_and_order)
        and nitrogen_oxygen_order_allowed(*atoms_and_order)
        and phosphorus_carbon_order_allowed(*atoms_and_order)
    )


def eliminate_negative_charges(
    omol: pybel.Molecule,
    given_charge: int,
    total_radical_electrons: int | None = None,
    metal_distances: tuple[int, ...] = (),
) -> tuple[pybel.Molecule, int, bool]:
    """Consume classified electron sources into negative formal charge.

    At zero budget this may intentionally generate a charge-separated resonance
    contributor from a real radical. When the optional global radical target is
    supplied, the target count is checked before every action so that a charge
    assignment cannot consume a radical needed by the target. A pure unresolved
    center is eligible only when at least two negative charge units remain and is
    consumed directly as ``-2``. With both negative charge and radical budget,
    Bv3 instead becomes B- with one new unpaired electron. Both budgets are
    rechecked after each assignment. Active lone pairs are not consumed.
    """

    obmol = cast(ob.OBMol, omol.OBMol)
    hit = False

    if given_charge > 0:
        return omol, given_charge, hit
    sites, real_unpaired = _negative_charge_sites(obmol, given_charge, total_radical_electrons)
    while sites and given_charge <= 0:
        if (
            given_charge == 0
            and total_radical_electrons is not None
            and real_unpaired <= total_radical_electrons
        ):
            break
        actions = [
            action
            for site in sites.values()
            if (action := site.action(given_charge, real_unpaired, total_radical_electrons))
            is not None
        ]
        if not actions:
            break
        if metal_distances and len(actions) > 1:
            action = min(
                actions, key=lambda a: (*a.score_key[:4], metal_distances[a.atom_idx], a.atom_idx)
            )
        else:
            action = min(actions, key=lambda a: a.score_key)
        if not _apply_charge_assignment_action(obmol, action):
            break
        del sites[action.atom_idx]
        given_charge += action.charge_delta
        real_unpaired += action.spin_created - action.spin_consumed
        hit = True

    return omol, given_charge, hit


__all__ = [
    "eliminate_1_3_dipole_postive",
    "eliminate_NNN",
    "eliminate_carboxyl",
    "eliminate_carbene_neighbor_heteroatom",
    "eliminate_possible_cp_like_radical_anion",
    "eliminate_high_positive_charge_atoms",
    "eliminate_negative_charges",
    "eliminate_positive_charges",
]
