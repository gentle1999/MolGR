"""Charge and radical refresh rules for fallback."""

from __future__ import annotations

from typing import cast

from openbabel import openbabel as ob
from openbabel import pybel

from molgr.fallback.utils import consts
from molgr.fallback.utils.bond_geometry import allows_seesaw_phosphoranide
from molgr.fallback.utils.electrons import (
    get_lone_pair_count,
    get_unpaired_electron_count,
    has_unresolved_two_electron_center,
    set_lone_pair_count,
    set_unpaired_electron_count,
    set_unresolved_two_electron_center,
)
from molgr.fallback.utils.ob_graph import mol_atoms


def assign_radical_dots(atom: ob.OBAtom) -> int:
    """Return the local bond-valence deficit, before classifying its electrons.

    The result is not yet an unpaired-electron count. In particular, two missing
    bond-valence units may later become one active lone pair, two real unpaired
    electrons, or an unresolved C/N/Si/P two-electron center. Boron uses its
    nonbonding electron budget, excluding the closed-shell anionic lone pairs
    that are not reconstruction-active. An odd electron left on B- remains
    active, including the three-bond radical anion.
    """
    atomic_num = int(atom.GetAtomicNum())
    total_valence = int(atom.GetTotalValence())
    if atomic_num == 5:
        charge = int(atom.GetFormalCharge())
        nonbonding_electrons = max(0, 3 - charge - total_valence)
        inactive_lone_pairs = min(max(0, -charge), nonbonding_electrons // 2)
        return nonbonding_electrons - 2 * inactive_lone_pairs
    typical_valence = int(ob.GetTypicalValence(atomic_num, total_valence, atom.GetFormalCharge()))
    return max(0, typical_valence - total_valence)


def _infer_active_electron_occupancy(atom: ob.OBAtom, electron_count: int) -> tuple[int, int]:
    """Classify active electrons into unpaired electrons and active lone pairs.

    Occupancy is derived from sigma degree, pi bond-order increments, and the
    remaining main-group valence-orbital slots. It does not read Open Babel
    hybridization and does not apply atom-wide parity as an orbital rule. The
    returned lone-pair count contains only reconstruction-active pairs, not every
    conventional Lewis lone pair on the atom.
    """

    if electron_count <= 0:
        return 0, 0
    total_valence = int(atom.GetTotalValence())
    sigma_bond_count = int(atom.GetTotalDegree())
    pi_bond_count = max(0, total_valence - sigma_bond_count)
    available_valence_orbitals = max(0, 4 - sigma_bond_count - pi_bond_count)

    lone_pair_count = 0
    unpaired_electron_count = 0

    valence_electrons = min(electron_count, 2 * available_valence_orbitals)
    if valence_electrons <= available_valence_orbitals:
        unpaired_electron_count += valence_electrons
    else:
        lone_pair_count += valence_electrons - available_valence_orbitals
        unpaired_electron_count += 2 * available_valence_orbitals - valence_electrons
    electron_count -= valence_electrons

    # Electrons beyond the four main-group valence slots are reconstruction
    # deficits rather than a basis for guessing Open Babel hybridization.
    if electron_count:
        lone_pair_count += electron_count // 2
        unpaired_electron_count += electron_count % 2
    return unpaired_electron_count, lone_pair_count


def _assign_active_electron_occupancy(atom: ob.OBAtom, electron_count: int) -> None:
    """Replace both explicit occupancy fields from one local electron deficit.

    This is a complete overwrite of ``unpaired`` and active ``lone_pair`` counts;
    callers must separately manage the unresolved-center marker.
    """

    unpaired_electron_count, lone_pair_count = _infer_active_electron_occupancy(
        atom, electron_count
    )
    set_unpaired_electron_count(atom, unpaired_electron_count)
    set_lone_pair_count(atom, lone_pair_count)


def assign_charge_radical_for_atom(atom: ob.OBAtom) -> bool:
    """Rebuild one atom's charge and explicit electron classification.

    Neutral C/N/Si/P atoms with a two-unit deficit are marked unresolved instead of
    being forced to singlet or triplet occupancy. Explicitly resolved ``(0, 1)``
    and ``(2, 0)`` states are preserved while the same two-electron topology
    remains. Other atoms overwrite unpaired/active-lone-pair counts from the local
    deficit and may also normalize formal charge. This is therefore a chemical
    state transformation, not a read-only label refresh.
    """

    old_charge = atom.GetFormalCharge()
    old_unpaired_electron_count = get_unpaired_electron_count(atom)
    old_lone_pair_count = get_lone_pair_count(atom)
    old_unresolved_center = has_unresolved_two_electron_center(atom)

    if old_charge == -1 and allows_seesaw_phosphoranide(atom):
        # Preserve an explicitly explored, geometry-supported phosphoranide.
        # Its closed-shell pair is not a reconstruction-active electron deficit.
        # The ordinary seed still starts as P+; geometry alone never selects P-.
        set_unpaired_electron_count(atom, 0)
        set_lone_pair_count(atom, 0)
        set_unresolved_two_electron_center(atom, False)
        return old_unpaired_electron_count != 0 or old_lone_pair_count != 0 or old_unresolved_center

    if consts.NON_METAL_DICT[atom.GetAtomicNum()].num_outer_electrons < atom.GetTotalValence():
        atom.SetFormalCharge(
            consts.NON_METAL_DICT[atom.GetAtomicNum()].num_outer_electrons - atom.GetTotalValence()
        )
    else:
        total_elec = (
            consts.NON_METAL_DICT[atom.GetAtomicNum()].num_outer_electrons + atom.GetTotalValence()
        )
        if total_elec > 8 and total_elec % 8 <= (
            consts.NON_METAL_DICT[atom.GetAtomicNum()].num_outer_electrons - atom.GetTotalValence()
        ):
            atom.SetFormalCharge(total_elec % 8)

    radical_dots = assign_radical_dots(atom)
    is_two_electron_center = (
        atom.GetAtomicNum() in (6, 7, 14, 15) and atom.GetFormalCharge() == 0 and radical_dots == 2
    )
    if is_two_electron_center:
        explicit_occupancy = (
            get_unpaired_electron_count(atom),
            get_lone_pair_count(atom),
        ) in {(0, 1), (2, 0)}
        if old_unresolved_center or not explicit_occupancy:
            set_unpaired_electron_count(atom, 0)
            set_lone_pair_count(atom, 0)
            set_unresolved_two_electron_center(atom, True)
        return (
            old_unpaired_electron_count != get_unpaired_electron_count(atom)
            or old_lone_pair_count != get_lone_pair_count(atom)
            or old_unresolved_center != has_unresolved_two_electron_center(atom)
        )

    set_unresolved_two_electron_center(atom, False)
    _assign_active_electron_occupancy(atom, radical_dots)
    return (
        old_charge != atom.GetFormalCharge()
        or old_unpaired_electron_count != get_unpaired_electron_count(atom)
        or old_lone_pair_count != get_lone_pair_count(atom)
        or old_unresolved_center != has_unresolved_two_electron_center(atom)
    )


def fresh_omol_charge_radical(omol: pybel.Molecule) -> tuple[pybel.Molecule, bool]:
    """Apply local charge/electron rebuilding independently to every atom.

    All explicit unpaired, active-lone-pair, and unresolved-center fields may be
    replaced, and formal charges may change. Use this for initial/recovery states;
    local resonance rules should prefer refreshing only atoms whose topology or
    charge they changed.
    """

    hit = False
    for atom in mol_atoms(cast(ob.OBMol, omol.OBMol)):
        hit = assign_charge_radical_for_atom(atom) or hit
    return omol, hit


__all__ = [
    "assign_charge_radical_for_atom",
    "assign_radical_dots",
    "fresh_omol_charge_radical",
]
