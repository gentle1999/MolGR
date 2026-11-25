"""Preparation helpers for no-metal fallback reconstruction."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import cast

from openbabel import openbabel as ob
from openbabel import pybel

from molgr.fallback.state import OmolStateMachine, ReconstructionState
from molgr.fallback.utils.bond_geometry import (
    allows_seesaw_phosphoranide,
    equalized_three_carbon_chains,
    has_conjugated_thioamide_donor,
    nitrogen_oxygen_order_allowed,
    short_ortho_quinodiimine_sites,
)
from molgr.fallback.utils.coordination_geometry import detach_carbonyl_hydride_contacts
from molgr.fallback.utils.dataclasses import MetalSite
from molgr.fallback.utils.electrons import (
    get_lone_pair_count,
    get_unpaired_electron_count,
    has_unresolved_two_electron_center,
    set_lone_pair_count,
    set_unpaired_electron_count,
    set_unresolved_two_electron_center,
)
from molgr.fallback.utils.ob_graph import atom_neighbors, mol_atoms, mol_bonds, neighbor_bond_order
from molgr.fallback.utils.xyz import read_xyz_seed


def _normalize_seed_electronic_labels(omol: pybel.Molecule) -> pybel.Molecule:
    """Clear all inferred seed charges and explicit MolGR electron classifications.

    XYZ/Open Babel perception is not trusted as an electronic-state source. The
    no-metal pipeline therefore resets formal charge, real unpaired electrons,
    active lone pairs, and unresolved-center markers before rebuilding them from
    topology and the supplied global budgets.
    """

    for atom in mol_atoms(cast(ob.OBMol, omol.OBMol)):
        atom.SetFormalCharge(0)
        set_unpaired_electron_count(atom, 0)
        set_lone_pair_count(atom, 0)
        set_unresolved_two_electron_center(atom, False)
    return omol


def _seed_state(
    xyz_block: str,
    total_charge: int,
    total_radical_electrons: int,
) -> ReconstructionState:
    """Create the initial reconstruction state from the input XYZ block."""

    return _seed_state_from_omol(
        _seed_omol_from_xyz(xyz_block),
        total_charge,
        total_radical_electrons,
    )


def _seed_omol_from_xyz(xyz_block: str, metal_sites: Sequence[MetalSite] = ()) -> pybel.Molecule:
    """Parse and normalize the shared initial no-metal molecule from XYZ."""

    seed = _normalize_seed_electronic_labels(read_xyz_seed(xyz_block))
    if metal_sites:
        detach_carbonyl_hydride_contacts(seed, metal_sites)
    return seed


def _seed_state_from_omol(
    seed_omol: pybel.Molecule,
    total_charge: int,
    total_radical_electrons: int,
) -> ReconstructionState:
    """Create an independent initial state by cloning a normalized seed molecule."""

    omol = _normalize_seed_electronic_labels(
        pybel.Molecule(ob.OBMol(cast(ob.OBMol, seed_omol.OBMol)))
    )
    return ReconstructionState(
        omol=omol,
        given_charge=0,
        total_charge=total_charge,
        total_radical_electrons=total_radical_electrons,
        phase_history=("read_xyz", "normalize_seed_electronic_labels"),
        metadata={"source": "xyz_to_omol_no_metal_state"},
    )


def prepare_no_metal_seed(state: ReconstructionState) -> ReconstructionState:
    """Compatibility entry point for the pipeline-owned linear reconstruction."""

    from molgr.fallback.pipeline.reconstruct_without_metals import _run_linear_preparation

    return _run_linear_preparation(state)


def _explore_phosphoranide(
    omol: pybel.Molecule, given_charge: int, atom_idx: int
) -> tuple[pybel.Molecule, int, bool]:
    """Spend two negative charge units to explore a seesaw P(−1) alternative to P(+1)."""
    atom = cast(ob.OBAtom, omol.OBMol.GetAtom(atom_idx))
    if atom.GetFormalCharge() != 1 or not allows_seesaw_phosphoranide(atom):
        return omol, given_charge, False
    atom.SetFormalCharge(-1)
    set_unpaired_electron_count(atom, 0)
    set_lone_pair_count(atom, 0)
    set_unresolved_two_electron_center(atom, False)
    return omol, given_charge + 2, True


def _is_anionic_diborene_bond(bond: ob.OBBond) -> bool:
    """Recognize a closed-shell B(−1)=B(−1) pair with three neighbors per boron."""
    return (
        bond.GetBondOrder() == 2
        and not bond.IsAromatic()
        and all(
            atom.GetAtomicNum() == 5
            and atom.GetFormalCharge() == -1
            and atom.GetTotalValence() == 4
            and atom.GetTotalDegree() == 3
            and get_unpaired_electron_count(atom) == 0
            and get_lone_pair_count(atom) == 0
            and not has_unresolved_two_electron_center(atom)
            for atom in (cast(ob.OBAtom, bond.GetBeginAtom()), cast(ob.OBAtom, bond.GetEndAtom()))
        )
    )


def _explore_neutral_diboron(
    omol: pybel.Molecule, given_charge: int, first: int, second: int
) -> tuple[pybel.Molecule, int, bool]:
    """Replace the dianionic B=B contributor with a neutral B–B single bond and update the charge budget."""
    bond = cast("ob.OBBond | None", omol.OBMol.GetBond(first, second))
    if not bond or not _is_anionic_diborene_bond(bond):
        return omol, given_charge, False
    bond.SetBondOrder(1)
    bond.GetBeginAtom().SetFormalCharge(0)
    bond.GetEndAtom().SetFormalCharge(0)
    return omol, given_charge - 2, True


def _localize_oxime_oxygen(
    omol: pybel.Molecule, charge: int, o_idx: int, n_idx: int
) -> tuple[pybel.Molecule, int, bool]:
    """Apply one available negative charge to the O radical of an oxime N/O radical pair."""
    oxygen, nitrogen = (cast(ob.OBAtom, omol.OBMol.GetAtom(i)) for i in (o_idx, n_idx))
    if charge >= 0 or any(
        a.GetFormalCharge()
        or a.GetImplicitHCount()
        or get_unpaired_electron_count(a) != 1
        or get_lone_pair_count(a)
        or has_unresolved_two_electron_center(a)
        for a in (oxygen, nitrogen)
    ):
        return omol, charge, False
    oxygen.SetFormalCharge(-1)
    set_unpaired_electron_count(oxygen, 0)
    return omol, charge + 1, True


def _dithiooxamide_sites(mol: ob.OBMol) -> Iterator[tuple[int, ...]]:
    """Yield adjacent thioamide carbons and their terminal sulfur indices for paired C=S closure."""
    for bond in mol_bonds(mol):
        left, right = cast(ob.OBAtom, bond.GetBeginAtom()), cast(ob.OBAtom, bond.GetEndAtom())
        if (
            left.GetAtomicNum() != 6
            or right.GetAtomicNum() != 6
            or bond.GetBondOrder() != 1
            or not 1.35 <= left.GetDistance(right) <= 1.60
        ):
            continue
        sulfurs = []
        for carbon in (left, right):
            if not has_conjugated_thioamide_donor(carbon):
                break
            matches = [
                s.GetIdx()
                for s in atom_neighbors(carbon)
                if s.GetAtomicNum() == 16
                and s.GetExplicitDegree() == 1
                and neighbor_bond_order(carbon, s) == 1
                and 1.60 < carbon.GetDistance(s) < 1.82
            ]
            if not matches:
                break
            sulfurs.append(matches[-1])
        if len(sulfurs) == 2:
            yield (left.GetIdx(), right.GetIdx(), *sulfurs)


def _close_dithiooxamide(omol: pybel.Molecule, site: Sequence[int]) -> tuple[pybel.Molecule, bool]:
    """Pair four monoradicals into two thiocarbonyl bonds after checking all four valences."""
    mol = cast(ob.OBMol, omol.OBMol)
    for idx, degree in zip(site, (3, 3, 1, 1)):
        a = cast(ob.OBAtom, mol.GetAtom(idx))
        if (
            a.GetFormalCharge()
            or a.GetTotalValence() != degree
            or a.GetExplicitDegree() != degree
            or a.GetImplicitHCount()
            or get_unpaired_electron_count(a) != 1
            or has_unresolved_two_electron_center(a)
        ):
            return omol, False
    for c, s in zip(site[:2], site[2:]):
        bond = cast(ob.OBBond, mol.GetBond(c, s))
        bond.SetBondOrder(2)
        bond.SetAromatic(False)
        for a in (cast(ob.OBAtom, mol.GetAtom(c)), cast(ob.OBAtom, mol.GetAtom(s))):
            a.SetAromatic(False)
            set_unpaired_electron_count(a, 0)
            set_lone_pair_count(a, 0)
    mol.SetAromaticPerceived(False)
    return omol, True


def _close_short_quinodiimine(
    omol: pybel.Molecule, site: Sequence[int]
) -> tuple[pybel.Molecule, bool]:
    """Close the two exocyclic imines and redistribute the ring double bonds to the quinoid pattern."""
    mol = cast(ob.OBMol, omol.OBMol)
    for j, idx in enumerate(site):
        atom = cast(ob.OBAtom, mol.GetAtom(idx))
        if (
            atom.GetFormalCharge()
            or atom.GetTotalValence() != (4 if j < 6 else 2)
            or get_unpaired_electron_count(atom) != (0 if j < 6 else 1)
            or has_unresolved_two_electron_center(atom)
        ):
            return omol, False
    for j, idx in enumerate(site[:6]):
        bond = cast(ob.OBBond, mol.GetBond(idx, site[(j + 1) % 6]))
        bond.SetBondOrder(2 if j in (2, 4) else 1)
        bond.SetAromatic(False)
        mol.GetAtom(idx).SetAromatic(False)
    for carbon, nitrogen in zip(site[:2], site[6:]):
        mol.GetBond(carbon, nitrogen).SetBondOrder(2)
        n = cast(ob.OBAtom, mol.GetAtom(nitrogen))
        set_unpaired_electron_count(n, 0)
        set_lone_pair_count(n, 0)
    mol.SetAromaticPerceived(False)
    return omol, True


def expand_geometry_electronic_seeds(state: ReconstructionState) -> list[ReconstructionState]:
    """Retain the ordinary seed and explore geometry-supported electronic states.

    Each branch balances the change in formal charges with an equal and
    opposite change in the remaining charge budget. The normal search/validation must still satisfy
    the original total charge and radical count. Enumerate combinations in
    deterministic atom order, with the same 256-state bound as neighbor seeds.
    """
    pool = [state]
    mol = cast(ob.OBMol, state.omol.OBMol)
    for oxygen in mol_atoms(mol):
        if oxygen.GetAtomicNum() != 8 or oxygen.GetExplicitDegree() != 1:
            continue
        nitrogen = next(atom_neighbors(oxygen))
        if (
            nitrogen.GetAtomicNum() != 7
            or nitrogen.GetExplicitDegree() != 2
            or neighbor_bond_order(oxygen, nitrogen) != 1
            or nitrogen_oxygen_order_allowed(nitrogen, oxygen, 2)
        ):
            continue
        for source in tuple(pool):
            if len(pool) >= 256:
                return pool
            if source.given_charge >= 0:
                continue
            machine = OmolStateMachine.from_reconstruction_state(source).branch(
                None, omol=pybel.Molecule(ob.OBMol(source.omol.OBMol))
            )
            if machine.run_omol_charge_stage(
                "explore_long_oxime_oxygen_anion",
                _localize_oxime_oxygen,
                oxygen.GetIdx(),
                nitrogen.GetIdx(),
            ):
                pool.append(machine.freeze_like(source))
    for sites, phase, stage in (
        (_dithiooxamide_sites(mol), "explore_neutral_dithiooxamide", _close_dithiooxamide),
        (
            short_ortho_quinodiimine_sites(mol),
            "explore_short_quinodiimine",
            _close_short_quinodiimine,
        ),
    ):
        for site in sites:
            for source in tuple(pool):
                if len(pool) >= 256:
                    return pool
                machine = OmolStateMachine.from_reconstruction_state(source).branch(
                    None, omol=pybel.Molecule(ob.OBMol(source.omol.OBMol))
                )
                if machine.run_omol_stage(phase, stage, site):
                    pool.append(machine.freeze_like(source))
    for atom in mol_atoms(state.omol.OBMol):
        if atom.GetFormalCharge() != 1 or not allows_seesaw_phosphoranide(atom):
            continue
        for source in tuple(pool):
            if len(pool) >= 256:
                return pool
            machine = OmolStateMachine.from_reconstruction_state(source).branch(
                "branch_geometry_electronic_seed",
                omol=pybel.Molecule(ob.OBMol(source.omol.OBMol)),
            )
            if machine.run_omol_charge_stage(
                "explore_seesaw_phosphoranide", _explore_phosphoranide, atom.GetIdx()
            ):
                pool.append(machine.freeze_like(source))
    for bond in mol_bonds(state.omol.OBMol):
        if not _is_anionic_diborene_bond(bond):
            continue
        for source in tuple(pool):
            if len(pool) >= 256:
                return pool
            machine = OmolStateMachine.from_reconstruction_state(source).branch(
                "branch_geometry_electronic_seed",
                omol=pybel.Molecule(ob.OBMol(source.omol.OBMol)),
            )
            if machine.run_omol_charge_stage(
                "explore_neutral_diboron_bond",
                _explore_neutral_diboron,
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
            ):
                pool.append(machine.freeze_like(source))
    if not any(source.given_charge <= -2 for source in pool):
        return pool
    for left, middle, right in equalized_three_carbon_chains(state.omol.OBMol):
        for first, other in ((left, right), (right, left)):
            for source in tuple(pool):
                if len(pool) >= 256:
                    return pool
                if not _can_reduce_carbon_pair(
                    source.omol.OBMol,
                    source.given_charge,
                    first.GetIdx(),
                    middle.GetIdx(),
                    other.GetIdx(),
                ):
                    continue
                machine = OmolStateMachine.from_reconstruction_state(source).branch(
                    "branch_geometry_electronic_seed",
                    omol=pybel.Molecule(ob.OBMol(source.omol.OBMol)),
                )
                machine.run_omol_charge_stage(
                    "explore_equalized_carbon_pair",
                    _reduce_carbon_pair,
                    first.GetIdx(),
                    middle.GetIdx(),
                    other.GetIdx(),
                )
                pool.append(machine.freeze_like(source))
    return pool


def _can_reduce_carbon_pair(
    mol: ob.OBMol, given_charge: int, first_idx: int, middle_idx: int, other_idx: int
) -> bool:
    """Require two charge units and two C radicals on the double-bonded end of the C₃ fragment."""
    first, middle, other = (
        cast(ob.OBAtom, mol.GetAtom(i)) for i in (first_idx, middle_idx, other_idx)
    )
    return (
        given_charge <= -2
        and neighbor_bond_order(first, middle) == 2
        and neighbor_bond_order(middle, other) == 1
        and other.GetFormalCharge() == 0
        and other.GetTotalValence() == 2
        and all(
            atom.GetFormalCharge() == 0 and get_unpaired_electron_count(atom) == 1
            for atom in (first, middle)
        )
    )


def _reduce_carbon_pair(
    omol: pybel.Molecule, given_charge: int, first_idx: int, middle_idx: int, other_idx: int
) -> tuple[pybel.Molecule, int, bool]:
    """Spend two negative charges before a radical pair closes into C#C."""
    if not _can_reduce_carbon_pair(omol.OBMol, given_charge, first_idx, middle_idx, other_idx):
        return omol, given_charge, False
    for idx in (first_idx, middle_idx):
        atom = cast(ob.OBAtom, omol.OBMol.GetAtom(idx))
        atom.SetFormalCharge(-1)
        set_unpaired_electron_count(atom, 0)
        set_lone_pair_count(atom, 0)
        set_unresolved_two_electron_center(atom, False)
    return omol, given_charge + 2, True


__all__ = [
    "_seed_omol_from_xyz",
    "_seed_state",
    "_seed_state_from_omol",
    "prepare_no_metal_seed",
    "expand_geometry_electronic_seeds",
]
