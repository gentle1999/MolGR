"""Ligand electronic alternatives supported by nearby, unassigned metal sites."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import cast

from openbabel import openbabel as ob
from openbabel import pybel

from molgr.config import MetalScoringConfig
from molgr.fallback.state import OmolStateMachine, ReconstructionState
from molgr.fallback.utils.coordination_geometry import (
    build_coordination_blocker_arrays,
    inner_visible_atoms_to_metal,
    terminal_nitrosyl_contacts,
)
from molgr.fallback.utils.dataclasses import MetalSite
from molgr.fallback.utils.electrons import (
    get_lone_pair_count,
    get_unpaired_electron_count,
    has_unresolved_two_electron_center,
    set_lone_pair_count,
    set_unpaired_electron_count,
    set_unresolved_two_electron_center,
)
from molgr.fallback.utils.ob_graph import atom_neighbors, mol_atoms, neighbor_bond_order
from molgr.fallback.utils.search_constraints import require_nitrosonium


def hydrazido_sites(mol: ob.OBMol) -> Iterator[tuple[ob.OBAtom, ob.OBAtom, tuple[int, int, int]]]:
    """Yield terminal N–NR₂ pairs with either neutral isodiazene or dianionic hydrazido labels."""
    for first in mol_atoms(mol):
        if first.GetAtomicNum() != 7 or first.GetExplicitDegree() != 1 or first.GetImplicitHCount():
            continue
        second = next(atom_neighbors(first))
        if (
            second.GetAtomicNum() != 7
            or second.GetExplicitDegree() != 3
            or second.GetImplicitHCount()
        ):
            continue
        if any(
            a.GetAtomicNum() != 6 for a in atom_neighbors(second) if a.GetIdx() != first.GetIdx()
        ):
            continue
        if any(
            get_unpaired_electron_count(a)
            or get_lone_pair_count(a)
            or has_unresolved_two_electron_center(a)
            for a in (first, second)
        ):
            continue
        state = (
            first.GetFormalCharge(),
            second.GetFormalCharge(),
            neighbor_bond_order(first, second),
        )
        if state in ((-1, 1, 2), (-2, 0, 1)):
            yield first, second, state


def supports_hydrazido(first: ob.OBAtom, second: ob.OBAtom, metal: MetalSite) -> bool:
    """Require a short metal–N contact nearly collinear with the terminal N–N bond."""
    mn = (
        metal.position_x - first.GetX(),
        metal.position_y - first.GetY(),
        metal.position_z - first.GetZ(),
    )
    nn = (second.GetX() - first.GetX(), second.GetY() - first.GetY(), second.GetZ() - first.GetZ())
    mn_length = math.sqrt(sum(x * x for x in mn))
    nn_length = math.sqrt(sum(x * x for x in nn))
    radius = ob.GetCovalentRad(metal.element_idx) + ob.GetCovalentRad(7)
    if not (1.20 < nn_length < 1.45 and 0.65 * radius < mn_length < 0.85 * radius):
        return False
    cosine = sum(x * y for x, y in zip(mn, nn)) / (mn_length * nn_length)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine)))) > 160.0


def _hydrazonido_sites(mol: ob.OBMol, metal: MetalSite) -> Iterator[tuple[int, int, int]]:
    """Find N=C closure partners in terminal N=N–C fragments with radicals at both ends."""
    for first in mol_atoms(mol):
        if first.GetAtomicNum() != 7 or first.GetExplicitDegree() != 1:
            continue
        second = next(atom_neighbors(first))
        if (
            second.GetAtomicNum() != 7
            or second.GetExplicitDegree() != 2
            or neighbor_bond_order(first, second) != 2
            or not supports_hydrazido(first, second, metal)
        ):
            continue
        carbon = next(a for a in atom_neighbors(second) if a.GetIdx() != first.GetIdx())
        if (
            carbon.GetAtomicNum() != 6
            or carbon.GetExplicitDegree() != 3
            or carbon.GetTotalValence() != 3
            or neighbor_bond_order(second, carbon) != 1
            or not 1.20 < carbon.GetDistance(second) < 1.38
            or not 90 < mol.GetAngle(first, second, carbon) < 150
        ):
            continue
        if any(
            a.GetFormalCharge()
            or a.GetImplicitHCount()
            or get_lone_pair_count(a)
            or has_unresolved_two_electron_center(a)
            or get_unpaired_electron_count(a) != radicals
            for a, radicals in ((first, 1), (second, 0), (carbon, 1))
        ):
            continue
        if all(
            a.GetAtomicNum() in (1, 6) or a.GetIdx() == second.GetIdx()
            for a in atom_neighbors(carbon)
        ):
            yield first.GetIdx(), second.GetIdx(), carbon.GetIdx()


def _reduce_hydrazonido(
    omol: pybel.Molecule, charge: int, first: int, second: int, carbon: int
) -> tuple[pybel.Molecule, int, bool]:
    """Shift N=N–C to N–N=C and pair the terminal radicals using two negative charge units."""
    mol = cast(ob.OBMol, omol.OBMol)
    mol.GetBond(first, second).SetBondOrder(1)
    mol.GetBond(second, carbon).SetBondOrder(2)
    mol.GetAtom(first).SetFormalCharge(-2)
    for idx in (first, carbon):
        set_unpaired_electron_count(mol.GetAtom(idx), 0)
    return omol, charge + 2, True


def expand_hydrazido_seeds(
    state: ReconstructionState, metals: Sequence[MetalSite]
) -> list[ReconstructionState]:
    """Preserve the seed and spend only an available two-charge deficit.

    Metal positions are supplied explicitly by the metal pipeline. The ordinary
    metal-free reconstruction neither assumes metal atoms nor calls this helper.
    """
    pool = [state]
    if len(metals) != 1:
        return pool
    for site in _hydrazonido_sites(state.omol.OBMol, metals[0]):
        for source in tuple(pool):
            if len(pool) >= 256:
                return pool
            if source.given_charge <= -2:
                machine = OmolStateMachine.from_reconstruction_state(source).branch(
                    None, omol=pybel.Molecule(ob.OBMol(source.omol.OBMol))
                )
                machine.run_omol_charge_stage(
                    "explore_terminal_hydrazonido", _reduce_hydrazonido, *site
                )
                pool.append(machine.freeze_like(source))
    for first, second, kind in hydrazido_sites(state.omol.OBMol):
        if kind != (-1, 1, 2) or not supports_hydrazido(first, second, metals[0]):
            continue
        for source in tuple(pool):
            if len(pool) >= 256:
                return pool
            if source.given_charge > -2:
                continue
            machine = OmolStateMachine.from_reconstruction_state(source).branch(
                "branch_terminal_hydrazido_seed", omol=pybel.Molecule(ob.OBMol(source.omol.OBMol))
            )
            machine.run_omol_charge_stage(
                "explore_terminal_hydrazido", _reduce, first.GetIdx(), second.GetIdx()
            )
            pool.append(machine.freeze_like(source))
    return pool


def _reduce(
    mol: pybel.Molecule, charge: int, first_idx: int, second_idx: int
) -> tuple[pybel.Molecule, int, bool]:
    """Convert the charge-separated N=N fragment to a dianionic terminal N with a single N–N bond."""
    first, second = (
        cast(ob.OBAtom, mol.OBMol.GetAtom(first_idx)),
        cast(ob.OBAtom, mol.OBMol.GetAtom(second_idx)),
    )
    first.GetBond(second).SetBondOrder(1)
    first.SetFormalCharge(-2)
    second.SetFormalCharge(0)
    return mol, charge + 2, True


def expand_coordination_seeds(
    state: ReconstructionState, metals: Sequence[MetalSite], config: MetalScoringConfig
) -> list[ReconstructionState]:
    """Keep the ordinary seed and compose geometry-supported ligand occupations."""
    if not metals:
        return [state]
    seeds = [state]
    mol = cast(ob.OBMol, state.omol.OBMol)
    # Only terminal NO can acquire this bond order. Avoid a full visibility
    # calculation for the common case without a terminal N–O fragment.
    terminal_atoms = tuple(atom for atom in mol_atoms(mol) if atom.GetExplicitDegree() == 1)
    if any(
        atom.GetAtomicNum() == 7 and next(atom_neighbors(atom)).GetAtomicNum() == 8
        for atom in terminal_atoms
    ):
        blockers = build_coordination_blocker_arrays(mol, metal_scoring_config=config)
        visible = [
            inner_visible_atoms_to_metal(
                terminal_atoms, site, metal_scoring_config=config, blocker_arrays=blockers
            )
            for site in metals
        ]
        contacts = list(terminal_nitrosyl_contacts(mol, metals, visible))
        if contacts:
            alternative = ob.OBMol(mol)
            remaining_charge = state.given_charge
            for nitrogen, oxygen, _ in contacts:
                n = cast(ob.OBAtom, alternative.GetAtom(nitrogen.GetIdx()))
                o = cast(ob.OBAtom, alternative.GetAtom(oxygen.GetIdx()))
                remaining_charge += n.GetFormalCharge() + o.GetFormalCharge() - 1
                n.SetFormalCharge(0)
                o.SetFormalCharge(1)
                n.GetBond(o).SetBondOrder(3)
                for atom in (n, o):
                    set_unpaired_electron_count(atom, 0)
                    set_lone_pair_count(atom, 0)
                    set_unresolved_two_electron_center(atom, False)
                require_nitrosonium(n, o)
            branch = OmolStateMachine.from_reconstruction_state(state).branch(
                "geometry_nitrosonium_seed",
                omol=pybel.Molecule(alternative),
                given_charge=remaining_charge,
            )
            seeds.append(branch.freeze_like(state))
    return [alternative for seed in seeds for alternative in expand_hydrazido_seeds(seed, metals)]
