"""Utility helpers for preparing metal-aware fallback inputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import List, cast

from openbabel import openbabel as ob
from openbabel import pybel

from molgr.config import MolGRConfig
from molgr.fallback.state import MetalPreparationState
from molgr.fallback.utils import consts, dataclasses
from molgr.fallback.utils.electrons import set_unpaired_electron_count
from molgr.fallback.utils.metal_radical_inference import infer_metal_radical_counts
from molgr.fallback.utils.ob_graph import mol_atoms
from molgr.fallback.utils.xyz import read_xyz_seed


def _build_metal_states(
    obatom: ob.OBAtom,
    *,
    config: MolGRConfig | None = None,
) -> List[dataclasses.MetalAtomPosition]:
    symbol = ob.GetSymbol(obatom.GetAtomicNum())

    def _default_state() -> dataclasses.MetalAtomPosition:
        return dataclasses.MetalAtomPosition(
            idx=obatom.GetIdx(),
            symbol=symbol,
            element_idx=obatom.GetAtomicNum(),
            valence=0,
            radical_num=0,
            position_x=obatom.GetX(),
            position_y=obatom.GetY(),
            position_z=obatom.GetZ(),
        )

    prior = consts.METAL_VALENCE_AVAILABLE_PRIOR.get(symbol, [])
    minor = consts.METAL_VALENCE_AVAILABLE_MINOR.get(symbol, [])
    seen_valences = set()
    valences: List[int] = []
    for valence in prior + minor:
        if valence in seen_valences:
            continue
        seen_valences.add(valence)
        valences.append(valence)
    if not valences:
        valences = [0]

    if symbol not in consts.METAL_F_D_S_P_ELECTRONS:
        return [_default_state()]

    states: List[dataclasses.MetalAtomPosition] = []
    for valence in valences:
        radicals = infer_metal_radical_counts(obatom, valence, config=config)
        for radical_num in radicals:
            states.append(
                dataclasses.MetalAtomPosition(
                    idx=obatom.GetIdx(),
                    symbol=symbol,
                    element_idx=obatom.GetAtomicNum(),
                    valence=valence,
                    radical_num=radical_num,
                    position_x=obatom.GetX(),
                    position_y=obatom.GetY(),
                    position_z=obatom.GetZ(),
                )
            )

    if not states:
        return [_default_state()]
    return states


def combine_metal_with_omol(
    omol: pybel.Molecule,
    metal_list: Sequence[dataclasses.MetalAtomPosition],
) -> pybel.Molecule:
    """Insert selected metal charge/spin states into the no-metal winner.

    ``metal.radical_num`` is copied only to the real unpaired-electron field.
    Metals receive no active-lone-pair or unresolved-center label; those concepts
    are not inferred during metal-state reinsertion. Organic electron fields are
    preserved by cloning the no-metal molecule.
    """

    obmol = cast(ob.OBMol, omol.clone.OBMol)
    obmol.BeginModify()
    try:
        num_organic = obmol.NumAtoms()
        num_metals = len(metal_list)
        total_atoms = num_organic + num_metals

        for metal in metal_list:
            atom = cast(ob.OBAtom, obmol.NewAtom())
            atom.SetAtomicNum(metal.element_idx)
            atom.SetFormalCharge(metal.valence)
            set_unpaired_electron_count(atom, metal.radical_num)
            atom.SetVector(metal.position_x, metal.position_y, metal.position_z)

        new_order = [0] * total_atoms
        has_error = False
        for i, metal in enumerate(metal_list):
            current_idx = num_organic + 1 + i
            target_slot = metal.idx - 1
            if target_slot < 0 or target_slot >= total_atoms:
                has_error = True
                continue
            if new_order[target_slot] != 0:
                has_error = True
                continue
            new_order[target_slot] = current_idx

        if not has_error:
            current_organic_idx = 1
            for i in range(total_atoms):
                if new_order[i] != 0:
                    continue
                if current_organic_idx > num_organic:
                    has_error = True
                    break
                new_order[i] = current_organic_idx
                current_organic_idx += 1

        if not has_error and all(idx > 0 for idx in new_order):
            obmol.RenumberAtoms(new_order)
    finally:
        obmol.EndModify()
    return pybel.Molecule(obmol)


def prepare_metal_state(
    xyz_block: str,
    total_charge: int,
    total_radical_electrons: int,
    *,
    config: MolGRConfig | None = None,
) -> MetalPreparationState:
    """Split the input into a no-metal XYZ block plus per-metal state options."""

    omol = read_xyz_seed(xyz_block)
    obmol = cast(ob.OBMol, omol.OBMol)
    removable_metal_atoms = [atom for atom in mol_atoms(obmol) if atom.IsMetal()]
    available_valence_radical_states = tuple(
        tuple(_build_metal_states(obatom, config=config)) for obatom in removable_metal_atoms
    )
    if not removable_metal_atoms:
        return MetalPreparationState(
            no_metal_xyz_block=xyz_block,
            available_valence_radical_states=available_valence_radical_states,
            total_charge=total_charge,
            total_radical_electrons=total_radical_electrons,
            phase_history=(
                "read_xyz",
                "build_metal_state_options",
                "preserve_no_metal_xyz",
            ),
            metadata={"metal_atom_count": 0},
        )
    metal_sites = tuple(
        dataclasses.MetalSite(atom.GetAtomicNum(), atom.GetX(), atom.GetY(), atom.GetZ())
        for atom in removable_metal_atoms
    )
    for obatom in removable_metal_atoms:
        obmol.DeleteAtom(obatom)
    return MetalPreparationState(
        no_metal_xyz_block=cast(str, omol.write("xyz")),
        available_valence_radical_states=available_valence_radical_states,
        total_charge=total_charge,
        total_radical_electrons=total_radical_electrons,
        phase_history=(
            "read_xyz",
            "build_metal_state_options",
            "remove_metal_atoms",
            "serialize_no_metal_xyz",
        ),
        metadata={"metal_atom_count": len(removable_metal_atoms)},
        metal_sites=metal_sites,
    )


__all__ = [
    "_build_metal_states",
    "combine_metal_with_omol",
    "prepare_metal_state",
]
