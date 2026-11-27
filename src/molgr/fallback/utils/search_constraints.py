"""Molecule-local constraints for geometry-supported ligand seeds."""

from typing import Optional, cast

from openbabel import openbabel as ob

from molgr.fallback.utils.electrons import (
    _get_int_data,
    _set_int_data,
    get_unpaired_electron_count,
    has_unresolved_two_electron_center,
)
from molgr.fallback.utils.ob_graph import indexed_atoms


REQUIRED_NITROSONIUM_OXYGEN = "MOLGR_REQUIRED_NITROSONIUM_OXYGEN"


def require_nitrosonium(nitrogen: ob.OBAtom, oxygen: ob.OBAtom) -> None:
    """Carry an approved NO+ hypothesis through copies of a ligand seed."""
    _set_int_data(nitrogen, REQUIRED_NITROSONIUM_OXYGEN, oxygen.GetIdx())


def search_constraints_satisfied(mol: ob.OBMol) -> bool:
    """Check final candidates only; intermediate electronic states may differ."""
    for nitrogen in indexed_atoms(mol):
        index = _get_int_data(nitrogen, REQUIRED_NITROSONIUM_OXYGEN)
        if index == 0:
            continue
        if not 0 < index <= mol.NumAtoms():
            return False
        oxygen = cast(ob.OBAtom, mol.GetAtom(index))
        bond = cast(Optional[ob.OBBond], nitrogen.GetBond(oxygen))
        if (
            nitrogen.GetAtomicNum() != 7
            or oxygen.GetAtomicNum() != 8
            or nitrogen.GetExplicitDegree() != 1
            or oxygen.GetExplicitDegree() != 1
            or nitrogen.GetFormalCharge() != 0
            or oxygen.GetFormalCharge() != 1
            or bond is None
            or bond.GetBondOrder() != 3
            or any(
                atom.GetImplicitHCount()
                or get_unpaired_electron_count(atom)
                or has_unresolved_two_electron_center(atom)
                for atom in (nitrogen, oxygen)
            )
        ):
            return False
    return True
