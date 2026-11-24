"""
Fallback-aligned fresh stage helpers
"""

from __future__ import annotations

import typing

__all__: list[str] = [
    "assign_charge_radical_for_atom_ptr",
    "assign_radical_dots_ptr",
    "fresh_omol_charge_radical_ptr",
]

def assign_charge_radical_for_atom_ptr(
    mol_ptr: typing.SupportsInt | typing.SupportsIndex,
    atom_idx: typing.SupportsInt | typing.SupportsIndex,
) -> bool:
    """
    Assign charge/radical state for a specific atom on an existing OBMol.

    Args:
        mol_ptr: int address of OpenBabel::OBMol
        atom_idx: 1-based atom index
    """

def assign_radical_dots_ptr(
    mol_ptr: typing.SupportsInt | typing.SupportsIndex,
    atom_idx: typing.SupportsInt | typing.SupportsIndex,
) -> int:
    """
    Assign radical dots for a specific atom on an existing OBMol.

    Args:
        mol_ptr: int address of OpenBabel::OBMol
        atom_idx: 1-based atom index
    """

def fresh_omol_charge_radical_ptr(mol_ptr: typing.SupportsInt | typing.SupportsIndex) -> bool:
    """
    Apply local charge/electron rebuilding independently to every atom.
    """
