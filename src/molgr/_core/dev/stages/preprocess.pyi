"""
Fallback-aligned preprocess stage helpers
"""

from __future__ import annotations

import typing

__all__: list[str] = ["make_connections_ptr", "pre_clean_ptr", "validate_omol_ptr"]

def make_connections_ptr(
    mol_ptr: typing.SupportsInt | typing.SupportsIndex,
    extra_tolerance_angstrom: typing.SupportsFloat | typing.SupportsIndex = 0.15,
) -> bool:
    """
    Remove overlong seed bonds and reconnect partners using their updated valences.

    Args:
        mol_ptr: int address of OpenBabel::OBMol
        extra_tolerance_angstrom: additive distance tolerance in Angstrom
    """

def pre_clean_ptr(mol_ptr: typing.SupportsInt | typing.SupportsIndex) -> bool:
    """
    Normalize labile seed bonds and charge-separated groups before assigning electrons.
    """

def validate_omol_ptr(
    mol_ptr: typing.SupportsInt | typing.SupportsIndex,
    total_charge: typing.SupportsInt | typing.SupportsIndex,
    total_radical: typing.SupportsInt | typing.SupportsIndex,
) -> bool:
    """
    Validate conservation of total charge and radical electrons.

    Args:
        mol_ptr: int address of OpenBabel::OBMol
        total_charge: expected total formal charge
        total_radical: expected total radical electrons
    """
