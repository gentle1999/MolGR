"""
Fallback-aligned break_bond stage helpers
"""

from __future__ import annotations

import typing

__all__: list[str] = ["break_deformed_ene_ptr", "break_one_bond_ptr"]

def break_deformed_ene_ptr(
    mol_ptr: typing.SupportsInt | typing.SupportsIndex,
    given_charge: typing.SupportsInt | typing.SupportsIndex = 0,
    given_radical: typing.SupportsInt | typing.SupportsIndex = 0,
    tolerance: typing.SupportsFloat | typing.SupportsIndex = 5.0,
) -> bool:
    """
    Homolytically reduce deformed multiple bonds by one bond-order unit.

    Args:
        mol_ptr: int address of OpenBabel::OBMol
        given_charge: target charge budget used by the stage
        given_radical: target radical budget used by the stage
        tolerance: torsion-angle tolerance threshold
    """

def break_one_bond_ptr(
    mol_ptr: typing.SupportsInt | typing.SupportsIndex,
    given_charge: typing.SupportsInt | typing.SupportsIndex = 0,
    given_radical: typing.SupportsInt | typing.SupportsIndex = 0,
) -> tuple:
    """
    Apply last-resort homolytic and charge-transfer bond-breaking templates.

    Args:
        mol_ptr: int address of OpenBabel::OBMol
        given_charge: charge budget to be updated in place and returned
        given_radical: target radical budget used by the stage
    """
