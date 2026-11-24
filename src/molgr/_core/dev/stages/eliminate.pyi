"""
Fallback-aligned eliminate stage helpers
"""

from __future__ import annotations

import typing

__all__: list[str] = [
    "eliminate_1_3_dipole_postive_ptr",
    "eliminate_carbene_neighbor_heteroatom_ptr",
    "eliminate_carboxyl_ptr",
    "eliminate_possible_cp_like_radical_anion_ptr",
    "eliminate_high_positive_charge_atoms_ptr",
    "eliminate_negative_charges_ptr",
    "eliminate_nnn_ptr",
    "eliminate_positive_charges_ptr",
]

def eliminate_1_3_dipole_postive_ptr(
    mol_ptr: typing.SupportsInt | typing.SupportsIndex,
    given_charge: typing.SupportsInt | typing.SupportsIndex,
) -> tuple:
    """
    Use positive-charge budget to close a bond in a 1,3-dipole motif.

    Args:
        mol_ptr: int address of OpenBabel::OBMol
        given_charge: charge deficit to be updated in place and returned
    """

def eliminate_carbene_neighbor_heteroatom_ptr(
    mol_ptr: typing.SupportsInt | typing.SupportsIndex,
    given_charge: typing.SupportsInt | typing.SupportsIndex,
) -> tuple:
    """
    Close an unresolved two-electron center with a heteroatom donor.

    Args:
        mol_ptr: int address of OpenBabel::OBMol
        given_charge: charge deficit to be updated in place and returned
    """

def eliminate_carboxyl_ptr(
    mol_ptr: typing.SupportsInt | typing.SupportsIndex,
    given_charge: typing.SupportsInt | typing.SupportsIndex,
) -> tuple:
    """
    Convert a carboxyl-like oxygen monoradical into an O- center.

    Args:
        mol_ptr: int address of OpenBabel::OBMol
        given_charge: charge deficit to be updated in place and returned
    """

def eliminate_possible_cp_like_radical_anion_ptr(
    mol_ptr: typing.SupportsInt | typing.SupportsIndex,
    given_charge: typing.SupportsInt | typing.SupportsIndex,
    total_radical_electrons: typing.SupportsInt | typing.SupportsIndex,
) -> tuple:
    """
    Localize an excess cyclopentadienyl-like radical as an aromatic anion.

    Args:
        mol_ptr: int address of OpenBabel::OBMol
        given_charge: charge deficit to be updated in place and returned
        total_radical_electrons: target number of unpaired electrons to preserve
    """

def eliminate_high_positive_charge_atoms_ptr(
    mol_ptr: typing.SupportsInt | typing.SupportsIndex,
    given_charge: typing.SupportsInt | typing.SupportsIndex,
) -> tuple:
    """
    Neutralize positive centers from their most electronegative neighbors.

    Args:
        mol_ptr: int address of OpenBabel::OBMol
        given_charge: charge deficit to be updated in place and returned
    """

def eliminate_negative_charges_ptr(
    mol_ptr: typing.SupportsInt | typing.SupportsIndex,
    given_charge: typing.SupportsInt | typing.SupportsIndex,
    total_radical_electrons: typing.SupportsInt | typing.SupportsIndex = -1,
) -> tuple:
    """
    Consume classified electron sources into negative formal charge.

    Args:
        mol_ptr: int address of OpenBabel::OBMol
        given_charge: charge deficit to be updated in place and returned
        total_radical_electrons: optional target number of radical electrons to preserve;
            use -1 to disable target preservation
    """

def eliminate_nnn_ptr(
    mol_ptr: typing.SupportsInt | typing.SupportsIndex,
    given_charge: typing.SupportsInt | typing.SupportsIndex,
    positive: bool = False,
) -> tuple:
    """
    Close N-N-N motifs using explicitly classified endpoint electrons.

    Args:
        mol_ptr: int address of OpenBabel::OBMol
        given_charge: charge deficit to be updated in place and returned
        positive: whether to run positive-direction elimination
    """

def eliminate_positive_charges_ptr(
    mol_ptr: typing.SupportsInt | typing.SupportsIndex,
    given_charge: typing.SupportsInt | typing.SupportsIndex,
) -> tuple:
    """
    Consume classified electron sources into positive formal charge.

    Args:
        mol_ptr: int address of OpenBabel::OBMol
        given_charge: charge deficit to be updated in place and returned
    """
