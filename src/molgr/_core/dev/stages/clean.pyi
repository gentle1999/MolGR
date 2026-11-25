"""
Fallback-aligned clean stage helpers
"""

from __future__ import annotations

import typing

__all__: list[str] = [
    "clean_carbene_neighbor_unsaturated_ptr",
    "clean_1_4_radicals_ptr",
    "clean_1_6_radicals_ptr",
    "clean_neighbor_radicals_ptr",
    "clean_possible_1_3_dipole_ptr",
    "clean_resonances_14_ptr",
    "clean_resonances_16_ptr",
    "clean_resonances_17_ptr",
    "clean_resonances_18_ptr",
    "clean_resonances_ptr",
]

def clean_carbene_neighbor_unsaturated_ptr(
    mol_ptr: typing.SupportsInt | typing.SupportsIndex,
) -> bool:
    """
    Move an unresolved two-electron center through an adjacent pi bond.
    """

def clean_neighbor_radicals_ptr(
    mol_ptr: typing.SupportsInt | typing.SupportsIndex,
    given_charge: typing.SupportsInt | typing.SupportsIndex,
    total_radical_electrons: typing.SupportsInt | typing.SupportsIndex,
) -> bool:
    """Pair adjacent excess radicals without consuming reserved electrons."""

def clean_1_4_radicals_ptr(
    mol_ptr: typing.SupportsInt | typing.SupportsIndex,
    given_charge: typing.SupportsInt | typing.SupportsIndex,
    total_radical_electrons: typing.SupportsInt | typing.SupportsIndex,
) -> bool:
    """Resolve separated terminal radicals or unresolved centers in A-B=C-D."""

def clean_1_6_radicals_ptr(
    mol_ptr: typing.SupportsInt | typing.SupportsIndex,
    given_charge: typing.SupportsInt | typing.SupportsIndex,
    total_radical_electrons: typing.SupportsInt | typing.SupportsIndex,
) -> bool:
    """Resolve separated endpoint states in A-B=C-D=E-F."""

def clean_possible_1_3_dipole_ptr(
    mol_ptr: typing.SupportsInt | typing.SupportsIndex,
    given_charge: typing.SupportsInt | typing.SupportsIndex,
    total_radical_electrons: typing.SupportsInt | typing.SupportsIndex,
) -> bool:
    """Convert an eligible excess-radical fragment into a neutral 1,3-dipole."""

def clean_resonances_ptr(mol_ptr: typing.SupportsInt | typing.SupportsIndex) -> bool:
    """
    Run the ordered resonance normalization rule set after candidate generation.
    """

def clean_resonances_14_ptr(mol_ptr: typing.SupportsInt | typing.SupportsIndex) -> bool:
    """Neutralize the rule-14 triple-bond ion pair and rebuild both endpoints."""

def clean_resonances_16_ptr(mol_ptr: typing.SupportsInt | typing.SupportsIndex) -> bool:
    """Neutralize the rule-16 conjugated ion pair and rebuild both endpoints."""

def clean_resonances_17_ptr(mol_ptr: typing.SupportsInt | typing.SupportsIndex) -> bool:
    """Replace a ring allene with its charge-shifted conjugated resonance form."""

def clean_resonances_18_ptr(mol_ptr: typing.SupportsInt | typing.SupportsIndex) -> bool:
    """Convert an unresolved terminal diazene fragment to a charge-separated azide."""
