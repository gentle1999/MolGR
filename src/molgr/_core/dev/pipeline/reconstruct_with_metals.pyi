"""
Development-only helpers for metal reconstruction internals
"""

from __future__ import annotations

import collections.abc
import typing

import molgr.config

__all__: list[str] = [
    "MetalAtomPosition",
    "build_metal_states_ptr",
    "combine_metal_with_omol_ptr",
    "debug_metal_search_summaries",
    "debug_ring_allene_resonance_ptr",
    "debug_scored_candidate_summaries",
]

class MetalAtomPosition:
    """A candidate metal charge and unpaired-electron count at the original one-based atom index and Cartesian position."""

    symbol: str
    """Chemical element symbol."""
    def __init__(self) -> None: ...
    def __repr__(self) -> str:
        """
        Return a concise debug representation of MetalAtomPosition.
        """
    @property
    def element_idx(self) -> int:
        """Atomic number of the metal."""
    @element_idx.setter
    def element_idx(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None: ...
    @property
    def idx(self) -> int:
        """One-based index of the metal in the original molecule."""
    @idx.setter
    def idx(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None: ...
    @property
    def position_x(self) -> float:
        """Cartesian x coordinate in angstroms."""
    @position_x.setter
    def position_x(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None: ...
    @property
    def position_y(self) -> float:
        """Cartesian y coordinate in angstroms."""
    @position_y.setter
    def position_y(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None: ...
    @property
    def position_z(self) -> float:
        """Cartesian z coordinate in angstroms."""
    @position_z.setter
    def position_z(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None: ...
    @property
    def radical_num(self) -> int:
        """Number of unpaired electrons assigned to the metal."""
    @radical_num.setter
    def radical_num(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None: ...
    @property
    def valence(self) -> int:
        """Candidate signed formal charge assigned to the metal."""
    @valence.setter
    def valence(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None: ...

def build_metal_states_ptr(
    mol_ptr: typing.SupportsInt | typing.SupportsIndex,
    atom_idx: typing.SupportsInt | typing.SupportsIndex,
) -> list[MetalAtomPosition]:
    """
    Build candidate metal states for a specific atom index on an OBMol pointer.
    """

def combine_metal_with_omol_ptr(
    mol_ptr: typing.SupportsInt | typing.SupportsIndex,
    metals: collections.abc.Sequence[MetalAtomPosition],
) -> None:
    """
    Combine metal states with an existing OBMol pointer in place.
    """

def debug_metal_search_summaries(
    xyz_block: str,
    total_charge: typing.SupportsInt | typing.SupportsIndex = 0,
    total_radical_electrons: typing.SupportsInt | typing.SupportsIndex = 0,
    *,
    config: molgr.config.MolGRConfig | None = None,
) -> dict:
    """
    Return C++ metal-search groups, layers, and target buckets for parity debugging.
    """

def debug_ring_allene_resonance_ptr(
    mol_ptr: typing.SupportsInt | typing.SupportsIndex,
    atom_index: typing.SupportsInt | typing.SupportsIndex,
) -> bool:
    """Check whether an alternating charge shift restores aromaticity at this ring allene."""

def debug_scored_candidate_summaries(
    xyz_block: str,
    total_charge: typing.SupportsInt | typing.SupportsIndex = 0,
    total_radical_electrons: typing.SupportsInt | typing.SupportsIndex = 0,
    *,
    config: molgr.config.MolGRConfig | None = None,
) -> dict:
    """
    Return scored metal candidates for the first successful search layer.
    """
