"""
Development-only utility helpers
"""

from __future__ import annotations

import typing

import molgr._core.utils
import molgr.config

__all__: list[str] = [
    "compute_organic_topology_metrics_ptr",
    "debug_roundtrip_xyz_block",
    "debug_vendor_uff_ptr",
    "debug_xyz_seed_molecule_data",
    "organic_force_field_energy_ptr",
    "test_deviation_score",
    "test_physchem_penalty",
    "test_symmetry_penalty",
    "test_total_score",
]

def debug_roundtrip_xyz_block(xyz_block: str) -> str:
    """Read and write an XYZ block with the native plugin-free converter."""

def compute_organic_topology_metrics_ptr(
    mol_ptr: typing.SupportsInt | typing.SupportsIndex,
    *,
    config: molgr.config.MolGRConfig | None = None,
) -> dict:
    """
    Compute C++ organic topology metrics for an OBMol pointer.
    """

def debug_vendor_uff_ptr(mol_ptr: typing.SupportsInt | typing.SupportsIndex) -> dict:
    """
    Return MolGR vendor UFF atom types and energy terms for an OBMol pointer.
    """

def debug_xyz_seed_molecule_data(xyz_block: str) -> molgr._core.utils.MoleculeData:
    """
    Return the C++ vendor-perceived seed molecule data for an XYZ block.
    """

def organic_force_field_energy_ptr(
    mol_ptr: typing.SupportsInt | typing.SupportsIndex,
    *,
    config: molgr.config.MolGRConfig | None = None,
) -> float:
    """
    Compute C++ organic force-field energy for an OBMol pointer.
    """

def test_deviation_score(
    xyz_block: str, atom_idx: typing.SupportsInt | typing.SupportsIndex
) -> float:
    """
    Calculate geometry deviation for atom (1-based index) from XYZ (For Testing)
    """

def test_physchem_penalty(smiles: str) -> float:
    """
    Calculate PhysChem penalty from SMILES (For Testing)
    """

def test_symmetry_penalty(smiles: str) -> float:
    """
    Calculate symmetry penalty from SMILES (For Testing)
    """

def test_total_score(xyz_block: str) -> float:
    """
    Calculate total OMolScore from XYZ block (For Testing)
    """
