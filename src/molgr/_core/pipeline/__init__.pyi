"""
Pipeline-level helpers
"""

from __future__ import annotations

import typing

import molgr._core.utils
import molgr.config
from molgr._native_types import NativeDiagnostics as _NativeDiagnostics

from . import reconstruct_with_metals, reconstruct_without_metals

__all__: list[str] = [
    "get_last_reconstruction_diagnostics",
    "get_last_run_timing_breakdown_ms",
    "reconstruct_with_metals",
    "reconstruct_without_metals",
    "xyz2omol",
]

def get_last_reconstruction_diagnostics() -> _NativeDiagnostics:
    """Return structured diagnostics from the most recent C++ failure."""

def get_last_run_timing_breakdown_ms() -> dict[str, float | int]:
    """
    Return timing breakdown (milliseconds) for the most recent reconstruction run.
    """

def xyz2omol(
    xyz_block: str,
    total_charge: typing.SupportsInt | typing.SupportsIndex = 0,
    total_radical_electrons: typing.SupportsInt | typing.SupportsIndex = 0,
    *,
    config: molgr.config.MolGRConfig | None = None,
) -> molgr._core.utils.MoleculeData | None:
    """
    Reconstruct molecule data from XYZ with metal-aware pipeline.
    """
