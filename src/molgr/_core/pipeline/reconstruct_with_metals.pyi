"""
Fallback-aligned reconstruction helpers with metals
"""

from __future__ import annotations

import typing

import molgr._core.utils
import molgr.config
from molgr._native_types import NativeBatchResult as _NativeBatchResult

__all__: list[str] = ["ReconstructionBatchIterator", "batch_xyz2omol", "xyz2omol"]

class ReconstructionBatchIterator(typing.Iterator[_NativeBatchResult]):
    """Stream indexed reconstruction results from the native workers. Close the iterator to stop outstanding work."""
    def __iter__(self) -> ReconstructionBatchIterator:
        """Return this native result iterator."""
    def __next__(self) -> _NativeBatchResult:
        """Yield the input index, molecule data and diagnostics for one request."""
    def close(self) -> None:
        """Stop scheduling requests and wait for active tasks to finish."""

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

def batch_xyz2omol(
    requests: typing.Iterable[tuple[str, int, int]],
    *,
    config: molgr.config.MolGRConfig | None = None,
    max_workers: int = 0,
    queue_size: int = 16,
    ordered: bool = False,
) -> ReconstructionBatchIterator:
    """
    Reconstruct a finite XYZ batch with a bounded native result queue.
    """
