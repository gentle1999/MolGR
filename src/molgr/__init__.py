from __future__ import annotations  # noqa: I001

import importlib.metadata
import os
from pathlib import Path

# RDKit must initialize before Open Babel's pybel on cp313 manylinux.
# isort: off
from rdkit import Chem as _rdkit_chem  # noqa: F401
from openbabel import pybel
# isort: on


def _configure_openbabel_data_dir() -> None:
    """Correct the data path used by the Windows Open Babel wheel."""

    if os.name != "nt":
        return

    configured_dir = os.environ.get("BABEL_DATADIR")
    if configured_dir and (Path(configured_dir) / "UFF.prm").is_file():
        return

    candidate = Path(pybel.__file__).resolve().parent / "bin" / "data"
    if (candidate / "UFF.prm").is_file():
        os.environ["BABEL_DATADIR"] = str(candidate)


_configure_openbabel_data_dir()

from . import _core as core  # noqa: E402
from . import config  # noqa: E402
from ._core import pipeline  # noqa: E402
from .batch import (  # noqa: E402
    ReconstructionBatchRequest,
    ReconstructionBatchResult,
    iter_xyz_to_rdmol_batch,
)
from .config import (  # noqa: E402
    CONFIG,
    CppBackendConfig,
    MetalRadicalInferenceConfig,
    MetalScoringConfig,
    MolGRConfig,
    OrganicTopologyConfig,
    PythonInterfaceConfig,
    ReconstructionFailurePolicy,
    ResonanceConfig,
    ResonanceTraversalScore,
)
from .diagnostics import (  # noqa: E402
    RECONSTRUCTION_FAILURE_CODES,
    ReconstructionDiagnosticCollector,
    ReconstructionDiagnostics,
    ReconstructionError,
    ReconstructionFailureCode,
)
from .utils.post_process import make_dative_bond  # noqa: E402
from .utils.smiles import (  # noqa: E402
    SmilesRoundtripError,
    mol_from_smiles,
    validate_smiles_roundtrip,
)


try:
    __version__ = importlib.metadata.version("molgr")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"


pybel.ob.obErrorLog.StopLogging()


def set_log_level(level: core.LogLevel) -> None:
    """
    Set C++ backend logging level.

    Args:
        level: Can be LogLevel.DEBUG, LogLevel.INFO, etc., or int (0-4).
    """
    # Pybind11 enums accept either integers or enum instances.
    core.set_log_level(level)


__all__ = [
    "CONFIG",
    "CppBackendConfig",
    "MetalRadicalInferenceConfig",
    "MetalScoringConfig",
    "MolGRConfig",
    "OrganicTopologyConfig",
    "PythonInterfaceConfig",
    "RECONSTRUCTION_FAILURE_CODES",
    "ReconstructionFailurePolicy",
    "ReconstructionDiagnosticCollector",
    "ReconstructionDiagnostics",
    "ReconstructionError",
    "ReconstructionFailureCode",
    "ReconstructionBatchRequest",
    "ReconstructionBatchResult",
    "ResonanceConfig",
    "ResonanceTraversalScore",
    "config",
    "iter_xyz_to_rdmol_batch",
    "make_dative_bond",
    "mol_from_smiles",
    "set_log_level",
    "SmilesRoundtripError",
    "validate_smiles_roundtrip",
    "pipeline",
]

set_log_level(core.LogLevel.WARN)
