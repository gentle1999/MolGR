from __future__ import annotations

import sys
from importlib import import_module


# Load the RDKit modules used by xyz2mol_tm before adapters load MolGR's
# native extension. The reverse order crashes MolStandardize initialization
# with the benchmark environment's RDKit 2026.03.1 Linux wheel.
if "molgr._core.pipeline" not in sys.modules and "openbabel._openbabel" not in sys.modules:
    try:
        from rdkit.Chem import rdEHTTools  # noqa: F401
        from rdkit.Chem.MolStandardize import rdMolStandardize  # noqa: F401
    except ImportError:
        # Optional RDKit builds without Hückel support still run other baselines.
        pass

from benchmarks.smiles_xyz_benchmark.methods.base import BenchmarkMethod


METHOD_IDS: tuple[str, ...] = (
    "rdkit_determine_bonds",
    "openbabel_read_xyz",
    "cell2mol_v2",
    "molgr_fallback",
    "molgr_cpp",
    "xyzgraph_cheminf_full",
    "xyz2mol_tm",
)

_METHOD_CLASS_IMPORTS: dict[str, tuple[str, str]] = {
    "xyz2mol_tm": (
        "benchmarks.smiles_xyz_benchmark.methods.xyz2mol_tm",
        "XYZ2MolTMMethod",
    ),
    "rdkit_determine_bonds": (
        "benchmarks.smiles_xyz_benchmark.methods.rdkit_determine_bonds",
        "RDKitDetermineBondsMethod",
    ),
    "openbabel_read_xyz": (
        "benchmarks.smiles_xyz_benchmark.methods.openbabel_read_xyz",
        "OpenBabelReadXYZMethod",
    ),
    "cell2mol_v2": (
        "benchmarks.smiles_xyz_benchmark.methods.cell2mol_v2",
        "Cell2MolV2Method",
    ),
    "molgr_fallback": (
        "benchmarks.smiles_xyz_benchmark.methods.molgr_fallback",
        "MolGRFallbackMethod",
    ),
    "molgr_cpp": (
        "benchmarks.smiles_xyz_benchmark.methods.molgr_cpp",
        "MolGRCppMethod",
    ),
    "xyzgraph_cheminf_full": (
        "benchmarks.smiles_xyz_benchmark.methods.xyzgraph_cheminf_full",
        "XYZGraphCheminfFullMethod",
    ),
}


def get_method_registry(method_ids: tuple[str, ...] | None = None) -> list[BenchmarkMethod]:
    selected_ids = METHOD_IDS if method_ids is None else method_ids
    registry: list[BenchmarkMethod] = []
    for method_id in selected_ids:
        if method_id not in _METHOD_CLASS_IMPORTS:
            continue
        module_name, class_name = _METHOD_CLASS_IMPORTS[method_id]
        method_cls = getattr(import_module(module_name), class_name)
        registry.append(method_cls())
    return registry


__all__ = ["METHOD_IDS", "get_method_registry", "BenchmarkMethod"]
