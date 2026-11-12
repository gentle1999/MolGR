"""
Author: TMJ
Date: 2026-02-21 23:08:39
LastEditors: TMJ
LastEditTime: 2026-04-28 23:25:31
Description: 请填写简介
"""

from __future__ import annotations  # noqa: I001

from typing import Literal

# Keep RDKit ahead of Open Babel: importing pybel first segfaults on cp313 manylinux.
# isort: off
from rdkit import Chem
from openbabel import pybel
# isort: on

from molgr.config import CONFIG, MolGRConfig
from molgr.diagnostics import (
    ReconstructionDiagnostics,
    ReconstructionError,
    ReconstructionFailureCode,
)
from molgr.fallback import xyz2omol
from molgr.process_guard import ensure_current_process
from molgr.utils.converter import mol_data_to_rdkit, pybel_to_rdmol, refresh_rdkit_aromaticity
from molgr.utils.post_process import make_dative_bond
from molgr.utils.post_process import make_stereochemistry as restore_stereochemistry
from molgr.utils.smiles import (
    SmilesRoundtripError,
    normalize_smiles_hydrogens,
    validate_smiles_roundtrip,
)

from . import _core as core


def _suspicious_rdmol_from_input_xyz(
    xyz_block: str,
    diagnostics: ReconstructionDiagnostics | None = None,
) -> Chem.Mol:
    """Build an untrusted RDKit molecule from OpenBabel's initial bond perception."""

    omol = pybel.readstring("xyz", xyz_block)
    omol.OBMol.ConnectTheDots()
    omol.OBMol.PerceiveBondOrders()
    rdmol = pybel_to_rdmol(omol, sanitize=False, kekulize=False)
    rdmol.SetProp("_MolGRReconstructionStatus", "suspicious_fallback")
    if diagnostics is not None:
        rdmol.SetProp("_MolGRReconstructionFailureCode", diagnostics.code.value)
        rdmol.SetProp("_MolGRReconstructionFailureStage", diagnostics.stage)
        rdmol.SetProp("_MolGRReconstructionDiagnostics", diagnostics.as_json())
    return rdmol


def _should_return_suspicious_on_reconstruction_failure(config: MolGRConfig) -> bool:
    policy = config.interface.reconstruction_failure_policy
    if policy == "return_suspicious":
        return True
    if policy == "raise":
        return False
    raise ValueError(f"Unknown reconstruction failure policy: {policy!r}")


def _handle_reconstruction_failure(
    xyz_block: str,
    *,
    config: MolGRConfig,
    diagnostics: ReconstructionDiagnostics | None = None,
) -> Chem.Mol:
    resolved_diagnostics = diagnostics or ReconstructionDiagnostics(
        code=ReconstructionFailureCode.NO_VALID_RECONSTRUCTION,
        stage="reconstruction",
        backend="unknown",
        message="The reconstruction backend returned no molecule.",
    )
    if _should_return_suspicious_on_reconstruction_failure(config):
        return _suspicious_rdmol_from_input_xyz(xyz_block, resolved_diagnostics)
    raise ReconstructionError(resolved_diagnostics)


def _diagnostics_from_exception(exc: BaseException, *, backend: str) -> ReconstructionDiagnostics:
    if isinstance(exc, ReconstructionError):
        return exc.diagnostics
    return ReconstructionDiagnostics(
        code=ReconstructionFailureCode.BACKEND_EXCEPTION,
        stage="reconstruction",
        backend=backend,
        message=f"The {backend} reconstruction backend raised an exception.",
        cause_type=type(exc).__name__,
        cause_message=str(exc),
    )


def _cpp_reconstruction_diagnostics() -> ReconstructionDiagnostics | None:
    getter = getattr(core.pipeline, "get_last_reconstruction_diagnostics", None)
    if getter is None:
        return None
    try:
        raw = getter()
    except Exception:
        return None
    if not raw:
        return None
    return ReconstructionDiagnostics.from_mapping(raw, backend="cpp")


def _validate_spin_multiplicity(
    xyz_block: str,
    total_charge: int,
    spin_multiplicity: int,
) -> None:
    """Reject spin states that are impossible for the input electron count."""

    if spin_multiplicity < 1:
        raise ValueError("spin_multiplicity must be >= 1")

    omol = pybel.readstring("xyz", xyz_block)
    total_electrons = sum(int(atom.atomicnum) for atom in omol.atoms) - total_charge
    if total_electrons < 0:
        raise ValueError(
            f"total_charge={total_charge} leaves a negative total electron count "
            f"({total_electrons})"
        )
    if spin_multiplicity > total_electrons + 1:
        raise ValueError(
            f"spin_multiplicity={spin_multiplicity} is impossible for "
            f"{total_electrons} total electrons; the maximum is {total_electrons + 1}"
        )
    if total_electrons % 2 != (spin_multiplicity - 1) % 2:
        required_parity = "odd" if total_electrons % 2 == 0 else "even"
        raise ValueError(
            f"spin_multiplicity={spin_multiplicity} is impossible for "
            f"{total_electrons} total electrons; the multiplicity must be {required_parity}"
        )


def _finalize_rdmol(
    rdmol: Chem.Mol,
    xyz_block: str,
    *,
    backend: Literal["cpp", "python"],
    make_dative_bonds: bool,
    make_stereochemistry: bool,
    config: MolGRConfig,
) -> Chem.Mol:
    try:
        if make_dative_bonds:
            rdmol = make_dative_bond(rdmol, config=config)

        for atom_idx in range(rdmol.GetNumAtoms()):
            rd_atom = rdmol.GetAtomWithIdx(atom_idx)
            rd_atom.SetNoImplicit(True)
        if make_stereochemistry:
            rdmol = restore_stereochemistry(rdmol)
        else:
            refresh_rdkit_aromaticity(rdmol)
        normalize_smiles_hydrogens(rdmol)
        if config.interface.require_smiles_roundtrip:
            # Stereo belongs to the input coordinates. Toolkit SMILES stereo
            # limitations must not reject or rewrite those assignments.
            validate_smiles_roundtrip(rdmol)
    except Exception as exc:
        roundtrip_failed = isinstance(exc, SmilesRoundtripError)
        diagnostics = ReconstructionDiagnostics(
            code=(
                ReconstructionFailureCode.SMILES_ROUNDTRIP_FAILED
                if roundtrip_failed
                else ReconstructionFailureCode.RDKIT_POSTPROCESS_FAILED
            ),
            stage="rdkit.smiles_roundtrip" if roundtrip_failed else "rdkit.postprocess",
            backend=backend,
            message=(
                "The reconstructed molecule cannot round trip through RDKit SMILES with explicit hydrogens retained."
                if roundtrip_failed
                else "RDKit post-processing failed after graph reconstruction."
            ),
            details=exc.details if roundtrip_failed else {},
            cause_type=type(exc).__name__,
            cause_message=str(exc),
        )
        if _should_return_suspicious_on_reconstruction_failure(config):
            return _suspicious_rdmol_from_input_xyz(xyz_block, diagnostics)
        raise ReconstructionError(diagnostics) from exc
    return rdmol


def xyz_to_rdmol(
    xyz_block: str,
    total_charge: int = 0,
    spin_multiplicity: int = 1,
    *,
    backend: Literal["cpp", "python"] = "cpp",
    make_dative_bonds: bool = True,
    make_stereochemistry: bool = True,
    config: MolGRConfig | None = None,
) -> Chem.Mol:
    """
    Convert XYZ block to RDKit Mol.
    """
    ensure_current_process("molgr.xyz_to_rdmol")
    resolved_config = CONFIG if config is None else config
    try:
        _validate_spin_multiplicity(xyz_block, total_charge, spin_multiplicity)
    except OSError as exc:
        diagnostics = ReconstructionDiagnostics(
            code=ReconstructionFailureCode.INVALID_XYZ,
            stage="input.parse",
            backend=backend,
            message="The XYZ block could not be parsed by Open Babel.",
            cause_type=type(exc).__name__,
            cause_message=str(exc),
        )
        raise ReconstructionError(diagnostics) from exc
    except ValueError as exc:
        diagnostics = ReconstructionDiagnostics(
            code=ReconstructionFailureCode.INVALID_ELECTRONIC_TARGET,
            stage="input.electronic_state",
            backend=backend,
            message=str(exc),
        )
        raise ReconstructionError(diagnostics) from exc
    total_radical_electrons = spin_multiplicity - 1
    if backend == "cpp":
        try:
            moldata = core.pipeline.reconstruct_with_metals.xyz2omol(
                xyz_block,
                total_charge,
                total_radical_electrons,
                config=resolved_config,
            )
        except Exception as exc:
            diagnostics = _diagnostics_from_exception(exc, backend="cpp")
            if _should_return_suspicious_on_reconstruction_failure(resolved_config):
                return _suspicious_rdmol_from_input_xyz(xyz_block, diagnostics)
            raise ReconstructionError(diagnostics) from exc
        if moldata is None:
            return _handle_reconstruction_failure(
                xyz_block,
                config=resolved_config,
                diagnostics=_cpp_reconstruction_diagnostics(),
            )
        try:
            rdmol = mol_data_to_rdkit(moldata)
        except Exception as exc:
            diagnostics = ReconstructionDiagnostics(
                code=ReconstructionFailureCode.RDKIT_POSTPROCESS_FAILED,
                stage="rdkit.conversion",
                backend="cpp",
                message="The C++ molecule could not be converted to an RDKit molecule.",
                cause_type=type(exc).__name__,
                cause_message=str(exc),
            )
            if _should_return_suspicious_on_reconstruction_failure(resolved_config):
                return _suspicious_rdmol_from_input_xyz(xyz_block, diagnostics)
            raise ReconstructionError(diagnostics) from exc
    elif backend == "python":
        diagnostic_holder: dict[str, object] = {}
        try:
            omol = xyz2omol(
                xyz_block,
                total_charge,
                total_radical_electrons,
                config=resolved_config,
                _diagnostics=diagnostic_holder,
            )
        except Exception as exc:
            diagnostics = (
                ReconstructionDiagnostics.from_mapping(
                    diagnostic_holder,
                    backend="python",
                )
                if diagnostic_holder
                else _diagnostics_from_exception(exc, backend="python")
            )
            if _should_return_suspicious_on_reconstruction_failure(resolved_config):
                return _suspicious_rdmol_from_input_xyz(xyz_block, diagnostics)
            raise ReconstructionError(diagnostics) from exc
        if omol is None:
            diagnostics = ReconstructionDiagnostics.from_mapping(
                diagnostic_holder,
                backend="python",
            )
            return _handle_reconstruction_failure(
                xyz_block,
                config=resolved_config,
                diagnostics=diagnostics,
            )
        try:
            rdmol = pybel_to_rdmol(omol)
        except Exception as exc:
            diagnostics = ReconstructionDiagnostics(
                code=ReconstructionFailureCode.RDKIT_POSTPROCESS_FAILED,
                stage="rdkit.conversion",
                backend="python",
                message="The Python molecule could not be converted to an RDKit molecule.",
                cause_type=type(exc).__name__,
                cause_message=str(exc),
            )
            if _should_return_suspicious_on_reconstruction_failure(resolved_config):
                return _suspicious_rdmol_from_input_xyz(xyz_block, diagnostics)
            raise ReconstructionError(diagnostics) from exc
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return _finalize_rdmol(
        rdmol,
        xyz_block,
        backend=backend,
        make_dative_bonds=make_dative_bonds,
        make_stereochemistry=make_stereochemistry,
        config=resolved_config,
    )
