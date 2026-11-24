"""Streaming batch reconstruction APIs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Literal, NamedTuple, Sequence

from rdkit import Chem

from molgr._native_types import NativeBatchResult
from molgr.config import CONFIG, MolGRConfig
from molgr.diagnostics import (
    ReconstructionDiagnostics,
    ReconstructionError,
    ReconstructionFailureCode,
)
from molgr.interface import (
    _finalize_rdmol,
    _should_return_suspicious_on_reconstruction_failure,
    _suspicious_rdmol_from_input_xyz,
    xyz_to_rdmol,
)
from molgr.process_guard import ensure_current_process
from molgr.utils.converter import mol_data_to_rdkit

from . import _core as core


@dataclass(frozen=True)
class ReconstructionBatchRequest:
    """One XYZ frame and its target electronic state for batch reconstruction.

    Parameters
    ----------
    xyz_block : str
        Full XYZ text with coordinates in angstroms and all hydrogen atoms.
    total_charge : int, default 0
        Net charge of the entire frame in units of the elementary charge.
    spin_multiplicity : int, default 1
        Total spin multiplicity (2S + 1), as in ``xyz_to_rdmol``.
    """

    xyz_block: str
    total_charge: int = 0
    spin_multiplicity: int = 1


class ReconstructionBatchResult(NamedTuple):
    """Outcome of one request from ``iter_xyz_to_rdmol_batch``.

    Attributes
    ----------
    input : ReconstructionBatchRequest
        Normalized request, including the original XYZ text, charge and spin.
    result : Chem.Mol or None
        Reconstructed molecule on success. A failed request can contain a
        marked initial-perception molecule if the failure policy allows it.
    status : ReconstructionDiagnostics or None
        None means success. For a failure, inspect ``status.code``,
        ``status.stage`` and ``status.message``, including when a molecule is present.

    The fields can also be unpacked as ``request, molecule, diagnostics``.
    """

    input: ReconstructionBatchRequest
    result: Chem.Mol | None
    status: ReconstructionDiagnostics | None = None

    @property
    def request(self) -> ReconstructionBatchRequest:
        """Compatibility alias for the first tuple element."""

        return self.input

    @property
    def molecule(self) -> Chem.Mol | None:
        """Compatibility alias for the reconstructed molecule."""

        return self.result

    @property
    def diagnostics(self) -> ReconstructionDiagnostics | None:
        """Compatibility alias for the item status."""

        return self.status

    def as_pair(self) -> tuple[ReconstructionBatchRequest, Chem.Mol | None]:
        """Return the input request and reconstructed molecule as a pair."""

        return self.input, self.result

    def as_dict(self) -> dict[str, object]:
        """Return a self-describing representation for unstructured consumers."""

        return {
            "input": self.input,
            "result": self.result,
            "status": self.status,
        }


def _normalize_request(
    value: ReconstructionBatchRequest | Sequence[Any],
) -> ReconstructionBatchRequest:
    if isinstance(value, ReconstructionBatchRequest):
        return value
    if len(value) != 3:
        raise ValueError(
            "each batch request must contain exactly three values: "
            "xyz_block, total_charge, spin_multiplicity"
        )
    return ReconstructionBatchRequest(
        xyz_block=str(value[0]),
        total_charge=int(value[1]),
        spin_multiplicity=int(value[2]),
    )


def _diagnostics_from_suspicious_mol(
    molecule: Chem.Mol,
    *,
    backend: Literal["cpp", "python"],
) -> ReconstructionDiagnostics:
    try:
        raw = json.loads(molecule.GetProp("_MolGRReconstructionDiagnostics"))
    except Exception:
        return ReconstructionDiagnostics(
            code=ReconstructionFailureCode.NO_VALID_RECONSTRUCTION,
            stage="reconstruction",
            backend=backend,
            message="The reconstruction backend returned a suspicious molecule without diagnostics.",
        )
    return ReconstructionDiagnostics.from_mapping(raw, backend=backend)


def _batch_result_from_native(
    raw: NativeBatchResult,
    request: ReconstructionBatchRequest,
    *,
    config: MolGRConfig,
    make_dative_bonds: bool,
    make_stereochemistry: bool,
    kekulize: bool,
    raise_on_error: bool,
) -> ReconstructionBatchResult:
    molecule_data = raw.get("molecule_data")
    raw_diagnostics = raw.get("diagnostics")
    diagnostics = (
        ReconstructionDiagnostics.from_mapping(raw_diagnostics, backend="cpp")
        if isinstance(raw_diagnostics, dict) and raw_diagnostics
        else None
    )
    if molecule_data is None:
        diagnostics = diagnostics or ReconstructionDiagnostics(
            code=ReconstructionFailureCode.NO_VALID_RECONSTRUCTION,
            stage="reconstruction",
            backend="cpp",
            message="The C++ reconstruction backend returned no molecule.",
        )
        if raise_on_error:
            raise ReconstructionError(diagnostics)
        if _should_return_suspicious_on_reconstruction_failure(config):
            molecule = _suspicious_rdmol_from_input_xyz(request.xyz_block, diagnostics)
            return ReconstructionBatchResult(request, molecule, diagnostics)
        return ReconstructionBatchResult(request, None, diagnostics)

    try:
        molecule = mol_data_to_rdkit(molecule_data)
        molecule = _finalize_rdmol(
            molecule,
            request.xyz_block,
            backend="cpp",
            make_dative_bonds=make_dative_bonds,
            make_stereochemistry=make_stereochemistry,
            kekulize=kekulize,
            config=config,
        )
    except ReconstructionError as exc:
        if raise_on_error:
            raise
        diagnostics = exc.diagnostics
        if _should_return_suspicious_on_reconstruction_failure(config):
            molecule = _suspicious_rdmol_from_input_xyz(request.xyz_block, diagnostics)
            return ReconstructionBatchResult(request, molecule, diagnostics)
        return ReconstructionBatchResult(request, None, diagnostics)
    except Exception as exc:
        diagnostics = ReconstructionDiagnostics(
            code=ReconstructionFailureCode.RDKIT_POSTPROCESS_FAILED,
            stage="rdkit.conversion",
            backend="cpp",
            message="The C++ molecule could not be converted to an RDKit molecule.",
            cause_type=type(exc).__name__,
            cause_message=str(exc),
        )
        if raise_on_error:
            raise ReconstructionError(diagnostics) from exc
        if _should_return_suspicious_on_reconstruction_failure(config):
            molecule = _suspicious_rdmol_from_input_xyz(request.xyz_block, diagnostics)
            return ReconstructionBatchResult(request, molecule, diagnostics)
        return ReconstructionBatchResult(request, None, diagnostics)
    if molecule.HasProp("_MolGRReconstructionDiagnostics"):
        diagnostics = _diagnostics_from_suspicious_mol(molecule, backend="cpp")
        if raise_on_error:
            raise ReconstructionError(diagnostics)
    return ReconstructionBatchResult(request, molecule, diagnostics)


def _iter_cpp_batch(
    requests: list[ReconstructionBatchRequest],
    *,
    config: MolGRConfig,
    max_workers: int | None,
    queue_size: int,
    ordered: bool,
    make_dative_bonds: bool,
    make_stereochemistry: bool,
    kekulize: bool,
    raise_on_error: bool,
) -> Iterator[ReconstructionBatchResult]:
    native_requests = [
        (request.xyz_block, request.total_charge, request.spin_multiplicity - 1)
        for request in requests
    ]
    native_iterator = core.pipeline.reconstruct_with_metals.batch_xyz2omol(
        native_requests,
        config=config,
        max_workers=0 if max_workers is None else max_workers,
        queue_size=queue_size,
        ordered=ordered,
    )
    try:
        for raw in native_iterator:
            request = requests[int(raw["index"])]
            yield _batch_result_from_native(
                raw,
                request,
                config=config,
                make_dative_bonds=make_dative_bonds,
                make_stereochemistry=make_stereochemistry,
                kekulize=kekulize,
                raise_on_error=raise_on_error,
            )
    finally:
        native_iterator.close()


def _iter_python_batch(
    requests: list[ReconstructionBatchRequest],
    *,
    config: MolGRConfig,
    make_dative_bonds: bool,
    make_stereochemistry: bool,
    kekulize: bool,
    raise_on_error: bool,
) -> Iterator[ReconstructionBatchResult]:
    for request in requests:
        diagnostics: ReconstructionDiagnostics | None = None
        try:
            molecule = xyz_to_rdmol(
                request.xyz_block,
                request.total_charge,
                request.spin_multiplicity,
                backend="python",
                make_dative_bonds=make_dative_bonds,
                make_stereochemistry=make_stereochemistry,
                kekulize=kekulize,
                config=config,
            )
        except ReconstructionError as exc:
            if raise_on_error:
                raise
            diagnostics = exc.diagnostics
            if _should_return_suspicious_on_reconstruction_failure(config):
                molecule = _suspicious_rdmol_from_input_xyz(request.xyz_block, diagnostics)
                yield ReconstructionBatchResult(request, molecule, diagnostics)
            else:
                yield ReconstructionBatchResult(request, None, diagnostics)
            continue
        if molecule.HasProp("_MolGRReconstructionDiagnostics"):
            diagnostics = _diagnostics_from_suspicious_mol(molecule, backend="python")
            if raise_on_error:
                raise ReconstructionError(diagnostics)
        yield ReconstructionBatchResult(request, molecule, diagnostics)


def iter_xyz_to_rdmol_batch(
    requests: Iterable[ReconstructionBatchRequest | Sequence[object]],
    *,
    backend: Literal["cpp", "python"] = "cpp",
    max_workers: int | None = None,
    queue_size: int = 16,
    ordered: bool = False,
    make_dative_bonds: bool = True,
    make_stereochemistry: bool = True,
    kekulize: bool = True,
    config: MolGRConfig | None = None,
    raise_on_error: bool = False,
) -> Iterator[ReconstructionBatchResult]:
    r"""Yield an explicit-H molecule and its diagnostics for each XYZ request.

    Parameters
    ----------
    requests : iterable
        A finite iterable of ReconstructionBatchRequest objects or three-element
        sequences ``(xyz_block, total_charge, spin_multiplicity)``. Each block
        contains one frame in angstroms. The iterator collects all requests
        before scheduling work. Submit large datasets in chunks to bound input
        memory.
    backend : {"cpp", "python"}, default "cpp"
        The C++ backend schedules frames in a native worker pool. The Python
        backend reconstructs one molecule at a time.
    max_workers : int, optional
        Maximum number of concurrent C++ workers. None uses the native
        configuration. The Python backend accepts None or 1.
    queue_size : int, default 16
        Positive capacity setting for the native result queue.
    ordered : bool, default False
        With C++, yield in completion order by default or input order when True.
        Each result carries its request. Python yields in input order.
    make_dative_bonds : bool, default True
        Infer additional metal–ligand contacts from the reconstructed graph and
        geometry, as in ``molgr.interface.xyz_to_rdmol``.
    make_stereochemistry : bool, default True
        Assign stereo from each input conformer.
    kekulize : bool, default True
        Use localized integer bond orders. False allows RDKit aromatic bonds.
    config : MolGRConfig, optional
        None uses ``molgr.CONFIG``. The interface failure policy controls
        whether a failed item can carry a marked initial-perception graph.
    raise_on_error : bool, default False
        Yield reconstruction failures with diagnostics by default. True stops
        iteration and raises ReconstructionError on an item failure.

    Yields
    ------
    ReconstructionBatchResult
        One ``(input, result, status)`` triple per completed request. Check
        ``status is None`` before using the molecule as a reconstruction.

    Raises
    ------
    ValueError
        Invalid backend, worker count, queue size or three-element request.
    ReconstructionError
        An item failed and ``raise_on_error=True``.
    RuntimeError
        Iteration began in a forked child after MolGR was imported in the parent.

    Examples
    --------
    >>> from molgr import ReconstructionBatchRequest, iter_xyz_to_rdmol_batch
    >>> chloride = ReconstructionBatchRequest("1\nchloride\nCl 0 0 0\n", -1, 1)
    >>> item = next(iter_xyz_to_rdmol_batch([chloride], max_workers=1))
    >>> item.input == chloride, item.status is None, item.result.GetNumAtoms()
    (True, True, 1)
    """

    ensure_current_process("molgr.iter_xyz_to_rdmol_batch")
    resolved_config = CONFIG if config is None else config
    if queue_size < 1:
        raise ValueError("queue_size must be >= 1")
    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be >= 1 when provided")
    if backend == "python":
        if max_workers not in (None, 1):
            raise ValueError("backend='python' only supports max_workers=1")
        normalized_requests = [_normalize_request(value) for value in requests]
        yield from _iter_python_batch(
            normalized_requests,
            config=resolved_config,
            make_dative_bonds=make_dative_bonds,
            make_stereochemistry=make_stereochemistry,
            kekulize=kekulize,
            raise_on_error=raise_on_error,
        )
        return
    if backend != "cpp":
        raise ValueError(f"Unknown backend: {backend}")
    normalized_requests = [_normalize_request(value) for value in requests]
    yield from _iter_cpp_batch(
        normalized_requests,
        config=resolved_config,
        max_workers=max_workers,
        queue_size=queue_size,
        ordered=ordered,
        make_dative_bonds=make_dative_bonds,
        make_stereochemistry=make_stereochemistry,
        kekulize=kekulize,
        raise_on_error=raise_on_error,
    )


__all__ = [
    "ReconstructionBatchRequest",
    "ReconstructionBatchResult",
    "iter_xyz_to_rdmol_batch",
]
