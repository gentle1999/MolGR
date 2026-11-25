"""Reconstruct one explicit-hydrogen RDKit molecule from Cartesian coordinates.

Import ``xyz_to_rdmol`` from this module for single XYZ frames. For a finite
collection, ``molgr.iter_xyz_to_rdmol_batch`` yields a molecule and a diagnostic
status alongside each request.
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
    """Return Open Babel's initial bond perception for inspection after a failure.

    Parameters
    ----------
    xyz_block : str
        Original XYZ frame, including hydrogens, with coordinates in angstroms.
    diagnostics : ReconstructionDiagnostics, optional
        If supplied, attach the failure code, stage and JSON diagnostic record.

    Returns
    -------
    Chem.Mol
        Graph converted without sanitization or Kekulization, marked with
        ``_MolGRReconstructionStatus="suspicious_fallback"``. Its bond orders and
        electron assignments have not passed MolGR's charge and spin checks.
    """

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
    """Read the single-molecule failure policy from ``config.interface``.

    Return True for ``"return_suspicious"`` and False for ``"raise"``. An
    unrecognized policy raises ValueError when a failure needs to be handled.
    """
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
    """Apply the configured policy when a backend produces no candidate.

    Parameters
    ----------
    xyz_block : str
        Original coordinates used to perceive a fallback graph if requested.
    config : MolGRConfig
        Supplies ``interface.reconstruction_failure_policy``.
    diagnostics : ReconstructionDiagnostics, optional
        Backend explanation. A missing record receives the generic
        ``NO_VALID_RECONSTRUCTION`` code.

    Returns
    -------
    Chem.Mol
        A marked, initial-perception graph under the ``"return_suspicious"`` policy.

    Raises
    ------
    ReconstructionError
        Under the default ``"raise"`` policy. Its ``diagnostics`` carries the reason.
    """
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
    """Recover a backend failure record for use by the public interface.

    Preserve ``exc.diagnostics`` when ``exc`` is a ReconstructionError. For any
    other exception, return a BACKEND_EXCEPTION record with the supplied backend
    name and the original exception's type and message.
    """
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
    """Read the native diagnostic record immediately after an unsuccessful C++ call.

    Return a ReconstructionDiagnostics tagged with the C++ backend, or None
    when the extension has no getter, the getter fails, or no record is present.
    """
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
    """Check whether the requested spin can occur at the specified net charge.

    Parameters
    ----------
    xyz_block : str
        XYZ text supplying every nuclear charge, including those of hydrogen.
    total_charge : int
        Net charge of the system. The electron count is the sum of atomic
        numbers minus this charge.
    spin_multiplicity : int
        Target ``2S + 1``. Subtracting one gives the search's unpaired-electron
        count, which must have the same parity as the total electron count.

    Raises
    ------
    ValueError
        The electron count is negative, the multiplicity is outside
        ``1 .. electron_count + 1``, or the two counts have incompatible parity.
    OSError
        Open Babel could not read the XYZ frame.
    """

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
    kekulize: bool = True,
) -> Chem.Mol:
    """Prepare a reconstructed graph for return to a single or batch caller.

    Parameters
    ----------
    rdmol : Chem.Mol
        Backend graph with explicit atoms and a conformer in XYZ atom order.
    xyz_block : str
        Original XYZ text used to rebuild an initial-perception graph on failure.
    backend : {"cpp", "python"}
        Backend name to include in failure diagnostics.
    make_dative_bonds : bool
        Infer metal coordination contacts before assigning stereo.
    make_stereochemistry : bool
        Perceive stereo from the conformer. Both settings refresh aromaticity.
    config : MolGRConfig
        Controls coordination, SMILES round-trip validation and failure handling.
    kekulize : bool, default True
        Retain localized integer bond orders. False allows RDKit aromatic bonds.

    Returns
    -------
    Chem.Mol
        The finalized graph, or the marked fallback allowed by the configuration.
        This operation can edit the argument in place or return a new molecule.

    Raises
    ------
    ReconstructionError
        RDKit post-processing or the requested SMILES graph check failed and
        the failure policy is ``"raise"``.
    """
    try:
        if make_dative_bonds:
            rdmol = make_dative_bond(rdmol, config=config)

        for atom_idx in range(rdmol.GetNumAtoms()):
            rd_atom = rdmol.GetAtomWithIdx(atom_idx)
            rd_atom.SetNoImplicit(True)
        if make_stereochemistry:
            rdmol = restore_stereochemistry(rdmol, kekulize=kekulize)
        else:
            refresh_rdkit_aromaticity(rdmol, kekulize=kekulize)
        normalize_smiles_hydrogens(rdmol)
        if config.interface.require_smiles_roundtrip:
            # Stereo belongs to the input coordinates. Toolkit SMILES stereo
            # limitations must not reject or rewrite those assignments.
            validate_smiles_roundtrip(rdmol, kekulize=kekulize)
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
            details=exc.details if isinstance(exc, SmilesRoundtripError) else {},
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
    kekulize: bool = True,
    config: MolGRConfig | None = None,
) -> Chem.Mol:
    r"""Reconstruct an explicit-hydrogen RDKit molecule from one XYZ frame.

    Parameters
    ----------
    xyz_block : str
        Contents of one XYZ frame: atom count, comment line, then one
        ``element x y z`` row per atom in angstroms. Include all hydrogens.
        Supply charge and spin as arguments even if they appear in the comment.
    total_charge : int, default 0
        Net charge of the entire system, including metals, ligands and any
        disconnected fragments. Positive values remove electrons.
    spin_multiplicity : int, default 1
        Total spin multiplicity ``2S + 1``. The search assigns
        ``spin_multiplicity - 1`` unpaired electrons (0 for a singlet, 1 for
        a doublet). A singlet target therefore cannot specify a pair of
        antiferromagnetically coupled local radicals.
    backend : {"cpp", "python"}, default "cpp"
        Run the compiled reconstruction pipeline or its Python reference
        implementation. Both return an RDKit molecule.
    make_dative_bonds : bool, default True
        Infer additional metal–ligand contacts using the reconstructed donor
        states and 3D geometry. The explicit ligand bonds orient localized
        lone pairs, for example the open face of a phosphine or the two
        possible donor directions of an ether. A bridging donor can acquire
        contacts to several visible metals. False skips this post-processing.
    make_stereochemistry : bool, default True
        Assign stereochemistry from the supplied coordinates after coordination
        processing. False skips this 3D assignment step.
    kekulize : bool, default True
        Return localized integer bond orders, including alternating single and
        double bonds in aromatic rings. Set False for RDKit aromatic bond types.
    config : MolGRConfig, optional
        None uses the package-level ``molgr.CONFIG``. Pass a fresh
        ``MolGRConfig()`` for per-call settings. Set
        ``config.interface.reconstruction_failure_policy = "return_suspicious"``
        to inspect failed reconstructions. The default
        ``config.interface.require_smiles_roundtrip = True`` validates that
        atom and bond labels survive writing and reading explicit-H SMILES.

    Returns
    -------
    Chem.Mol
        On success, the inferred graph with explicit H atoms, input atom order
        and a conformer carrying the XYZ geometry. With failure policy
        ``"return_suspicious"``, a backend or post-processing failure can instead
        return Open Babel's initial perception. Inspect
        ``mol.HasProp("_MolGRReconstructionDiagnostics")`` to identify it.

    Raises
    ------
    ReconstructionError
        The XYZ input cannot be parsed, or the charge and multiplicity imply
        an impossible electron count or parity. These errors precede fallback.
        With the default failure policy, reconstruction, conversion and
        post-processing failures also raise this exception. Inspect
        ``exc.diagnostics.code`` and ``exc.diagnostics.stage`` for the cause.
    ValueError
        An unknown backend or an unrecognized failure policy was selected.
    RuntimeError
        Called in a forked child after MolGR was imported in the parent.
        Use the multiprocessing ``spawn`` start method in that case.

    Examples
    --------
    >>> from molgr.interface import xyz_to_rdmol
    >>> water = "3\nwater\nO 0 0 0\nH 0.7586 0 0.5043\nH -0.7586 0 0.5043\n"
    >>> mol = xyz_to_rdmol(water)
    >>> mol.GetNumAtoms(), mol.GetNumBonds(), mol.GetConformer().Is3D()
    (3, 2, True)

    Pass the electronic state explicitly for an ion. A chloride ion has 18
    electrons and a singlet ground state.

    >>> from rdkit.Chem.rdmolops import GetFormalCharge
    >>> chloride = xyz_to_rdmol("1\nchloride\nCl 0 0 0\n", total_charge=-1, spin_multiplicity=1)
    >>> GetFormalCharge(chloride)
    -1

    Water has ten electrons. Requesting a doublet fails input validation
    before either backend runs.

    >>> from molgr import ReconstructionError
    >>> try:
    ...     xyz_to_rdmol(water, spin_multiplicity=2)
    ... except ReconstructionError as exc:
    ...     print(exc.diagnostics.code.value, exc.diagnostics.stage)
    INVALID_ELECTRONIC_TARGET input.electronic_state
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
        kekulize=kekulize,
        config=resolved_config,
    )
