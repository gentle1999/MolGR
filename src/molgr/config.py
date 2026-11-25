"""Configuration for single and batch XYZ reconstruction.

Pass a fresh ``MolGRConfig()`` as ``config=`` to keep changes local to a call.
When ``config=None``, the public entry points use the shared ``CONFIG``.
Field descriptions are available at runtime through ``dataclasses.fields``.

Examples
--------
>>> from dataclasses import fields
>>> from molgr.config import MolGRConfig
>>> config = MolGRConfig()
>>> config.cpp_backend.max_threads = 1
>>> options = {f.name: f for f in fields(config.metal_radical_inference)}
>>> print(options["linear_angle_min_degrees"].metadata["description"])
Minimum donor–metal–donor angle in degrees for a two-coordinate site to be classified as linear.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Literal, Optional, TypeVar


ResonanceTraversalScore = Literal[
    "uff_lite_gain",
    "input_order",
]
ReconstructionFailurePolicy = Literal[
    "raise",
    "return_suspicious",
]


_T = TypeVar("_T")


def _option(default: _T, description: str) -> _T:
    """Declare a scalar dataclass field with a default and runtime help text."""
    return field(default=default, metadata={"description": description})


@dataclass
class ResonanceConfig:
    """Configure how ligand reconstruction explores alternative electron assignments."""

    max_depth: int = _option(
        3,
        "Maximum number of successive moves along a reconstruction resonance path. Ligand "
        "targets initially try up to two steps and deepen when needed.",
    )

    limited_discrepancy_max_discrepancy: int = _option(
        2,
        "Maximum number of nonpreferred branch choices along a resonance walk. Increasing "
        "it admits more alternatives at the same depth.",
    )

    traversal_score: ResonanceTraversalScore = _option(
        "uff_lite_gain",
        'Order moves by estimated UFF-lite gain ("uff_lite_gain") or keep their '
        'generation order ("input_order"). This selects the preferred branch in the '
        "discrepancy-limited walk.",
    )

    neighbor_radical_max_discrepancy: int = _option(
        2,
        "Maximum neighboring radical pairs per seed resolved by charge separation instead "
        "of bond formation. This budget is independent of the resonance walk. Zero or a "
        "negative value permits no such choices.",
    )


@dataclass
class CppBackendConfig:
    """Set native concurrency and reuse of UFF atom typing during reconstruction.

    A C++ batch with multiple workers parallelizes across molecules and disables
    both kinds of within-molecule parallelism. A single-worker batch honors the
    target-bucket and candidate-scoring settings.
    """

    max_threads: Optional[int] = _option(
        1 if sys.platform == "win32" else None,
        "Cap worker counts chosen automatically by the native backend. None uses "
        "available hardware parallelism, bounded by the number of tasks. The default is 1 "
        "on Windows and None elsewhere.",
    )

    enable_target_bucket_parallelism: bool = _option(
        True,
        "Allow independent ligand charge and radical targets within a molecule to be "
        "reconstructed concurrently when the task threshold is met.",
    )

    enable_candidate_scoring_parallelism: bool = _option(
        False,
        "Allow native scoring of independent metal–ligand candidates to run concurrently "
        "when enough score jobs are available.",
    )

    enable_uff_atom_typing_cache: bool = _option(
        False,
        "Reuse cached Universal Force Field atom types during native scoring. Enable to "
        "reduce repeated typing work.",
    )

    target_bucket_parallel_threshold: int = _option(
        1,
        "Minimum number of ligand target tasks required before native target-bucket "
        "parallelism is considered. Effective minimum is one task.",
    )

    target_bucket_parallel_max_threads: Optional[int] = _option(
        None,
        "Additional worker cap for target buckets within one molecule. None leaves worker "
        "selection to the native thread and task limits.",
    )

    candidate_score_parallel_threshold: int = _option(
        32,
        "Minimum number of score jobs required for parallel candidate scoring when "
        "enabled. Effective minimum is one job.",
    )


@dataclass
class OrganicTopologyConfig:
    """Weight aromatic ring stability and decide when ligand geometry supports conjugation."""

    aromatic_stability_benzene_score: float = _option(
        1.0,
        "Stability contribution of a six-carbon aromatic ring with no formally charged "
        "atoms and no unpaired electrons.",
    )

    aromatic_stability_other_ring_max_score: float = _option(
        0.99,
        "Upper bound on the per-ring stability contribution after multiplying the size, "
        "heteroatom, charge and radical factors for other aromatic rings.",
    )

    aromatic_stability_ring_size_6_factor: float = _option(
        0.92,
        "Base stability multiplier for a six-heavy-atom aromatic ring outside the "
        "neutral, closed-shell all-carbon case.",
    )

    aromatic_stability_ring_size_5_factor: float = _option(
        0.84,
        "Base stability multiplier for a five-heavy-atom aromatic ring before electronic "
        "penalties.",
    )

    aromatic_stability_other_ring_size_factor: float = _option(
        0.76,
        "Base stability multiplier for an aromatic ring with a heavy-atom count other "
        "than five or six.",
    )

    aromatic_stability_hetero_atom_penalty: float = _option(
        0.10,
        "Amount subtracted from the heteroatom factor for each noncarbon heavy atom in an "
        "aromatic ring. The factor starts at one.",
    )

    aromatic_stability_min_hetero_factor: float = _option(
        0.62,
        "Lower bound on the aromatic ring multiplier after the heteroatom penalty.",
    )

    aromatic_stability_formal_charge_penalty: float = _option(
        0.16,
        "Amount subtracted from the charge factor for each ring atom with a nonzero "
        "formal charge. Each charged atom counts once regardless of charge magnitude.",
    )

    aromatic_stability_min_charge_factor: float = _option(
        0.50,
        "Lower bound on the aromatic ring multiplier after the formal-charge penalty.",
    )

    aromatic_stability_radical_penalty: float = _option(
        0.20,
        "Amount subtracted from the radical factor for each ring atom carrying unpaired "
        "electrons. Each radical-bearing atom counts once.",
    )

    aromatic_stability_min_radical_factor: float = _option(
        0.50,
        "Lower bound on the aromatic ring multiplier after the radical penalty.",
    )

    conjugation_normalized_tetrahedron_volume_tolerance: float = _option(
        0.075,
        "Largest dimensionless tetrahedron volume accepted by the donor–acceptor "
        "coplanarity check. Volume is normalized by the cube of the RMS edge length. "
        "Larger values allow more distortion.",
    )

    conjugation_min_plane_overlap: float = _option(
        0.5,
        "Minimum mean squared cosine between observable local pi-plane normals for "
        "conjugation. A value of 0.5 corresponds to 45 degrees for two planar "
        "environments.",
    )


@dataclass
class MetalScoringConfig:
    """Select oxidation and spin assignments, score ligand charge localization, and infer metal contacts."""

    open_shell_multimetal_state_penalty_window: float = _option(
        10.0,
        "Penalty interval used to widen successive metal-state layers in open-shell "
        "multimetal search. Zero advances one distinct penalty at a time. A negative "
        "value opens all options in the first layer.",
    )

    open_shell_multimetal_min_state_options: int = _option(
        6,
        "Minimum number of available state choices admitted per group in the first "
        "open-shell multimetal layer, including ties at the cutoff. Values below one are "
        "treated as one.",
    )

    same_element_multimetal_unify_threshold: int = _option(
        3,
        "When the number of metals of one element exceeds this threshold, group them if "
        "they share valence and radical-count options. A negative value disables "
        "grouping.",
    )

    max_mixed_valence_spread: Optional[int] = _option(
        3,
        "Maximum difference between the highest and lowest assigned valences for metals "
        "of the same element. None or a negative value disables this limit.",
    )

    max_assignments_per_target: int = _option(
        64,
        "Maximum number of ranked metal-state assignments retained for a given charge and "
        "radical target. At least one slot is retained.",
    )

    metal_coordination_extra_tolerance_angstrom: float = _option(
        0.75,
        "Distance in angstroms added to the scaled sum of metal and donor covalent radii "
        "when screening coordination contacts.",
    )

    pi_dative_distance_difference_tolerance_angstrom: float = _option(
        0.10,
        "Largest allowed difference in angstroms between the two metal-to-endpoint "
        "distances when considering a ligand bond as a pi donor during dative-bond "
        "post-processing.",
    )

    metal_access_radius_scale: float = _option(
        1.0,
        "Dimensionless multiplier for the covalent radii in coordination cutoffs and in "
        "the spheres used to test whether a donor is visible from the metal.",
    )

    metal_access_clearance_angstrom: float = _option(
        0.0,
        "Distance in angstroms added to each scaled blocker radius in metal-to-donor "
        "visibility tests. Increasing it makes more donor paths count as obstructed.",
    )

    charge_localization_selection_margin: float = _option(
        0.3,
        "Allowed increase in the organic charge-localization penalty relative to the "
        "reference candidate used in metal-state ranking. Reaching the margin can demote "
        "a candidate. A large oxidation-state jump is checked separately.",
    )


@dataclass
class MetalRadicalInferenceConfig:
    """Infer metal unpaired-electron options from donor geometry and ligand-field strength."""

    max_considered_donors: int = _option(
        6,
        "Maximum number of candidate donors used to characterize a metal site. The "
        "closest donors are considered first.",
    )

    square_planar_planarity_tolerance_angstrom: float = _option(
        0.45,
        "Maximum out-of-plane distance in angstroms for classifying four donors as square "
        "planar around the metal.",
    )

    trigonal_planar_planarity_tolerance_angstrom: float = _option(
        0.35,
        "Maximum out-of-plane distance in angstroms for classifying three donors as "
        "trigonal planar around the metal.",
    )

    linear_angle_min_degrees: float = _option(
        150.0,
        "Minimum donor–metal–donor angle in degrees for a two-coordinate site to be "
        "classified as linear.",
    )

    strong_field_threshold: float = _option(
        1.10,
        "Base ligand-field score for strong-field classification. The score must reach "
        "this threshold plus the ambiguity margin.",
    )

    weak_field_threshold: float = _option(
        0.75,
        "Base ligand-field score for weak-field classification. The score must be at or "
        "below this threshold minus the ambiguity margin.",
    )

    field_ambiguity_margin: float = _option(
        0.10,
        "Buffer in ligand-field score units added to the strong-field threshold and "
        "subtracted from the weak-field threshold. Increasing it widens the ambiguous "
        "interval.",
    )


@dataclass
class PythonInterfaceConfig:
    """Control how reconstruction entry points validate the returned graph and report failures."""

    reconstruction_failure_policy: ReconstructionFailurePolicy = _option(
        "raise",
        'Use "raise" to raise ReconstructionError on single-frame reconstruction failure '
        'or "return_suspicious" to return a marked initial-perception graph. Single-frame '
        "input validation still raises for an impossible charge or spin.",
    )

    require_smiles_roundtrip: bool = _option(
        True,
        "Verify that the finalized explicit-H atom and bond labels survive RDKit SMILES "
        "writing and reading. Coordinate-derived stereo stays on the returned graph. To "
        "test stereo serialization, use validate_smiles_roundtrip with "
        "check_stereochemistry=True.",
    )


@dataclass
class MolGRConfig:
    """Group the settings for a reconstruction run. Each new instance creates independent sections."""

    resonance: ResonanceConfig = field(
        default_factory=ResonanceConfig,
        metadata={
            "description": "Depth, branch ordering and discrepancy budgets used to find ligand electron assignments."
        },
    )

    cpp_backend: CppBackendConfig = field(
        default_factory=CppBackendConfig,
        metadata={
            "description": "Worker caps, parallel scheduling and UFF atom-typing cache for native reconstruction."
        },
    )

    organic_topology: OrganicTopologyConfig = field(
        default_factory=OrganicTopologyConfig,
        metadata={
            "description": "Aromatic-ring stability factors and geometry thresholds for conjugation in organic fragments."
        },
    )

    metal_scoring: MetalScoringConfig = field(
        default_factory=MetalScoringConfig,
        metadata={
            "description": "Metal-state search limits, candidate ranking and geometry thresholds for metal–ligand contacts."
        },
    )

    metal_radical_inference: MetalRadicalInferenceConfig = field(
        default_factory=MetalRadicalInferenceConfig,
        metadata={
            "description": "Donor sampling, coordination geometry and ligand-field thresholds used to infer metal spin."
        },
    )

    interface: PythonInterfaceConfig = field(
        default_factory=PythonInterfaceConfig,
        metadata={
            "description": "Failure policy and explicit-hydrogen SMILES validation for the Python reconstruction entry points."
        },
    )


CONFIG = MolGRConfig()


__all__ = [
    "CONFIG",
    "CppBackendConfig",
    "MetalRadicalInferenceConfig",
    "MetalScoringConfig",
    "MolGRConfig",
    "OrganicTopologyConfig",
    "PythonInterfaceConfig",
    "ReconstructionFailurePolicy",
    "ResonanceConfig",
    "ResonanceTraversalScore",
]
