from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Literal, Optional


ResonanceTraversalScore = Literal[
    "uff_lite_gain",
    "input_order",
]
ReconstructionFailurePolicy = Literal[
    "raise",
    "return_suspicious",
]


@dataclass
class ResonanceConfig:
    max_depth: int = 2
    limited_discrepancy_max_discrepancy: int = 1
    traversal_score: ResonanceTraversalScore = "uff_lite_gain"


@dataclass
class CppBackendConfig:
    max_threads: Optional[int] = 1 if sys.platform == "win32" else None
    enable_target_bucket_parallelism: bool = True
    enable_candidate_scoring_parallelism: bool = False
    enable_uff_atom_typing_cache: bool = False
    enable_target_bucket_score_bundle_preheat: bool = True
    target_bucket_parallel_threshold: int = 1
    target_bucket_parallel_max_threads: Optional[int] = None
    candidate_score_parallel_threshold: int = 32


@dataclass
class OrganicTopologyConfig:
    aromatic_stability_benzene_score: float = 1.0
    aromatic_stability_other_ring_max_score: float = 0.99
    aromatic_stability_ring_size_6_factor: float = 0.92
    aromatic_stability_ring_size_5_factor: float = 0.84
    aromatic_stability_other_ring_size_factor: float = 0.76
    aromatic_stability_hetero_atom_penalty: float = 0.10
    aromatic_stability_min_hetero_factor: float = 0.62
    aromatic_stability_formal_charge_penalty: float = 0.16
    aromatic_stability_min_charge_factor: float = 0.50
    aromatic_stability_radical_penalty: float = 0.20
    aromatic_stability_min_radical_factor: float = 0.50
    conjugation_normalized_tetrahedron_volume_tolerance: float = 0.075


@dataclass
class MetalScoringConfig:
    open_shell_multimetal_state_penalty_window: float = 10.0
    open_shell_multimetal_min_state_options: int = 6
    same_element_multimetal_unify_threshold: int = 3
    max_mixed_valence_spread: Optional[int] = 3
    max_assignments_per_target: int = 64
    metal_coordination_extra_tolerance_angstrom: float = 0.75
    pi_dative_distance_difference_tolerance_angstrom: float = 0.10
    metal_access_radius_scale: float = 1.0
    metal_access_clearance_angstrom: float = 0.0
    charge_localization_selection_margin: float = 0.3


@dataclass
class MetalRadicalInferenceConfig:
    max_considered_donors: int = 6
    square_planar_planarity_tolerance_angstrom: float = 0.45
    trigonal_planar_planarity_tolerance_angstrom: float = 0.35
    linear_angle_min_degrees: float = 150.0
    strong_field_threshold: float = 1.10
    weak_field_threshold: float = 0.75
    field_ambiguity_margin: float = 0.10


@dataclass
class PythonInterfaceConfig:
    reconstruction_failure_policy: ReconstructionFailurePolicy = "raise"
    require_smiles_roundtrip: bool = True


@dataclass
class MolGRConfig:
    resonance: ResonanceConfig = field(default_factory=ResonanceConfig)
    cpp_backend: CppBackendConfig = field(default_factory=CppBackendConfig)
    organic_topology: OrganicTopologyConfig = field(default_factory=OrganicTopologyConfig)
    metal_scoring: MetalScoringConfig = field(default_factory=MetalScoringConfig)
    metal_radical_inference: MetalRadicalInferenceConfig = field(
        default_factory=MetalRadicalInferenceConfig
    )
    interface: PythonInterfaceConfig = field(default_factory=PythonInterfaceConfig)


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
