#pragma once

#include <optional>
#include <string>

namespace molgr::config
{
    /// Search depth, ordering and discrepancy budgets for exploring electronic assignments.
    struct ResonanceConfig
    {
        int max_depth = 3;
        int limited_discrepancy_max_discrepancy = 2;
        std::string traversal_score = "uff_lite_gain";
        // Charge-separation choices during neighbor-radical seed generation.
        // Independent of limited discrepancy in the resonance walk.
        int neighbor_radical_max_discrepancy = 2;
    };

    /// Worker limits and cache settings for native reconstruction and candidate scoring.
    struct CppBackendConfig
    {
#if defined(_WIN32)
        std::optional<int> max_threads = 1;
#else
        std::optional<int> max_threads;
#endif
        bool enable_target_bucket_parallelism = true;
        bool enable_candidate_scoring_parallelism = false;
        bool enable_uff_atom_typing_cache = false;
        int target_bucket_parallel_threshold = 1;
        std::optional<int> target_bucket_parallel_max_threads;
        int candidate_score_parallel_threshold = 32;
    };

    /// Aromatic stability weights and geometric cutoffs for conjugation in organic ligands.
    struct OrganicTopologyConfig
    {
        double aromatic_stability_benzene_score = 1.0;
        double aromatic_stability_other_ring_max_score = 0.99;
        double aromatic_stability_ring_size_6_factor = 0.92;
        double aromatic_stability_ring_size_5_factor = 0.84;
        double aromatic_stability_other_ring_size_factor = 0.76;
        double aromatic_stability_hetero_atom_penalty = 0.10;
        double aromatic_stability_min_hetero_factor = 0.62;
        double aromatic_stability_formal_charge_penalty = 0.16;
        double aromatic_stability_min_charge_factor = 0.50;
        double aromatic_stability_radical_penalty = 0.20;
        double aromatic_stability_min_radical_factor = 0.50;
        double conjugation_normalized_tetrahedron_volume_tolerance = 0.075;
        double conjugation_min_plane_overlap = 0.5;
    };

    /// Coordination visibility, contact cutoffs and weights for ranking metal assignments.
    struct MetalScoringConfig
    {
        double open_shell_multimetal_state_penalty_window = 10.0;
        int open_shell_multimetal_min_state_options = 6;
        int same_element_multimetal_unify_threshold = 3;
        std::optional<int> max_mixed_valence_spread = 3;
        int max_assignments_per_target = 64;
        double metal_coordination_extra_tolerance_angstrom = 0.75;
        double pi_dative_distance_difference_tolerance_angstrom = 0.10;
        double metal_access_radius_scale = 1.0;
        double metal_access_clearance_angstrom = 0.0;
        double charge_localization_selection_margin = 0.3;
    };

    /// Spin-state inference settings for metal sites with incomplete electronic labels.
    struct MetalRadicalInferenceConfig
    {
        int max_considered_donors = 6;
        double square_planar_planarity_tolerance_angstrom = 0.45;
        double trigonal_planar_planarity_tolerance_angstrom = 0.35;
        double linear_angle_min_degrees = 150.0;
        double strong_field_threshold = 1.10;
        double weak_field_threshold = 0.75;
        double field_ambiguity_margin = 0.10;
    };

    /// Run-level settings shared by native search, topology and metal scoring.
    struct MolGRConfig
    {
        ResonanceConfig resonance;
        CppBackendConfig cpp_backend;
        OrganicTopologyConfig organic_topology;
        MetalScoringConfig metal_scoring;
        MetalRadicalInferenceConfig metal_radical_inference;
    };

    /// @return The shared default reconstruction settings when no per-run configuration is supplied.
    const MolGRConfig &GetDefaultConfig();
}
