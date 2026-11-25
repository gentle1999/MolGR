#pragma once

#include "molgr/context.h"
#include "molgr/state.h"
#include "molgr/types.h"

#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace molgr
{
    namespace metal
    {
        namespace search
        {
            using ValenceBoundsEntry = std::tuple<std::string, int, int>;
            using ValenceBoundsKey = std::vector<ValenceBoundsEntry>;
            using MetalStateChoice = std::vector<molgr::MetalAtomPosition>;
            using MetalStateChoiceGroup = std::vector<MetalStateChoice>;
            using MetalStateSearchLayer = std::vector<MetalStateChoiceGroup>;

            /// Accumulated charge, radicals and metal choices for a partially assigned search frontier.
            struct PartialMetalAssignment
            {
                std::vector<molgr::MetalAtomPosition> metal_states;
                int total_metal_charge = 0;
                int total_metal_radicals = 0;
                double metal_assignment_rank = 0.0;
                ValenceBoundsKey valence_bounds;
                int order = 0;
            };

            /// Group partial assignments with compatible charge, radicals and mixed-valence bounds.
            struct FrontierKey
            {
                int charge = 0;
                int radicals = 0;
                ValenceBoundsKey valence_bounds;

                /// Compare charge, radical and valence constraints for deterministic frontier lookup.
                /// @param other Other instance participating in the comparison.
                /// @return True if this frontier key precedes the other in lexicographic order.
                bool operator<(const FrontierKey &other) const
                {
                    return std::tie(charge, radicals, valence_bounds) <
                           std::tie(other.charge, other.radicals, other.valence_bounds);
                }
            };

            using PartialAssignmentFrontier = std::map<FrontierKey, std::vector<PartialMetalAssignment>>;
            using ChargeGroupedAssignments =
                std::map<int, std::vector<std::pair<ValenceBoundsKey, std::vector<PartialMetalAssignment>>>>;
            using RadicalBucketIndex = std::map<int, ChargeGroupedAssignments>;
            using TargetBucket = molgr::context::TargetBucket;
            using TargetCandidateBuckets =
                std::map<TargetBucket, std::vector<molgr::state::MetalCandidateState>>;

            /// Work item for ligand reconstruction at one residual charge and spin target.
            struct TargetBucketTask
            {
                TargetBucket target;
                std::vector<molgr::state::MetalCandidateState> candidates;
            };

            /// Prepared ligand result for reuse by the metal assignments that share its target.
            struct PreparedTargetBucket
            {
                TargetBucket target;
                std::shared_ptr<molgr::state::ReconstructionState> no_metal_state;
            };

            /// Pair a ligand solution with its metal-state combination for scoring.
            struct CandidateScoreJob
            {
                std::size_t bucket_index = 0;
                std::size_t candidate_index = 0;
            };

            /// Rank the formal charge and spin configuration of a partial metal assignment.
            /// @param metal_state Metal site with its coordinates and proposed electronic assignment.
            /// @return Numeric preference penalty used to rank this metal assignment.
            double MetalStateAssignmentPenalty(const molgr::metal::MetalAtomPosition &metal_state);

            /// Extend per-element minimum and maximum valences to include the selected metals.
            /// @param bounds Per-element minimum and maximum valences of the partial assignment.
            /// @param metal_state Metal site with its coordinates and proposed electronic assignment.
            /// @param max_mixed_valence_spread Largest permitted difference in valences of equivalent metal
            ///     elements.
            /// @return Updated bounds, or an empty optional if the mixed-valence limit is exceeded.
            std::optional<ValenceBoundsKey> UpdateValenceBounds(
                const ValenceBoundsKey &bounds,
                const molgr::metal::MetalAtomPosition &metal_state,
                const std::optional<int> &max_mixed_valence_spread);

            /// Retain only the configured number of preferred partial assignments.
            /// @param entries Partial assignments available for ranking and retention.
            /// @param max_assignments_per_target Maximum ranked assignments to retain for the indicated
            ///     search bucket.
            /// @return Preferred partial assignments in rank order, within the retention limit.
            std::vector<PartialMetalAssignment> TrimPartialAssignments(
                std::vector<PartialMetalAssignment> entries,
                int max_assignments_per_target);

            /// Build compatible choices for metal sites before enumerating the joint charge and spin.
            /// @param available_valence_radical_states Eligible valence and unpaired-electron combinations
            ///     for each metal.
            /// @param config Settings for the current reconstruction or scoring operation.
            /// @return Groups of metal choices to expand during the search.
            std::vector<MetalStateChoiceGroup> BuildMetalStateSearchGroups(
                const std::vector<std::vector<molgr::metal::MetalAtomPosition>> &available_valence_radical_states,
                const molgr::config::MolGRConfig &config);

            /// Organize metal choices into layers for bounded frontier expansion.
            /// @param available_state_search_groups Grouped metal-state choices available to extend the
            ///     frontier.
            /// @param total_radical_electrons Target number of unpaired electrons.
            /// @param config Settings for the current reconstruction or scoring operation.
            /// @return Groups of metal choices to expand during the search.
            std::vector<MetalStateSearchLayer> BuildLayeredMetalStateSearchGroups(
                const std::vector<MetalStateChoiceGroup> &available_state_search_groups,
                int total_radical_electrons,
                const molgr::config::MolGRConfig &config);

            /// Enumerate one half of the metal search space while pruning dominated prefixes.
            /// @param available_valence_radical_states Eligible valence and unpaired-electron combinations
            ///     for each metal.
            /// @param max_mixed_valence_spread Largest permitted difference in valences of equivalent metal
            ///     elements.
            /// @param max_total_metal_radicals Upper bound on the sum of metal unpaired-electron counts.
            /// @param max_assignments_per_state Maximum ranked assignments to retain for the indicated
            ///     search bucket.
            /// @return Reachable partial assignments keyed by charge, radicals and mixed-valence bounds.
            PartialAssignmentFrontier EnumeratePartialAssignmentFrontier(
                const std::vector<std::vector<molgr::metal::MetalAtomPosition>> &available_valence_radical_states,
                const std::optional<int> &max_mixed_valence_spread,
                const std::optional<int> &max_total_metal_radicals,
                int max_assignments_per_state);

            /// Enumerate one half of the metal search space while pruning dominated prefixes.
            /// @param available_state_search_groups Grouped metal-state choices available to extend the
            ///     frontier.
            /// @param max_mixed_valence_spread Largest permitted difference in valences of equivalent metal
            ///     elements.
            /// @param max_total_metal_radicals Upper bound on the sum of metal unpaired-electron counts.
            /// @param max_assignments_per_state Maximum ranked assignments to retain for the indicated
            ///     search bucket.
            /// @return Reachable partial assignments keyed by charge, radicals and mixed-valence bounds.
            PartialAssignmentFrontier EnumeratePartialAssignmentFrontier(
                const std::vector<MetalStateChoiceGroup> &available_state_search_groups,
                const std::optional<int> &max_mixed_valence_spread,
                const std::optional<int> &max_total_metal_radicals,
                int max_assignments_per_state);

            /// Combine the minimum and maximum valences observed in two partial assignments.
            /// @param left_bounds Per-element minimum and maximum valences of the partial assignment.
            /// @param right_bounds Per-element minimum and maximum valences of the partial assignment.
            /// @param max_mixed_valence_spread Largest permitted difference in valences of equivalent metal
            ///     elements.
            /// @return Updated bounds, or an empty optional if the mixed-valence limit is exceeded.
            std::optional<ValenceBoundsKey> MergeValenceBounds(
                const ValenceBoundsKey &left_bounds,
                const ValenceBoundsKey &right_bounds,
                const std::optional<int> &max_mixed_valence_spread);

            /// Group partial assignments by total metal charge and unpaired-electron count.
            /// @param frontier Partial metal assignments grouped by their reachable electronic targets.
            /// @return Index of frontier entries by charge and radical count.
            RadicalBucketIndex BucketPartialAssignmentsByChargeRadicals(
                const PartialAssignmentFrontier &frontier);

            /// Join the left/right DP frontiers by compatible charge and radical buckets.
            /// @param left_frontier Partial metal assignments grouped by their reachable electronic
            ///     targets.
            /// @param right_frontier Partial metal assignments grouped by their reachable electronic
            ///     targets.
            /// @param total_charge Target net formal charge of the full input.
            /// @param total_radical_electrons Target number of unpaired electrons.
            /// @param max_mixed_valence_spread Largest permitted difference in valences of equivalent metal
            ///     elements.
            /// @param max_total_metal_radicals Upper bound on the sum of metal unpaired-electron counts.
            /// @param max_assignments_per_target Maximum ranked assignments to retain for the indicated
            ///     search bucket.
            /// @return Candidate combinations grouped by the residual electronic target for the ligand.
            std::map<TargetBucket, std::vector<PartialMetalAssignment>> CombinePartialAssignmentFrontiers(
                const PartialAssignmentFrontier &left_frontier,
                const PartialAssignmentFrontier &right_frontier,
                int total_charge,
                int total_radical_electrons,
                const std::optional<int> &max_mixed_valence_spread,
                const std::optional<int> &max_total_metal_radicals,
                int max_assignments_per_target);

            /// Build target buckets directly, without enumerating the full metal Cartesian product.
            /// @param base_phase_history Ordered reconstruction stages already applied.
            /// @param available_valence_radical_states Eligible valence and unpaired-electron combinations
            ///     for each metal.
            /// @param total_charge Target net formal charge of the full input.
            /// @param total_radical_electrons Target number of unpaired electrons.
            /// @param config Settings for the current reconstruction or scoring operation.
            /// @return Candidate combinations grouped by the residual electronic target for the ligand.
            TargetCandidateBuckets GroupCandidatesByTargetDp(
                const std::vector<std::string> &base_phase_history,
                const std::vector<std::vector<molgr::metal::MetalAtomPosition>> &available_valence_radical_states,
                int total_charge,
                int total_radical_electrons,
                const molgr::config::MolGRConfig &config);

            /// Build target buckets directly, without enumerating the full metal Cartesian product.
            /// @param base_phase_history Ordered reconstruction stages already applied.
            /// @param available_state_search_groups Grouped metal-state choices available to extend the
            ///     frontier.
            /// @param total_charge Target net formal charge of the full input.
            /// @param total_radical_electrons Target number of unpaired electrons.
            /// @param config Settings for the current reconstruction or scoring operation.
            /// @return Candidate combinations grouped by the residual electronic target for the ligand.
            TargetCandidateBuckets GroupCandidatesByTargetDp(
                const std::vector<std::string> &base_phase_history,
                const std::vector<MetalStateChoiceGroup> &available_state_search_groups,
                int total_charge,
                int total_radical_electrons,
                const molgr::config::MolGRConfig &config);

            /// @param grouped_candidates Assignments grouped by residual ligand charge and radical targets.
            /// @return Ligand reconstruction tasks for reachable residual charge and radical targets.
            std::vector<TargetBucketTask> BuildTargetBucketTasks(TargetCandidateBuckets &&grouped_candidates);
        }
    }
}
