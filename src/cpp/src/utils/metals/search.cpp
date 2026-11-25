#include "molgr/utils/metals/search.h"

#include "molgr/utils/consts.h"
#include "molgr/utils/parallel.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <tuple>

namespace molgr
{
    namespace metal
    {
        namespace search
        {
            double MetalStateAssignmentPenalty(const molgr::metal::MetalAtomPosition &metal_state)
            {
                double penalty = 0.0;
                if (metal_state.valence <= 0)
                {
                    penalty += 10.0 * std::max(std::abs(metal_state.valence), 1);
                }

                const auto prior_it = kMetalValencePrior.find(metal_state.symbol);
                const auto minor_it = kMetalValenceMinor.find(metal_state.symbol);
                /// Check whether the proposed valence is represented in this metal's preference tier.
                const auto contains = [](const std::vector<int> &values, int target)
                {
                    return std::find(values.begin(), values.end(), target) != values.end();
                };

                const bool in_prior =
                    prior_it != kMetalValencePrior.end() && contains(prior_it->second, metal_state.valence);
                const bool in_minor =
                    minor_it != kMetalValenceMinor.end() && contains(minor_it->second, metal_state.valence);

                if (!in_prior)
                {
                    penalty += in_minor ? 10.0 : 20.0;
                }
                return penalty;
            }

            namespace
            {
                /// @param choice Electronic states selected together for one or more metals.
                /// @return The sum of penalties for the metal states in one assignment choice.
                double MetalStateChoicePenalty(const MetalStateChoice &choice)
                {
                    double penalty = 0.0;
                    for (const auto &metal_state : choice)
                    {
                        penalty += MetalStateAssignmentPenalty(metal_state);
                    }
                    return penalty;
                }

                /// @param choice Electronic states selected together for one or more metals.
                /// @return The total number of unpaired electrons assigned to the metals in this choice.
                int MetalStateChoiceRadicals(const MetalStateChoice &choice)
                {
                    int radicals = 0;
                    for (const auto &metal_state : choice)
                    {
                        radicals += metal_state.radical_num;
                    }
                    return radicals;
                }

                /// @param choice Electronic states selected together for one or more metals.
                /// @return The assignment ordering key combining penalty, radical count and explicit metal
                ///     labels.
                auto MetalStateChoiceSortKey(const MetalStateChoice &choice)
                {
                    std::vector<std::tuple<std::string, int, int, int>> signature;
                    signature.reserve(choice.size());
                    for (const auto &metal_state : choice)
                    {
                        signature.emplace_back(
                            metal_state.symbol,
                            metal_state.valence,
                            metal_state.radical_num,
                            metal_state.idx);
                    }
                    return std::make_tuple(
                        MetalStateChoicePenalty(choice),
                        MetalStateChoiceRadicals(choice),
                        signature);
                }

                /// Merge compatible choices for equivalent metal elements, or report that unification is
                /// unavailable.
                /// @param grouped_state_options Electronic alternatives for metal sites grouped by element.
                /// @return Unified choices, or an empty optional if the element-wise merge cannot be
                ///     applied.
                std::optional<MetalStateChoiceGroup> BuildUnifiedSameElementStateOptions(
                    const std::vector<std::vector<molgr::metal::MetalAtomPosition>> &grouped_state_options)
                {
                    std::vector<std::map<std::pair<int, int>, molgr::metal::MetalAtomPosition>> signature_maps;
                    std::optional<std::set<std::pair<int, int>>> shared_signatures;
                    for (const auto &state_options : grouped_state_options)
                    {
                        std::map<std::pair<int, int>, molgr::metal::MetalAtomPosition> signature_map;
                        for (const auto &state : state_options)
                        {
                            signature_map.emplace(
                                std::make_pair(state.valence, state.radical_num),
                                state);
                        }
                        if (signature_map.empty())
                        {
                            return std::nullopt;
                        }

                        std::set<std::pair<int, int>> signatures;
                        for (const auto &entry : signature_map)
                        {
                            signatures.insert(entry.first);
                        }
                        if (!shared_signatures.has_value())
                        {
                            shared_signatures = signatures;
                        }
                        else
                        {
                            std::set<std::pair<int, int>> intersection;
                            std::set_intersection(
                                shared_signatures->begin(),
                                shared_signatures->end(),
                                signatures.begin(),
                                signatures.end(),
                                std::inserter(intersection, intersection.begin()));
                            shared_signatures = std::move(intersection);
                        }
                        signature_maps.push_back(std::move(signature_map));
                    }

                    if (!shared_signatures.has_value() || shared_signatures->empty())
                    {
                        return std::nullopt;
                    }

                    std::vector<std::pair<std::pair<int, int>, MetalStateChoice>> ranked_choices;
                    ranked_choices.reserve(shared_signatures->size());
                    for (const auto &signature : *shared_signatures)
                    {
                        MetalStateChoice choice;
                        choice.reserve(signature_maps.size());
                        for (const auto &signature_map : signature_maps)
                        {
                            choice.push_back(signature_map.at(signature));
                        }
                        ranked_choices.emplace_back(signature, std::move(choice));
                    }
                    std::sort(
                        ranked_choices.begin(),
                        ranked_choices.end(),
                        [](const auto &lhs, const auto &rhs)
                        {
                            return MetalStateChoiceSortKey(lhs.second) <
                                   MetalStateChoiceSortKey(rhs.second);
                        });

                    MetalStateChoiceGroup unified_state_options;
                    unified_state_options.reserve(ranked_choices.size());
                    for (auto &entry : ranked_choices)
                    {
                        unified_state_options.push_back(std::move(entry.second));
                    }
                    return unified_state_options;
                }
            }

            std::optional<ValenceBoundsKey> UpdateValenceBounds(
                const ValenceBoundsKey &bounds,
                const molgr::metal::MetalAtomPosition &metal_state,
                const std::optional<int> &max_mixed_valence_spread)
            {
                if (!max_mixed_valence_spread.has_value() || *max_mixed_valence_spread < 0)
                {
                    return ValenceBoundsKey{};
                }

                ValenceBoundsKey updated_bounds = bounds;
                for (auto &entry : updated_bounds)
                {
                    if (std::get<0>(entry) != metal_state.symbol)
                    {
                        continue;
                    }

                    const int next_lower = std::min(std::get<1>(entry), metal_state.valence);
                    const int next_upper = std::max(std::get<2>(entry), metal_state.valence);
                    if (next_upper - next_lower > *max_mixed_valence_spread)
                    {
                        return std::nullopt;
                    }
                    entry = std::make_tuple(metal_state.symbol, next_lower, next_upper);
                    return updated_bounds;
                }

                updated_bounds.emplace_back(
                    metal_state.symbol,
                    metal_state.valence,
                    metal_state.valence);
                std::sort(updated_bounds.begin(), updated_bounds.end());
                return updated_bounds;
            }

            std::vector<PartialMetalAssignment> TrimPartialAssignments(
                std::vector<PartialMetalAssignment> entries,
                int max_assignments_per_target)
            {
                const int limit = std::max(1, max_assignments_per_target);
                std::sort(
                    entries.begin(),
                    entries.end(),
                    [](const PartialMetalAssignment &lhs, const PartialMetalAssignment &rhs)
                    {
                        return std::tie(lhs.metal_assignment_rank, lhs.order) <
                               std::tie(rhs.metal_assignment_rank, rhs.order);
                    });
                if (static_cast<int>(entries.size()) > limit)
                {
                    entries.resize(limit);
                }
                return entries;
            }

            std::vector<MetalStateChoiceGroup> BuildMetalStateSearchGroups(
                const std::vector<std::vector<molgr::metal::MetalAtomPosition>> &available_valence_radical_states,
                const molgr::config::MolGRConfig &config)
            {
                const int unify_threshold =
                    config.metal_scoring.same_element_multimetal_unify_threshold;
                std::map<std::string, std::vector<int>> grouped_indices_by_symbol;
                for (std::size_t idx = 0; idx < available_valence_radical_states.size(); ++idx)
                {
                    const auto &state_options = available_valence_radical_states[idx];
                    if (!state_options.empty())
                    {
                        grouped_indices_by_symbol[state_options.front().symbol].push_back(
                            static_cast<int>(idx));
                    }
                }

                std::vector<MetalStateChoiceGroup> search_groups;
                search_groups.reserve(available_valence_radical_states.size());
                std::set<int> skipped_indices;
                for (std::size_t idx = 0; idx < available_valence_radical_states.size(); ++idx)
                {
                    if (skipped_indices.find(static_cast<int>(idx)) != skipped_indices.end())
                    {
                        continue;
                    }

                    const auto &state_options = available_valence_radical_states[idx];
                    if (state_options.empty())
                    {
                        search_groups.push_back({});
                        continue;
                    }

                    const std::string &symbol = state_options.front().symbol;
                    const auto grouped_it = grouped_indices_by_symbol.find(symbol);
                    if (grouped_it != grouped_indices_by_symbol.end() &&
                        unify_threshold >= 0 &&
                        static_cast<int>(grouped_it->second.size()) > unify_threshold &&
                        static_cast<int>(idx) == grouped_it->second.front())
                    {
                        std::vector<std::vector<molgr::metal::MetalAtomPosition>> grouped_state_options;
                        grouped_state_options.reserve(grouped_it->second.size());
                        for (const int grouped_idx : grouped_it->second)
                        {
                            grouped_state_options.push_back(
                                available_valence_radical_states[static_cast<std::size_t>(grouped_idx)]);
                        }
                        auto unified_state_options =
                            BuildUnifiedSameElementStateOptions(grouped_state_options);
                        if (unified_state_options.has_value())
                        {
                            search_groups.push_back(std::move(*unified_state_options));
                            for (std::size_t grouped_pos = 1; grouped_pos < grouped_it->second.size();
                                 ++grouped_pos)
                            {
                                skipped_indices.insert(grouped_it->second[grouped_pos]);
                            }
                            continue;
                        }
                    }

                    MetalStateChoiceGroup state_group;
                    state_group.reserve(state_options.size());
                    for (const auto &state : state_options)
                    {
                        state_group.push_back(MetalStateChoice{state});
                    }
                    search_groups.push_back(std::move(state_group));
                }
                return search_groups;
            }

            std::vector<MetalStateSearchLayer> BuildLayeredMetalStateSearchGroups(
                const std::vector<MetalStateChoiceGroup> &available_state_search_groups,
                int total_radical_electrons,
                const molgr::config::MolGRConfig &config)
            {
                MetalStateSearchLayer reachable_state_search_groups;
                reachable_state_search_groups.reserve(available_state_search_groups.size());
                const int radical_budget = std::max(0, total_radical_electrons);
                for (const auto &state_search_group : available_state_search_groups)
                {
                    MetalStateChoiceGroup reachable_state_options;
                    for (const auto &state_choice : state_search_group)
                    {
                        if (MetalStateChoiceRadicals(state_choice) <= radical_budget)
                        {
                            reachable_state_options.push_back(state_choice);
                        }
                    }
                    reachable_state_search_groups.push_back(std::move(reachable_state_options));
                }
                if (total_radical_electrons <= 0 || available_state_search_groups.size() < 2)
                {
                    return {reachable_state_search_groups};
                }

                const double penalty_window =
                    config.metal_scoring.open_shell_multimetal_state_penalty_window;
                const int min_state_options =
                    std::max(1, config.metal_scoring.open_shell_multimetal_min_state_options);

                std::vector<std::vector<std::pair<MetalStateChoice, double>>> ranked_group_entries;
                std::vector<std::vector<double>> group_thresholds;
                std::size_t layer_count = 1;
                for (const auto &state_search_group : reachable_state_search_groups)
                {
                    const MetalStateChoiceGroup &candidate_state_options = state_search_group;

                    std::vector<std::pair<MetalStateChoice, double>> ranked_state_entries;
                    ranked_state_entries.reserve(candidate_state_options.size());
                    for (const auto &state_choice : candidate_state_options)
                    {
                        ranked_state_entries.emplace_back(
                            state_choice,
                            MetalStateChoicePenalty(state_choice));
                    }
                    std::sort(
                        ranked_state_entries.begin(),
                        ranked_state_entries.end(),
                        [](const auto &lhs, const auto &rhs)
                        {
                            return MetalStateChoiceSortKey(lhs.first) <
                                   MetalStateChoiceSortKey(rhs.first);
                        });
                    ranked_group_entries.push_back(ranked_state_entries);

                    if (ranked_state_entries.empty())
                    {
                        group_thresholds.push_back({0.0});
                        continue;
                    }

                    std::vector<double> penalties;
                    penalties.reserve(ranked_state_entries.size());
                    for (const auto &entry : ranked_state_entries)
                    {
                        penalties.push_back(entry.second);
                    }

                    if (static_cast<int>(ranked_state_entries.size()) <= min_state_options ||
                        penalty_window < 0.0)
                    {
                        group_thresholds.push_back({penalties.back()});
                        continue;
                    }

                    const double initial_limit =
                        penalties[std::min(static_cast<int>(penalties.size()), min_state_options) - 1];
                    std::vector<double> thresholds{initial_limit};
                    std::vector<double> unique_penalties = penalties;
                    std::sort(unique_penalties.begin(), unique_penalties.end());
                    unique_penalties.erase(
                        std::unique(unique_penalties.begin(), unique_penalties.end()),
                        unique_penalties.end());

                    double current_limit = initial_limit;
                    const double max_penalty = unique_penalties.back();
                    while (current_limit < max_penalty)
                    {
                        double next_limit = max_penalty;
                        bool found = false;
                        if (penalty_window == 0.0)
                        {
                            for (const double penalty : unique_penalties)
                            {
                                if (penalty > current_limit)
                                {
                                    next_limit = penalty;
                                    found = true;
                                    break;
                                }
                            }
                        }
                        else
                        {
                            for (const double penalty : unique_penalties)
                            {
                                if (penalty > current_limit &&
                                    penalty <= current_limit + penalty_window)
                                {
                                    next_limit = penalty;
                                    found = true;
                                }
                            }
                            if (!found)
                            {
                                for (const double penalty : unique_penalties)
                                {
                                    if (penalty > current_limit)
                                    {
                                        next_limit = penalty;
                                        found = true;
                                        break;
                                    }
                                }
                            }
                        }
                        if (!found)
                        {
                            break;
                        }
                        thresholds.push_back(next_limit);
                        current_limit = next_limit;
                    }

                    layer_count = std::max(layer_count, thresholds.size());
                    group_thresholds.push_back(std::move(thresholds));
                }

                std::vector<MetalStateSearchLayer> layers;
                layers.reserve(layer_count);
                for (std::size_t layer_idx = 0; layer_idx < layer_count; ++layer_idx)
                {
                    MetalStateSearchLayer layer_groups;
                    layer_groups.reserve(ranked_group_entries.size());
                    for (std::size_t group_idx = 0; group_idx < ranked_group_entries.size(); ++group_idx)
                    {
                        const auto &ranked_state_entries = ranked_group_entries[group_idx];
                        const auto &threshold_values = group_thresholds[group_idx];
                        const double threshold =
                            threshold_values[std::min(layer_idx, threshold_values.size() - 1)];

                        MetalStateChoiceGroup group_choices;
                        for (const auto &entry : ranked_state_entries)
                        {
                            if (entry.second <= threshold)
                            {
                                group_choices.push_back(entry.first);
                            }
                        }
                        layer_groups.push_back(std::move(group_choices));
                    }
                    layers.push_back(std::move(layer_groups));
                }
                if (layers.empty())
                {
                    layers.push_back(available_state_search_groups);
                }
                return layers;
            }

            PartialAssignmentFrontier EnumeratePartialAssignmentFrontier(
                const std::vector<std::vector<molgr::metal::MetalAtomPosition>> &available_valence_radical_states,
                const std::optional<int> &max_mixed_valence_spread,
                const std::optional<int> &max_total_metal_radicals,
                int max_assignments_per_state)
            {
                std::vector<MetalStateChoiceGroup> normalized_state_groups;
                normalized_state_groups.reserve(available_valence_radical_states.size());
                for (const auto &state_options : available_valence_radical_states)
                {
                    MetalStateChoiceGroup state_group;
                    state_group.reserve(state_options.size());
                    for (const auto &state : state_options)
                    {
                        state_group.push_back(MetalStateChoice{state});
                    }
                    normalized_state_groups.push_back(std::move(state_group));
                }
                return EnumeratePartialAssignmentFrontier(
                    normalized_state_groups,
                    max_mixed_valence_spread,
                    max_total_metal_radicals,
                    max_assignments_per_state);
            }

            PartialAssignmentFrontier EnumeratePartialAssignmentFrontier(
                const std::vector<MetalStateChoiceGroup> &available_state_search_groups,
                const std::optional<int> &max_mixed_valence_spread,
                const std::optional<int> &max_total_metal_radicals,
                int max_assignments_per_state)
            {
                PartialAssignmentFrontier partial_assignments;
                partial_assignments[FrontierKey{}] = {
                    PartialMetalAssignment{{}, 0, 0, 0.0, {}, 0}};
                int next_order = 1;

                for (const auto &metal_state_options : available_state_search_groups)
                {
                    PartialAssignmentFrontier next_partial_assignments;
                    for (const auto &frontier_entry : partial_assignments)
                    {
                        for (const auto &entry : frontier_entry.second)
                        {
                            for (const auto &metal_state_choice : metal_state_options)
                            {
                                PartialMetalAssignment next_entry;
                                next_entry.metal_states = entry.metal_states;
                                next_entry.total_metal_charge = entry.total_metal_charge;
                                next_entry.total_metal_radicals = entry.total_metal_radicals;
                                next_entry.metal_assignment_rank = entry.metal_assignment_rank;
                                auto next_valence_bounds =
                                    std::optional<ValenceBoundsKey>(entry.valence_bounds);

                                for (const auto &metal_state : metal_state_choice)
                                {
                                    next_entry.metal_states.push_back(metal_state);
                                    next_entry.total_metal_charge += metal_state.valence;
                                    next_entry.total_metal_radicals += metal_state.radical_num;
                                    next_entry.metal_assignment_rank +=
                                        MetalStateAssignmentPenalty(metal_state);
                                    next_valence_bounds = UpdateValenceBounds(
                                        *next_valence_bounds,
                                        metal_state,
                                        max_mixed_valence_spread);
                                    if (!next_valence_bounds.has_value())
                                    {
                                        break;
                                    }
                                }

                                if (!next_valence_bounds.has_value())
                                {
                                    continue;
                                }
                                if (max_total_metal_radicals.has_value() &&
                                    next_entry.total_metal_radicals > *max_total_metal_radicals)
                                {
                                    continue;
                                }

                                next_entry.valence_bounds = *next_valence_bounds;
                                next_entry.order = next_order++;

                                FrontierKey next_key{
                                    next_entry.total_metal_charge,
                                    next_entry.total_metal_radicals,
                                    next_entry.valence_bounds};
                                next_partial_assignments[next_key].push_back(std::move(next_entry));
                            }
                        }
                    }

                    PartialAssignmentFrontier trimmed_frontier;
                    for (auto &entry : next_partial_assignments)
                    {
                        trimmed_frontier.emplace(
                            entry.first,
                            TrimPartialAssignments(
                                std::move(entry.second),
                                max_assignments_per_state));
                    }
                    partial_assignments = std::move(trimmed_frontier);
                    if (partial_assignments.empty())
                    {
                        break;
                    }
                }

                return partial_assignments;
            }

            std::optional<ValenceBoundsKey> MergeValenceBounds(
                const ValenceBoundsKey &left_bounds,
                const ValenceBoundsKey &right_bounds,
                const std::optional<int> &max_mixed_valence_spread)
            {
                if (!max_mixed_valence_spread.has_value() || *max_mixed_valence_spread < 0)
                {
                    return ValenceBoundsKey{};
                }

                std::map<std::string, std::pair<int, int>> merged_bounds;
                for (const auto &entry : left_bounds)
                {
                    merged_bounds[std::get<0>(entry)] = {std::get<1>(entry), std::get<2>(entry)};
                }

                for (const auto &entry : right_bounds)
                {
                    const std::string &symbol = std::get<0>(entry);
                    const int lower = std::get<1>(entry);
                    const int upper = std::get<2>(entry);
                    const auto it = merged_bounds.find(symbol);
                    if (it == merged_bounds.end())
                    {
                        merged_bounds.emplace(symbol, std::make_pair(lower, upper));
                        continue;
                    }

                    const int next_lower = std::min(it->second.first, lower);
                    const int next_upper = std::max(it->second.second, upper);
                    if (next_upper - next_lower > *max_mixed_valence_spread)
                    {
                        return std::nullopt;
                    }
                    it->second = {next_lower, next_upper};
                }

                ValenceBoundsKey result;
                result.reserve(merged_bounds.size());
                for (const auto &entry : merged_bounds)
                {
                    result.emplace_back(entry.first, entry.second.first, entry.second.second);
                }
                return result;
            }

            RadicalBucketIndex BucketPartialAssignmentsByChargeRadicals(
                const PartialAssignmentFrontier &frontier)
            {
                RadicalBucketIndex bucket_index;
                for (const auto &frontier_entry : frontier)
                {
                    bucket_index[frontier_entry.first.radicals][frontier_entry.first.charge].push_back(
                        std::make_pair(frontier_entry.first.valence_bounds, frontier_entry.second));
                }
                return bucket_index;
            }

            std::map<TargetBucket, std::vector<PartialMetalAssignment>> CombinePartialAssignmentFrontiers(
                const PartialAssignmentFrontier &left_frontier,
                const PartialAssignmentFrontier &right_frontier,
                int total_charge,
                int total_radical_electrons,
                const std::optional<int> &max_mixed_valence_spread,
                const std::optional<int> &max_total_metal_radicals,
                int max_assignments_per_target)
            {
                std::map<TargetBucket, std::vector<PartialMetalAssignment>> grouped_entries;
                const int trim_trigger = std::max(1, max_assignments_per_target) * 4;
                const RadicalBucketIndex left_bucket_index =
                    BucketPartialAssignmentsByChargeRadicals(left_frontier);
                const RadicalBucketIndex right_bucket_index =
                    BucketPartialAssignmentsByChargeRadicals(right_frontier);
                int right_order_stride = 1;
                for (const auto &entry : right_frontier)
                {
                    for (const auto &assignment : entry.second)
                    {
                        right_order_stride =
                            std::max(right_order_stride, assignment.order + 1);
                    }
                }

                int max_left_radicals = 0;
                int max_right_radicals = 0;
                for (const auto &entry : right_bucket_index)
                {
                    max_right_radicals = std::max(max_right_radicals, entry.first);
                }
                for (const auto &entry : left_bucket_index)
                {
                    max_left_radicals = std::max(max_left_radicals, entry.first);
                }
                int max_combined_metal_radicals = std::max(0, total_radical_electrons);
                if (max_total_metal_radicals.has_value())
                {
                    max_combined_metal_radicals = std::min(
                        max_combined_metal_radicals,
                        std::max(0, *max_total_metal_radicals));
                }

                for (const auto &left_radical_bucket : left_bucket_index)
                {
                    const int left_radicals = left_radical_bucket.first;
                    const int max_right_radicals = max_combined_metal_radicals - left_radicals;
                    if (max_right_radicals < 0)
                    {
                        continue;
                    }

                    for (auto right_it = right_bucket_index.begin();
                         right_it != right_bucket_index.end() && right_it->first <= max_right_radicals;
                         ++right_it)
                    {
                        const int right_radicals = right_it->first;
                        const int total_metal_radicals = left_radicals + right_radicals;
                        const int target_radicals =
                            total_radical_electrons - total_metal_radicals;

                        for (const auto &left_charge_group : left_radical_bucket.second)
                        {
                            const int left_charge = left_charge_group.first;
                            for (const auto &right_charge_group : right_it->second)
                            {
                                const int right_charge = right_charge_group.first;
                                const TargetBucket target{
                                    total_charge - (left_charge + right_charge),
                                    target_radicals};
                                auto &bucket = grouped_entries[target];
                                const std::size_t initial_bucket_size = bucket.size();

                                for (const auto &left_bounds_group : left_charge_group.second)
                                {
                                    for (const auto &right_bounds_group : right_charge_group.second)
                                    {
                                        const auto merged_bounds = MergeValenceBounds(
                                            left_bounds_group.first,
                                            right_bounds_group.first,
                                            max_mixed_valence_spread);
                                        if (!merged_bounds.has_value())
                                        {
                                            continue;
                                        }

                                        for (const auto &left_entry : left_bounds_group.second)
                                        {
                                            for (const auto &right_entry : right_bounds_group.second)
                                            {
                                                PartialMetalAssignment combined_entry;
                                                combined_entry.metal_states = left_entry.metal_states;
                                                combined_entry.metal_states.insert(
                                                    combined_entry.metal_states.end(),
                                                    right_entry.metal_states.begin(),
                                                    right_entry.metal_states.end());
                                                combined_entry.total_metal_charge =
                                                    left_entry.total_metal_charge +
                                                    right_entry.total_metal_charge;
                                                combined_entry.total_metal_radicals =
                                                    left_entry.total_metal_radicals +
                                                    right_entry.total_metal_radicals;
                                                combined_entry.metal_assignment_rank =
                                                    left_entry.metal_assignment_rank +
                                                    right_entry.metal_assignment_rank;
                                                combined_entry.valence_bounds = *merged_bounds;
                                                combined_entry.order =
                                                    left_entry.order * right_order_stride +
                                                    right_entry.order;
                                                bucket.push_back(std::move(combined_entry));
                                            }
                                        }
                                    }
                                }

                                if (bucket.size() == initial_bucket_size)
                                {
                                    if (initial_bucket_size == 0)
                                    {
                                        grouped_entries.erase(target);
                                    }
                                    continue;
                                }

                                if (static_cast<int>(bucket.size()) > trim_trigger)
                                {
                                    bucket = TrimPartialAssignments(
                                        std::move(bucket),
                                        max_assignments_per_target);
                                }
                            }
                        }
                    }
                }

                for (auto &entry : grouped_entries)
                {
                    entry.second = TrimPartialAssignments(
                        std::move(entry.second),
                        max_assignments_per_target);
                }
                return grouped_entries;
            }

            TargetCandidateBuckets GroupCandidatesByTargetDp(
                const std::vector<std::string> &base_phase_history,
                const std::vector<std::vector<molgr::metal::MetalAtomPosition>> &available_valence_radical_states,
                int total_charge,
                int total_radical_electrons,
                const molgr::config::MolGRConfig &config)
            {
                auto normalized_state_groups =
                    BuildMetalStateSearchGroups(available_valence_radical_states, config);
                return GroupCandidatesByTargetDp(
                    base_phase_history,
                    normalized_state_groups,
                    total_charge,
                    total_radical_electrons,
                    config);
            }

            TargetCandidateBuckets GroupCandidatesByTargetDp(
                const std::vector<std::string> &base_phase_history,
                const std::vector<MetalStateChoiceGroup> &available_state_search_groups,
                int total_charge,
                int total_radical_electrons,
                const molgr::config::MolGRConfig &config)
            {
                const std::size_t split_index = available_state_search_groups.size() / 2;
                auto left_options = std::vector<MetalStateChoiceGroup>(
                    available_state_search_groups.begin(),
                    available_state_search_groups.begin() + static_cast<std::ptrdiff_t>(split_index));
                auto right_options = std::vector<MetalStateChoiceGroup>(
                    available_state_search_groups.begin() + static_cast<std::ptrdiff_t>(split_index),
                    available_state_search_groups.end());

                PartialAssignmentFrontier left_frontier;
                PartialAssignmentFrontier right_frontier;
                const std::optional<int> max_mixed_valence_spread =
                    config.metal_scoring.max_mixed_valence_spread;
                const int max_assignments_per_target =
                    std::max(1, config.metal_scoring.max_assignments_per_target);
                left_frontier = EnumeratePartialAssignmentFrontier(
                    left_options,
                    max_mixed_valence_spread,
                    total_radical_electrons,
                    max_assignments_per_target);
                right_frontier = EnumeratePartialAssignmentFrontier(
                    right_options,
                    max_mixed_valence_spread,
                    total_radical_electrons,
                    max_assignments_per_target);
                if (left_frontier.empty() || right_frontier.empty())
                {
                    return {};
                }

                const auto grouped_entries = CombinePartialAssignmentFrontiers(
                    left_frontier,
                    right_frontier,
                    total_charge,
                    total_radical_electrons,
                    max_mixed_valence_spread,
                    total_radical_electrons,
                    max_assignments_per_target);

                TargetCandidateBuckets grouped_candidates;
                int combination_index = 0;
                for (const auto &target_entry : grouped_entries)
                {
                    auto &bucket = grouped_candidates[target_entry.first];
                    for (const auto &entry : target_entry.second)
                    {
                        molgr::state::MetalCandidateStateMachine machine(
                            entry.metal_states,
                            target_entry.first.no_metal_charge,
                            target_entry.first.no_metal_radicals,
                            base_phase_history,
                            {{"combination_index", combination_index}});
                        machine.Annotate("enumerate_metal_combination");
                        machine.Annotate("reconstruct_no_metal_candidate");
                        machine.metadata["metal_assignment_rank"] = entry.metal_assignment_rank;
                        machine.Annotate("rank_metal_assignment_for_target");
                        bucket.push_back(machine.Freeze());
                        ++combination_index;
                    }
                }
                return grouped_candidates;
            }

            std::vector<TargetBucketTask> BuildTargetBucketTasks(TargetCandidateBuckets &&grouped_candidates)
            {
                std::vector<TargetBucketTask> tasks;
                tasks.reserve(grouped_candidates.size());
                for (auto &entry : grouped_candidates)
                {
                    tasks.push_back(TargetBucketTask{
                        entry.first,
                        std::move(entry.second),
                    });
                }
                return tasks;
            }
        }
    }
}
