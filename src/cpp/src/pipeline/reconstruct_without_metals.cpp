#include "molgr/utils/coordination_seeds.h"
#include "molgr/utils/coordination_geometry.h"
#include "molgr/pipeline/reconstruct_without_metals.h"

#include "molgr/pipeline/resonance.h"
#include "molgr/process_guard.h"
#include "molgr/stages/clean.h"
#include "molgr/stages/preprocess.h"
#include "molgr/utils/conversions.h"
#include "molgr/utils/bond_geometry.h"
#include "molgr/utils/no_metals/neighbor_radicals.h"
#include "molgr/utils/no_metals/preparation.h"
#include "molgr/utils/no_metals/recovery.h"
#include "molgr/utils/no_metals/resonance.h"
#include "molgr/utils/no_metals/selection.h"
#include "molgr/utils/perf.h"
#include "molgr/utils/utils.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <variant>

namespace molgr
{
    namespace pipeline
    {
        namespace reconstruct_without_metals
        {
            namespace
            {
                /// Add the elapsed time since the stage started to its run-level accumulator.
                /// @param started Steady-clock start of the measured stage.
                /// @param timing_reducer Run-level collector for timing and work counters.
                void RecordNoMetalElapsed(
                    const std::chrono::steady_clock::time_point &started,
                    perf::RunTimingReducer *timing_reducer)
                {
                    const auto now = std::chrono::steady_clock::now();
                    const double elapsed_ms =
                        std::chrono::duration<double, std::milli>(now - started).count();
                    if (timing_reducer != nullptr)
                    {
                        timing_reducer->AddNoMetalPipelineMs(elapsed_ms);
                    }
                }

                /// Add the elapsed time since the stage started to its run-level accumulator.
                /// @param started Steady-clock start of the measured stage.
                /// @param timing_reducer Run-level collector for timing and work counters.
                void RecordResonanceElapsed(
                    const std::chrono::steady_clock::time_point &started,
                    perf::RunTimingReducer *timing_reducer)
                {
                    const auto now = std::chrono::steady_clock::now();
                    const double elapsed_ms =
                        std::chrono::duration<double, std::milli>(now - started).count();
                    if (timing_reducer != nullptr)
                    {
                        timing_reducer->AddResonanceHandlingEnumerationMs(elapsed_ms);
                    }
                }
            }

            /// Run ligand reconstruction from an existing electronic seed and retain valid candidates on
            /// request.
            /// @param state Ligand graph and electronic budgets at this stage of reconstruction.
            /// @param config Settings for the current reconstruction or scoring operation.
            /// @param timing_reducer Run-level collector for timing and work counters.
            /// @param preheat_score_bundle Whether to calculate reusable ligand scoring terms before
            ///     returning.
            /// @param no_metal_started Steady-clock start of the measured stage.
            /// @param retained_candidates Optional destination for ligand or metal candidate states
            ///     retained during the search.
            /// @param metal_sites Metal sites used to evaluate ligand coordination.
            /// @param collect_only Whether to collect candidate states without selecting a final ligand.
            /// @return Selected ligand state, or an empty optional if no acceptable state was found.
            std::optional<molgr::state::ReconstructionState> RunNoMetalPipelineFromSeedState(
                molgr::state::ReconstructionState state,
                const molgr::config::MolGRConfig &config,
                perf::RunTimingReducer *timing_reducer,
                bool preheat_score_bundle,
                const std::chrono::steady_clock::time_point &no_metal_started,
                std::vector<molgr::state::ReconstructionState> *retained_candidates = nullptr,
                const std::vector<molgr::MetalSite> &metal_sites = {},
                bool collect_only = false)
            {
                if (state.total_radical_electrons < 0)
                {
                    RecordNoMetalElapsed(no_metal_started, timing_reducer);
                    return std::nullopt;
                }

                if (!state.omol)
                {
                    RecordNoMetalElapsed(no_metal_started, timing_reducer);
                    return std::nullopt;
                }
                const auto linear_started = std::chrono::steady_clock::now();
                const auto metal_distances = molgr::coordination::MetalContactDistances(
                    *state.omol, metal_sites, config.metal_scoring);
                auto prepared_seed = molgr::no_metals::preparation::PrepareNoMetalSeed(state, metal_distances);
                if (timing_reducer != nullptr)
                {
                    const auto linear_now = std::chrono::steady_clock::now();
                    timing_reducer->AddNoMetalLinearPipelineMs(
                        std::chrono::duration<double, std::milli>(linear_now - linear_started).count());
                }

                const auto resonance_started = std::chrono::steady_clock::now();
                std::vector<molgr::state::ReconstructionState> candidate_pool;
                auto local_seeds = molgr::no_metals::preparation::ExpandGeometryElectronicSeeds(prepared_seed);
                if (!metal_sites.empty())
                {
                    std::vector<molgr::state::ReconstructionState> expanded;
                    for (const auto &seed : local_seeds)
                        for (auto &alternative : molgr::coordination::ExpandCoordinationSeeds(
                                 seed, metal_sites, config.metal_scoring))
                            expanded.push_back(std::move(alternative));
                    local_seeds = std::move(expanded);
                }
                for (const auto &local_seed : local_seeds)
                {
                    std::vector<molgr::state::ReconstructionState> resonance_seeds;
                    std::vector<molgr::state::ReconstructionState> local_candidates;
                    molgr::no_metals::resonance::ResonanceSearchSession search_session;
                    std::set<molgr::no_metals::resonance::RawStateKey> budget_raw_keys;
                    auto primary_config = config;
                    primary_config.resonance.max_depth = std::min(2, config.resonance.max_depth);
                    primary_config.resonance.limited_discrepancy_max_discrepancy = std::min(1, config.resonance.limited_discrepancy_max_discrepancy);
                    const int max_discrepancy = std::max(
                        0,
                        config.resonance.neighbor_radical_max_discrepancy);
                    for (int discrepancy = 0; discrepancy <= max_discrepancy; ++discrepancy)
                    {
                        const auto layer_started = std::chrono::steady_clock::now();
                        auto neighbor_seeds =
                            molgr::no_metals::neighbor_radicals::EnumerateNeighborRadicalSeeds(
                                local_seed,
                                discrepancy);
                        if (discrepancy == 0 &&
                            molgr::no_metals::neighbor_radicals::HasSubstitutedNORadicalPair(
                                local_seed.Mol()))
                        {
                            // Keep the unclosed hypothesis beside normal closure.
                            // Common resonance/validation still enforces charge/spin.
                            neighbor_seeds.push_back(local_seed);
                        }
                        auto layer_seeds =
                            molgr::no_metals::resonance::BuildResonanceSeedPool(
                                std::move(neighbor_seeds));
                        if (timing_reducer != nullptr)
                        {
                            const auto layer_now = std::chrono::steady_clock::now();
                            timing_reducer->AddNoMetalLinearPipelineMs(
                                std::chrono::duration<double, std::milli>(
                                    layer_now - layer_started)
                                    .count());
                        }
                        local_candidates =
                            molgr::no_metals::resonance::SearchResonanceCandidates(
                                layer_seeds,
                                primary_config,
                                timing_reducer,
                                &search_session, metal_distances, &budget_raw_keys);
                        for (auto &layer_seed : layer_seeds)
                        {
                            resonance_seeds.push_back(std::move(layer_seed));
                        }
                        if (!local_candidates.empty())
                        {
                            break;
                        }
                    }

                    if (local_candidates.empty() && config.resonance.max_depth > 2)
                    {
                        auto depth_config = config;
                        depth_config.resonance.limited_discrepancy_max_discrepancy = primary_config.resonance.limited_discrepancy_max_discrepancy;
                        local_candidates = molgr::no_metals::resonance::SearchResonanceCandidates(
                            resonance_seeds, depth_config, timing_reducer, nullptr, metal_distances, &budget_raw_keys);
                    }
                    if (local_candidates.empty() && config.resonance.limited_discrepancy_max_discrepancy > 1)
                    {
                        local_candidates = molgr::no_metals::resonance::SearchResonanceCandidates(
                            resonance_seeds, config, timing_reducer, nullptr, metal_distances, &budget_raw_keys);
                    }


                    std::vector<molgr::state::ReconstructionState> deformed_pi_seeds;
                    if (local_candidates.empty())
                    {
                        deformed_pi_seeds =
                            molgr::no_metals::recovery::EnumerateDeformedPiRecoverySeeds(
                                resonance_seeds);
                        if (!deformed_pi_seeds.empty())
                        {
                            auto recovery_pool =
                                molgr::no_metals::resonance::BuildResonanceSeedPool(
                                    deformed_pi_seeds);
                            local_candidates =
                                molgr::no_metals::resonance::SearchResonanceCandidates(
                                    recovery_pool,
                                    config,
                                    timing_reducer, nullptr, metal_distances);
                        }
                    }

                    if (local_candidates.empty())
                    {
                        std::vector<molgr::state::ReconstructionState> bond_break_sources =
                            resonance_seeds;
                        for (auto &deformed_seed : deformed_pi_seeds)
                        {
                            bond_break_sources.push_back(std::move(deformed_seed));
                        }
                        auto bond_break_seeds =
                            molgr::no_metals::recovery::EnumerateBondBreakRecoverySeeds(
                                bond_break_sources);
                        if (!bond_break_seeds.empty())
                        {
                            auto recovery_pool =
                                molgr::no_metals::resonance::BuildResonanceSeedPool(
                                    std::move(bond_break_seeds));
                            local_candidates =
                                molgr::no_metals::resonance::SearchResonanceCandidates(
                                    recovery_pool,
                                    config,
                                    timing_reducer, nullptr, metal_distances);
                        }
                    }

                    for (auto &candidate : local_candidates)
                        candidate_pool.push_back(std::move(candidate));
                }

                RecordResonanceElapsed(resonance_started, timing_reducer);
                if (candidate_pool.empty())
                {
                    RecordNoMetalElapsed(no_metal_started, timing_reducer);
                    return std::nullopt;
                }

                if (collect_only)
                {
                    *retained_candidates = std::move(candidate_pool);
                    RecordNoMetalElapsed(no_metal_started, timing_reducer);
                    return std::nullopt;
                }
                std::size_t best_idx = 0;
                std::optional<
                    std::tuple<int, int, int, int, int, double, double, double, double, int, int, double>>
                    best_selection_key;
                std::optional<molgr::no_metals::selection::NoMetalGraphTieBreakKey>
                    best_graph_tie_break_key;
                for (std::size_t i = 0; i < candidate_pool.size(); ++i)
                {
                    const auto selection_key =
                        molgr::no_metals::selection::NoMetalCandidateSelectionKey(
                            candidate_pool[i],
                            config);
                    const auto graph_tie_break_key =
                        molgr::no_metals::selection::NoMetalCandidateGraphTieBreakKey(
                            candidate_pool[i]);
                    if (!best_selection_key.has_value() ||
                        selection_key < *best_selection_key ||
                        (selection_key == *best_selection_key &&
                         (!best_graph_tie_break_key.has_value() ||
                          graph_tie_break_key < *best_graph_tie_break_key)))
                    {
                        best_selection_key = selection_key;
                        best_graph_tie_break_key = graph_tie_break_key;
                        best_idx = i;
                    }
                }

                if (!best_selection_key.has_value())
                {
                    RecordNoMetalElapsed(no_metal_started, timing_reducer);
                    return std::nullopt;
                }

                if (retained_candidates != nullptr)
                {
                    *retained_candidates = candidate_pool;
                }
                auto result_state = std::move(candidate_pool[best_idx]);
                auto result_machine =
                    molgr::state::OmolStateMachine::FromReconstructionState(result_state);
                result_machine.Annotate("select_best_no_metal_candidate");
                result_state = result_machine.FreezeLike(result_state);
                if (preheat_score_bundle)
                {
                    result_state.PreheatScoreBundle(config);
                }
                RecordNoMetalElapsed(no_metal_started, timing_reducer);
                return result_state;
            }

            std::optional<molgr::state::ReconstructionState> XyzToOmolNoMetalState(
                const std::string &xyz_block,
                int total_charge,
                int total_radical_electrons,
                const molgr::config::MolGRConfig &config,
                perf::RunTimingReducer *timing_reducer,
                bool preheat_score_bundle,
                const std::vector<molgr::MetalSite> &metal_sites)
            {
                const auto no_metal_started = std::chrono::steady_clock::now();
                auto state = molgr::no_metals::preparation::SeedState(
                    xyz_block,
                    total_charge,
                    total_radical_electrons);
                if (state.omol && !metal_sites.empty())
                    molgr::coordination::DetachCarbonylHydrideContacts(state.MutableMol(), metal_sites);
                return RunNoMetalPipelineFromSeedState(
                    std::move(state),
                    config,
                    timing_reducer,
                    preheat_score_bundle,
                    no_metal_started, nullptr, metal_sites);
            }

            std::optional<molgr::state::ReconstructionState> SeedOmolToOmolNoMetalState(
                const OpenBabel::OBMol &seed_omol,
                int total_charge,
                int total_radical_electrons,
                const molgr::config::MolGRConfig &config,
                perf::RunTimingReducer *timing_reducer,
                bool preheat_score_bundle,
                const std::vector<molgr::MetalSite> &metal_sites)
            {
                const auto no_metal_started = std::chrono::steady_clock::now();
                auto state = molgr::no_metals::preparation::SeedStateFromOmol(
                    seed_omol,
                    total_charge,
                    total_radical_electrons);
                if (state.omol && !metal_sites.empty())
                    molgr::coordination::DetachCarbonylHydrideContacts(state.MutableMol(), metal_sites);
                return RunNoMetalPipelineFromSeedState(
                    std::move(state),
                    config,
                    timing_reducer,
                    preheat_score_bundle,
                    no_metal_started, nullptr, metal_sites);
            }

            std::optional<molgr::state::ReconstructionState> SeedOmolCopyToOmolNoMetalState(
                std::shared_ptr<OpenBabel::OBMol> seed_omol,
                int total_charge,
                int total_radical_electrons,
                const molgr::config::MolGRConfig &config,
                perf::RunTimingReducer *timing_reducer,
                bool preheat_score_bundle,
                std::vector<molgr::state::ReconstructionState> *candidate_pool,
                const std::vector<molgr::MetalSite> &metal_sites)
            {
                if (!seed_omol)
                {
                    return std::nullopt;
                }
                const auto no_metal_started = std::chrono::steady_clock::now();
                auto state = molgr::no_metals::preparation::BuildSeedState(
                    std::move(seed_omol),
                    total_charge,
                    total_radical_electrons);
                return RunNoMetalPipelineFromSeedState(
                    std::move(state),
                    config,
                    timing_reducer,
                    preheat_score_bundle,
                    no_metal_started,
                    candidate_pool, metal_sites);
            }

            std::vector<molgr::state::ReconstructionState> CollectLigandCandidates(
                std::shared_ptr<OpenBabel::OBMol> seed, int charge, int radicals,
                const molgr::config::MolGRConfig &config, perf::RunTimingReducer *timing,
                const std::vector<molgr::MetalSite> &sites)
            {
                std::vector<molgr::state::ReconstructionState> candidates;
                const auto started = std::chrono::steady_clock::now();
                if (seed)
                    RunNoMetalPipelineFromSeedState(
                        molgr::no_metals::preparation::BuildSeedState(std::move(seed), charge, radicals),
                        config, timing, false, started, &candidates, sites, true);
                return candidates;
            }

            std::unique_ptr<molgr::utils::MoleculeData> XyzToMolDataNoMetal(
                const std::string &xyz_block,
                int total_charge,
                int total_radical_electrons,
                const molgr::config::MolGRConfig &config)
            {
                molgr::EnsureCurrentProcess("molgr.pipeline.reconstruct_without_metals.xyz_to_omol_no_metal");
                molgr::pipeline::perf::RunTimingScope timing_scope;
                auto state = XyzToOmolNoMetalState(
                    xyz_block,
                    total_charge,
                    total_radical_electrons,
                    config,
                    &timing_scope.Reducer(),
                    false);
                if (!state.has_value())
                {
                    return nullptr;
                }
                return std::make_unique<molgr::utils::MoleculeData>(
                    molgr::utils::MoleculeDataFromOBMol(state->Mol(), true));
            }

            std::vector<DebugNoMetalCandidateSummary> DebugNoMetalResonanceCandidateSummaries(
                const std::string &xyz_block,
                int total_charge,
                int total_radical_electrons,
                const molgr::config::MolGRConfig &config)
            {
                std::vector<DebugNoMetalCandidateSummary> summaries;
                if (total_radical_electrons < 0)
                {
                    return summaries;
                }

                auto state = molgr::no_metals::preparation::SeedState(
                    xyz_block,
                    total_charge,
                    total_radical_electrons);
                if (!state.omol)
                {
                    return summaries;
                }
                // Report the production search, including geometry-supported
                // electronic branches and the same recovery/early-stop policy.
                std::vector<molgr::state::ReconstructionState> candidates;
                RunNoMetalPipelineFromSeedState(
                    std::move(state), config, nullptr, false,
                    std::chrono::steady_clock::now(), &candidates);
                summaries.reserve(candidates.size());
                for (auto &candidate : candidates)
                {
                    molgr::no_metals::selection::NoMetalCandidateSelectionKey(candidate, config);
                    /// Read a numeric candidate diagnostic for the Python summary, or use the supplied
                    /// fallback.
                    const auto get_int = [&](const std::string &key, int fallback = 0)
                    {
                        const auto it = candidate.metadata.find(key);
                        if (it == candidate.metadata.end())
                        {
                            return fallback;
                        }
                        if (const auto *value = std::get_if<int>(&it->second))
                        {
                            return *value;
                        }
                        if (const auto *value = std::get_if<double>(&it->second))
                        {
                            return static_cast<int>(*value);
                        }
                        return fallback;
                    };
                    /// Read a numeric candidate diagnostic for the Python summary, or use the supplied
                    /// fallback.
                    const auto get_double = [&](const std::string &key, double fallback = 0.0)
                    {
                        const auto it = candidate.metadata.find(key);
                        if (it == candidate.metadata.end())
                        {
                            return fallback;
                        }
                        if (const auto *value = std::get_if<double>(&it->second))
                        {
                            return *value;
                        }
                        if (const auto *value = std::get_if<int>(&it->second))
                        {
                            return static_cast<double>(*value);
                        }
                        return fallback;
                    };

                    summaries.push_back(DebugNoMetalCandidateSummary{
                        reconstruct::SmilesFirstToken(candidate.Mol()),
                        get_int("resonance_index", -1),
                        get_double("score"),
                        get_double("organic_aromatic_stability_score"),
                        get_int("organic_aromatic_atom_count"),
                        get_int("organic_aromatic_ring_count"),
                        get_int("organic_max_conjugated_component_size"),
                        get_int("organic_conjugated_atom_count"),
                        get_int("organic_conjugated_bond_count"),
                        get_int("organic_hyperconjugative_donor_count"),
                        get_int("organic_hyperconjugation_score"),
                        get_int("organic_azinium_geometry_conflict_count"),
                        get_int("organic_formal_charge_absolute_sum"),
                        get_int("organic_long_carbonyl_multiple_bond_count"),
                        get_double("organic_conjugation_charge_penalty"),
                        get_double("organic_adjusted_max_conjugated_component_size"),
                        get_double("organic_adjusted_conjugated_atom_count"),
                        get_double("organic_adjusted_conjugated_bond_count"),
                        get_int("organic_excess_radical_labels"),
                    });
                }
                return summaries;
            }
        }
    }
}
