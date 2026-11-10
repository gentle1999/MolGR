#include "molgr/pipeline/reconstruct_without_metals.h"

#include "molgr/pipeline/resonance.h"
#include "molgr/process_guard.h"
#include "molgr/stages/clean.h"
#include "molgr/stages/preprocess.h"
#include "molgr/utils/conversions.h"
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

            std::optional<molgr::state::ReconstructionState> RunNoMetalPipelineFromSeedState(
                molgr::state::ReconstructionState state,
                const molgr::config::MolGRConfig &config,
                perf::RunTimingReducer *timing_reducer,
                bool preheat_score_bundle,
                const std::chrono::steady_clock::time_point &no_metal_started)
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
                auto prepared_seed = molgr::no_metals::preparation::PrepareNoMetalSeed(state);
                if (timing_reducer != nullptr)
                {
                    const auto linear_now = std::chrono::steady_clock::now();
                    timing_reducer->AddNoMetalLinearPipelineMs(
                        std::chrono::duration<double, std::milli>(linear_now - linear_started).count());
                }

                const auto resonance_started = std::chrono::steady_clock::now();
                std::vector<molgr::state::ReconstructionState> resonance_seeds;
                std::vector<molgr::state::ReconstructionState> candidate_pool;
                molgr::no_metals::resonance::ResonanceSearchSession search_session;
                const int max_discrepancy = std::max(
                    0,
                    config.resonance.limited_discrepancy_max_discrepancy);
                for (int discrepancy = 0; discrepancy <= max_discrepancy; ++discrepancy)
                {
                    const auto layer_started = std::chrono::steady_clock::now();
                    auto neighbor_seeds =
                        molgr::no_metals::neighbor_radicals::EnumerateNeighborRadicalSeeds(
                            prepared_seed,
                            discrepancy);
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
                    candidate_pool =
                        molgr::no_metals::resonance::SearchResonanceCandidates(
                            layer_seeds,
                            config,
                            timing_reducer,
                            &search_session);
                    for (auto &layer_seed : layer_seeds)
                    {
                        resonance_seeds.push_back(std::move(layer_seed));
                    }
                    if (!candidate_pool.empty())
                    {
                        break;
                    }
                }

                std::vector<molgr::state::ReconstructionState> deformed_pi_seeds;
                if (candidate_pool.empty())
                {
                    deformed_pi_seeds =
                        molgr::no_metals::recovery::EnumerateDeformedPiRecoverySeeds(
                            resonance_seeds);
                    if (!deformed_pi_seeds.empty())
                    {
                        auto recovery_pool =
                            molgr::no_metals::resonance::BuildResonanceSeedPool(
                                deformed_pi_seeds);
                        candidate_pool =
                            molgr::no_metals::resonance::SearchResonanceCandidates(
                                recovery_pool,
                                config,
                                timing_reducer);
                    }
                }

                if (candidate_pool.empty())
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
                        candidate_pool =
                            molgr::no_metals::resonance::SearchResonanceCandidates(
                                recovery_pool,
                                config,
                                timing_reducer);
                    }
                }

                RecordResonanceElapsed(resonance_started, timing_reducer);
                if (candidate_pool.empty())
                {
                    RecordNoMetalElapsed(no_metal_started, timing_reducer);
                    return std::nullopt;
                }

                std::size_t best_idx = 0;
                std::optional<
                    std::tuple<int, int, int, double, double, double, double, int, int, double>>
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
                bool preheat_score_bundle)
            {
                const auto no_metal_started = std::chrono::steady_clock::now();
                auto state = molgr::no_metals::preparation::SeedState(
                    xyz_block,
                    total_charge,
                    total_radical_electrons);
                return RunNoMetalPipelineFromSeedState(
                    std::move(state),
                    config,
                    timing_reducer,
                    preheat_score_bundle,
                    no_metal_started);
            }

            std::optional<molgr::state::ReconstructionState> SeedOmolToOmolNoMetalState(
                const OpenBabel::OBMol &seed_omol,
                int total_charge,
                int total_radical_electrons,
                const molgr::config::MolGRConfig &config,
                perf::RunTimingReducer *timing_reducer,
                bool preheat_score_bundle)
            {
                const auto no_metal_started = std::chrono::steady_clock::now();
                auto state = molgr::no_metals::preparation::SeedStateFromOmol(
                    seed_omol,
                    total_charge,
                    total_radical_electrons);
                return RunNoMetalPipelineFromSeedState(
                    std::move(state),
                    config,
                    timing_reducer,
                    preheat_score_bundle,
                    no_metal_started);
            }

            std::optional<molgr::state::ReconstructionState> SeedOmolCopyToOmolNoMetalState(
                std::shared_ptr<OpenBabel::OBMol> seed_omol,
                int total_charge,
                int total_radical_electrons,
                const molgr::config::MolGRConfig &config,
                perf::RunTimingReducer *timing_reducer,
                bool preheat_score_bundle)
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
                    no_metal_started);
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
                auto prepared = molgr::no_metals::preparation::PrepareNoMetalSeed(state);
                auto neighbor_seeds =
                    molgr::no_metals::neighbor_radicals::EnumerateNeighborRadicalSeeds(
                        prepared);
                auto resonance_seeds =
                    molgr::no_metals::resonance::BuildResonanceSeedPool(
                        std::move(neighbor_seeds));
                auto candidates =
                    molgr::no_metals::resonance::SearchResonanceCandidates(
                        resonance_seeds,
                        config);
                summaries.reserve(candidates.size());
                for (auto &candidate : candidates)
                {
                    molgr::no_metals::selection::NoMetalCandidateSelectionKey(candidate, config);
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
                        get_int("organic_formal_charge_absolute_sum"),
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
