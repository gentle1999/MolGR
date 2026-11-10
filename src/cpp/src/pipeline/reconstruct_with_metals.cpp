#include "molgr/pipeline/reconstruct_with_metals.h"

#include "molgr/context.h"
#include "molgr/process_guard.h"
#include "molgr/pipeline/reconstruct_without_metals.h"
#include "molgr/state.h"
#include "molgr/utils/conversions.h"
#include "molgr/utils/consts.h"
#include "molgr/utils/metals/preparation.h"
#include "molgr/utils/metals/scoring.h"
#include "molgr/utils/metals/search.h"
#include "molgr/utils/no_metals/preparation.h"
#include "molgr/utils/parallel.h"
#include "molgr/utils/utils.h"
#include "molgr/vendor/openbabel_copy.h"

#include <openbabel/mol.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace molgr
{
    namespace pipeline
    {
        namespace reconstruct_with_metals
        {
            namespace
            {
                bool IsXyzWhitespace(char value)
                {
                    return value == ' ' || value == '\t' || value == '\r' ||
                           value == '\n' || value == '\f' || value == '\v';
                }

                std::size_t FindLineEnd(const std::string &text, std::size_t offset)
                {
                    const std::size_t line_end = text.find('\n', offset);
                    return line_end == std::string::npos ? text.size() : line_end;
                }

                bool IsKnownNonMetalSymbol(std::string_view symbol)
                {
                    for (const auto &entry : molgr::kNonMetalDict)
                    {
                        const std::string &known_symbol = entry.second.symbol;
                        if (symbol == std::string_view(known_symbol.data(), known_symbol.size()))
                        {
                            return true;
                        }
                    }
                    return false;
                }

                bool ParseAtomCountLine(
                    const std::string &xyz_block,
                    std::size_t *cursor,
                    std::size_t *atom_count)
                {
                    const std::size_t line_end = FindLineEnd(xyz_block, 0);
                    std::size_t pos = 0;
                    while (pos < line_end && IsXyzWhitespace(xyz_block[pos]))
                    {
                        ++pos;
                    }

                    std::size_t count = 0;
                    const std::size_t count_start = pos;
                    while (pos < line_end && xyz_block[pos] >= '0' && xyz_block[pos] <= '9')
                    {
                        count = count * 10 + static_cast<std::size_t>(xyz_block[pos] - '0');
                        ++pos;
                    }
                    if (pos == count_start || count == 0)
                    {
                        return false;
                    }

                    while (pos < line_end && IsXyzWhitespace(xyz_block[pos]))
                    {
                        ++pos;
                    }
                    if (pos != line_end)
                    {
                        return false;
                    }

                    *cursor = line_end < xyz_block.size() ? line_end + 1 : line_end;
                    *atom_count = count;
                    return true;
                }

                bool ReadFirstTokenFromLine(
                    const std::string &xyz_block,
                    std::size_t *cursor,
                    std::string_view *token)
                {
                    if (*cursor >= xyz_block.size())
                    {
                        return false;
                    }

                    const std::size_t line_start = *cursor;
                    const std::size_t line_end = FindLineEnd(xyz_block, line_start);
                    *cursor = line_end < xyz_block.size() ? line_end + 1 : line_end;

                    std::size_t token_start = line_start;
                    while (token_start < line_end && IsXyzWhitespace(xyz_block[token_start]))
                    {
                        ++token_start;
                    }
                    if (token_start == line_end)
                    {
                        return false;
                    }

                    std::size_t token_end = token_start;
                    while (token_end < line_end && !IsXyzWhitespace(xyz_block[token_end]))
                    {
                        ++token_end;
                    }

                    *token = std::string_view(
                        xyz_block.data() + token_start,
                        token_end - token_start);
                    return true;
                }

                bool XyzBlockIsDefinitelyMetalFree(const std::string &xyz_block)
                {
                    std::size_t cursor = 0;
                    std::size_t atom_count = 0;
                    if (!ParseAtomCountLine(xyz_block, &cursor, &atom_count))
                    {
                        return false;
                    }

                    if (cursor >= xyz_block.size())
                    {
                        return false;
                    }
                    cursor = FindLineEnd(xyz_block, cursor);
                    cursor = cursor < xyz_block.size() ? cursor + 1 : cursor;

                    for (std::size_t atom_index = 0; atom_index < atom_count; ++atom_index)
                    {
                        std::string_view symbol;
                        if (!ReadFirstTokenFromLine(xyz_block, &cursor, &symbol))
                        {
                            return false;
                        }
                        if (!IsKnownNonMetalSymbol(symbol))
                        {
                            return false;
                        }
                    }
                    return true;
                }
            }

            std::unique_ptr<molgr::utils::MoleculeData> Xyz2OmolMolData(
                const std::string &xyz_block,
                int total_charge,
                int total_radical_electrons,
                const molgr::config::MolGRConfig &config,
                molgr::diagnostics::ReconstructionDiagnostics *diagnostics)
            {
                molgr::EnsureCurrentProcess("molgr.pipeline.reconstruct_with_metals.xyz2omol");
                if (diagnostics != nullptr)
                {
                    diagnostics->Reset();
                    diagnostics->details["total_charge"] = std::to_string(total_charge);
                    diagnostics->details["total_radical_electrons"] =
                        std::to_string(total_radical_electrons);
                }
                const auto fail =
                    [diagnostics](
                        const std::string &code,
                        const std::string &stage,
                        const std::string &message)
                {
                    if (diagnostics != nullptr)
                    {
                        diagnostics->Fail(code, stage, message);
                    }
                };
                const molgr::context::ReconstructionContext run_context{
                    xyz_block,
                    total_charge,
                    total_radical_electrons,
                    &config};
                const molgr::config::MolGRConfig &run_config = run_context.Config();

                molgr::pipeline::perf::RunTimingScope timing_scope;
                auto &timing_reducer = timing_scope.Reducer();
                if (!run_context.HasValidRadicalTarget())
                {
                    fail(
                        "INVALID_ELECTRONIC_TARGET",
                        "input.electronic_state",
                        "The requested radical-electron count is invalid.");
                    return nullptr;
                }

                if (XyzBlockIsDefinitelyMetalFree(run_context.xyz_block))
                {
                    auto no_metal_state =
                        molgr::pipeline::reconstruct_without_metals::XyzToOmolNoMetalState(
                            run_context.xyz_block,
                            run_context.total_charge,
                            run_context.total_radical_electrons,
                            run_config,
                            &timing_reducer,
                            false);
                    if (!no_metal_state.has_value())
                    {
                        fail(
                            "NO_VALID_ORGANIC_CANDIDATE",
                            "no_metal.reconstruction",
                            "No valid organic graph was produced for the requested target.");
                        return nullptr;
                    }
                    return std::make_unique<molgr::utils::MoleculeData>(
                        molgr::utils::MoleculeDataFromOBMol(no_metal_state->Mol(), true));
                }

                const auto metal_enum_started = std::chrono::steady_clock::now();
                const auto base_state = molgr::metal::preparation::PrepareMetalState(
                    run_context.xyz_block,
                    run_context.total_charge,
                    run_context.total_radical_electrons,
                    run_config);
                if (base_state.phase_history.empty())
                {
                    fail(
                        "NO_REACHABLE_METAL_STATE",
                        "metal.preparation",
                        "Metal preparation produced no usable metal-state options.");
                    return nullptr;
                }
                if (diagnostics != nullptr)
                {
                    diagnostics->details["metal_atom_count"] =
                        std::to_string(base_state.available_valence_radical_states.size());
                }

                const auto state_search_groups =
                    molgr::metal::search::BuildMetalStateSearchGroups(
                        base_state.available_valence_radical_states,
                        run_config);
                const auto layered_state_search_groups =
                    molgr::metal::search::BuildLayeredMetalStateSearchGroups(
                        state_search_groups,
                        run_context.total_radical_electrons,
                        run_config);
                const auto metal_enum_now = std::chrono::steady_clock::now();
                const double metal_enum_ms =
                    std::chrono::duration<double, std::milli>(metal_enum_now - metal_enum_started).count();
                timing_reducer.AddMetalEnumerationCombinationMs(metal_enum_ms);

                std::shared_ptr<OpenBabel::OBMol> no_metal_seed_omol;
                std::vector<molgr::state::MetalCandidateState> possible_candidates;
                int winning_layer_index = 0;
                for (std::size_t layer_index = 0; layer_index < layered_state_search_groups.size();
                     ++layer_index)
                {
                    if (diagnostics != nullptr)
                    {
                        diagnostics->Count("search_layers");
                    }
                    auto grouped_candidates = molgr::metal::search::GroupCandidatesByTargetDp(
                        base_state.phase_history,
                        layered_state_search_groups[layer_index],
                        run_context.total_charge,
                        run_context.total_radical_electrons,
                        run_config);
                    if (diagnostics != nullptr)
                    {
                        diagnostics->Count(
                            "target_buckets",
                            static_cast<long long>(grouped_candidates.size()));
                        long long candidate_count = 0;
                        for (const auto &entry : grouped_candidates)
                        {
                            candidate_count += static_cast<long long>(entry.second.size());
                        }
                        diagnostics->Count("metal_candidates_enumerated", candidate_count);
                    }
                    auto target_bucket_tasks =
                        molgr::metal::search::BuildTargetBucketTasks(std::move(grouped_candidates));
                    if (target_bucket_tasks.empty())
                    {
                        continue;
                    }

                    if (!no_metal_seed_omol)
                    {
                        no_metal_seed_omol =
                            molgr::no_metals::preparation::SeedOmolFromXyzBlock(
                                base_state.no_metal_xyz_block);
                        if (!no_metal_seed_omol)
                        {
                            fail(
                                "INVALID_XYZ",
                                "no_metal.parse",
                                "The no-metal XYZ seed could not be parsed.");
                            return nullptr;
                        }
                    }

                    std::vector<std::optional<molgr::metal::search::PreparedTargetBucket>> prepared_buckets(
                        target_bucket_tasks.size());
                    std::vector<std::shared_ptr<OpenBabel::OBMol>> bucket_seed_omols;
                    bucket_seed_omols.reserve(target_bucket_tasks.size());
                    for (std::size_t bucket_index = 0; bucket_index < target_bucket_tasks.size(); ++bucket_index)
                    {
                        bucket_seed_omols.push_back(
                            molgr::no_metals::preparation::NormalizeSeedOmolCopy(*no_metal_seed_omol));
                    }
                    const std::size_t target_bucket_parallel_threshold =
                        static_cast<std::size_t>(std::max(
                            1,
                            run_config.cpp_backend.target_bucket_parallel_threshold));
                    std::size_t bucket_parallelism = 1;
                    if (run_config.cpp_backend.enable_target_bucket_parallelism &&
                        target_bucket_tasks.size() >= target_bucket_parallel_threshold)
                    {
                        bucket_parallelism =
                            molgr::utils::parallel::ConfiguredParallelism(
                                run_config,
                                target_bucket_tasks.size());
                        if (run_config.cpp_backend.target_bucket_parallel_max_threads.has_value())
                        {
                            bucket_parallelism = std::min<std::size_t>(
                                bucket_parallelism,
                                static_cast<std::size_t>(std::max(
                                    1,
                                    *run_config.cpp_backend.target_bucket_parallel_max_threads)));
                        }
                    }
                    molgr::utils::parallel::ParallelForIndices(
                        target_bucket_tasks.size(),
                        bucket_parallelism,
                        [&](std::size_t bucket_index)
                        {
                            molgr::pipeline::perf::ActiveRunTimingReducerScope active_timing(
                                &timing_reducer);
                            const auto &task = target_bucket_tasks[bucket_index];
                            auto no_metal_state =
                                molgr::pipeline::reconstruct_without_metals::SeedOmolCopyToOmolNoMetalState(
                                    std::move(bucket_seed_omols[bucket_index]),
                                    task.target.no_metal_charge,
                                    task.target.no_metal_radicals,
                                    run_config,
                                    &timing_reducer,
                                    false);
                            if (!no_metal_state.has_value())
                            {
                                return;
                            }

                            prepared_buckets[bucket_index] = molgr::metal::search::PreparedTargetBucket{
                                task.target,
                                std::make_shared<molgr::state::ReconstructionState>(std::move(*no_metal_state)),
                            };
                        });

                    std::vector<molgr::metal::search::CandidateScoreJob> score_jobs;
                    std::vector<std::vector<molgr::state::MetalCandidateState>> scored_buckets(
                        target_bucket_tasks.size());
                    for (std::size_t bucket_index = 0; bucket_index < target_bucket_tasks.size(); ++bucket_index)
                    {
                        if (!prepared_buckets[bucket_index].has_value())
                        {
                            continue;
                        }
                        auto &scored_bucket = scored_buckets[bucket_index];
                        scored_bucket.resize(target_bucket_tasks[bucket_index].candidates.size());
                        for (std::size_t candidate_index = 0;
                             candidate_index < target_bucket_tasks[bucket_index].candidates.size();
                             ++candidate_index)
                        {
                            score_jobs.push_back(molgr::metal::search::CandidateScoreJob{
                                bucket_index,
                                candidate_index,
                            });
                        }
                    }

                    if (score_jobs.empty())
                    {
                        continue;
                    }

                    const std::size_t score_parallelism =
                        run_config.cpp_backend.enable_candidate_scoring_parallelism &&
                                score_jobs.size() >= static_cast<std::size_t>(std::max(
                                                         1,
                                                         run_config.cpp_backend.candidate_score_parallel_threshold))
                            ? molgr::utils::parallel::ConfiguredParallelism(run_config, score_jobs.size())
                            : 1;
                    molgr::utils::parallel::ParallelForIndices(
                        score_jobs.size(),
                        score_parallelism,
                        [&](std::size_t job_index)
                        {
                            molgr::pipeline::perf::ActiveRunTimingReducerScope active_timing(
                                &timing_reducer);
                            const auto &job = score_jobs[job_index];
                            const auto &prepared_bucket = *prepared_buckets[job.bucket_index];
                            scored_buckets[job.bucket_index][job.candidate_index] =
                                molgr::metal::scoring::PrepareCandidateWithNoMetalState(
                                    target_bucket_tasks[job.bucket_index].candidates[job.candidate_index],
                                    prepared_bucket.no_metal_state,
                                    run_config);
                        });

                    std::vector<molgr::state::MetalCandidateState> current_layer_scored_candidates;
                    current_layer_scored_candidates.reserve(score_jobs.size());
                    for (auto &scored_bucket : scored_buckets)
                    {
                        current_layer_scored_candidates.insert(
                            current_layer_scored_candidates.end(),
                            std::make_move_iterator(scored_bucket.begin()),
                            std::make_move_iterator(scored_bucket.end()));
                    }
                    if (diagnostics != nullptr)
                    {
                        diagnostics->Count(
                            "no_metal_reconstruction_successes",
                            static_cast<long long>(
                                std::count_if(
                                    prepared_buckets.begin(),
                                    prepared_buckets.end(),
                                    [](const auto &bucket) { return bucket.has_value(); })));
                        diagnostics->Count(
                            "metal_candidates_scored",
                            static_cast<long long>(current_layer_scored_candidates.size()));
                    }
                    if (current_layer_scored_candidates.empty())
                    {
                        continue;
                    }

                    possible_candidates = std::move(current_layer_scored_candidates);
                    winning_layer_index = static_cast<int>(layer_index);
                    break;
                }

                if (possible_candidates.empty())
                {
                    if (diagnostics != nullptr)
                    {
                        const auto target_count = diagnostics->counts["target_buckets"];
                        const auto organic_count = diagnostics->counts["no_metal_reconstruction_successes"];
                        if (target_count == 0)
                        {
                            fail(
                                "NO_REACHABLE_METAL_STATE",
                                "metal.search",
                                "No metal-state assignment reached a valid charge/radical target.");
                        }
                        else if (organic_count == 0)
                        {
                            fail(
                                "NO_VALID_ORGANIC_CANDIDATE",
                                "no_metal.reconstruction",
                                "Every reachable metal target failed organic graph reconstruction.");
                        }
                        else
                        {
                            fail(
                                "ALL_METAL_CANDIDATES_REJECTED",
                                "metal.scoring",
                                "Organic targets were reconstructed, but every metal candidate was rejected.");
                        }
                    }
                    return nullptr;
                }

                for (auto &scored_candidate : possible_candidates)
                {
                    scored_candidate.metadata["search_layer_index"] = winning_layer_index;
                }

                auto selected_candidate =
                    molgr::metal::scoring::SelectBestCandidate(possible_candidates, run_config);
                if (!selected_candidate.has_value())
                {
                    fail(
                        "ALL_METAL_CANDIDATES_REJECTED",
                        "metal.selection",
                        "No scored metal candidate could be selected.");
                    return nullptr;
                }

                auto best_candidate = *selected_candidate;
                int candidate_total_charge = best_candidate.no_metal_charge_target;
                int candidate_total_radicals = best_candidate.no_metal_radical_target;
                for (const auto &metal_state : best_candidate.metal_states)
                {
                    candidate_total_charge += metal_state.valence;
                    candidate_total_radicals += metal_state.radical_num;
                }
                if (candidate_total_charge != run_context.total_charge ||
                    candidate_total_radicals != run_context.total_radical_electrons)
                {
                    fail(
                        "OUTPUT_INVARIANT_BROKEN",
                        "metal.selection",
                        "The selected candidate does not satisfy the requested charge/radical budget.");
                    return nullptr;
                }
                if (!best_candidate.combined_omol)
                {
                    best_candidate.MaterializeCombinedOmol(
                        [](const OpenBabel::OBMol &no_metal_omol,
                           const std::vector<molgr::metal::MetalAtomPosition> &metal_states)
                        {
                            auto combined_omol = molgr::utils::CloneMolTopologyShared(no_metal_omol);
                            molgr::metal::preparation::CombineMetalWithOmol(
                                *combined_omol,
                                metal_states);
                            return combined_omol;
                        });
                    auto winner_machine =
                        molgr::state::MetalCandidateStateMachine::FromCandidateState(best_candidate);
                    winner_machine.Annotate("combine_metal_with_omol");
                    best_candidate = winner_machine.Freeze();
                }

                auto winner_machine =
                    molgr::state::MetalCandidateStateMachine::FromCandidateState(best_candidate);
                winner_machine.Annotate("select_best_candidate");
                best_candidate = winner_machine.Freeze();

                if (!best_candidate.combined_omol)
                {
                    fail(
                        "OUTPUT_INVARIANT_BROKEN",
                        "metal.materialization",
                        "The selected candidate could not be materialized as an OBMol.");
                    return nullptr;
                }
                return std::make_unique<molgr::utils::MoleculeData>(
                    molgr::utils::MoleculeDataFromOBMol(*best_candidate.combined_omol, true));
            }
        }
    }
}
