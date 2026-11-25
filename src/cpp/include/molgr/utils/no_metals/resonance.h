#pragma once

#include "molgr/config.h"
#include "molgr/state.h"
#include "molgr/utils/perf.h"
#include "molgr/utils/resonance.h"

#include <cstddef>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace molgr
{
    namespace no_metals
    {
        namespace resonance
        {
            using RawStateKey =
                std::pair<molgr::resonance::ResonanceStateKey, int>;
            using ProcessedStateKey =
                std::pair<molgr::resonance::ResonanceStateKey, int>;

            /// Track raw and processed ligand states across seeds so shared resonances can reuse search work.
            struct ResonanceSearchSession
            {
                std::set<RawStateKey> seen_raw_states;
                std::map<RawStateKey, std::vector<std::pair<int, int>>> labels_by_state;
                std::set<ProcessedStateKey> seen_processed_states;
                std::size_t next_raw_index = 0;
            };

            /// Add optional budgeted electronic states and resonance normalizations.
            /// @param neighbor_seeds Ligand states from which to generate or select electronic
            ///     alternatives.
            /// @return Electronic states surviving this seed expansion or search stage.
            std::vector<molgr::state::ReconstructionState> BuildResonanceSeedPool(
                std::vector<molgr::state::ReconstructionState> neighbor_seeds);

            /// Search radical resonance states, then resolve deferred centers at validation.
            /// @param states Ligand states from which to generate or select electronic alternatives.
            /// @param config Settings for the current reconstruction or scoring operation.
            /// @param timing_reducer Run-level collector for timing and work counters.
            /// @param session Deduplication and traversal state shared across resonance seeds.
            /// @param metal_distances Nearest eligible metal-contact distances indexed by ligand atom, in
            ///     milliangstroms.
            /// @param budget_raw_keys Raw graph fingerprints counted against the resonance search budget.
            /// @return Electronic states surviving this seed expansion or search stage.
            std::vector<molgr::state::ReconstructionState> SearchResonanceCandidates(
                const std::vector<molgr::state::ReconstructionState> &states,
                const molgr::config::MolGRConfig &config,
                molgr::pipeline::perf::RunTimingReducer *timing_reducer = nullptr,
                ResonanceSearchSession *session = nullptr,
                const std::vector<int> &metal_distances = {}, std::set<RawStateKey> *budget_raw_keys = nullptr);
        }
    }
}
