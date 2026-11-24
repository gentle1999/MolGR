#pragma once

#include "molgr/config.h"
#include "molgr/state.h"

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace molgr
{
    namespace metal
    {
        namespace scoring
        {
            /// Copy-only diagnostic used by ring-cumulene scoring and regression tests.
            /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
            /// @param atom_index One-based Open Babel atom index.
            /// @return True if an aromatic alternative relieves the bent ring-allene assignment.
            bool HasAromaticRingAlleneResonance(const OpenBabel::OBMol &mol, int atom_index);

            /// Read the index of the metal-state combination that produced this candidate.
            /// @param candidate Candidate electronic assignment being evaluated.
            /// @return Original combination index, defaulting to zero when absent.
            int CandidateCombinationIndex(const molgr::state::MetalCandidateState &candidate);

            /// Read an integer candidate diagnostic, returning the fallback when the key is absent.
            /// @param candidate Candidate electronic assignment being evaluated.
            /// @param key Lookup key for the recorded state or property.
            /// @param fallback Value used when the requested information is unavailable.
            /// @return Stored numeric metadata value, or the supplied fallback.
            int MetadataInt(
                const molgr::state::MetalCandidateState &candidate,
                const std::string &key,
                int fallback = 0);

            /// Read a floating-point candidate diagnostic, returning the fallback when the key is absent.
            /// @param candidate Candidate electronic assignment being evaluated.
            /// @param key Lookup key for the recorded state or property.
            /// @param fallback Value used when the requested information is unavailable.
            /// @return Stored numeric metadata value, or the supplied fallback.
            double MetadataDouble(
                const molgr::state::MetalCandidateState &candidate,
                const std::string &key,
                double fallback = 0.0);

            /// Resolve ligand contributors before comparing different metal assignments.
            /// @param candidates Candidate assignments under comparison.
            /// @param config Settings for the current reconstruction or scoring operation.
            /// @return Ligand assignments retained after geometry, electronic-state and score comparison.
            std::vector<molgr::state::MetalCandidateState> SelectLigandAlternatives(
                std::vector<molgr::state::MetalCandidateState> candidates,
                const molgr::config::MolGRConfig &config);

            /// Rank metal–ligand assignments and return an empty optional when none can be selected.
            /// @param candidates Candidate assignments under comparison.
            /// @param config Settings for the current reconstruction or scoring operation.
            /// @return Best scored metal-ligand candidate, or an empty optional when none is available.
            std::optional<molgr::state::MetalCandidateState> SelectBestCandidate(
                const std::vector<molgr::state::MetalCandidateState> &candidates,
                const molgr::config::MolGRConfig &config);

            /// Update pool-dependent metal and ligand scores before selecting the best admissible assignment.
            /// @param candidates Candidate assignments under comparison.
            /// @param config Settings for the current reconstruction or scoring operation.
            /// @return Best scored metal-ligand candidate, or an empty optional when none is available.
            std::optional<molgr::state::MetalCandidateState> SelectBestCandidateInPlace(
                std::vector<molgr::state::MetalCandidateState> *candidates,
                const molgr::config::MolGRConfig &config);

            /// Attach the shared no-metal reconstruction and count discordance features.
            /// @param candidate Candidate electronic assignment being evaluated.
            /// @param no_metal_state Ligand graph and electronic budgets at this stage of reconstruction.
            /// @param config Settings for the current reconstruction or scoring operation.
            /// @return Metal candidate carrying the supplied ligand state and its derived scoring data.
            molgr::state::MetalCandidateState PrepareCandidateWithNoMetalState(
                const molgr::state::MetalCandidateState &candidate,
                const std::shared_ptr<molgr::state::ReconstructionState> &no_metal_state,
                const molgr::config::MolGRConfig &config);

        }
    }
}
